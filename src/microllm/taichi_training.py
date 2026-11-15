"""Обучение GPU-модели Taichi с совместимым форматом checkpoint."""

import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import structlog
import taichi as ti

from microllm.checkpoint import load_checkpoint, save_checkpoint
from microllm.model import Transformer
from microllm.taichi_model import TaichiTransformer
from microllm.training import TokenDataset, TrainConfig, learning_rate_at

LOGGER = structlog.get_logger()


@ti.kernel
def _accumulate(source: ti.types.ndarray(), total: ti.types.ndarray()):  # noqa: ANN202
    for index in ti.grouped(source):
        total[index] += source[index]


@ti.kernel
def _sum_squares(source: ti.types.ndarray(), total: ti.template()):  # noqa: ANN202
    subtotal = ti.cast(0, ti.f64)
    for index in ti.grouped(source):
        value = ti.cast(source[index], ti.f64)
        subtotal += value * value
    total[None] += subtotal


@ti.kernel
def _update(  # noqa: ANN202
    parameter: ti.types.ndarray(),
    gradient: ti.types.ndarray(),
    first: ti.types.ndarray(),
    second: ti.types.ndarray(),
    scale: ti.f32,
    beta1: ti.f32,
    beta2: ti.f32,
    correction1: ti.f32,
    correction2: ti.f32,
    epsilon: ti.f32,
    rate: ti.f32,
    decay: ti.f32,
):
    for index in ti.grouped(parameter):
        grad = gradient[index] * scale
        m = beta1 * first[index] + (1.0 - beta1) * grad
        v = beta2 * second[index] + (1.0 - beta2) * grad * grad
        first[index] = m
        second[index] = v
        parameter[index] = parameter[index] * (1.0 - rate * decay) - rate * (
            (m / correction1) / (ti.sqrt(v / correction2) + epsilon)
        )


class TaichiAdamW:
    """Держать градиенты и моменты оптимизатора в памяти устройства."""

    def __init__(
        self,
        model: TaichiTransformer,
        config: TrainConfig,
        *,
        parallel_accumulation: bool = False,
    ) -> None:
        self.config = config
        self.parallel_accumulation = parallel_accumulation
        self.step = 0
        self.first = {}
        self.second = {}
        self.accumulated = {}
        for name, parameter in model.parameters.items():
            self.first[name] = ti.ndarray(dtype=ti.f32, shape=parameter.shape)
            self.second[name] = ti.ndarray(dtype=ti.f32, shape=parameter.shape)
            self.first[name].fill(0)
            self.second[name].fill(0)
            if config.accumulation_steps > 1 and not parallel_accumulation:
                self.accumulated[name] = ti.ndarray(
                    dtype=ti.f32, shape=parameter.shape
                )
        self.squared_norm = ti.field(dtype=ti.f64, shape=())

    def begin(self) -> None:
        for array in self.accumulated.values():
            array.fill(0)

    def accumulate(self, model: TaichiTransformer) -> None:
        if (
            self.config.accumulation_steps > 1
            and not self.parallel_accumulation
        ):
            for name, gradient in model.gradients.items():
                _accumulate(gradient, self.accumulated[name])

    def update(self, model: TaichiTransformer, rate: float) -> float:
        gradients = (
            self.accumulated
            if self.config.accumulation_steps > 1
            and not self.parallel_accumulation
            else model.gradients
        )
        divisor = (
            1 if self.parallel_accumulation else self.config.accumulation_steps
        )
        self.squared_norm[None] = 0.0
        for gradient in gradients.values():
            _sum_squares(gradient, self.squared_norm)
        norm = math.sqrt(float(self.squared_norm[None]))
        norm /= divisor
        if not math.isfinite(norm):
            raise FloatingPointError("Non-finite gradient norm")
        scale = min(1.0, self.config.grad_clip / (norm + 1e-12))
        scale /= divisor
        self.step += 1
        for name, parameter in model.parameters.items():
            _update(
                parameter,
                gradients[name],
                self.first[name],
                self.second[name],
                scale,
                self.config.beta1,
                self.config.beta2,
                1.0 - self.config.beta1**self.step,
                1.0 - self.config.beta2**self.step,
                self.config.epsilon,
                rate,
                self.config.weight_decay if len(parameter.shape) >= 2 else 0.0,
            )
        return norm

    def export_moments(self) -> dict[str, np.ndarray]:
        result = {
            f"m/{name}": value.to_numpy() for name, value in self.first.items()
        }
        result.update(
            {
                f"v/{name}": value.to_numpy()
                for name, value in self.second.items()
            }
        )
        return result

    def load_moments(self, arrays: dict[str, np.ndarray]) -> None:
        expected = {
            f"{prefix}/{name}" for prefix in ("m", "v") for name in self.first
        }
        if set(arrays) != expected:
            raise ValueError("Checkpoint optimizer moments are incomplete")
        for name in self.first:
            self.first[name].from_numpy(arrays[f"m/{name}"])
            self.second[name].from_numpy(arrays[f"v/{name}"])


def train_taichi(  # noqa: C901
    model: Transformer,
    data_dir: Path,
    output_dir: Path,
    config: TrainConfig,
    *,
    resume: Path | None = None,
    stop_after: int | None = None,
    parallel_accumulation: bool = False,
) -> Path:
    """Обучить модель на инициализированном Taichi-устройстве."""
    if config.sequence_length > model.config.context_length:
        raise ValueError("Training sequence exceeds model context length")
    if stop_after is not None and stop_after < 1:
        raise ValueError("stop_after must be positive")
    dataset = TokenDataset(
        data_dir, config.sequence_length, model.config.vocab_size
    )
    rng = np.random.default_rng(config.seed)
    eval_rng = np.random.default_rng(config.seed + 1)
    loaded = None
    step = 0
    if resume is not None:
        loaded = load_checkpoint(resume, load_optimizer=True, model=model)
        state = loaded.training_state
        if state.get("train_config") != asdict(config):
            raise ValueError(
                "Resume requires the original training configuration"
            )
        if state.get("data_fingerprint") != dataset.fingerprint:
            raise ValueError("Resume data differs from checkpoint data")
        saved_step = state.get("step")
        if (
            not isinstance(saved_step, int)
            or isinstance(saved_step, bool)
            or saved_step < 1
        ):
            raise ValueError(
                "Checkpoint must contain a positive optimizer step"
            )
        step = saved_step
        if step > config.steps:
            raise ValueError(
                "Checkpoint step exceeds configured training horizon"
            )
        rng.bit_generator.state = state["rng_state"]
        eval_rng.bit_generator.state = state["eval_rng_state"]

    gpu_model = TaichiTransformer(model.config, parameters=model.parameters)
    optimizer = TaichiAdamW(
        gpu_model, config, parallel_accumulation=parallel_accumulation
    )
    if loaded is not None:
        optimizer.step = step
        optimizer.load_moments(loaded.optimizer_arrays)
        del loaded

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "checkpoint.npz"
    final_step = min(config.steps, stop_after or config.steps)

    LOGGER.info(
        "training_started",
        backend="taichi",
        start_step=optimizer.step,
        final_step=final_step,
        output_dir=str(output_dir),
        resume=str(resume) if resume is not None else None,
    )

    def save() -> None:
        for name, array in gpu_model.export_parameters().items():
            np.copyto(model.parameters[name], array)
        save_checkpoint(
            checkpoint,
            model,
            optimizer_arrays=optimizer.export_moments(),
            training_state={
                "step": optimizer.step,
                "train_config": asdict(config),
                "data_fingerprint": dataset.fingerprint,
                "rng_state": rng.bit_generator.state,
                "eval_rng_state": eval_rng.bit_generator.state,
            },
        )
        LOGGER.info(
            "checkpoint_saved", step=optimizer.step, path=str(checkpoint)
        )

    with (output_dir / "metrics.jsonl").open("a", encoding="utf-8") as log:
        while optimizer.step < final_step:
            started = time.perf_counter()
            if parallel_accumulation and config.accumulation_steps > 1:
                batches = [
                    dataset.batch("train", config.batch_size, rng)
                    for _ in range(config.accumulation_steps)
                ]
                tokens = np.concatenate([batch[0] for batch in batches])
                targets = np.concatenate([batch[1] for batch in batches])
                loss = gpu_model.loss_and_gradients(tokens, targets)
            else:
                optimizer.begin()
                loss = 0.0
                for _ in range(config.accumulation_steps):
                    tokens, targets = dataset.batch(
                        "train", config.batch_size, rng
                    )
                    loss += gpu_model.loss_and_gradients(tokens, targets)
                    optimizer.accumulate(gpu_model)
                loss /= config.accumulation_steps
            if not math.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            rate = learning_rate_at(optimizer.step + 1, config)
            norm = optimizer.update(gpu_model, rate)
            ti.sync()
            metrics: dict[str, Any] = {
                "step": optimizer.step,
                "train_loss": loss,
                "learning_rate": rate,
                "gradient_norm": norm,
                "seconds": time.perf_counter() - started,
                "tokens": config.batch_size
                * config.sequence_length
                * config.accumulation_steps,
            }
            if optimizer.step % config.eval_every == 0:
                losses = []
                for _ in range(config.eval_batches):
                    tokens, targets = dataset.batch(
                        "val", config.batch_size, eval_rng
                    )
                    losses.append(gpu_model.loss(tokens, targets))
                metrics["val_loss"] = sum(losses) / len(losses)
            line = json.dumps(metrics, allow_nan=False)
            log.write(line + "\n")
            log.flush()
            LOGGER.info("training_step", backend="taichi", **metrics)
            if (
                optimizer.step % config.save_every == 0
                and optimizer.step < final_step
            ):
                save()
    save()
    return checkpoint
