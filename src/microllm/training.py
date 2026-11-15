"""Ручной AdamW, выборка батчей и воспроизводимый цикл обучения."""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from microllm.checkpoint import load_checkpoint, save_checkpoint
from microllm.model import Transformer

type FloatArray = NDArray[np.float32]
type IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 1000
    batch_size: int = 1
    sequence_length: int = 128
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_steps: int = 100
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    grad_clip: float = 1.0
    accumulation_steps: int = 1
    eval_every: int = 100
    eval_batches: int = 4
    save_every: int = 100
    seed: int = 42

    def __post_init__(self) -> None:
        """Проверить область допустимых значений гиперпараметров."""
        positive = (
            self.steps,
            self.batch_size,
            self.sequence_length,
            self.accumulation_steps,
            self.eval_every,
            self.eval_batches,
            self.save_every,
        )

        if any(value < 1 for value in positive) or self.warmup_steps < 0:
            raise ValueError("Steps, sizes and intervals must be positive")

        values = (
            self.learning_rate,
            self.min_learning_rate,
            self.weight_decay,
            self.beta1,
            self.beta2,
            self.epsilon,
            self.grad_clip,
        )

        if not all(math.isfinite(value) for value in values):
            raise ValueError("Training hyperparameters must be finite")

        if not 0 <= self.min_learning_rate <= self.learning_rate:
            raise ValueError("Require 0 <= min_learning_rate <= learning_rate")

        if self.learning_rate <= 0 or self.epsilon <= 0 or self.grad_clip <= 0:
            raise ValueError(
                "Learning rate, epsilon and clip must be positive"
            )

        if self.weight_decay < 0 or not 0 <= self.beta1 < 1:
            raise ValueError("Invalid weight decay or beta1")

        if not 0 <= self.beta2 < 1:
            raise ValueError("Invalid beta2")


def learning_rate_at(step: int, config: TrainConfig) -> float:
    """Номер шага начинается с 1; после warmup работает cosine decay."""
    if step < 1:
        raise ValueError("Optimizer step must start at 1")

    if config.warmup_steps and step <= config.warmup_steps:
        return config.learning_rate * step / config.warmup_steps

    decay_steps = max(config.steps - config.warmup_steps, 1)
    progress = min(max((step - config.warmup_steps) / decay_steps, 0), 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))

    return config.min_learning_rate + cosine * (
        config.learning_rate - config.min_learning_rate
    )


def clip_gradients(gradients: dict[str, FloatArray], maximum: float) -> float:
    if maximum <= 0 or not math.isfinite(maximum):
        raise ValueError("Gradient clip must be finite and positive")

    # ||g||₂ = sqrt(sum_i g_i²). Сумма float64 не теряет малые слагаемые.
    squared = sum(
        float(
            np.einsum(
                "i,i->", gradient.ravel(), gradient.ravel(), dtype=np.float64
            )
        )
        for gradient in gradients.values()
    )
    norm = math.sqrt(squared)

    if not math.isfinite(norm):
        raise FloatingPointError("Non-finite gradient norm")

    scale = min(1.0, maximum / (norm + 1e-12))

    for gradient in gradients.values():
        gradient *= np.float32(scale)

    return norm


class AdamW:
    def __init__(self, config: TrainConfig) -> None:
        self.config = config
        self.step = 0
        self.moments: dict[str, FloatArray] = {}

    def update(
        self,
        parameters: dict[str, FloatArray],
        gradients: dict[str, FloatArray],
        learning_rate: float,
    ) -> None:
        if parameters.keys() != gradients.keys():
            raise ValueError("Every parameter must have a gradient")

        self.step += 1
        config = self.config
        correction1 = 1.0 - config.beta1**self.step
        correction2 = 1.0 - config.beta2**self.step

        for name, parameter in parameters.items():
            gradient = gradients[name]

            if gradient.shape != parameter.shape:
                raise ValueError(f"Gradient shape mismatch: {name}")

            if f"m/{name}" not in self.moments:
                self.moments[f"m/{name}"] = np.zeros_like(parameter)
                self.moments[f"v/{name}"] = np.zeros_like(parameter)

            first = self.moments[f"m/{name}"]
            second = self.moments[f"v/{name}"]

            # m_t = β₁m_(t-1)+(1-β₁)g; v_t = β₂v_(t-1)+(1-β₂)g².
            first *= config.beta1
            first += np.float32(1 - config.beta1) * gradient
            second *= config.beta2
            second += np.float32(1 - config.beta2) * gradient * gradient

            # Деление на (1-β^t) исправляет начальное смещение к нулю.
            denominator = np.sqrt(second / correction2) + config.epsilon
            update = (first / correction1) / denominator

            # Decoupled weight decay: θ ← (1-ηλ)θ - η m̂/(sqrt(v̂)+ε).
            # Векторы RMSNorm не штрафуем, как обычно делают для нормировок.
            if parameter.ndim >= 2:
                parameter *= 1 - learning_rate * config.weight_decay

            parameter -= np.float32(learning_rate) * update


class TokenDataset:
    def __init__(
        self, directory: Path, sequence_length: int, vocab_size: int = 259
    ) -> None:
        if sequence_length < 1:
            raise ValueError("Sequence length must be positive")

        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.arrays: dict[str, NDArray[np.uint16]] = {}
        self.fingerprint: dict[str, Any] = {}
        metadata = json.loads(
            (directory / "metadata.json").read_text(encoding="utf-8")
        )

        if (
            metadata.get("dtype") != "uint16"
            or metadata.get("byteorder") != "little"
            or metadata.get("tokenizer") != "utf8-bytes-v1"
            or metadata.get("vocab_size") != vocab_size
        ):
            raise ValueError("Dataset encoding does not match model")

        for split in ("train", "val"):
            path = directory / f"{split}.bin"
            size = path.stat().st_size

            if size % 2 or size // 2 <= sequence_length:
                raise ValueError(f"Invalid or too short token file: {path}")

            self.arrays[split] = np.memmap(path, mode="r", dtype="<u2")
            checksum = _sha256(path)
            details = metadata["train" if split == "train" else "validation"]

            if details["tokens"] != size // 2 or details["sha256"] != checksum:
                raise ValueError(f"Dataset metadata mismatch: {path}")

            self.fingerprint[split] = {"bytes": size, "sha256": checksum}

    def batch(
        self,
        split: Literal["train", "val"],
        batch_size: int,
        rng: np.random.Generator,
    ) -> tuple[IntArray, IntArray]:
        if batch_size < 1:
            raise ValueError("Batch size must be positive")

        source = self.arrays[split]
        length = self.sequence_length

        # Вход x[t] предсказывает следующий байт y[t] = x[t+1].
        starts = rng.integers(0, len(source) - length, size=batch_size)
        tokens = np.empty((batch_size, length), dtype=np.int64)
        targets = np.empty_like(tokens)

        for index, start in enumerate(starts):
            tokens[index] = source[start : start + length]
            targets[index] = source[start + 1 : start + length + 1]

        if np.any(tokens >= self.vocab_size) or np.any(
            targets >= self.vocab_size
        ):
            raise ValueError("Token ID exceeds model vocabulary")

        return tokens, targets


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest()


def cross_entropy(logits: FloatArray, targets: IntArray) -> float:
    if logits.shape[:-1] != targets.shape:
        raise ValueError("Logits and target shapes do not match")

    flat = logits.reshape(-1, logits.shape[-1])
    labels = targets.ravel()

    if np.any(labels < 0) or np.any(labels >= flat.shape[-1]):
        raise ValueError("Target ID exceeds logits vocabulary")

    # log(sum(exp(z-max(z))))+max(z)-z_target избегает overflow.
    maximum = np.max(flat, axis=-1)
    normalizer = np.log(np.sum(np.exp(flat - maximum[:, None]), axis=-1))
    losses = normalizer + maximum - flat[np.arange(len(labels)), labels]

    return float(np.mean(losses, dtype=np.float64))


def evaluate(
    model: Transformer,
    dataset: TokenDataset,
    config: TrainConfig,
    rng: np.random.Generator,
) -> float:
    losses: list[float] = []

    for _ in range(config.eval_batches):
        tokens, targets = dataset.batch("val", config.batch_size, rng)
        losses.append(model.loss(tokens, targets))

    return sum(losses) / len(losses)


def _accumulate_gradients(
    model: Transformer,
    dataset: TokenDataset,
    config: TrainConfig,
    rng: np.random.Generator,
) -> float:
    accumulated: dict[str, FloatArray] = {}
    total_loss = 0.0

    for _ in range(config.accumulation_steps):
        tokens, targets = dataset.batch("train", config.batch_size, rng)
        total_loss += model.loss_and_gradients(tokens, targets)

        if config.accumulation_steps > 1:
            for name, gradient in model.gradients.items():
                if name not in accumulated:
                    accumulated[name] = gradient.copy()
                else:
                    accumulated[name] += gradient

    if config.accumulation_steps > 1:
        for name, gradient in model.gradients.items():
            np.divide(
                accumulated[name], config.accumulation_steps, out=gradient
            )

    return total_loss / config.accumulation_steps


def _restore(
    path: Path,
    model: Transformer,
    optimizer: AdamW,
    dataset: TokenDataset,
    rng: np.random.Generator,
    eval_rng: np.random.Generator,
) -> None:
    loaded = load_checkpoint(path, load_optimizer=True, model=model)
    state = loaded.training_state

    if state.get("train_config") != asdict(optimizer.config):
        raise ValueError("Resume requires the original training configuration")

    if state.get("data_fingerprint") != dataset.fingerprint:
        raise ValueError("Resume data differs from checkpoint data")

    step = state.get("step")

    if not isinstance(step, int) or isinstance(step, bool) or step < 1:
        raise ValueError("Checkpoint must contain a positive optimizer step")

    optimizer.step = step
    expected = {
        f"{prefix}/{name}"
        for prefix in ("m", "v")
        for name in model.parameters
    }

    if set(loaded.optimizer_arrays) != expected:
        raise ValueError("Checkpoint optimizer moments are incomplete")

    optimizer.moments = loaded.optimizer_arrays
    rng.bit_generator.state = state["rng_state"]
    eval_rng.bit_generator.state = state["eval_rng_state"]


def _save_training(
    path: Path,
    model: Transformer,
    optimizer: AdamW,
    dataset: TokenDataset,
    rng: np.random.Generator,
    eval_rng: np.random.Generator,
) -> None:
    save_checkpoint(
        path,
        model,
        optimizer_arrays=optimizer.moments,
        training_state={
            "step": optimizer.step,
            "train_config": asdict(optimizer.config),
            "data_fingerprint": dataset.fingerprint,
            "rng_state": rng.bit_generator.state,
            "eval_rng_state": eval_rng.bit_generator.state,
        },
    )


def train(
    model: Transformer,
    data_dir: Path,
    output_dir: Path,
    config: TrainConfig,
    *,
    resume: Path | None = None,
    stop_after: int | None = None,
) -> Path:
    if config.sequence_length > model.config.context_length:
        raise ValueError("Training sequence exceeds model context length")

    if stop_after is not None and stop_after < 1:
        raise ValueError("stop_after must be positive")

    dataset = TokenDataset(
        data_dir,
        config.sequence_length,
        model.config.vocab_size,
    )
    rng = np.random.default_rng(config.seed)

    # Валидация не изменяет последовательность обучающих батчей.
    eval_rng = np.random.default_rng(config.seed + 1)
    optimizer = AdamW(config)

    if resume is not None:
        _restore(resume, model, optimizer, dataset, rng, eval_rng)

    if optimizer.step > config.steps:
        raise ValueError("Checkpoint step exceeds configured training horizon")

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "checkpoint.npz"
    final_step = min(config.steps, stop_after or config.steps)

    with (output_dir / "metrics.jsonl").open("a", encoding="utf-8") as log:
        while optimizer.step < final_step:
            started = time.perf_counter()
            loss = _accumulate_gradients(model, dataset, config, rng)

            if not math.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")

            norm = clip_gradients(model.gradients, config.grad_clip)
            rate = learning_rate_at(optimizer.step + 1, config)

            optimizer.update(model.parameters, model.gradients, rate)

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
                metrics["val_loss"] = evaluate(
                    model, dataset, config, eval_rng
                )

            line = json.dumps(metrics, allow_nan=False)

            log.write(line + "\n")
            log.flush()
            print(line, flush=True)

            if (
                optimizer.step % config.save_every == 0
                and optimizer.step < final_step
            ):
                _save_training(
                    checkpoint, model, optimizer, dataset, rng, eval_rng
                )

    _save_training(checkpoint, model, optimizer, dataset, rng, eval_rng)

    return checkpoint
