"""Проверка GPU-оптимизатора относительно NumPy."""

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

ti = pytest.importorskip("taichi")

from test_training import make_data, make_model  # noqa: E402

from microllm.checkpoint import load_checkpoint  # noqa: E402
from microllm.taichi_training import TaichiAdamW, train_taichi  # noqa: E402
from microllm.training import AdamW, TrainConfig, clip_gradients, train  # noqa: E402


def test_optimizer_matches_numpy(tmp_path: Path) -> None:
    ti.init(
        arch=ti.cpu,
        offline_cache=False,
        offline_cache_file_path=str(tmp_path / "taichi-cache"),
    )
    config = TrainConfig(accumulation_steps=2, grad_clip=0.3)
    parameters = {
        "matrix": np.array([[1.0, -2.0], [0.4, 0.5]], dtype=np.float32),
        "norm": np.array([0.3, -0.1], dtype=np.float32),
    }
    batches = [
        {
            "matrix": np.array([[1.0, 2.0], [-1.0, 0.5]], dtype=np.float32),
            "norm": np.array([0.2, -0.4], dtype=np.float32),
        },
        {
            "matrix": np.array([[0.1, -1.0], [0.4, 0.3]], dtype=np.float32),
            "norm": np.array([0.7, -0.3], dtype=np.float32),
        },
    ]

    class Model:
        def __init__(self) -> None:
            self.parameters = {}
            self.gradients = {}
            for name, value in parameters.items():
                self.parameters[name] = ti.ndarray(
                    dtype=ti.f32, shape=value.shape
                )
                self.parameters[name].from_numpy(value)
                self.gradients[name] = ti.ndarray(
                    dtype=ti.f32, shape=value.shape
                )

    model = cast(Any, Model())
    gpu = TaichiAdamW(model, config)
    cpu = AdamW(config)
    for _ in range(2):
        gpu.begin()
        for batch in batches:
            for name, value in batch.items():
                model.gradients[name].from_numpy(value)
            gpu.accumulate(model)
        gradients = {
            name: np.asarray(
                (batches[0][name] + batches[1][name]) / 2, dtype=np.float32
            )
            for name in parameters
        }
        expected_norm = clip_gradients(gradients, config.grad_clip)
        cpu.update(parameters, gradients, 0.001)
        actual_norm = gpu.update(model, 0.001)
        np.testing.assert_allclose(actual_norm, expected_norm, rtol=1e-6)
        for name in parameters:
            np.testing.assert_allclose(
                model.parameters[name].to_numpy(),
                parameters[name],
                rtol=1e-6,
                atol=1e-7,
            )
    for name, value in gpu.export_moments().items():
        np.testing.assert_allclose(
            value, cpu.moments[name], rtol=1e-5, atol=1e-7
        )
    ti.reset()


def test_training_matches_cpu_and_resume(tmp_path: Path) -> None:
    ti.init(
        arch=ti.cpu,
        offline_cache=False,
        offline_cache_file_path=str(tmp_path / "taichi-cache"),
    )
    data = make_data(tmp_path / "data")
    config = TrainConfig(
        steps=3,
        batch_size=2,
        sequence_length=4,
        warmup_steps=1,
        accumulation_steps=2,
        eval_every=2,
        eval_batches=1,
        save_every=2,
    )
    cpu = train(make_model(), data, tmp_path / "cpu", config)
    cpu_partial = train(
        make_model(), data, tmp_path / "cpu-partial", config, stop_after=1
    )
    full = train_taichi(make_model(), data, tmp_path / "full", config)
    partial = train_taichi(
        make_model(), data, tmp_path / "partial", config, stop_after=1
    )
    resumed = train_taichi(
        make_model(), data, tmp_path / "resume", config, resume=partial
    )
    resumed_cpu = train_taichi(
        make_model(), data, tmp_path / "cpu-resume", config, resume=cpu_partial
    )
    expected = load_checkpoint(full, load_optimizer=True)
    actual = load_checkpoint(resumed, load_optimizer=True)
    numpy_reference = load_checkpoint(cpu, load_optimizer=True)
    resumed_cpu_state = load_checkpoint(resumed_cpu, load_optimizer=True)
    for name, value in expected.model.parameters.items():
        np.testing.assert_allclose(
            actual.model.parameters[name], value, rtol=1e-5, atol=1e-7
        )
        np.testing.assert_allclose(
            numpy_reference.model.parameters[name], value, rtol=1e-3, atol=3e-6
        )
        np.testing.assert_allclose(
            resumed_cpu_state.model.parameters[name],
            value,
            rtol=1e-3,
            atol=3e-6,
        )
    for name, value in expected.optimizer_arrays.items():
        np.testing.assert_allclose(
            actual.optimizer_arrays[name], value, rtol=1e-5, atol=1e-7
        )
    assert expected.training_state == actual.training_state
    ti.reset()


def test_parallel_accumulation_matches_and_resumes(tmp_path: Path) -> None:
    ti.init(
        arch=ti.cpu,
        offline_cache=False,
        offline_cache_file_path=str(tmp_path / "taichi-cache"),
    )
    data = make_data(tmp_path / "data")
    config = TrainConfig(
        steps=3,
        batch_size=2,
        sequence_length=4,
        warmup_steps=1,
        accumulation_steps=3,
        eval_every=2,
        eval_batches=1,
        save_every=2,
    )
    sequential = train_taichi(make_model(), data, tmp_path / "seq", config)
    grouped = train_taichi(
        make_model(),
        data,
        tmp_path / "grouped",
        config,
        parallel_accumulation=True,
    )
    partial = train_taichi(
        make_model(), data, tmp_path / "partial", config, stop_after=1
    )
    resumed = train_taichi(
        make_model(),
        data,
        tmp_path / "resumed",
        config,
        resume=partial,
        parallel_accumulation=True,
    )
    expected = load_checkpoint(sequential, load_optimizer=True)
    for path in (grouped, resumed):
        actual = load_checkpoint(path, load_optimizer=True)
        assert actual.training_state == expected.training_state
        for name, value in expected.model.parameters.items():
            np.testing.assert_allclose(
                actual.model.parameters[name], value, rtol=1e-3, atol=3e-6
            )
        for name, value in expected.optimizer_arrays.items():
            np.testing.assert_allclose(
                actual.optimizer_arrays[name], value, rtol=1e-3, atol=3e-6
            )
    original = [
        json.loads(line)
        for line in (tmp_path / "seq" / "metrics.jsonl")
        .read_text()
        .splitlines()
    ]
    combined = [
        json.loads(line)
        for line in (tmp_path / "grouped" / "metrics.jsonl")
        .read_text()
        .splitlines()
    ]
    for left, right in zip(original, combined, strict=True):
        assert right["tokens"] == left["tokens"]
        assert right["train_loss"] == pytest.approx(
            left["train_loss"], abs=1e-6
        )
        assert right["gradient_norm"] == pytest.approx(
            left["gradient_norm"], rel=1e-4, abs=1e-6
        )
    ti.reset()
