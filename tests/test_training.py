"""Проверки оптимизатора, формата весов и точного продолжения обучения."""

from __future__ import annotations

import hashlib
import io
import json
import math
import zipfile
from pathlib import Path

import numpy as np
import pytest

from microllm.checkpoint import (
    load_checkpoint,
    read_checkpoint_metadata,
    save_checkpoint,
)
from microllm.model import ModelConfig, Transformer
from microllm.training import (
    AdamW,
    TokenDataset,
    TrainConfig,
    clip_gradients,
    cross_entropy,
    learning_rate_at,
    train,
)


def make_model() -> Transformer:
    return Transformer(
        ModelConfig(
            context_length=8,
            d_model=8,
            n_heads=2,
            n_layers=1,
            d_ff=16,
            seed=7,
        )
    )


def make_data(directory: Path) -> Path:
    directory.mkdir()
    metadata: dict[str, object] = {
        "dtype": "uint16",
        "byteorder": "little",
        "tokenizer": "utf8-bytes-v1",
        "vocab_size": 259,
    }

    for split, pattern in (
        ("train", [97, 98, 99, 256]),
        ("val", [100, 101, 256]),
    ):
        values = np.array(pattern * 50, dtype="<u2")
        payload = values.tobytes()
        (directory / f"{split}.bin").write_bytes(payload)
        metadata["train" if split == "train" else "validation"] = {
            "tokens": len(values),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }

    (directory / "metadata.json").write_text(json.dumps(metadata), "utf-8")
    return directory


def test_adamw_matches_scalar_formula() -> None:
    config = TrainConfig(beta1=0.7, beta2=0.8, weight_decay=0.2)
    optimizer = AdamW(config)
    parameter = np.array([[1.5, -0.5]], dtype=np.float32)
    reference = [1.5, -0.5]
    first = [0.0, 0.0]
    second = [0.0, 0.0]

    for step, values in enumerate(([0.4, -0.8], [-0.2, 0.3]), 1):
        gradient = np.array([values], dtype=np.float32)
        optimizer.update({"weight": parameter}, {"weight": gradient}, 0.01)

        for index, value in enumerate(values):
            first[index] = 0.7 * first[index] + 0.3 * value
            second[index] = 0.8 * second[index] + 0.2 * value * value
            corrected_first = first[index] / (1 - 0.7**step)
            corrected_second = second[index] / (1 - 0.8**step)
            reference[index] = reference[index] * (
                1 - 0.01 * 0.2
            ) - 0.01 * corrected_first / (
                math.sqrt(corrected_second) + config.epsilon
            )

        np.testing.assert_allclose(parameter[0], reference, atol=1e-7)

    assert optimizer.step == 2


def test_gradient_clip_and_schedule() -> None:
    gradients = {"a": np.array([3, 4], dtype=np.float32)}
    assert clip_gradients(gradients, 2) == 5
    np.testing.assert_allclose(gradients["a"], [1.2, 1.6])
    config = TrainConfig(steps=10, warmup_steps=2)
    assert learning_rate_at(1, config) == config.learning_rate / 2
    assert learning_rate_at(2, config) == config.learning_rate
    assert learning_rate_at(10, config) == config.min_learning_rate

    with pytest.raises(FloatingPointError):
        clip_gradients({"a": np.array([np.inf], dtype=np.float32)}, 1)


def test_sampler_and_data_integrity(tmp_path: Path) -> None:
    directory = make_data(tmp_path / "data")
    dataset = TokenDataset(directory, 4)
    tokens, targets = dataset.batch("train", 3, np.random.default_rng(0))
    np.testing.assert_array_equal(tokens[:, 1:], targets[:, :-1])
    assert tokens.dtype == np.int64
    del dataset
    path = directory / "train.bin"
    payload = bytearray(path.read_bytes())
    payload[0] ^= 1
    path.write_bytes(payload)

    with pytest.raises(ValueError, match="metadata mismatch"):
        TokenDataset(directory, 4)


def test_checkpoint_round_trip_and_no_optimizer_allocation(
    tmp_path: Path,
) -> None:
    model = make_model()
    path = tmp_path / "weights.npz"
    save_checkpoint(
        path,
        model,
        optimizer_arrays={
            "m/token_embedding": np.ones_like(
                model.parameters["token_embedding"]
            )
        },
        training_state={"step": 4},
    )
    loaded = load_checkpoint(path)
    assert loaded.optimizer_arrays == {}
    assert loaded.training_state["step"] == 4
    metadata = read_checkpoint_metadata(path)
    assert metadata["config"]["d_model"] == 8

    for name, parameter in model.parameters.items():
        np.testing.assert_array_equal(parameter, loaded.model.parameters[name])


def test_checkpoint_rejects_object_array(tmp_path: Path) -> None:
    path = tmp_path / "bad.npz"
    save_checkpoint(path, make_model())

    with zipfile.ZipFile(path) as archive:
        entries = {name: archive.read(name) for name in archive.namelist()}

    metadata = json.loads(entries["metadata.json"])
    name = metadata["parameters"]["token_embedding"]
    stream = io.BytesIO()
    np.save(stream, np.array([{"unsafe": True}], dtype=object))
    entries[name] = stream.getvalue()

    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in entries.items():
            archive.writestr(name, payload)

    with pytest.raises(ValueError, match="Object arrays"):
        load_checkpoint(path)


def test_resume_matches_uninterrupted_training(tmp_path: Path) -> None:
    data = make_data(tmp_path / "data")
    config = TrainConfig(
        steps=6,
        batch_size=2,
        sequence_length=4,
        learning_rate=0.01,
        min_learning_rate=0.001,
        warmup_steps=1,
        accumulation_steps=2,
        eval_every=2,
        eval_batches=2,
        save_every=3,
    )
    full = train(make_model(), data, tmp_path / "full", config)
    partial = train(
        make_model(), data, tmp_path / "partial", config, stop_after=3
    )
    resumed = train(
        make_model(), data, tmp_path / "resumed", config, resume=partial
    )
    expected = load_checkpoint(full, load_optimizer=True)
    actual = load_checkpoint(resumed, load_optimizer=True)

    for name, parameter in expected.model.parameters.items():
        np.testing.assert_array_equal(parameter, actual.model.parameters[name])

    for name, moment in expected.optimizer_arrays.items():
        np.testing.assert_array_equal(moment, actual.optimizer_arrays[name])

    assert expected.training_state == actual.training_state


def test_training_reduces_next_token_loss() -> None:
    model = make_model()
    tokens = np.array([[97, 98, 97, 98]], dtype=np.int64)
    targets = np.array([[98, 97, 98, 97]], dtype=np.int64)
    initial = model.loss(tokens, targets)
    optimizer = AdamW(TrainConfig(weight_decay=0))

    for _ in range(30):
        model.loss_and_gradients(tokens, targets)
        clip_gradients(model.gradients, 1)
        optimizer.update(model.parameters, model.gradients, 0.03)

    final = model.loss(tokens, targets)
    assert final < 0.25 * initial
    assert cross_entropy(model.forward(tokens), targets) == pytest.approx(
        final, abs=1e-5
    )
