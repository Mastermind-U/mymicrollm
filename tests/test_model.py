"""Проверки математических свойств и численного градиента всей модели."""

from __future__ import annotations

import numpy as np
import pytest

from microllm.model import (
    ModelConfig,
    Transformer,
    estimate_parameter_count,
    rms_norm,
    rms_norm_backward,
    sigmoid,
)


def small_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=11,
        context_length=5,
        d_model=8,
        n_heads=2,
        n_layers=2,
        d_ff=12,
        seed=7,
    )


def test_parameter_count_without_allocating_large_model() -> None:
    assert estimate_parameter_count(ModelConfig()) == 250_392_576

    model = Transformer(small_config())

    assert model.parameter_count() == estimate_parameter_count(model.config)
    assert not model.gradients


def test_causal_attention_cannot_read_future_tokens() -> None:
    model = Transformer(small_config())
    first = np.array([[1, 2, 3, 4]], dtype=np.int64)
    second = np.array([[1, 2, 9, 8]], dtype=np.int64)

    np.testing.assert_array_equal(
        model.forward(first)[:, :2],
        model.forward(second)[:, :2],
    )

    assert not model.gradients


def test_all_parameter_gradients_against_finite_differences() -> None:
    """Численная dL/dw=(L(w+h)-L(w-h))/(2h) проверяет правило цепочки."""
    model = Transformer(small_config())
    rng = np.random.default_rng(61)

    # Чуть большие матрицы дают заметные градиенты даже Q/K и SwiGLU.
    for name, parameter in model.parameters.items():
        if "norm" not in name:
            parameter *= 8

    tokens = np.array([[1, 2, 1, 3], [4, 1, 5, 4]], dtype=np.int64)
    targets = np.array([[2, 1, 3, 6], [1, 5, 4, 7]], dtype=np.int64)
    loss = model.loss_and_gradients(tokens, targets)

    assert np.isfinite(loss)
    assert set(model.gradients) == set(model.parameters)

    epsilon = 0.002

    for name, parameter in model.parameters.items():
        analytic = model.gradients[name]

        assert analytic.shape == parameter.shape
        assert analytic.dtype == np.float32

        # В каждом массиве проверяем сильнейшую производную и случайную.
        indices = [
            int(np.argmax(np.abs(analytic))),
            int(rng.integers(parameter.size)),
        ]

        for flat_index in indices:
            index = np.unravel_index(flat_index, parameter.shape)
            original = float(parameter[index])
            parameter[index] = original + epsilon
            plus = model.loss(tokens, targets)
            parameter[index] = original - epsilon
            minus = model.loss(tokens, targets)
            parameter[index] = original
            numeric = (plus - minus) / (2 * epsilon)

            np.testing.assert_allclose(
                analytic[index],
                numeric,
                rtol=0.025,
                atol=0.00012,
                err_msg=f"Ошибка ручной производной: {name}{index}",
            )


def test_tiny_gradient_descent_reduces_loss() -> None:
    model = Transformer(small_config())
    tokens = np.array([[1, 2, 1, 2]], dtype=np.int64)
    targets = np.array([[2, 1, 2, 1]], dtype=np.int64)
    initial = model.loss(tokens, targets)

    for _ in range(40):
        model.loss_and_gradients(tokens, targets)

        for name, parameter in model.parameters.items():
            parameter -= 0.02 * model.gradients[name]

    assert model.loss(tokens, targets) < initial * 0.5


def test_extreme_sigmoid_and_near_zero_norm_stay_finite() -> None:
    values = np.array([[[-1000, 0, 1000]]], dtype=np.float32)

    with np.errstate(over="raise", invalid="raise"):
        result = sigmoid(values)

    np.testing.assert_array_equal(result, [[[0, 0.5, 1]]])
    values.fill(0)

    scale = np.ones(3, dtype=np.float32)
    normalized, cache = rms_norm(values, scale, 1e-5)
    gradient, scale_gradient = rms_norm_backward(
        np.ones_like(values),
        scale,
        cache,
    )

    assert np.isfinite(normalized).all()
    assert np.isfinite(gradient).all()
    assert np.isfinite(scale_gradient).all()


@pytest.mark.parametrize(
    "tokens",
    [
        np.array([[11]], dtype=np.int64),
        np.array([[-1]], dtype=np.int64),
        np.array([[1.2]], dtype=np.float32),
        np.zeros((1, 6), dtype=np.int64),
        np.zeros((0, 1), dtype=np.int64),
    ],
)
def test_invalid_input_is_rejected(tokens: np.ndarray) -> None:
    model = Transformer(small_config())

    with pytest.raises(ValueError):
        model.forward(tokens)
