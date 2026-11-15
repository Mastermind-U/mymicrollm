"""Трансформер: только NumPy и вручную выведенные производные.

Обозначения в комментариях: B — размер пакета, T — длина текста,
D — размер скрытого вектора, H — число голов, F — размер FFN.
Все обучаемые массивы хранятся в float32; готового autograd здесь нет.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np
from numpy.typing import NDArray

type FloatArray = NDArray[np.float32]
type IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class ModelConfig:
    """Размеры модели; конфигурация по умолчанию содержит 250 392 576 весов."""

    vocab_size: int = 259
    context_length: int = 1024
    d_model: int = 1024
    n_heads: int = 16
    n_layers: int = 20
    d_ff: int = 2688
    norm_eps: float = 1e-5
    seed: int = 42

    def __post_init__(self) -> None:
        """Проверить размерности до выделения памяти."""
        dimensions = (
            self.vocab_size,
            self.context_length,
            self.d_model,
            self.n_heads,
            self.n_layers,
            self.d_ff,
        )

        if any(value < 1 for value in dimensions):
            raise ValueError("Все размеры модели должны быть положительными")

        if self.vocab_size < 2:
            raise ValueError("Для предсказания нужно минимум два токена")

        if self.d_model % self.n_heads:
            raise ValueError("d_model должен делиться на n_heads")

        if not np.isfinite(self.norm_eps) or self.norm_eps <= 0:
            raise ValueError("norm_eps должен быть конечным и положительным")

        if self.seed < 0:
            raise ValueError("seed должен быть неотрицательным")

    def to_dict(self) -> dict[str, int | float]:
        return {
            "vocab_size": self.vocab_size,
            "context_length": self.context_length,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "d_ff": self.d_ff,
            "norm_eps": self.norm_eps,
            "seed": self.seed,
        }


def estimate_parameter_count(config: ModelConfig) -> int:
    """Посчитать веса без выделения памяти под большую модель."""
    width = config.d_model
    embeddings = (config.vocab_size + config.context_length) * width
    # Четыре проекции внимания, три матрицы SwiGLU, две RMSNorm.
    block = 4 * width**2 + 3 * width * config.d_ff + 2 * width

    return embeddings + config.n_layers * block + width


def softmax(values: FloatArray) -> FloatArray:
    """Softmax(x)_i = exp(x_i - max(x)) / sum_j exp(x_j - max(x))."""
    shifted = values - np.max(values, axis=-1, keepdims=True)
    exponentials = np.exp(shifted, dtype=np.float32)

    return np.asarray(
        exponentials / np.sum(exponentials, axis=-1, keepdims=True),
        dtype=np.float32,
    )


def sigmoid(values: FloatArray) -> FloatArray:
    """Вычислить сигмоиду без exp(большого положительного числа)."""
    exponentials = np.exp(-np.abs(values), dtype=np.float32)

    return np.where(
        values >= 0,
        1 / (1 + exponentials),
        exponentials / (1 + exponentials),
    )


def cross_entropy(
    logits: FloatArray,
    targets: IntArray,
) -> tuple[float, FloatArray]:
    """Средний -log p(верный токен) и его производная по logits."""
    if logits.ndim != 3 or targets.shape != logits.shape[:2]:
        raise ValueError("targets должны совпадать с первыми осями logits")

    if targets.dtype.kind not in "iu":
        raise ValueError("targets должны содержать целые индексы")

    if targets.size == 0 or np.any(targets < 0):
        raise ValueError("targets должны быть непустыми и неотрицательными")

    if np.any(targets >= logits.shape[-1]):
        raise ValueError("Индекс targets выходит за размер словаря")

    flat = logits.reshape(-1, logits.shape[-1])
    shifted = flat - flat.max(axis=-1, keepdims=True)
    exponentials = np.exp(shifted, dtype=np.float32)
    normalizers = exponentials.sum(axis=-1, keepdims=True)
    positions = np.arange(targets.size)
    correct = shifted[positions, targets.reshape(-1)]
    loss = float(np.mean(np.log(normalizers[:, 0]) - correct))
    # dL/dlogits = (p - one_hot(target)) / (B*T).
    gradient = exponentials / normalizers
    gradient[positions, targets.reshape(-1)] -= 1
    gradient /= targets.size

    return loss, np.asarray(gradient.reshape(logits.shape), dtype=np.float32)


@dataclass
class NormCache:
    values: FloatArray
    inverse_rms: FloatArray


def rms_norm(
    values: FloatArray,
    scale: FloatArray,
    epsilon: float,
) -> tuple[FloatArray, NormCache]:
    """Y = g*x/sqrt(mean(x**2) + epsilon), по последней оси."""
    inverse_rms = np.asarray(
        1
        / np.sqrt(
            np.mean(values * values, axis=-1, keepdims=True) + epsilon,
        ),
        dtype=np.float32,
    )

    return values * inverse_rms * scale, NormCache(values, inverse_rms)


def rms_norm_backward(
    gradient: FloatArray,
    scale: FloatArray,
    cache: NormCache,
) -> tuple[FloatArray, FloatArray]:
    """Применить правило цепочки к RMSNorm, включая обучаемый g."""
    values, inverse = cache.values, cache.inverse_rms
    weighted = gradient * scale
    projection = np.mean(weighted * values, axis=-1, keepdims=True)
    input_gradient = inverse * (weighted - values * inverse**2 * projection)
    reduction_axes = tuple(range(gradient.ndim - 1))
    scale_gradient = np.sum(
        gradient * values * inverse,
        axis=reduction_axes,
    )

    return input_gradient, scale_gradient


@dataclass
class AttentionCache:
    inputs: FloatArray
    query: FloatArray
    key: FloatArray
    value: FloatArray
    probabilities: FloatArray
    combined: FloatArray


@dataclass
class BlockCache:
    attention_norm: NormCache
    attention: AttentionCache
    feedforward_norm: NormCache
    normalized: FloatArray
    gate: FloatArray
    up: FloatArray
    gate_sigmoid: FloatArray
    activated: FloatArray


class Transformer:
    """Decoder-only LM с причинным вниманием и явным backward."""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.parameters: dict[str, FloatArray] = {}
        # На инференсе производные не нужны: не выделяем ещё гигабайт памяти.
        self.gradients: dict[str, FloatArray] = {}
        rng = np.random.default_rng(config.seed)
        width = config.d_model

        def matrix(
            shape: tuple[int, ...], deviation: float = 0.02
        ) -> FloatArray:
            return rng.standard_normal(shape, dtype=np.float32) * deviation

        self.parameters["token_embedding"] = matrix((config.vocab_size, width))
        self.parameters["position_embedding"] = matrix(
            (config.context_length, width),
        )
        residual_deviation = 0.02 / sqrt(2 * config.n_layers)

        for layer in range(config.n_layers):
            prefix = f"layers.{layer}."

            for name in ("attention_norm", "feedforward_norm"):
                self.parameters[prefix + name] = np.ones(
                    width, dtype=np.float32
                )

            for name in ("query", "key", "value"):
                self.parameters[prefix + name] = matrix((width, width))

            self.parameters[prefix + "output"] = matrix(
                (width, width),
                residual_deviation,
            )

            for name in ("gate", "up"):
                self.parameters[prefix + name] = matrix((width, config.d_ff))

            self.parameters[prefix + "down"] = matrix(
                (config.d_ff, width),
                residual_deviation,
            )

        self.parameters["final_norm"] = np.ones(width, dtype=np.float32)

    def parameter_count(self) -> int:
        return sum(parameter.size for parameter in self.parameters.values())

    def _validate_tokens(self, tokens: IntArray) -> None:
        if tokens.ndim != 2 or tokens.dtype.kind not in "iu":
            raise ValueError("tokens должны быть целочисленной матрицей B*T")

        if not tokens.size or tokens.shape[1] > self.config.context_length:
            raise ValueError("Пустой пакет или превышена длина контекста")

        if np.any(tokens < 0) or np.any(tokens >= self.config.vocab_size):
            raise ValueError("Индекс tokens выходит за размер словаря")

    def _heads(self, values: FloatArray) -> FloatArray:
        batch, length, _ = values.shape

        return values.reshape(
            batch,
            length,
            self.config.n_heads,
            -1,
        ).transpose(0, 2, 1, 3)

    def _combine_heads(self, values: FloatArray) -> FloatArray:
        batch, _, length, _ = values.shape

        return values.transpose(0, 2, 1, 3).reshape(
            batch,
            length,
            self.config.d_model,
        )

    def _attention(
        self,
        inputs: FloatArray,
        prefix: str,
    ) -> tuple[FloatArray, AttentionCache]:
        weights = self.parameters
        query = self._heads(inputs @ weights[prefix + "query"])
        key = self._heads(inputs @ weights[prefix + "key"])
        value = self._heads(inputs @ weights[prefix + "value"])
        head_width = self.config.d_model // self.config.n_heads
        # A = softmax(Q K^T / sqrt(D/H) + causal_mask).
        scores = (query @ key.swapaxes(-1, -2)) / sqrt(head_width)
        length = inputs.shape[1]
        future = np.triu(np.ones((length, length), dtype=np.bool_), k=1)
        scores[..., future] = -np.inf
        probabilities = softmax(scores)
        combined = self._combine_heads(probabilities @ value)
        output = combined @ weights[prefix + "output"]

        return output, AttentionCache(
            inputs,
            query,
            key,
            value,
            probabilities,
            combined,
        )

    def _block(
        self,
        inputs: FloatArray,
        prefix: str,
    ) -> tuple[FloatArray, BlockCache]:
        weights = self.parameters
        normalized, attention_norm = rms_norm(
            inputs,
            weights[prefix + "attention_norm"],
            self.config.norm_eps,
        )
        attention, attention_cache = self._attention(normalized, prefix)
        residual = inputs + attention
        normalized, feedforward_norm = rms_norm(
            residual,
            weights[prefix + "feedforward_norm"],
            self.config.norm_eps,
        )
        # SwiGLU(x) = (SiLU(x Wgate) * (x Wup)) Wdown.
        gate = normalized @ weights[prefix + "gate"]
        up = normalized @ weights[prefix + "up"]
        gate_sigmoid = sigmoid(gate)
        activated = gate * gate_sigmoid * up
        output = np.asarray(
            residual + activated @ weights[prefix + "down"],
            dtype=np.float32,
        )

        return output, BlockCache(
            attention_norm,
            attention_cache,
            feedforward_norm,
            normalized,
            gate,
            up,
            gate_sigmoid,
            activated,
        )

    def _forward(
        self,
        tokens: IntArray,
        *,
        retain_cache: bool,
    ) -> tuple[FloatArray, FloatArray, NormCache, list[BlockCache]]:
        self._validate_tokens(tokens)

        hidden = (
            self.parameters["token_embedding"][tokens]
            + self.parameters["position_embedding"][None, : tokens.shape[1]]
        )
        caches: list[BlockCache] = []

        for layer in range(self.config.n_layers):
            hidden, cache = self._block(hidden, f"layers.{layer}.")

            if retain_cache:
                caches.append(cache)

        normalized, norm_cache = rms_norm(
            hidden,
            self.parameters["final_norm"],
            self.config.norm_eps,
        )
        # Одна E используется и для входных embedding, и для выхода (tying).
        logits = normalized @ self.parameters["token_embedding"].T

        return logits, normalized, norm_cache, caches

    def forward(self, tokens: IntArray) -> FloatArray:
        """Вернуть B*T*V logits; внутри нет softmax и случайности."""
        logits, _, _, _ = self._forward(tokens, retain_cache=False)

        return logits

    def loss(self, tokens: IntArray, targets: IntArray) -> float:
        """Посчитать loss без сохранения активаций всех слоёв."""
        logits = self.forward(tokens)
        loss, _ = cross_entropy(logits, targets)

        return loss

    @staticmethod
    def _weight_gradient(
        inputs: FloatArray, gradient: FloatArray
    ) -> FloatArray:
        # Для Y=XW: dW=X^T dY; первые оси — независимые строки.
        return inputs.reshape(-1, inputs.shape[-1]).T @ gradient.reshape(
            -1,
            gradient.shape[-1],
        )

    def _attention_backward(
        self,
        gradient: FloatArray,
        prefix: str,
        cache: AttentionCache,
    ) -> FloatArray:
        weights, gradients = self.parameters, self.gradients
        gradients[prefix + "output"] = self._weight_gradient(
            cache.combined,
            gradient,
        )
        combined_gradient = gradient @ weights[prefix + "output"].T
        context_gradient = self._heads(combined_gradient)
        probabilities_gradient = context_gradient @ cache.value.swapaxes(
            -1, -2
        )
        value_gradient = (
            cache.probabilities.swapaxes(-1, -2) @ context_gradient
        )
        # Jacobian softmax не строим: ds=p*(dp-sum(dp*p)).
        scores_gradient = cache.probabilities * (
            probabilities_gradient
            - np.sum(
                probabilities_gradient * cache.probabilities,
                axis=-1,
                keepdims=True,
            )
        )
        scores_gradient /= sqrt(self.config.d_model // self.config.n_heads)
        query_gradient = scores_gradient @ cache.key
        key_gradient = scores_gradient.swapaxes(-1, -2) @ cache.query
        input_gradient = np.zeros_like(cache.inputs)

        for name, head_gradient in (
            ("query", query_gradient),
            ("key", key_gradient),
            ("value", value_gradient),
        ):
            projection_gradient = self._combine_heads(head_gradient)
            gradients[prefix + name] = self._weight_gradient(
                cache.inputs,
                projection_gradient,
            )
            input_gradient += projection_gradient @ weights[prefix + name].T

        return input_gradient

    def _block_backward(
        self,
        gradient: FloatArray,
        prefix: str,
        cache: BlockCache,
    ) -> FloatArray:
        weights, gradients = self.parameters, self.gradients
        gradients[prefix + "down"] = self._weight_gradient(
            cache.activated,
            gradient,
        )
        activated_gradient = gradient @ weights[prefix + "down"].T
        # d SiLU(z)/dz = sigmoid(z) * (1 + z*(1-sigmoid(z))).
        gate_gradient = (
            activated_gradient
            * cache.up
            * cache.gate_sigmoid
            * (1 + cache.gate * (1 - cache.gate_sigmoid))
        )
        up_gradient = activated_gradient * cache.gate * cache.gate_sigmoid
        normalized_gradient = np.zeros_like(cache.normalized)

        for name, projection_gradient in (
            ("gate", gate_gradient),
            ("up", up_gradient),
        ):
            gradients[prefix + name] = self._weight_gradient(
                cache.normalized,
                projection_gradient,
            )
            normalized_gradient += (
                projection_gradient @ weights[prefix + name].T
            )

        residual_gradient, norm_gradient = rms_norm_backward(
            normalized_gradient,
            weights[prefix + "feedforward_norm"],
            cache.feedforward_norm,
        )
        gradients[prefix + "feedforward_norm"] = norm_gradient
        # Для y=x+f(x) градиент из короткого пути равен dy.
        residual_gradient += gradient
        normalized_gradient = self._attention_backward(
            residual_gradient,
            prefix,
            cache.attention,
        )
        input_gradient, norm_gradient = rms_norm_backward(
            normalized_gradient,
            weights[prefix + "attention_norm"],
            cache.attention_norm,
        )
        gradients[prefix + "attention_norm"] = norm_gradient

        return input_gradient + residual_gradient

    def loss_and_gradients(self, tokens: IntArray, targets: IntArray) -> float:
        """Посчитать mean loss и заменить self.gradients свежими dL/dW."""
        logits, normalized, norm_cache, caches = self._forward(
            tokens,
            retain_cache=True,
        )
        loss, gradient = cross_entropy(logits, targets)

        self.gradients.clear()

        # Выходная роль E даёт первую часть производной связанных весов.
        embedding_gradient = self._weight_gradient(gradient, normalized)
        hidden_gradient = gradient @ self.parameters["token_embedding"]
        hidden_gradient, norm_gradient = rms_norm_backward(
            hidden_gradient,
            self.parameters["final_norm"],
            norm_cache,
        )
        self.gradients["final_norm"] = norm_gradient

        for layer in reversed(range(self.config.n_layers)):
            hidden_gradient = self._block_backward(
                hidden_gradient,
                f"layers.{layer}.",
                caches[layer],
            )

        # Embedding — выбор строк. Повторяющиеся токены должны СУММИРОВАТЬ
        # вклады; обычное advanced-index += этого не гарантирует.
        np.add.at(
            embedding_gradient,
            tokens.reshape(-1),
            hidden_gradient.reshape(-1, self.config.d_model),
        )

        self.gradients["token_embedding"] = embedding_gradient
        position_gradient = np.zeros_like(
            self.parameters["position_embedding"]
        )
        position_gradient[: tokens.shape[1]] = hidden_gradient.sum(axis=0)
        self.gradients["position_embedding"] = position_gradient

        return loss
