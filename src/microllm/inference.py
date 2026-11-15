"""Авторегрессионная генерация: предсказываем по одному следующему байту."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from microllm.model import Transformer
from microllm.tokenizer import ByteTokenizer


def sample_token(
    logits: NDArray[np.float32],
    rng: np.random.Generator,
    *,
    temperature: float = 0.8,
    top_k: int = 40,
) -> int:
    """Выбрать токен из softmax(logits / T); T=0 даёт argmax."""
    if temperature < 0 or not math.isfinite(temperature):
        raise ValueError("temperature must be finite and nonnegative")

    if top_k < 0:
        raise ValueError("top_k must be nonnegative (0 disables filtering)")

    if logits.ndim != 1 or logits.size != ByteTokenizer.vocab_size:
        raise ValueError("Expected one logit per byte-tokenizer token")

    if not np.isfinite(logits).all():
        raise ValueError("Non-finite logits: check the checkpoint/training")

    # BOS/PAD нужны для входа и служебных операций, но не для вывода.
    scores = logits.astype(np.float64, copy=True)
    scores[ByteTokenizer.bos_id] = -np.inf
    scores[ByteTokenizer.pad_id] = -np.inf

    if temperature == 0:
        return int(np.argmax(scores))

    candidates = np.arange(scores.size)

    if 0 < top_k < scores.size:
        candidates = np.argpartition(scores, -top_k)[-top_k:]

    selected = scores[candidates]

    # Вычитание максимума не меняет softmax и предотвращает overflow exp.
    selected = (selected - np.max(selected)) / temperature
    probabilities = np.exp(selected)
    probabilities /= np.sum(probabilities)

    return int(rng.choice(candidates, p=probabilities))


def generate(
    model: Transformer,
    prompt: str,
    *,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 40,
    seed: int = 42,
    context_length: int | None = None,
) -> str:
    """Продолжить строку; контекст пересчитывается без KV-cache."""
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be nonnegative")

    if model.config.vocab_size != ByteTokenizer.vocab_size:
        raise ValueError("Checkpoint vocabulary does not match tokenizer")

    if temperature < 0 or not math.isfinite(temperature) or top_k < 0:
        raise ValueError("Invalid sampling temperature or top_k")

    window = (
        model.config.context_length
        if context_length is None
        else context_length
    )

    if not 1 <= window <= model.config.context_length:
        raise ValueError("context_length must fit the model context")

    tokenizer = ByteTokenizer()
    token_ids = tokenizer.encode(prompt, add_bos=True)
    rng = np.random.default_rng(seed)

    for _ in range(max_new_tokens):
        # Learned positions нумеруются заново в скользящем окне.
        context = token_ids[-window:]
        inputs = np.array([context], dtype=np.int64)
        logits = model.forward(inputs)[0, -1]
        token = sample_token(logits, rng, temperature=temperature, top_k=top_k)

        if token == tokenizer.eos_id:
            break

        token_ids.append(token)

    # Незавершённая UTF-8 последовательность может дать символ замены.
    return tokenizer.decode(token_ids)
