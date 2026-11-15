"""Проверки распределения, специальных токенов и генерации."""

import numpy as np
import pytest

from microllm.inference import generate, sample_token
from microllm.model import ModelConfig, Transformer
from microllm.tokenizer import ByteTokenizer


def test_sampling_filters_service_tokens_and_is_stable() -> None:
    scores = np.full(259, -10000, dtype=np.float32)
    scores[97] = 10000
    scores[257:] = 20000
    rng = np.random.default_rng(42)
    assert sample_token(scores, rng, temperature=0) == 97
    assert sample_token(scores, rng, temperature=0.01, top_k=1) == 97
    assert sample_token(scores, rng, temperature=0.01, top_k=0) == 97


def test_generation_is_reproducible_and_handles_long_utf8_prompt() -> None:
    config = ModelConfig(
        context_length=4, d_model=8, n_heads=2, n_layers=1, d_ff=12
    )
    model = Transformer(config)
    prompt = "Привет, мир!"
    first = generate(model, prompt, max_new_tokens=8, seed=4)
    second = generate(model, prompt, max_new_tokens=8, seed=4)
    assert first == second
    assert first.startswith(prompt)
    assert model.gradients == {}
    assert generate(model, prompt, max_new_tokens=0) == prompt
    assert generate(
        model, prompt, max_new_tokens=1, context_length=2
    ).startswith(prompt)

    with pytest.raises(ValueError, match="context_length"):
        generate(model, prompt, context_length=5)


def test_sampling_top_k_and_temperature_reject_invalid_values() -> None:
    rng = np.random.default_rng(4)
    scores = np.ones(ByteTokenizer.vocab_size, dtype=np.float32)

    for value in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="temperature"):
            sample_token(scores, rng, temperature=value)

    with pytest.raises(ValueError, match="top_k"):
        sample_token(scores, rng, top_k=-1)

    scores[3] = np.nan

    with pytest.raises(ValueError, match="Non-finite"):
        sample_token(scores, rng)
