"""Cross-check the experimental backend against the NumPy implementation."""

from pathlib import Path

import numpy as np
import pytest

ti = pytest.importorskip("taichi")

from microllm.model import ModelConfig, Transformer  # noqa: E402
from microllm.taichi_model import TaichiTransformer  # noqa: E402


def test_loss_and_all_gradients_match_numpy(tmp_path: Path) -> None:
    ti.init(arch=ti.cpu, offline_cache_file_path=str(tmp_path / "cache"))
    config = ModelConfig(
        vocab_size=13,
        context_length=8,
        d_model=8,
        n_heads=2,
        n_layers=2,
        d_ff=16,
    )
    reference = Transformer(config)
    actual = TaichiTransformer(config)
    tokens = np.array([[1, 2, 3, 2], [3, 2, 1, 0]], dtype=np.int64)
    targets = np.array([[2, 3, 2, 1], [2, 1, 0, 4]], dtype=np.int64)

    expected_loss = reference.loss_and_gradients(tokens, targets)
    actual_loss = actual.loss_and_gradients(tokens, targets)

    assert actual_loss == pytest.approx(expected_loss, abs=1e-6)
    assert actual.loss(tokens, targets) == pytest.approx(
        expected_loss, abs=1e-6
    )
    assert set(actual.gradients) == set(reference.gradients)
    for name, expected in reference.gradients.items():
        np.testing.assert_allclose(
            actual.gradients[name].to_numpy(),
            expected,
            rtol=3e-4,
            atol=2e-7,
            err_msg=name,
        )
