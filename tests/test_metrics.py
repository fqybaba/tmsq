import numpy as np
import torch

from tmsq.metrics import (
    dimensionality,
    dims_per_feat,
    feats_per_dim,
    feature_norms,
    is_pentagon,
    margin,
    max_overlap,
    represented,
    superposition_score,
)
from tmsq.model import ToyModel


def pentagon_W() -> torch.Tensor:
    ang = 2 * np.pi * np.arange(5) / 5
    W = np.stack([np.cos(ang), np.sin(ang)])  # [m=2, n=5], unit columns
    return torch.tensor(W[None], dtype=torch.float64)


def test_dimensionality_sanity_values():
    D = dimensionality(pentagon_W())
    assert np.allclose(D, 0.4, atol=1e-9)
    antipodal = torch.tensor([[[1.0, -1.0]]])  # m=1, n=2
    assert np.allclose(dimensionality(antipodal), 0.5)
    orthogonal = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])  # m=2, n=2
    assert np.allclose(dimensionality(orthogonal), 1.0)


def test_dimensionality_nan_for_dead_features_and_sum_rule():
    W = torch.tensor([[[1.0, 0.0, 0.01], [0.0, 1.0, 0.0]]])
    D = dimensionality(W)
    assert np.isnan(D[0, 2])
    assert np.allclose(D[0, :2], 1.0, atol=1e-3)  # feature 0 overlaps the dead feature by 0.01
    assert np.isclose(np.nansum(D), 2.0, atol=1e-3)  # sums to m when packed efficiently


def test_feats_per_dim_and_dstar():
    W = pentagon_W()
    assert np.isclose(feats_per_dim(W)[0], 5 / 2)
    assert np.isclose(dims_per_feat(W)[0], 2 / 5)


def test_running_example_pentagon_bias_leak():
    """CLAUDE.md running example: pentagon, b = -0.25, x = (0, 0.7, 0, 0, 0)."""
    W = pentagon_W().float()
    b = torch.full((1, 5), -0.25)
    x = torch.zeros(1, 1, 5)
    x[0, 0, 1] = 0.7
    out = ToyModel.run(W, W, b, x)[0, 0]
    assert torch.allclose(out[[0, 2, 3, 4]], torch.zeros(4))
    assert abs(out[1].item() - 0.45) < 1e-6
    assert np.allclose(max_overlap(W), np.cos(2 * np.pi / 5), atol=1e-6)  # ~0.309
    m = margin(W, b)
    assert np.allclose(m, 0.25 - np.cos(2 * np.pi / 5), atol=1e-6)  # ~ -0.059, NEGATIVE
    assert (m < 0).all()


def test_margin_positive_when_bias_blocks_leak():
    W = pentagon_W()
    b = torch.full((1, 5), -0.5)
    assert (margin(W, b) > 0).all()
    orthogonal = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    assert np.allclose(margin(orthogonal, torch.zeros(1, 2)), 0.0)  # max(0, 0) and b = 0


def test_norms_represented_and_superposition_score():
    W = torch.tensor([[[1.0, 0.0, 0.3], [0.0, 1.0, 0.0]]])
    assert np.allclose(feature_norms(W), [[1.0, 1.0, 0.3]])
    assert represented(W).tolist() == [[True, True, False]]
    s = superposition_score(W)
    assert np.isclose(s[0, 0], 0.09) and np.isclose(s[0, 1], 0.0)  # (1*0.3)^2 ; orthogonal
    assert np.isclose(superposition_score(pentagon_W())[0, 0], 2 * np.cos(2 * np.pi / 5) ** 2 + 2 * 0.809017**2)


def test_is_pentagon():
    assert is_pentagon(pentagon_W()).tolist() == [True]
    W = pentagon_W().clone()
    W[0, :, 0] = 0.0  # kill one feature -> digon-like, not a pentagon
    assert is_pentagon(W).tolist() == [False]
