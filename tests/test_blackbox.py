"""Independent black-box tests for tmsq Phase 0/1 (data, model, metrics, train).

Every expected value below is derived from the project CLAUDE.md definitions only
(sections "Model and data", "Per-feature geometry metrics", design notes 4 and 9,
"Engineering rules"). No implementation body or existing test was consulted.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

import tmsq.metrics as M
from tmsq.data import FeatureSampler
from tmsq.model import ToyModel, instance_seed
from tmsq.train import TrainConfig, importance_weighted_loss, lr_at, plateau_ratio, train

COS72 = (math.sqrt(5.0) - 1.0) / 4.0  # cos(72 deg)  ~ 0.309017
COS144 = -(1.0 + math.sqrt(5.0)) / 4.0  # cos(144 deg) ~ -0.809017


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def pentagon(k: int = 1, scale: float = 1.0) -> np.ndarray:
    """Regular pentagon of unit vectors in R^2, columns i = 0..4 at angle 2*pi*i/5. Shape [k, 2, 5]."""
    ang = 2.0 * np.pi * np.arange(5) / 5.0
    W = np.stack([np.cos(ang), np.sin(ang)]) * scale  # [2, 5]
    return np.repeat(W[None], k, axis=0)


def ref_D(W: np.ndarray, min_norm: float = 0.1) -> np.ndarray:
    """Independent re-derivation of D_i via explicit loops over the Gram matrix."""
    K, m, n = W.shape
    out = np.full((K, n), np.nan)
    for k in range(K):
        for i in range(n):
            wi = W[k, :, i]
            norm = float(np.sqrt(np.sum(wi * wi)))
            if norm < min_norm:
                continue
            denom = 0.0
            for j in range(n):
                denom += float(np.dot(wi, W[k, :, j])) ** 2
            out[k, i] = norm**4 / denom
    return out


def ref_forward(W: np.ndarray, b: np.ndarray, x: np.ndarray) -> np.ndarray:
    """ReLU(W^T W x + b) per instance, float64, explicit loops."""
    K, m, n = W.shape
    B = x.shape[1]
    out = np.zeros((K, B, n))
    for k in range(K):
        for s in range(B):
            h = W[k] @ x[k, s]  # [m]
            pre = W[k].T @ h + b[k]  # [n]
            out[k, s] = np.maximum(pre, 0.0)
    return out


def zero_rate(x: torch.Tensor) -> torch.Tensor:
    """Fraction of exact zeros along the batch axis: [K, batch, n] -> [K, n]."""
    return (x == 0).float().mean(dim=1)


def active_stats(col: torch.Tensor) -> tuple[float, float, float, float]:
    """(mean, second moment, min, max) over the active (nonzero) entries."""
    a = col[col != 0].double()
    return float(a.mean()), float((a * a).mean()), float(a.min()), float(a.max())


# --------------------------------------------------------------------------------------
# 1. data: sparsity rate and conditional Uniform[0,1] distribution
# --------------------------------------------------------------------------------------
N_BIG = 100_000
RATE_TOL = 0.01  # ~7 binomial SEs at N = 1e5, p = 0.3
MOM_TOL = 0.01  # ~6 SEs of the uniform mean / second moment at ~30k active samples


def test_sample_shape_and_dtype():
    s = FeatureSampler(5, 0.3, n_instances=2, seed=0)
    for batch in (1, 7, 64, 1001):
        x = s.sample(batch)
        assert tuple(x.shape) == (2, batch, 5), (batch, tuple(x.shape))
        assert torch.isfinite(x).all()


def test_sparsity_rate_scalar_p():
    p = 0.3
    x = FeatureSampler(5, p, n_instances=2, seed=1).sample(N_BIG)
    zr = zero_rate(x)  # [2, 5]
    assert torch.allclose(zr, torch.full_like(zr, 1.0 - p), atol=RATE_TOL), zr
    # Uniform[0,1] given active: mean 1/2, second moment 1/3, 0 < x < 1
    for k in range(2):
        for i in range(5):
            mean, m2, lo, hi = active_stats(x[k, :, i])
            assert abs(mean - 0.5) < MOM_TOL, (k, i, mean)
            assert abs(m2 - 1.0 / 3.0) < MOM_TOL, (k, i, m2)
            assert lo > 0.0 and hi < 1.0, (k, i, lo, hi)


def test_sparsity_rate_per_feature_vector_including_p0_and_p1():
    p = torch.tensor([1.0, 0.5, 0.0, 0.25])
    x = FeatureSampler(4, p, n_instances=2, seed=2).sample(N_BIG)
    zr = zero_rate(x)  # [2, 4]
    expected = (1.0 - p)[None].expand(2, 4)
    assert torch.allclose(zr, expected, atol=RATE_TOL), zr
    # p = 0 column is exactly zero everywhere; p = 1 column is (almost surely) never zero
    assert torch.all(x[:, :, 2] == 0)
    assert float((x[:, :, 0] == 0).float().mean()) < 1e-5
    mean, m2, lo, hi = active_stats(x[:, :, 0])
    assert abs(mean - 0.5) < MOM_TOL and abs(m2 - 1.0 / 3.0) < MOM_TOL and hi < 1.0


def test_sparsity_rate_all_p1_and_all_p0():
    x1 = FeatureSampler(3, 1.0, n_instances=2, seed=3).sample(20_000)
    assert tuple(x1.shape) == (2, 20_000, 3)
    assert float((x1 == 0).float().mean()) < 1e-5
    assert float(x1.max()) < 1.0 and float(x1.min()) >= 0.0
    x0 = FeatureSampler(3, 0.0, n_instances=2, seed=3).sample(20_000)
    assert tuple(x0.shape) == (2, 20_000, 3)
    assert torch.all(x0 == 0)


def test_sparsity_rate_per_instance_per_feature_Kn():
    p = torch.tensor([[0.9, 0.1, 0.5], [0.2, 0.8, 1.0]])  # [K=2, n=3], distinct rows
    x = FeatureSampler(3, p, n_instances=2, seed=4).sample(N_BIG)
    zr = zero_rate(x)
    assert torch.allclose(zr, 1.0 - p, atol=RATE_TOL), zr


def test_sparsity_rate_per_instance_K1_broadcast():
    p = torch.tensor([[0.2], [0.8]])  # [K=2, 1]
    x = FeatureSampler(3, p, n_instances=2, seed=5).sample(N_BIG)
    zr = zero_rate(x)
    expected = (1.0 - p).expand(2, 3)
    assert torch.allclose(zr, expected, atol=RATE_TOL), zr


def test_instances_are_independent_draws():
    x = FeatureSampler(5, 0.5, n_instances=2, seed=6).sample(1000)
    assert not torch.equal(x[0], x[1])


def test_sample_conditional_active_and_inactive():
    p = 0.3
    s = FeatureSampler(5, p, n_instances=2, seed=7)
    xa = s.sample_conditional(N_BIG, 2, True)
    assert tuple(xa.shape) == (2, N_BIG, 5)
    assert torch.all(xa[:, :, 2] > 0), "forced-active feature must be nonzero everywhere"
    mean, m2, lo, hi = active_stats(xa[:, :, 2])
    assert abs(mean - 0.5) < MOM_TOL and abs(m2 - 1.0 / 3.0) < MOM_TOL and hi < 1.0
    others = [0, 1, 3, 4]
    zr = zero_rate(xa)[:, others]
    assert torch.allclose(zr, torch.full_like(zr, 1.0 - p), atol=RATE_TOL), zr

    xz = s.sample_conditional(N_BIG, 2, False)
    assert tuple(xz.shape) == (2, N_BIG, 5)
    assert torch.all(xz[:, :, 2] == 0), "forced-inactive feature must be exactly 0"
    zr = zero_rate(xz)[:, others]
    assert torch.allclose(zr, torch.full_like(zr, 1.0 - p), atol=RATE_TOL), zr
    for i in others:
        mean, m2, lo, hi = active_stats(xz[:, :, i])
        assert abs(mean - 0.5) < MOM_TOL and abs(m2 - 1.0 / 3.0) < MOM_TOL and hi < 1.0


# --------------------------------------------------------------------------------------
# 2. determinism under a fixed seed (sampler, model)
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("rep", range(5))
def test_sampler_determinism_same_seed(rep):
    a = FeatureSampler(6, 0.4, n_instances=3, seed=11).sample(2048)
    b = FeatureSampler(6, 0.4, n_instances=3, seed=11).sample(2048)
    assert torch.equal(a, b)


def test_sampler_sequence_and_seed_dependence():
    s = FeatureSampler(6, 0.4, n_instances=2, seed=11)
    a, b = s.sample(512), s.sample(512)
    assert not torch.equal(a, b), "successive batches must differ"
    c = FeatureSampler(6, 0.4, n_instances=2, seed=12).sample(512)
    assert not torch.equal(a, c), "different seeds must give different batches"


def test_sample_conditional_determinism():
    a = FeatureSampler(6, 0.4, n_instances=2, seed=13).sample_conditional(1024, 1, True)
    b = FeatureSampler(6, 0.4, n_instances=2, seed=13).sample_conditional(1024, 1, True)
    assert torch.equal(a, b)


def test_model_determinism_and_init():
    m1 = ToyModel(3, 7, 4, seed=5)
    m2 = ToyModel(3, 7, 4, seed=5)
    m3 = ToyModel(3, 7, 4, seed=6)
    assert torch.equal(m1.W, m2.W) and torch.equal(m1.b, m2.b)
    assert not torch.equal(m1.W, m3.W)
    assert not torch.equal(m1.W[0], m1.W[1]), "instances must get different inits"
    assert torch.all(m1.b == 0), "b = 0 at init"
    assert instance_seed(0, 1) == instance_seed(0, 1)
    assert len({instance_seed(0, k) for k in range(10)}) == 10


def test_model_shapes():
    K, n, m = 3, 7, 4
    model = ToyModel(K, n, m, seed=0)
    assert tuple(model.W.shape) == (K, m, n)
    assert tuple(model.b.shape) == (K, n)
    x = torch.rand(K, 11, n)
    out = model(x)
    assert tuple(out.shape) == (K, 11, n)


def test_model_xavier_normal_init_std():
    K, n, m = 3, 300, 100
    W = ToyModel(K, n, m, seed=0).W.detach().double()
    expected = math.sqrt(2.0 / (n + m))  # Xavier normal for an [m, n] matrix
    for k in range(K):
        std = float(W[k].std())
        assert abs(std - expected) / expected < 0.03, (k, std, expected)
        assert abs(float(W[k].mean())) < 0.01


# --------------------------------------------------------------------------------------
# 3. forward pass against hand-computed ReLU(W^T W x + b)
# --------------------------------------------------------------------------------------
def test_forward_matches_hand_computation_two_distinct_instances():
    W = np.array(
        [
            [[1.0, 0.5, -0.3], [0.0, 1.0, 0.8]],
            [[-0.7, 0.2, 0.9], [0.4, -1.1, 0.1]],
        ]
    )  # [K=2, m=2, n=3]
    b = np.array([[-0.2, 0.1, 0.0], [0.05, -0.5, -0.1]])  # [K, n]
    x = np.array(
        [
            [[0.0, 0.7, 0.3], [1.0, 0.0, 0.0]],
            [[0.5, 0.5, 0.0], [0.0, 0.0, 0.9]],
        ]
    )  # [K, B=2, n]
    model = ToyModel(2, 3, 2, seed=0)
    with torch.no_grad():
        model.W.copy_(torch.tensor(W, dtype=torch.float32))
        model.b.copy_(torch.tensor(b, dtype=torch.float32))
    out = model(torch.tensor(x, dtype=torch.float32)).detach().double().numpy()
    ref = ref_forward(W, b, x)
    assert ref.min() == 0.0 and (ref > 0).any(), "example should exercise both ReLU branches"
    np.testing.assert_allclose(out, ref, atol=1e-6)
    # run() with tied weights equals forward
    out_run = ToyModel.run(model.W, model.W, model.b, torch.tensor(x, dtype=torch.float32))
    np.testing.assert_allclose(out_run.detach().double().numpy(), ref, atol=1e-6)


def test_run_with_distinct_encoder_decoder():
    W_enc = np.array([[[1.0, 0.0, 0.5], [0.0, 1.0, 0.5]]])
    W_dec = np.array([[[0.5, 0.2, -1.0], [0.3, 0.9, 0.4]]])
    b = np.array([[-0.1, 0.0, 0.2]])
    x = np.array([[[0.4, 0.0, 0.8], [0.0, 1.0, 0.0]]])
    out = ToyModel.run(
        torch.tensor(W_enc, dtype=torch.float32),
        torch.tensor(W_dec, dtype=torch.float32),
        torch.tensor(b, dtype=torch.float32),
        torch.tensor(x, dtype=torch.float32),
    )
    h = np.einsum("kmn,kbn->kbm", W_enc, x)
    pre = np.einsum("kmn,kbm->kbn", W_dec, h) + b[:, None, :]
    ref = np.maximum(pre, 0.0)
    np.testing.assert_allclose(out.detach().double().numpy(), ref, atol=1e-6)


# --------------------------------------------------------------------------------------
# 4. running example: pentagon, b = -0.25, x = (0, 0.7, 0, 0, 0)
# --------------------------------------------------------------------------------------
def test_running_example_outputs_and_margin():
    W = pentagon()  # [1, 2, 5] unit vectors
    b = np.full((1, 5), -0.25)
    model = ToyModel(1, 5, 2, seed=0)
    with torch.no_grad():
        model.W.copy_(torch.tensor(W, dtype=torch.float32))
        model.b.copy_(torch.tensor(b, dtype=torch.float32))
    x = torch.zeros(1, 1, 5)
    x[0, 0, 1] = 0.7
    out = model(x).detach()[0, 0]
    for i in (0, 2, 3, 4):
        assert float(out[i]) == 0.0, (i, float(out[i]))
    assert abs(float(out[1]) - 0.45) < 1e-5, float(out[1])
    # pre-activation check of the binding adjacent case: cos72 * 0.7 - 0.25 < 0
    assert COS72 * 0.7 - 0.25 < 0
    mg = M.margin(W, b)
    assert mg.shape == (1, 5)
    np.testing.assert_allclose(mg, np.full((1, 5), 0.25 - COS72), atol=1e-9)
    assert np.all(mg < 0)
    assert abs(0.25 - COS72 - (-0.059017)) < 1e-6


# --------------------------------------------------------------------------------------
# 5. geometry metrics on hand-built matrices
# --------------------------------------------------------------------------------------
def test_gram_and_feature_norms_hand_built():
    W = np.array([[[1.0, 0.0, 3.0], [2.0, 2.0, 4.0]]])  # [1, 2, 3]
    G = M.gram(W)
    assert G.shape == (1, 3, 3)
    np.testing.assert_allclose(G[0], W[0].T @ W[0], atol=1e-12)
    norms = M.feature_norms(W)
    np.testing.assert_allclose(norms, np.array([[math.sqrt(5.0), 2.0, 5.0]]), atol=1e-12)


def test_dimensionality_sanity_values():
    np.testing.assert_allclose(M.dimensionality(pentagon()), np.full((1, 5), 0.4), atol=1e-9)
    antipodal = np.array([[[1.0, -1.0], [0.0, 0.0]]])
    np.testing.assert_allclose(M.dimensionality(antipodal), np.full((1, 2), 0.5), atol=1e-12)
    orth = np.eye(3)[None]
    np.testing.assert_allclose(M.dimensionality(orth), np.ones((1, 3)), atol=1e-12)
    # independent re-derivation agrees
    for W in (pentagon(), antipodal, orth):
        np.testing.assert_allclose(M.dimensionality(W), ref_D(W), atol=1e-9)


def test_dimensionality_nan_rule_and_neighbours_unaffected():
    W = np.array([[[1.0, 0.0, 0.05], [0.0, 1.0, 0.0]]])  # column 2 has norm 0.05 < 0.1
    D = M.dimensionality(W)
    assert D.shape == (1, 3)
    assert np.isnan(D[0, 2])
    assert np.isfinite(D[0, 0]) and np.isfinite(D[0, 1])
    # D_0 = 1 / (1 + 0.05^2), D_1 = 1
    np.testing.assert_allclose(D[0, :2], [1.0 / (1.0 + 0.0025), 1.0], atol=1e-12)
    np.testing.assert_allclose(D, ref_D(W), atol=1e-12, equal_nan=True)


def test_dimensionality_threshold_location():
    below = np.array([[[0.1 - 1e-6, 0.0], [0.0, 1.0]]])
    above = np.array([[[0.1 + 1e-6, 0.0], [0.0, 1.0]]])
    assert np.isnan(M.dimensionality(below)[0, 0])
    assert np.isfinite(M.dimensionality(above)[0, 0])
    # configurable min_norm
    assert np.isfinite(M.dimensionality(below, min_norm=0.05)[0, 0])


def test_represented_threshold():
    W = np.array([[[0.3, 0.5, 0.5 + 1e-6, 0.7], [0.0, 0.0, 0.0, 0.0]]])
    r = M.represented(W)
    assert r.shape == (1, 4) and r.dtype == bool
    assert r.tolist() == [[False, False, True, True]], r
    assert M.represented(W, threshold=0.2).tolist() == [[True, True, True, True]]


def test_feats_per_dim_and_dims_per_feat():
    W = np.array([[[1.0, 0.0, 2.0], [0.0, 1.0, 0.0]]])  # ||W||_F^2 = 6, m = 2
    np.testing.assert_allclose(M.feats_per_dim(W), [3.0], atol=1e-12)
    np.testing.assert_allclose(M.dims_per_feat(W), [1.0 / 3.0], atol=1e-12)
    np.testing.assert_allclose(M.feats_per_dim(pentagon()), [2.5], atol=1e-9)
    np.testing.assert_allclose(M.dims_per_feat(pentagon()), [0.4], atol=1e-9)
    np.testing.assert_allclose(M.feats_per_dim(np.eye(2)[None]), [1.0], atol=1e-12)
    # K = 2 stacking keeps per-instance values
    W2 = np.concatenate([W, pentagon()[:, :, :3] * 2.0], axis=0)  # second: 3 columns of norm 2 -> 12 / 2
    np.testing.assert_allclose(M.feats_per_dim(W2), [3.0, 6.0], atol=1e-9)


def test_max_overlap_excludes_diagonal():
    # columns of norm 2 and ~1.03: diagonal entries (4, 1.0625) exceed the cross term 0.5
    W = np.array([[[2.0, 0.25], [0.0, 1.0]]])
    np.testing.assert_allclose(M.max_overlap(W), [[0.5, 0.5]], atol=1e-12)
    # orthogonal: all off-diagonal dots are 0, diagonal would be 1
    np.testing.assert_allclose(M.max_overlap(np.eye(3)[None]), np.zeros((1, 3)), atol=1e-12)
    # antipodal: the only other dot is -1; must not be clamped to 0 here
    antipodal = np.array([[[1.0, -1.0], [0.0, 0.0]]])
    np.testing.assert_allclose(M.max_overlap(antipodal), [[-1.0, -1.0]], atol=1e-12)
    # pentagon: max over j != i is cos 72
    np.testing.assert_allclose(M.max_overlap(pentagon()), np.full((1, 5), COS72), atol=1e-9)
    # a 3-column case with a mix of signs: W_0 = (1, 0), W_1 = (-1, 0), W_2 = (0.5, 2)
    # dots: 01 -> -1, 02 -> 0.5, 12 -> -0.5
    W3 = np.array([[[1.0, -1.0, 0.5], [0.0, 0.0, 2.0]]])
    np.testing.assert_allclose(M.max_overlap(W3), [[0.5, -0.5, 0.5]], atol=1e-12)


def test_margin_definition_with_clamp():
    # orthogonal, b = -0.1 -> margin = 0.1 (max_overlap 0 -> max(0, 0) = 0)
    np.testing.assert_allclose(M.margin(np.eye(2)[None], np.full((1, 2), -0.1)), [[0.1, 0.1]], atol=1e-12)
    # antipodal, b = -0.1 -> max_overlap = -1 -> clamped to 0 -> margin = 0.1
    antipodal = np.array([[[1.0, -1.0], [0.0, 0.0]]])
    np.testing.assert_allclose(M.margin(antipodal, np.full((1, 2), -0.1)), [[0.1, 0.1]], atol=1e-12)
    # antipodal with b = 0 -> margin 0
    np.testing.assert_allclose(M.margin(antipodal, np.zeros((1, 2))), [[0.0, 0.0]], atol=1e-12)
    # per-feature bias and positive overlaps: W_0.W_1 = 0.5; b = (-0.7, 0.2) -> (0.7 - 0.5, -0.2 - 0.5)
    W = np.array([[[2.0, 0.25], [0.0, 1.0]]])
    np.testing.assert_allclose(M.margin(W, np.array([[-0.7, 0.2]])), [[0.2, -0.7]], atol=1e-12)


def test_is_pentagon_design_note_9():
    good = pentagon()
    assert M.is_pentagon(good).tolist() == [True]
    small = pentagon(scale=0.4)  # D_i still 0.4 but norms 0.4 < 0.5
    assert M.is_pentagon(small).tolist() == [False]
    dead = np.zeros((1, 2, 5))
    dead[0, 0, 0] = 1.0
    dead[0, 1, 1] = 1.0  # two orthogonal unit columns, three dead -> D = 1, 1, NaN x3
    assert M.is_pentagon(dead).tolist() == [False]
    stacked = np.concatenate([good, small, dead], axis=0)
    assert M.is_pentagon(stacked).tolist() == [True, False, False]
    # tolerance knob: a slightly squashed pentagon fails at tight d_tol
    squash = pentagon().copy()
    squash[0, 1, :] *= 0.8
    D = M.dimensionality(squash)
    tight = bool(np.all(np.abs(D - 0.4) <= 0.001))
    assert M.is_pentagon(squash, d_tol=0.001).tolist() == [tight]


def test_geometry_table_consistent_with_individual_metrics():
    W = pentagon(k=2)
    W[1] *= 0.7
    b = np.array([[-0.25] * 5, [-0.1] * 5])
    tab = M.geometry_table(W, b)
    assert isinstance(tab, dict)
    for key, val in tab.items():
        assert np.asarray(val).shape == (2, 5), (key, np.asarray(val).shape)
    np.testing.assert_allclose(tab["norm"], M.feature_norms(W), atol=1e-12)
    np.testing.assert_allclose(tab["D"], M.dimensionality(W), atol=1e-12, equal_nan=True)
    np.testing.assert_allclose(tab["max_overlap"], M.max_overlap(W), atol=1e-12)
    np.testing.assert_allclose(tab["margin"], M.margin(W, b), atol=1e-12)


def test_metrics_accept_model_parameters():
    model = ToyModel(2, 5, 2, seed=0)
    tab = M.geometry_table(model.W, model.b)
    assert np.asarray(tab["margin"]).shape == (2, 5)
    np.testing.assert_allclose(
        M.feature_norms(model.W),
        np.linalg.norm(model.W.detach().double().numpy(), axis=1),
        atol=1e-6,
    )


# --------------------------------------------------------------------------------------
# 6. loss definition on a tiny hand-computed example
# --------------------------------------------------------------------------------------
def test_importance_weighted_loss_hand_computed():
    x = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])  # [K=1, B=2, n=2]
    out = torch.tensor([[[0.5, 0.0], [0.0, 0.0]]])
    imp = torch.tensor([[1.0, 2.0]])  # [K, n]
    # sample 0: 1*(0.5)^2 + 2*0 = 0.25 ; sample 1: 0 + 2*(1)^2 = 2 ; mean = 1.125
    loss = importance_weighted_loss(x, out, imp)
    assert tuple(loss.shape) == (1,)
    assert abs(float(loss[0]) - 1.125) < 1e-6, float(loss[0])


def test_importance_weighted_loss_two_instances():
    x = torch.tensor(
        [
            [[1.0, 0.0], [0.5, 0.5]],
            [[0.0, 1.0], [0.0, 0.0]],
        ]
    )
    out = torch.tensor(
        [
            [[0.5, 0.25], [0.0, 1.0]],
            [[0.0, 0.0], [0.3, 0.0]],
        ]
    )
    imp = torch.tensor([[1.0, 0.5], [2.0, 1.0]])
    # K=0: s0 = 1*0.25 + 0.5*0.0625 = 0.28125 ; s1 = 1*0.25 + 0.5*0.25 = 0.375 ; mean = 0.328125
    # K=1: s0 = 2*0 + 1*1 = 1 ; s1 = 2*0.09 + 0 = 0.18 ; mean = 0.59
    loss = importance_weighted_loss(x, out, imp)
    assert tuple(loss.shape) == (2,)
    np.testing.assert_allclose(loss.numpy(), [0.328125, 0.59], atol=1e-6)


# --------------------------------------------------------------------------------------
# 7. training: a short run lowers the loss; reproducible; lr_at / plateau_ratio
# --------------------------------------------------------------------------------------
def _short_run(seed: int = 0) -> np.ndarray:
    K, n, m = 2, 5, 2
    model = ToyModel(K, n, m, seed=seed)
    sampler = FeatureSampler(n, 0.3, n_instances=K, seed=seed)
    importance = (0.9 ** torch.arange(n, dtype=torch.float32))[None].expand(K, n).clone()
    cfg = TrainConfig(steps=300, batch=256, progress=False)
    res = train(model, sampler, importance, cfg)
    return res.loss_history


def test_short_training_lowers_loss():
    hist = _short_run()
    assert hist.shape == (300, 2), hist.shape
    assert np.isfinite(hist).all()
    first = hist[:30].mean(axis=0)
    last = hist[-30:].mean(axis=0)
    assert np.all(last < first), (first, last)
    # a clear decrease, not a float-noise one (observed ratio ~0.68 / 0.75 at 300 steps, lr 1e-3)
    assert np.all(last < 0.9 * first), (first, last)


def test_training_reproducible_under_fixed_seeds():
    a = _short_run(seed=3)
    b = _short_run(seed=3)
    np.testing.assert_array_equal(a, b)


def test_lr_at_constant_schedule():
    cfg = TrainConfig(steps=100, lr=1e-3, lr_schedule="constant", progress=False)
    for step in (0, 1, 50, 99):
        assert lr_at(step, cfg) == pytest.approx(1e-3)


def test_plateau_ratio_hand_computed():
    flat = np.ones((100, 2))
    np.testing.assert_allclose(plateau_ratio(flat), [1.0, 1.0], atol=1e-12)
    falling = (100.0 - np.arange(100, dtype=float))[:, None].repeat(2, axis=1)
    # last 10 steps (i = 90..99): mean 5.5 ; preceding 10 (i = 80..89): mean 15.5
    np.testing.assert_allclose(plateau_ratio(falling, frac=0.1), [5.5 / 15.5] * 2, atol=1e-12)
