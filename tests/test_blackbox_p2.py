"""Independent black-box tests for Phase 2 of TMS-Q (quantize / metrics / analysis).

Specification: project CLAUDE.md ("Quantization (PTQ) definitions", "Per-feature damage metrics",
design notes 1-6 and 10) and notes/LOG.md "Phase 2 plan". Only the public API is exercised; the
implementation bodies of the modules under test were not read (signatures / docstrings only).
"""

from __future__ import annotations

import math
import pathlib

import numpy as np
import pandas as pd
import pytest
import torch

from tmsq import analysis as an
from tmsq import metrics as me
from tmsq import quantize as qz

BITS = [8, 6, 5, 4, 3, 2]
GRANS = ["tensor", "column", "row"]
ABLATIONS = ["quant_b", "enc_only", "dec_only"]
ABL_GRANS = ["column", "row"]
PROJECT = pathlib.Path(__file__).resolve().parents[1]
SMOKE_DIRS = sorted(PROJECT.glob("results/quant_sweep/*-A-smoke"))
P = an.EvalParams()
VAR_05 = 0.5 / 3 - 0.25 / 4  # Var(x) at p = 0.5


def qmax(bits: int) -> int:
    return 2 ** (bits - 1) - 1


def rand_w(K: int = 2, m: int = 3, n: int = 5, seed: int = 0, scale: float = 1.0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return scale * torch.randn(K, m, n, generator=g)


# --------------------------------------------------------------------------------------- quantizer


@pytest.mark.parametrize("bits", BITS)
def test_qmax_for(bits):
    assert qz.qmax_for(bits) == 2 ** (bits - 1) - 1


def _dense_input(gran: str, M: int = 100001):
    """Dense linspace spanning the range of EVERY scaling unit, distinct amplitudes per unit (note 1)."""
    lin = torch.linspace(-1.0, 1.0, M)
    if gran == "column":  # W [K, M, n]; column i has amplitude i+1
        amp = torch.arange(1, 5, dtype=torch.float32)
        W = lin[None, :, None] * amp[None, None, :]
        W = torch.cat([W, 0.5 * W], dim=0)

        def units(Q):
            return [Q[k, :, i] for k in range(Q.shape[0]) for i in range(Q.shape[2])]

    elif gran == "row":  # W [K, m, M]; row j has amplitude j+1
        amp = torch.arange(1, 4, dtype=torch.float32)
        W = lin[None, None, :] * amp[None, :, None]
        W = torch.cat([W, 0.5 * W], dim=0)

        def units(Q):
            return [Q[k, j, :] for k in range(Q.shape[0]) for j in range(Q.shape[1])]

    else:  # W [K, 1, M]
        W = lin[None, None, :]
        W = torch.cat([W, 0.5 * W], dim=0)

        def units(Q):
            return [Q[k] for k in range(Q.shape[0])]

    return W, units


@pytest.mark.parametrize("gran", GRANS)
@pytest.mark.parametrize("bits", BITS)
def test_codebook_size_on_dense_linspace(bits, gran):
    W, units = _dense_input(gran)
    Q, s = qz.rtn(W, bits, gran)
    assert Q.shape == W.shape
    for u in units(Q):
        assert torch.unique(u).numel() == 2**bits - 1
    k = Q / s
    assert torch.allclose(k, torch.round(k), atol=1e-3)  # on the lattice s * k
    assert k.abs().max().item() <= qmax(bits) + 1e-3  # clamped to [-qmax, qmax]


@pytest.mark.parametrize("gran", GRANS)
@pytest.mark.parametrize("bits", BITS)
def test_rounding_error_within_half_step(bits, gran):
    W = rand_w(K=3, m=6, n=7, seed=1)
    Q, s = qz.rtn(W, bits, gran)
    assert torch.all((Q - W).abs() <= s / 2 + 1e-6)


@pytest.mark.parametrize("gran", GRANS)
@pytest.mark.parametrize("bits", BITS)
def test_idempotence(bits, gran):
    W = rand_w(K=3, m=6, n=7, seed=2)
    Q, s = qz.rtn(W, bits, gran)
    Q2, s2 = qz.rtn(Q, bits, gran)
    assert torch.allclose(Q2, Q, atol=1e-6)
    assert torch.allclose(s2, s, rtol=1e-5)


@pytest.mark.parametrize("gran", GRANS)
@pytest.mark.parametrize("bits", BITS)
def test_zero_column_row_tensor_quantize_to_exact_zero(bits, gran):
    W = rand_w(K=2, m=4, n=5, seed=3)
    W[:, :, 2] = 0.0
    W[:, 1, :] = 0.0
    Q, s = qz.rtn(W, bits, gran)
    assert torch.isfinite(Q).all() and torch.isfinite(s).all()
    assert torch.all(Q[:, :, 2] == 0.0) and torch.all(Q[:, 1, :] == 0.0)
    assert torch.all(s > 0)
    Qz, sz = qz.rtn(torch.zeros(2, 4, 5), bits, gran)
    assert torch.all(Qz == 0.0) and torch.isfinite(Qz).all() and torch.isfinite(sz).all()
    assert torch.all(sz > 0)


@pytest.mark.parametrize("bits", BITS)
def test_scale_shapes_and_values(bits):
    K, m, n = 3, 4, 6
    W = rand_w(K, m, n, seed=4)
    q = qmax(bits)
    s_t = qz.rtn_scales(W, bits, "tensor")
    s_c = qz.rtn_scales(W, bits, "column")
    s_r = qz.rtn_scales(W, bits, "row")
    assert tuple(s_t.shape) == (K, 1, 1)
    assert tuple(s_c.shape) == (K, 1, n)
    assert tuple(s_r.shape) == (K, m, 1)
    assert torch.allclose(s_t, W.abs().amax(dim=(1, 2), keepdim=True) / q)
    assert torch.allclose(s_c, W.abs().amax(dim=1, keepdim=True) / q)
    assert torch.allclose(s_r, W.abs().amax(dim=2, keepdim=True) / q)
    for gran, s_ref in (("tensor", s_t), ("column", s_c), ("row", s_r)):
        Q, s = qz.rtn(W, bits, gran)
        assert tuple(s.shape) == tuple(s_ref.shape) and torch.allclose(s, s_ref)
        assert torch.allclose(Q, s_ref * torch.clamp(torch.round(W / s_ref), -q, q), atol=1e-6)


@pytest.mark.parametrize("bits", BITS)
def test_quantize_bias_per_tensor_scale(bits):
    b = torch.linspace(-1, 1, 100001)[None, :] * torch.tensor([[1.0], [0.3]])
    Qb = qz.quantize_bias(b, bits)
    assert Qb.shape == b.shape and Qb.dtype == torch.float32
    s = b.abs().amax(dim=1, keepdim=True) / qmax(bits)
    for k in range(b.shape[0]):
        assert torch.unique(Qb[k]).numel() == 2**bits - 1
    assert torch.all((Qb - b).abs() <= s / 2 + 1e-6)
    assert torch.allclose(qz.quantize_bias(Qb, bits), Qb, atol=1e-6)
    Qz = qz.quantize_bias(torch.zeros(2, 5), bits)
    assert torch.all(Qz == 0.0) and torch.isfinite(Qz).all()


# ---------------------------------------------------------------------------------- noise control


@pytest.mark.parametrize("gran", GRANS)
def test_noise_control_matches_per_column_and_frobenius_error(gran):
    W = rand_w(K=2, m=8, n=6, seed=5)
    Q, _ = qz.rtn(W, 4, gran)
    Wn = qz.noise_control(W, Q, seed=3)
    assert Wn.shape == W.shape
    delta, err = Wn - W, Q - W
    assert torch.allclose(delta.norm(dim=1), err.norm(dim=1), rtol=1e-4, atol=1e-6)
    assert torch.allclose(delta.flatten(1).norm(dim=1), err.flatten(1).norm(dim=1), rtol=1e-4)
    assert torch.equal(Wn, qz.noise_control(W, Q, seed=3))  # deterministic in the seed
    assert not torch.allclose(Wn, qz.noise_control(W, Q, seed=4))
    assert not torch.allclose(Wn, Q, atol=1e-4)  # random direction, not the rounding direction


def test_noise_control_zero_error_column_untouched():
    W = rand_w(K=2, m=8, n=6, seed=6)
    Q = W.clone()
    Q[:, :, 1] += 0.1  # only column 1 carries rounding error
    delta = qz.noise_control(W, Q, seed=0) - W
    keep = torch.ones(6, dtype=torch.bool)
    keep[1] = False
    assert torch.all(delta[:, :, keep].abs() <= 1e-7)
    assert torch.allclose(delta[:, :, 1].norm(dim=1), torch.full((2,), 0.1 * math.sqrt(8)), rtol=1e-4)


# ------------------------------------------------------------------------------------- apply_quant


def _wb(seed: int = 7):
    W = rand_w(K=2, m=5, n=7, seed=seed)
    b = 0.1 * rand_w(K=2, m=1, n=7, seed=seed + 1)[:, 0, :]
    return W, b


@pytest.mark.parametrize("gran", GRANS)
@pytest.mark.parametrize("bits", [8, 4, 2])
def test_apply_quant_default_is_tied_rtn_with_fp_bias(bits, gran):
    W, b = _wb()
    cfg = qz.QuantConfig(bits=bits, granularity=gran)
    assert (cfg.method, cfg.noise_seed, cfg.ablation) == ("rtn", 0, "none")
    res = qz.apply_quant(cfg, W, b)
    Q, s = qz.rtn(W, bits, gran)
    assert torch.equal(res.W_enc, Q) and torch.equal(res.W_dec, Q) and torch.equal(res.Wq, Q)
    assert torch.equal(res.b, b) and res.b.dtype == torch.float32
    assert torch.allclose(res.scales, s)
    assert torch.equal(W, _wb()[0])  # input not modified in place


@pytest.mark.parametrize("gran", ABL_GRANS)
def test_apply_quant_ablations(gran):
    W, b = _wb()
    Q, _ = qz.rtn(W, 4, gran)
    enc = qz.apply_quant(qz.QuantConfig(4, gran, ablation="enc_only"), W, b)
    assert torch.equal(enc.W_enc, Q) and torch.equal(enc.W_dec, W) and torch.equal(enc.b, b)
    assert torch.equal(enc.Wq, Q)  # covariates come from the perturbed side (LOG plan)
    dec = qz.apply_quant(qz.QuantConfig(4, gran, ablation="dec_only"), W, b)
    assert torch.equal(dec.W_enc, W) and torch.equal(dec.W_dec, Q) and torch.equal(dec.b, b)
    assert torch.equal(dec.Wq, Q)
    qb = qz.apply_quant(qz.QuantConfig(4, gran, ablation="quant_b"), W, b)
    assert torch.equal(qb.W_enc, Q) and torch.equal(qb.W_dec, Q) and torch.equal(qb.Wq, Q)
    assert torch.allclose(qb.b, qz.quantize_bias(b, 4)) and not torch.equal(qb.b, b)
    sb = b.abs().amax(dim=1, keepdim=True) / qmax(4)  # per-instance per-tensor scale on b
    assert torch.all((qb.b - b).abs() <= sb / 2 + 1e-6)
    for k in range(b.shape[0]):
        assert torch.unique(qb.b[k]).numel() <= 2**4 - 1


@pytest.mark.parametrize("gran", GRANS)
def test_apply_quant_noise_method_tied_and_norm_matched(gran):
    W, b = _wb()
    cfg = qz.QuantConfig(4, gran, method="noise", noise_seed=2)
    res = qz.apply_quant(cfg, W, b)
    Q, s = qz.rtn(W, 4, gran)
    assert torch.equal(res.W_enc, res.W_dec) and torch.equal(res.Wq, res.W_enc)
    assert torch.equal(res.b, b)
    assert torch.allclose((res.W_enc - W).norm(dim=1), (Q - W).norm(dim=1), rtol=1e-4, atol=1e-6)
    assert not torch.allclose(res.W_enc, Q, atol=1e-4)
    assert torch.equal(res.W_enc, qz.apply_quant(cfg, W, b).W_enc)
    other = qz.apply_quant(qz.QuantConfig(4, gran, method="noise", noise_seed=3), W, b)
    assert not torch.allclose(res.W_enc, other.W_enc, atol=1e-4)
    assert torch.allclose(res.scales, s)


def test_build_configs_grid_is_144_distinct():
    cfgs = qz.build_configs(BITS, GRANS, 5, ABLATIONS, ABL_GRANS)
    assert len(cfgs) == 144 and len({repr(c) for c in cfgs}) == 144
    plain = [c for c in cfgs if c.method == "rtn" and c.ablation == "none"]
    noise = [c for c in cfgs if c.method == "noise"]
    abl = [c for c in cfgs if c.ablation != "none"]
    assert (len(plain), len(noise), len(abl)) == (18, 90, 36)
    assert all(c.ablation == "none" and c.granularity in GRANS for c in noise)
    assert {(c.bits, c.granularity) for c in plain} == {(b, g) for b in BITS for g in GRANS}
    assert {(c.bits, c.granularity, c.noise_seed) for c in noise} == {
        (b, g, k) for b in BITS for g in GRANS for k in range(5)
    }
    assert all(c.method == "rtn" and c.noise_seed == 0 for c in abl)
    assert {(c.bits, c.granularity, c.ablation) for c in abl} == {
        (b, g, a) for b in BITS for g in ABL_GRANS for a in ABLATIONS
    }


# ---------------------------------------------------------------------------------------- metrics


def _synthetic_recon(p: float = 0.5, N: int = 60000, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = np.where(rng.random(N) < p, rng.random(N), 0.0)
    leak = 0.2 * np.abs(rng.standard_normal(N))
    x_hat = np.where(x > 0, 0.8 * x + 0.05 * rng.standard_normal(N), leak)
    return x, np.maximum(x_hat, 0.0)


def test_mse_decomposition_matches_direct_monte_carlo():
    x, x_hat = _synthetic_recon()
    act = x > 0
    cond = float(np.mean((x[act] - x_hat[act]) ** 2))
    leak = float(np.mean(x_hat[~act] ** 2))
    direct = float(np.mean((x - x_hat) ** 2))
    assert float(me.mse_from_strata(float(act.mean()), cond, leak)) == pytest.approx(direct, abs=1e-12)
    assert float(me.mse_from_strata(0.5, cond, leak)) == pytest.approx(direct, abs=2e-3)
    out = me.mse_from_strata(0.5, np.array([cond, 0.0]), np.array([leak, 0.0]))
    assert out.shape == (2,) and out[0] == pytest.approx(0.5 * cond + 0.5 * leak) and out[1] == 0.0


def test_mse_and_r2_at_p1_ignore_missing_inactive_stratum():
    assert float(me.mse_from_strata(1.0, 0.123, np.nan)) == pytest.approx(0.123)
    r = float(me.r2_from_strata(1.0, 0.01, np.nan))
    assert np.isfinite(r) and r == pytest.approx(1 - 0.01 / (1 / 12))


def test_var_x_formula():
    assert float(me.var_x(1.0)) == pytest.approx(1 / 12)
    assert float(me.var_x(0.5)) == pytest.approx(VAR_05)
    assert float(me.var_x(0.0)) == 0.0
    x, _ = _synthetic_recon(p=0.3, N=400000, seed=2)
    assert float(me.var_x(0.3)) == pytest.approx(float(np.var(x)), abs=1e-3)
    assert np.allclose(me.var_x(np.array([0.0, 0.5, 1.0])), [0.0, VAR_05, 1 / 12])


def test_r2_nan_at_p0_and_formula():
    r = np.asarray(me.r2_from_strata(0.0, 0.0, 0.0), dtype=float)
    assert np.isnan(r).all()
    assert float(me.r2_from_strata(0.5, 0.06, 0.06)) == pytest.approx(1 - 0.06 / VAR_05)
    arr = me.r2_from_strata(np.array([0.0, 0.5]), np.array([0.0, 0.06]), np.array([0.0, 0.06]))
    assert np.isnan(arr[0]) and arr[1] == pytest.approx(1 - 0.06 / VAR_05)


def _pearson_inputs():
    rng = np.random.default_rng(1)
    return rng.random((5000, 3)), rng.random((5000, 3))


def test_pearson_scale_blind():
    a, _ = _pearson_inputs()
    r = me.pearson_columns(a, 0.6 * a)  # a pure gain change gives correlation 1 (CLAUDE.md, survival)
    assert r.shape == (3,) and np.allclose(r, 1.0, atol=1e-6)
    assert np.allclose(me.pearson_columns(a, 2.0 - a), -1.0, atol=1e-6)


def test_pearson_matches_numpy_reference():
    a, c = _pearson_inputs()
    ref = [np.corrcoef(a[:, i], c[:, i])[0, 1] for i in range(3)]
    assert np.allclose(me.pearson_columns(a, c), ref, atol=1e-6)


def test_pearson_nan_on_exactly_zero_constant_column():
    a, _ = _pearson_inputs()
    b = 0.6 * a
    b[:, 1] = 0.0
    r = me.pearson_columns(a, b)
    assert np.isclose(r[0], 1.0, atol=1e-6) and np.isnan(r[1]) and np.isclose(r[2], 1.0, atol=1e-6)
    a2 = a.copy()
    a2[:, 0] = 0.0
    r = me.pearson_columns(a2, b)
    assert np.isnan(r[0]) and np.isnan(r[1]) and np.isclose(r[2], 1.0, atol=1e-6)


def test_pearson_nan_on_nonrepresentable_constant_column():
    # docstring: "NaN where either column is constant"; design note 3: NaN when FP or quantized x_hat_i is constant
    a, _ = _pearson_inputs()
    b = 0.6 * a
    b[:, 1] = 0.3  # constant, but 0.3 is not exactly representable -> mean subtraction leaves float dust
    r = me.pearson_columns(a, b)
    assert np.isclose(r[0], 1.0, atol=1e-6) and np.isclose(r[2], 1.0, atol=1e-6)
    assert np.isnan(r[1]), f"constant quantized column gave {r[1]!r}, expected NaN"
    a2 = a.copy()
    a2[:, 0] = 0.3
    assert np.isnan(me.pearson_columns(a2, b)[0]), "constant FP column must give NaN"


def test_perturbation_covariates_hand_built():
    W = torch.tensor([[[1.0, 0.0, 0.6], [0.0, 1.0, 0.8]]])
    Wq = torch.tensor([[[0.8, 0.0, 0.3], [0.0, 1.2, 0.0]]])
    b = torch.tensor([[-0.1, -0.2, 0.3]])
    cov = me.perturbation_covariates(W, Wq, b, b)
    assert {"norm_q", "killed", "d_norm2", "dW", "d_overlap", "margin_q", "margin_flip"} <= set(cov)
    assert np.allclose(cov["norm_q"], [[0.8, 1.2, 0.3]], atol=1e-6)
    assert np.asarray(cov["killed"]).tolist() == [[False, False, True]]
    assert np.allclose(cov["d_norm2"], [[-0.36, 0.44, -0.91]], atol=1e-6)
    assert np.allclose(cov["dW"], [[0.2, 0.2, math.sqrt(0.73)]], atol=1e-6)
    assert np.allclose(cov["d_overlap"], [[0.36, 0.8, 0.8]], atol=1e-6)
    assert np.allclose(cov["margin_q"], [[-0.14, 0.2, -0.54]], atol=1e-6)
    assert np.asarray(cov["margin_flip"]).tolist() == [[False, True, False]]
    b_used = torch.tensor([[-0.3, -0.2, 0.3]])  # quant_b ablation: margin_q from the bias actually used
    cov2 = me.perturbation_covariates(W, Wq, b, b_used)
    assert np.allclose(cov2["margin_q"], [[0.06, 0.2, -0.54]], atol=1e-6)
    assert np.asarray(cov2["margin_flip"]).tolist() == [[True, True, False]]
    for k in ("norm_q", "d_norm2", "dW", "d_overlap"):
        assert np.array_equal(cov2[k], cov[k])


# --------------------------------------------------------------------------------------- eval set


def test_eval_set_p05_natural_strata_are_first_rows():
    n, p = 4, 0.5
    es = an.build_eval_set(n, p, seed=123, params=P)
    assert es.n_natural == 44000 and tuple(es.X.shape) == (44000, n) and es.p == p
    assert (es.n_active_eval, es.n_inactive_eval, es.n_forced_active) == (2000, 20000, 0)
    assert tuple(es.idx_act.shape) == (2000, n) and tuple(es.idx_inact.shape) == (20000, n)  # [rows, n]
    X = es.X
    assert torch.all(X >= 0) and torch.all(X <= 1)
    for i in range(n):
        act_rows = torch.nonzero(X[:, i] > 0).flatten()[:2000]
        inact_rows = torch.nonzero(X[:, i] == 0).flatten()[:20000]
        assert torch.equal(es.idx_act[:, i].sort().values.long(), act_rows)
        assert torch.equal(es.idx_inact[:, i].sort().values.long(), inact_rows)
    zero_frac = (X == 0).float().mean(0)
    assert torch.allclose(zero_frac, torch.full((n,), 0.5), atol=0.02)
    assert abs(X[X > 0].mean().item() - 0.5) < 0.02
    assert torch.equal(an.build_eval_set(n, p, 123, P).X, X)
    assert not torch.equal(an.build_eval_set(n, p, 124, P).X, X)


def test_eval_set_p001_forced_active_batches():
    n, p = 3, 0.01
    es = an.build_eval_set(n, p, seed=5, params=P)
    assert es.n_natural == 40000
    assert es.X.shape[0] == es.n_natural + n * 2000
    assert es.n_forced_active == n
    assert (es.n_active_eval, es.n_inactive_eval) == (2000, 20000)
    for i in range(n):
        rows = es.idx_act[:, i].long()
        assert torch.all(rows >= es.n_natural)  # whole active batch is forced, not a top-up
        assert torch.all(es.X[rows, i] > 0) and abs(es.X[rows, i].mean().item() - 0.5) < 0.03
        others = [j for j in range(n) if j != i]
        assert abs((es.X[rows][:, others] == 0).float().mean().item() - 0.99) < 0.01
        inact = es.idx_inact[:, i].long()
        assert torch.all(inact < es.n_natural) and torch.all(es.X[inact, i] == 0)
        assert torch.equal(inact.sort().values, torch.nonzero(es.X[: es.n_natural, i] == 0).flatten()[:20000])
    assert abs((es.X[: es.n_natural] == 0).float().mean().item() - 0.99) < 0.005


def test_eval_set_p09_common_minimum_inactive_count():
    n, p = 3, 0.9
    es = an.build_eval_set(n, p, seed=9, params=P)
    assert es.n_natural == 120000
    X = es.X[: es.n_natural]
    zero_counts = (X == 0).sum(0)
    assert es.n_inactive_eval == int(zero_counts.min()) and es.n_inactive_eval < 20000
    assert tuple(es.idx_inact.shape) == (es.n_inactive_eval, n)
    for i in range(n):
        inact_rows = torch.nonzero(X[:, i] == 0).flatten()[: es.n_inactive_eval]
        assert torch.equal(es.idx_inact[:, i].sort().values.long(), inact_rows)
    assert es.n_forced_active == 0 and es.n_active_eval == 2000 and tuple(es.idx_act.shape) == (2000, n)


def test_eval_set_p1_has_no_inactive_stratum():
    n = 3
    es = an.build_eval_set(n, 1.0, seed=1, params=P)
    assert es.idx_inact is None and es.n_inactive_eval == 0
    assert tuple(es.idx_act.shape) == (2000, n) and es.n_active_eval == 2000
    assert torch.all(es.X > 0) and torch.all(es.X <= 1)


def test_eval_seed_formula():
    assert an.eval_seed(0, 0, P) == 10000
    assert an.eval_seed(3, 2, P) == 10000 + 7919 * 3 + 2


# --------------------------------------------------------------------------------- forward_single


def _pentagon() -> torch.Tensor:
    ang = torch.arange(5, dtype=torch.float32) * (2 * math.pi / 5)
    return torch.stack([torch.cos(ang), torch.sin(ang)])  # [m=2, n=5]


def test_forward_single_running_example():
    W = _pentagon()
    b = torch.full((5,), -0.25)
    X = torch.tensor([[0.0, 0.7, 0.0, 0.0, 0.0]])
    pre, out = an.forward_single(W, W, b, X, want_pre=True, chunk=10)
    assert tuple(out.shape) == (1, 5) and tuple(pre.shape) == (1, 5)
    assert out[0, 1].item() == pytest.approx(0.45, abs=1e-6)
    for j in (0, 2, 3, 4):
        assert out[0, j].item() == 0.0
    assert pre[0, 0].item() == pytest.approx(math.cos(2 * math.pi / 5) * 0.7 - 0.25, abs=1e-6)
    assert pre[0, 3].item() == pytest.approx(math.cos(4 * math.pi / 5) * 0.7 - 0.25, abs=1e-6)


def test_forward_single_enc_dec_order_and_chunking():
    g = torch.Generator().manual_seed(11)
    A = torch.randn(3, 5, generator=g)
    B = torch.randn(3, 5, generator=g)
    b = 0.1 * torch.randn(5, generator=g)
    X = torch.rand(23, 5, generator=g)
    pre, out = an.forward_single(A, B, b, X, want_pre=True, chunk=7)
    ref = (X @ A.T) @ B + b
    assert torch.allclose(pre, ref, atol=1e-5) and torch.allclose(out, torch.relu(ref), atol=1e-5)
    assert not torch.allclose(pre, (X @ B.T) @ A + b, atol=1e-3)  # a swap would invert enc/dec-only
    pre1, out1 = an.forward_single(A, B, b, X, want_pre=True, chunk=1000)
    assert torch.allclose(pre1, pre, atol=1e-6) and torch.allclose(out1, out, atol=1e-6)
    pre_none, out2 = an.forward_single(A, B, b, X, want_pre=False, chunk=7)
    assert pre_none is None and torch.allclose(out2, out, atol=1e-6)


# ------------------------------------------------------------------------------ evaluate_instance

CONFIGS_8 = qz.build_configs([8], GRANS, 5, ABLATIONS, ABL_GRANS)  # 24 configurations


def _check_identities(df: pd.DataFrame) -> None:
    """Column identities fixed by CLAUDE.md / LOG, checked on the table itself (float32 tolerances)."""
    f = {c: df[c].to_numpy().astype(np.float64) for c in df.columns if df[c].dtype.kind == "f"}
    p = f["p"]
    m = p < 1
    leak_fp = np.where(m, f["leak_energy_fp"], 0.0)
    leak_q = np.where(m, f["leak_energy_q"], 0.0)
    assert np.allclose(f["mse_fp"], p * f["cond_mse_fp"] + (1 - p) * leak_fp, atol=1e-6, rtol=1e-4)
    assert np.allclose(f["mse_q"], p * f["cond_mse_q"] + (1 - p) * leak_q, atol=1e-6, rtol=1e-4)
    var = p / 3 - p**2 / 4
    assert np.allclose(f["var_x"], var, atol=1e-6)
    assert np.allclose(f["R2_fp"], 1 - f["mse_fp"] / var, atol=1e-5, rtol=1e-4)
    assert np.allclose(f["R2_q"], 1 - f["mse_q"] / var, atol=1e-5, rtol=1e-4)
    assert np.allclose(f["dR2"], f["R2_fp"] - f["R2_q"], atol=1e-5)
    assert np.allclose(f["dLoss"], f["importance"] * (f["mse_q"] - f["mse_fp"]), atol=1e-6)
    assert np.allclose(f["dcond_mse"], f["cond_mse_q"] - f["cond_mse_fp"], atol=1e-6)
    for name in ("false_act", "leak_energy"):
        d = f["d" + name]
        assert np.allclose(d[m], (f[name + "_q"] - f[name + "_fp"])[m], atol=1e-6)
        assert np.isnan(d[~m]).all() and np.isnan(f[name + "_fp"][~m]).all()
    assert np.isnan(f["near_thr"][~m]).all()
    assert np.all((f["near_thr"][m] >= 0) & (f["near_thr"][m] <= 1))
    assert np.all((f["false_act_fp"][m] >= 0) & (f["false_act_fp"][m] <= 1))
    elig = (f["norm"] > 0.5) & (f["R2_fp"] >= 0.5)
    assert np.array_equal(df["eligible"].to_numpy(), elig)
    assert np.array_equal(df["represented"].to_numpy(), f["norm"] > 0.5)
    rel = f["rel_damage"]
    assert np.isnan(rel[~elig]).all()
    assert np.allclose(rel[elig], (f["dR2"] / f["R2_fp"])[elig], atol=1e-6, rtol=1e-4)
    assert np.allclose(f["margin_tau"], f["margin"] + 0.05, atol=1e-6)
    surv = f["surv_corr"] >= 0.9  # NaN -> False (design note 3)
    assert np.array_equal(df["survived"].to_numpy(), surv)
    comp = f["composite_survived"]
    assert np.isnan(comp[~m]).all()
    expected = (surv & (f["dR2"] <= 0.1) & (f["dfalse_act"] <= 0.05)).astype(float)
    assert np.array_equal(comp[m], expected[m])
    assert np.array_equal(df["killed"].to_numpy(), f["norm_q"] < 0.5)
    assert np.allclose(f["d_norm2"], f["norm_q"] ** 2 - f["norm"] ** 2, atol=1e-5)
    assert np.array_equal(df["margin_flip"].to_numpy(), np.sign(f["margin"]) != np.sign(f["margin_q"]))
    assert np.all(f["peak_out_fp"] >= f["p99_out_fp"] - 1e-6)


def _check_model_table(feats: pd.DataFrame, models: pd.DataFrame) -> None:
    key = ["bits", "granularity", "method", "noise_seed", "ablation"]
    if "instance" in models.columns:
        key = ["instance", "S"] + key
    ff = feats.assign(wl_fp=feats.importance * feats.mse_fp, wl_q=feats.importance * feats.mse_q)
    g = ff.groupby(key)

    def over_eligible(fn):
        return g.apply(lambda x: fn(x) if x.eligible.any() else np.nan, include_groups=False)

    agg = pd.DataFrame(
        {
            "wl_fp": g.wl_fp.sum(),
            "wl_q": g.wl_q.sum(),
            "n_elig": g.eligible.sum(),
            "surv_elig": over_eligible(lambda x: x.survived[x.eligible].mean()),
            "worst_elig": over_eligible(lambda x: x.dR2[x.eligible].max()),
            "mean_elig": over_eligible(lambda x: x.dR2[x.eligible].mean()),
            # n_killed_represented counts REPRESENTED features (norm > 0.5) whose quantized norm drops below 0.5; the
            # other model-level aggregates are over ELIGIBLE features (verified on runs A, B, C)
            "killed_rep": g.apply(lambda x: int((x.killed & x.represented).sum()), include_groups=False),
            "comp_elig": over_eligible(lambda x: x.composite_survived[x.eligible].mean()),
        }
    ).reset_index()
    mm = models.merge(agg, on=key)
    assert len(mm) == len(models) == len(agg)
    assert np.allclose(mm.loss_fp, mm.wl_fp, atol=1e-6, rtol=1e-5)
    assert np.allclose(mm.loss_q, mm.wl_q, atol=1e-6, rtol=1e-5)
    assert np.allclose(mm.loss_increase, mm.loss_q - mm.loss_fp, atol=1e-9)
    assert np.array_equal(mm.n_eligible.to_numpy(), mm.n_elig.to_numpy())
    assert np.allclose(mm.survived_frac, mm.surv_elig, atol=1e-9, equal_nan=True)
    assert np.allclose(mm.worst_dR2, mm.worst_elig, atol=1e-6, equal_nan=True)
    assert np.allclose(mm.mean_dR2, mm.mean_elig, atol=1e-6, equal_nan=True)
    assert np.array_equal(mm.n_killed_represented.to_numpy(), mm.killed_rep.to_numpy())
    if "composite_frac" in mm.columns:
        assert np.allclose(mm.composite_frac, mm.comp_elig, atol=1e-9, equal_nan=True)


def test_evaluate_instance_identity_model_8bit():
    W, b, p = torch.eye(2), torch.zeros(2), 0.5
    imp = np.array([1.0, 0.9])
    es = an.build_eval_set(2, p, seed=an.eval_seed(0, 0, P), params=P)
    feats, models = an.evaluate_instance(W, b, es, P, CONFIGS_8, imp, p)
    assert len(feats) == 48 and len(models) == 24 and set(feats.feature) == {0, 1}
    keys = {(c.bits, c.granularity, c.method, c.noise_seed, c.ablation) for c in CONFIGS_8}
    assert set(zip(feats.bits, feats.granularity, feats.method, feats.noise_seed, feats.ablation, strict=True)) == keys
    assert (
        set(zip(models.bits, models.granularity, models.method, models.noise_seed, models.ablation, strict=True))
        == keys
    )
    # FP: exact reconstruction, analytic values
    assert (feats.cond_mse_fp.abs() < 1e-6).all() and (feats.mse_fp.abs() < 1e-6).all()
    assert (feats.false_act_fp == 0).all() and (feats.leak_energy_fp == 0).all()
    assert (feats.R2_fp > 0.9999).all()
    assert np.allclose(feats.var_x, VAR_05, atol=1e-6)
    assert np.allclose(feats.importance, np.where(feats.feature == 0, 1.0, 0.9))
    assert (feats.p == 0.5).all()
    assert ((feats.peak_out_fp > 0.99) & (feats.peak_out_fp <= 1.0 + 1e-6)).all()
    assert ((feats.p99_out_fp > 0.97) & (feats.p99_out_fp <= 1.0 + 1e-6)).all()
    # FP geometry of two orthogonal unit features with b = 0
    assert np.allclose(feats.norm, 1.0, atol=1e-6) and feats.represented.all()
    assert np.allclose(feats.D, 1.0, atol=1e-6) and np.allclose(feats.D_plus, 1.0, atol=1e-6)
    assert np.allclose(feats.peak_ratio, 1.0, atol=1e-6)
    assert (feats.max_overlap == 0).all() and (feats.margin == 0).all()
    assert np.allclose(feats.margin_tau, 0.05, atol=1e-7)
    assert (feats.superposition == 0).all() and (feats.bias == 0).all() and (feats.leak_mean == 0).all()
    assert ((feats.near_thr >= 0) & (feats.near_thr <= 1)).all()
    # 8-bit: damage within tolerance for every granularity, control and ablation
    assert (feats.dR2.abs() < 1e-3).all() and (feats.dLoss.abs() < 1e-4).all()
    assert (feats.cond_mse_q < 1e-4).all() and (feats.false_act_q < 1e-3).all()
    assert (feats.leak_energy_q < 1e-5).all()
    assert (feats.surv_corr > 0.999).all() and feats.survived.all()
    assert (feats.composite_survived == 1.0).all()
    assert feats.eligible.all() and (~feats.killed).all()
    assert np.allclose(feats.rel_damage, feats.dR2 / feats.R2_fp, atol=1e-6)
    assert np.allclose(feats.norm_q, 1.0, atol=1e-5)
    assert (feats.dW.abs() < 1e-5).all() and (feats.d_overlap.abs() < 1e-5).all()
    assert (feats.d_norm2.abs() < 1e-5).all()
    assert (feats.n_active_eval == 2000).all() and (feats.n_inactive_eval == 20000).all()
    assert es.n_forced_active == 0
    _check_identities(feats)
    # model table
    assert (models.loss_fp.abs() < 1e-6).all() and (models.loss_q.abs() < 1e-4).all()
    assert (models.survived_frac == 1.0).all() and (models.n_eligible == 2).all()
    assert (models.n_killed_represented == 0).all() and (models.frob_err < 1e-5).all()
    _check_model_table(feats, models)


def test_evaluate_instance_leak_model_closed_forms():
    # x_hat_0 = ReLU(x0 + 0.6 x1), x_hat_1 = ReLU(0.6 x0 + 0.36 x1); p = 0.5, b = 0
    W = torch.tensor([[1.0, 0.6], [0.0, 0.0]])  # [m=2, n=2]: column 0 = (1,0), column 1 = (0.6,0)
    b, p = torch.zeros(2), 0.5
    imp = np.array([1.0, 0.9])
    es = an.build_eval_set(2, p, seed=777, params=P)
    feats, models = an.evaluate_instance(W, b, es, P, CONFIGS_8, imp, p)
    f0, f1 = feats[feats.feature == 0], feats[feats.feature == 1]
    cond1 = 0.4096 / 3 + 0.36 / 6 - 2 * 0.64 * 0.6 * 0.5 * 0.25  # E[(0.64 x1 - 0.6 x0)^2 | x1 > 0]
    # geometry
    assert np.allclose(f0.norm, 1.0, atol=1e-6) and np.allclose(f1.norm, 0.6, atol=1e-6)
    assert np.allclose(f0.D, 1 / 1.36, atol=1e-5) and np.allclose(f1.D, 0.1296 / 0.4896, atol=1e-5)
    assert np.allclose(f0.D_plus, 1 / 1.36, atol=1e-5) and np.allclose(f1.D_plus, 0.1296 / 0.4896, atol=1e-5)
    assert np.allclose(feats.max_overlap, 0.6, atol=1e-6) and np.allclose(feats.margin, -0.6, atol=1e-6)
    assert np.allclose(feats.margin_tau, -0.55, atol=1e-6)
    assert np.allclose(f0.superposition, 0.36, atol=1e-5) and np.allclose(f1.superposition, 1.0, atol=1e-5)
    assert np.allclose(feats.leak_mean, 0.15, atol=1e-6) and np.allclose(feats.peak_ratio, 1.0, atol=1e-6)
    assert (feats.bias == 0).all() and feats.represented.all()
    # FP damage metrics vs closed forms (MC tolerance: 2,000 active / 20,000 inactive rows)
    assert np.allclose(f0.cond_mse_fp, 0.06, atol=0.012)
    assert np.allclose(f1.cond_mse_fp, cond1, atol=0.015)
    assert np.allclose(feats.false_act_fp, 0.5 * 11 / 12, atol=0.02)
    assert np.allclose(feats.leak_energy_fp, 0.06, atol=0.005)
    assert np.allclose(f0.mse_fp, 0.06, atol=0.01)
    assert np.allclose(f1.mse_fp, 0.5 * cond1 + 0.03, atol=0.01)
    assert np.allclose(f0.R2_fp, 1 - 0.06 / VAR_05, atol=0.06)
    assert np.allclose(f1.R2_fp, 1 - (0.5 * cond1 + 0.03) / VAR_05, atol=0.06)
    assert (f0.peak_out_fp > 1.0).all()  # leak pushes the active-stratum maximum above 1
    assert (~feats.eligible).all() and feats.rel_damage.isna().all()  # R2_fp < 0.5 (design note 3)
    # 8-bit damage is small; identities hold on non-trivial numbers
    assert (feats.dR2.abs() < 0.02).all() and (feats.dfalse_act.abs() < 0.02).all()
    assert (feats.surv_corr > 0.99).all() and (~feats.killed).all()
    assert (feats.cond_mse_q > 0.03).all() and (feats.false_act_q > 0.4).all()  # not stuck at zero
    _check_identities(feats)
    _check_model_table(feats, models)
    assert np.allclose(models.loss_fp, 1.0 * 0.06 + 0.9 * (0.5 * cond1 + 0.03), atol=0.015)


@pytest.mark.skipif(not SMOKE_DIRS, reason="smoke run not present")
def test_smoke_tables_internally_consistent():
    d = SMOKE_DIRS[-1]
    feats = pd.read_parquet(d / "features.parquet")
    models = pd.read_parquet(d / "models.parquet")
    assert (feats.n_active_eval == 2000).all()
    assert (feats.loc[feats.p < 1, "n_inactive_eval"] == 20000).all()
    assert (feats.loc[feats.p == 1, "n_inactive_eval"] == 0).all()
    p1 = feats[feats.p == 1]
    assert np.allclose(p1.mse_fp, p1.cond_mse_fp, atol=1e-6)
    assert np.allclose(feats.S, 1 - feats.p, atol=1e-6)
    _check_identities(feats)
    _check_model_table(feats, models)


def test_evaluate_instance_2bit_kills_small_column_and_reports_signed_damage():
    # W columns (1,0) and (0.3,0): 2-bit per-tensor RTN has s = 1, so column 1 rounds to exactly 0.
    W = torch.tensor([[1.0, 0.3], [0.0, 0.0]])
    b, p = torch.zeros(2), 0.5
    es = an.build_eval_set(2, p, seed=31, params=P)
    cfgs = [qz.QuantConfig(8, "tensor"), qz.QuantConfig(2, "tensor")]
    feats, models = an.evaluate_instance(W, b, es, P, cfgs, np.array([1.0, 1.0]), p)
    f8, f2 = feats[feats.bits == 8], feats[feats.bits == 2]
    r0, r1 = f2[f2.feature == 0].iloc[0], f2[f2.feature == 1].iloc[0]
    # FP closed forms: x_hat_0 = x0 + 0.3 x1 -> cond_mse_0 = leak_energy_0 = 0.09/6 = 0.015, R2_0 = 0.856
    assert abs(r0.cond_mse_fp - 0.015) < 0.006 and abs(r0.leak_energy_fp - 0.015) < 0.003
    assert abs(r0.false_act_fp - 0.5 * 5 / 6) < 0.02
    assert abs(r0.R2_fp - (1 - 0.015 / VAR_05)) < 0.04 and bool(r0.eligible)
    assert not bool(r1.represented) and not bool(r1.eligible)
    # 2-bit: column 1 zeroed -> killed, constant (zero) quantized output -> NaN correlation, not survived
    assert bool(r1.killed) and r1.norm_q == 0.0 and abs(r1.d_norm2 + 0.09) < 1e-6 and abs(r1.dW - 0.3) < 1e-6
    assert np.isnan(r1.surv_corr) and not bool(r1.survived) and r1.composite_survived == 0.0
    assert np.isnan(r1.rel_damage)
    assert r1.cond_mse_q == pytest.approx(1 / 3, abs=0.02) and r1.false_act_q == 0.0 and r1.leak_energy_q == 0.0
    assert abs(r1.R2_q - (-0.6)) < 0.1 and abs(r1.d_overlap - 0.3) < 1e-6 and bool(r1.margin_flip)
    # feature 0 loses its leaker: exact reconstruction after quantization, SIGNED negative dR2 (improvement)
    assert r0.cond_mse_q < 1e-6 and r0.false_act_q == 0.0 and r0.leak_energy_q == 0.0
    assert r0.dR2 < -0.1 and r0.dfalse_act < -0.3 and r0.dLoss < 0
    assert 0.9 < r0.surv_corr < 0.99 and bool(r0.survived) and r0.composite_survived == 1.0
    assert r0.rel_damage == pytest.approx(r0.dR2 / r0.R2_fp, abs=1e-6)
    assert not bool(r0.killed) and abs(r0.d_overlap - 0.3) < 1e-6 and abs(r0.margin_q) < 1e-6
    # 8-bit leaves everything within tolerance
    assert (f8.dR2.abs() < 1e-2).all()
    # killed_i = (||Q(W)_i|| < 0.5) is defined on the quantized norm alone: the FP-unrepresented column
    # (norm 0.3) is 'killed' at every bit width; only the represented column must not be.
    assert f8.killed.tolist() == [False, True]
    _check_identities(feats)
    _check_model_table(feats, models)
    m2 = models[models.bits == 2].iloc[0]
    assert (
        m2.n_killed_represented == 0 and m2.n_eligible == 1 and m2.survived_frac == 1.0
    )  # killed column is ineligible
    assert abs(m2.frob_err - 0.3) < 1e-6
    # total loss: feature 0 improves (-0.015) but feature 1 now predicts 0 (+0.048) -> net +0.033
    assert abs(m2.loss_increase - 0.0328) < 0.01
    assert m2.loss_increase == pytest.approx(f2.dLoss.sum(), abs=1e-6)


def test_quant_b_ablation_margin_from_quantized_bias():
    W = torch.tensor([[1.0, 0.3], [0.0, 0.0]])
    b, p = torch.tensor([-0.1, -0.03]), 0.5
    es = an.build_eval_set(2, p, seed=32, params=P)
    cfgs = [qz.QuantConfig(2, "tensor", ablation="quant_b"), qz.QuantConfig(2, "tensor")]
    feats, _ = an.evaluate_instance(W, b, es, P, cfgs, np.array([1.0, 1.0]), p)
    qb = feats[feats.ablation == "quant_b"].sort_values("feature")
    plain = feats[feats.ablation == "none"].sort_values("feature")
    # FP geometry is the same in both rows; margin uses FP b
    assert np.allclose(qb.bias, [-0.1, -0.03]) and np.allclose(qb.margin, [-0.2, -0.27], atol=1e-6)
    assert np.allclose(plain.margin, [-0.2, -0.27], atol=1e-6)
    # Q(b) with per-tensor scale 0.1/1: (-0.1, 0); Q(W) zeroes column 1 -> max_overlap_q = 0
    assert np.allclose(qb.margin_q, [0.1, 0.0], atol=1e-6) and qb.margin_flip.tolist() == [True, True]
    assert np.allclose(plain.margin_q, [0.1, 0.03], atol=1e-6) and plain.margin_flip.tolist() == [True, True]
    # the quantized bias is actually used in the forward pass: feature 1 output is ReLU(0) in both,
    # feature 0 differs (b_0 = -0.1 in both, so only through column 1 which is dead): same cond_mse
    assert np.isclose(qb.cond_mse_q.iloc[0], plain.cond_mse_q.iloc[0], atol=1e-6)
    assert qb.false_act_q.iloc[1] == 0.0 and plain.false_act_q.iloc[1] == 0.0
    _check_identities(feats)


def test_evaluate_instance_p1_no_inactive_stratum():
    W, b, p = torch.eye(2), torch.zeros(2), 1.0
    es = an.build_eval_set(2, p, seed=3, params=P)
    feats, models = an.evaluate_instance(W, b, es, P, [qz.QuantConfig(8, "column")], np.array([1.0, 1.0]), p)
    assert (feats.n_inactive_eval == 0).all() and (feats.n_active_eval == 2000).all()
    assert feats.false_act_fp.isna().all() and feats.leak_energy_fp.isna().all() and feats.near_thr.isna().all()
    assert feats.dfalse_act.isna().all() and feats.composite_survived.isna().all()
    assert np.allclose(feats.mse_fp, feats.cond_mse_fp, atol=1e-7) and np.allclose(feats.var_x, 1 / 12)
    assert (feats.R2_fp > 0.9999).all() and feats.survived.all()
    _check_identities(feats)
    _check_model_table(feats, models)
