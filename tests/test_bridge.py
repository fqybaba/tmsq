"""Tests for the Phase 6b bridge core (tmsq.bridge): quantizers, streamed statistics, predictors, tests."""

import math

import numpy as np
import pytest
import torch

from tmsq import bridge as B


def test_rtn_lastdim_groups_and_grid():
    torch.manual_seed(0)
    w = torch.randn(8, 32, 16)
    for bits in (4, 6, 8):
        q = B.rtn_lastdim(w, bits)
        qmax = 2 ** (bits - 1) - 1
        s = w.abs().amax(dim=(0, 1), keepdim=True) / qmax
        codes = torch.round(q / s)
        assert torch.allclose(codes * s, q)
        assert codes.min() >= -qmax and codes.max() <= qmax  # the nominal -2^(b-1) is never reached
        assert ((q - w).abs() <= s / 2 + 1e-6).all()
        assert torch.allclose(B.rtn_lastdim(q, bits), q, atol=1e-6)  # idempotent
    # scales are shared across all but the last dim: scaling one head's slice changes every head's codes
    w2 = w.clone()
    w2[0, 0, 3] = 100.0
    assert not torch.allclose(B.rtn_lastdim(w2, 4)[1, :, 3], B.rtn_lastdim(w, 4)[1, :, 3])
    assert torch.allclose(B.rtn_lastdim(w2, 4)[1, :, 2], B.rtn_lastdim(w, 4)[1, :, 2])


def test_rtn_rows_per_row_and_zero_row():
    torch.manual_seed(1)
    w = torch.randn(10, 7)
    w[3] = 0.0
    q = B.rtn_rows(w, 4)
    s = w.abs().amax(dim=1, keepdim=True) / 7
    assert torch.equal(q[3], torch.zeros(7))
    keep = torch.arange(10) != 3
    assert ((q - w).abs()[keep] <= s[keep] / 2 + 1e-6).all()
    assert torch.round(q[keep] / s[keep]).abs().max() <= 7
    with pytest.raises(ValueError):
        B.rtn_rows(torch.randn(2, 3, 4), 4)


@pytest.mark.parametrize("arm,shape", [("code", (4, 6, 5)), ("stated", (9, 5))])
def test_noise_matched_group_norms(arm, shape):
    torch.manual_seed(2)
    w = torch.randn(*shape)
    q = B.rtn_lastdim(w, 4) if arm == "code" else B.rtn_rows(w, 4)
    v = B.noise_matched(w, q, arm, torch.Generator().manual_seed(0))
    assert torch.allclose(B.group_norms(v - w, arm), B.group_norms(q - w, arm), atol=1e-5)
    assert not torch.allclose(v, q)


def _sparse(rng, T, d, rate):
    return np.where(rng.random((T, d)) < rate, rng.random((T, d)) * 3, 0.0).astype(np.float32)


def test_pairstats_matches_numpy_and_groups_add():
    rng = np.random.default_rng(3)
    T, d = 600, 5
    x = _sparse(rng, T, d, 0.3)
    y = np.maximum(x + rng.normal(0, 0.3, (T, d)).astype(np.float32), 0)
    thr = np.full(d, 0.1)
    st = B.PairStats(3, thr)
    edges = [0, 200, 450, 600]
    for g in range(3):
        st.update(g, torch.from_numpy(x[edges[g] : edges[g + 1]]), torch.from_numpy(y[edges[g] : edges[g + 1]]))
    o = B.outcomes(st.merged())
    for j in range(d):
        assert math.isclose(o["corr"][j], np.corrcoef(x[:, j], y[:, j])[0, 1], rel_tol=1e-5)
        a = x[:, j] > 0
        assert math.isclose(o["corr_act"][j], np.corrcoef(x[a, j], y[a, j])[0, 1], rel_tol=1e-4)
        assert math.isclose(o["dfalse"][j], ((~a) & (y[:, j] > 0.1)).sum() / (~a).sum(), rel_tol=1e-9)
    sub = B.outcomes(st.merged([0, 2]))
    idx = np.r_[0:200, 450:600]
    for j in range(d):
        assert math.isclose(sub["corr"][j], np.corrcoef(x[idx, j], y[idx, j])[0, 1], rel_tol=1e-5)


def test_killed_features_get_zero_correlation():
    """A variant feature that never fires gets corr = 0 (Duan: damaged), over several groups and batches with
    non-dyadic values; one that is zero on every FP-active token gets corr_act = 0."""
    rng = np.random.default_rng(7)
    T, d = 3000, 4
    x = (_sparse(rng, T, d, 0.2) * 0.7).astype(np.float32)
    y = np.maximum(x + rng.normal(0, 0.05, (T, d)).astype(np.float32), 0)
    y[:, 1] = 0.0  # killed
    y[:, 2] = np.where(x[:, 2] > 0, 0.0, 0.3)  # zero on the active stratum, leaks elsewhere
    st = B.PairStats(4, np.full(d, 0.01))
    for g, (lo, hi) in enumerate([(0, 700), (700, 1500), (1500, 2600), (2600, 3000)]):
        for s in range(lo, hi, 256):
            e = min(s + 256, hi)
            st.update(g, torch.from_numpy(x[s:e]), torch.from_numpy(y[s:e]))
    o = B.outcomes(st.merged())
    assert o["corr"][1] == 0.0 and o["corr_act"][1] == 0.0 and B.label(o["corr"])[1] == 0
    assert o["corr_act"][2] == 0.0 and o["dfalse"][2] == 1.0
    assert o["corr"][0] > 0.95 and math.isclose(o["corr"][0], np.corrcoef(x[:, 0], y[:, 0])[0, 1], rel_tol=1e-9)
    assert B.label(np.array([0.95, 0.9, 0.7, 0.5, 0.49])).tolist() == [2, 1, 1, 1, 0]


def test_fpstats_and_nearthr():
    f = torch.tensor([[0.0, 1.0], [2.0, 0.0], [0.0, 3.0], [0.0, 0.0]])
    st = B.FPStats(2)
    st.update(f[:2])
    st.update(f[2:])
    t = st.table()
    assert np.allclose(t["rate"], [0.25, 0.5]) and np.allclose(t["mean"], [0.5, 1.0])
    assert np.allclose(t["peak"], [2.0, 3.0]) and np.allclose(t["mu_plus"], [2.0, 2.0])
    z = torch.tensor([[-0.05, -0.2], [0.0, -0.09], [-0.11, 0.3], [-0.01, -0.1]])
    nt = B.NearThrStats(np.array([0.1, 0.1]))
    nt.update(z)
    assert np.allclose(nt.value(), [3 / 4, 1 / 4])  # (-0.1, 0]: col 0: -0.05, 0, -0.01; col 1: -0.09


def test_duan_predictors_floors():
    p = B.duan_predictors(np.array([0.0, 0.01]), np.array([0.0, 0.02]), np.array([0.0, 0.01]))
    assert np.isclose(p["rarity"][0], -np.log(1e-8)) and np.isclose(p["log_peak"][1], np.log(0.01))
    assert p["log_conc"][1] == 0.0  # peak < mean -> floored at log(1)


def _unit(deg):
    return torch.tensor([math.cos(math.radians(deg)), math.sin(math.radians(deg))])


def test_margin_analog_reproduces_toy_margin():
    """With tied unit decoder/encoder, mu+ = 1 and b_enc = b, the analog equals the toy-model margin."""
    W = torch.stack([_unit(72 * i) for i in range(5)])  # pentagon rows d_k
    m = B.margin_analog(W.T.contiguous(), W, torch.full((5,), -0.25), np.ones(5), np.ones(5, bool), chunk=2)
    assert np.allclose(m["margin"], 0.25 - math.cos(math.radians(72)), atol=1e-6)  # the running example: -0.059
    A = torch.stack([_unit(0), _unit(180), _unit(90), _unit(270)])  # two antipodal pairs
    m = B.margin_analog(A.T.contiguous(), A, torch.zeros(4), np.ones(4), np.ones(4, bool))
    assert np.allclose(m["interf"], 0.0) and np.allclose(m["margin"], 0.0)
    # inactive features are ignored as leakers and get NaN
    act = np.array([True, True, False, True])
    m = B.margin_analog(A.T.contiguous(), A, torch.zeros(4), np.ones(4), act)
    assert np.isnan(m["margin"][2]) and np.isfinite(m["margin"][act]).all()


def test_margin_analog_scales_by_mu():
    W = torch.stack([_unit(0), _unit(60)])
    m = B.margin_analog(W.T.contiguous(), W, torch.tensor([-1.0, -1.0]), np.array([2.0, 4.0]), np.ones(2, bool))
    # feature 0: interf = cos60 * mu_1 = 2; margin = (1 - 2) / 2
    assert np.isclose(m["interf"][0], 2.0) and np.isclose(m["margin"][0], -0.5)
    assert np.isclose(m["interf"][1], 1.0) and np.isclose(m["margin"][1], 0.0)
    s = B.interference_sum(W.T.contiguous(), W, np.array([2.0, 4.0]), np.array([0.1, 0.5]), np.ones(2, bool))
    assert np.isclose(s[0], 0.5 * 0.5 * 4) and np.isclose(s[1], 0.1 * 0.5 * 2)  # plan D3: not divided by mu_j+


def test_partial_spearman_and_strata():
    rng = np.random.default_rng(4)
    c = rng.normal(size=3000)
    a = c + rng.normal(size=3000)
    b = c + rng.normal(size=3000)
    assert abs(B.partial_spearman(a, b, c)) < 0.06
    assert B.partial_spearman(a, a + 0.3 * rng.normal(size=3000), c) > 0.5
    bins = B.strata(c, 10)
    assert np.bincount(bins).tolist() == [300] * 10
    p = B.permute_within(a, bins, rng)
    for k in range(10):
        assert sorted(p[bins == k]) == sorted(a[bins == k])


def test_family_a_and_b_detect_signal():
    rng = np.random.default_rng(5)
    n = 1500
    lp = rng.normal(size=n)
    sig = rng.normal(size=n)
    y = (lp + 1.5 * sig + rng.normal(size=n) > 0).astype(int)
    reg = B.Registered(cv_repeats=2, perm_repeats=1, n_perm_a=30, n_perm_b=200)
    good = B.family_a(y, lp, sig, reg)
    null = B.family_a(y, lp, rng.normal(size=n), reg)
    assert good["dauc"] > 0.1 and good["p"] < 0.05 and good["coef_pred"] > 0
    assert abs(null["dauc"]) < 0.02 and null["p"] > 0.05
    df = np.abs(sig) + 0.1 * rng.normal(size=n)
    assert B.family_b(df, lp, np.abs(sig), reg)["p"] < 0.01
    assert B.family_b(df, lp, rng.normal(size=n), reg)["p"] > 0.01


def test_two_controls():
    rng = np.random.default_rng(6)
    n = 2000
    c1, c2 = rng.normal(size=n), rng.normal(size=n)
    C = np.column_stack([c1, c2])
    bins = B.strata(C, 10)
    assert len(np.unique(bins)) == 25 and np.bincount(bins).min() > 40
    # a predictor that is only a proxy of c2 adds to c1 alone but not to (c1, c2)
    proxy = c2 + 0.2 * rng.normal(size=n)
    y = (c1 + 1.5 * c2 + rng.normal(size=n) > 0).astype(int)
    reg = B.Registered(cv_repeats=2, perm_repeats=1, n_perm_a=30, n_perm_b=200)
    assert B.family_a(y, c1, proxy, reg)["dauc"] > 0.05
    assert abs(B.family_a(y, C, proxy, reg)["dauc"]) < 0.01
    out = c2 + 0.5 * rng.normal(size=n)
    assert B.partial_spearman(proxy, out, c1) > 0.5
    assert abs(B.partial_spearman(proxy, out, C, B.strata(C, 10))) < 0.06


def test_nonlinear_shared_confounder_is_not_a_finding():
    """Review finding: predictor and outcome both depend nonlinearly (U-shape) on the control only. The
    cell-consistent statistics must not reject more often than about alpha."""
    reg = B.Registered(cv_repeats=1, perm_repeats=1, n_perm_a=40, n_perm_b=200)
    rej_b, daucs = 0, []
    for s in range(10):
        rng = np.random.default_rng(100 + s)
        lp = rng.normal(size=2000)
        pred = lp**2 + 0.5 * rng.normal(size=2000)
        dfalse = np.maximum(lp**2 + 0.5 * rng.normal(size=2000) - 0.5, 0)
        r = B.family_b(dfalse, lp, pred, reg)
        rej_b += abs(r["rho"]) >= reg.rho_min and r["p"] < reg.alpha
        y = (lp**2 + rng.normal(size=2000) > 1).astype(int)
        daucs.append(B.family_a(y, lp, pred, reg)["dauc"])
    assert rej_b <= 2
    assert max(daucs) < reg.dauc_min
    # the old global-linear statistic is fooled by the same data
    assert B.partial_spearman(pred, dfalse, lp) > 0.3


def test_cell_projector_single_cell_equals_global_fit():
    rng = np.random.default_rng(8)
    a, b, c = rng.normal(size=(3, 500))
    g = B.partial_spearman(a + c, b + c, c)
    one = B.partial_spearman(a + c, b + c, c, np.zeros(500, dtype=int))
    assert math.isclose(g, one, rel_tol=1e-12)


def test_holm_and_decisions():
    h = B.holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.5})
    assert np.isclose(h["a"], 0.04) and np.isclose(h["c"], 0.09) and np.isclose(h["b"], 0.09) and h["d"] == 0.5
    reg = B.Registered()
    r = {"dauc": 0.05, "coef_pred": -0.4}
    assert B.decide_a(r, 0.001, reg, -1) == "supported"
    assert B.decide_a({**r, "coef_pred": 0.4}, 0.001, reg, -1) == "adds information, opposite to prediction"
    assert B.decide_a({**r, "dauc": 0.01}, 0.001, reg, -1) == "no added information beyond peak"
    assert B.decide_a(r, 0.2, reg, 0) == "no added information beyond peak"
    assert B.decide_a(r, 0.001, reg, 0).startswith("adds information beyond peak (coefficient -")
    assert B.decide_b({"rho": 0.2}, 0.01, reg, +1) == "supported"
    assert B.decide_b({"rho": -0.2}, 0.01, reg, +1) == "associated, opposite to prediction"
    assert B.decide_b({"rho": 0.05}, 0.01, reg, +1) == "no association beyond peak"
