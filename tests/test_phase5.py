"""Phase 5: mixed-precision quantizer, allocation rules, calibration data and greedy rounding."""

import numpy as np
import pytest
import torch

from tmsq import phase5 as P5
from tmsq.quantize import rtn


def _model(m=6, n=12, seed=0):
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(m, n, generator=g) / np.sqrt(m)
    W[:, -1] = 0.0  # a dead column
    b = -0.1 * torch.rand(n, generator=g)
    return W, b


@pytest.mark.parametrize("bits", [2, 3, 4, 5, 8])
def test_rtn_mixed_uniform_equals_phase2_column_rtn_bit_for_bit(bits):
    W, _ = _model()
    Q, codes, s = P5.rtn_mixed(W, np.full(W.shape[1], bits))
    Q2, s2 = rtn(W[None], bits, "column")
    assert torch.equal(Q, Q2[0]) and torch.equal(s, s2[0])
    assert torch.all(Q[:, -1] == 0)  # dead column: eps-guarded scale, no NaN


def test_rtn_mixed_column_codebooks():
    W, _ = _model()
    bits = np.array([3, 5] * 6)
    _, codes, _ = P5.rtn_mixed(W, bits)
    q = 2 ** (bits - 1) - 1
    assert torch.all(codes.abs() <= torch.as_tensor(q, dtype=torch.float32)[None, :])
    assert torch.all(codes == codes.round())


def test_allocation_split_ties_nan_and_odd_counts():
    rep = np.array([True] * 7 + [False] * 3)
    key = np.array([0.3, 0.1, 0.1, np.nan, 0.5, -0.2, 0.0, 9.0, 9.0, 9.0])
    bits = P5.allocation_bits(key, rep, descending=False)
    # pool = represented & finite = features 0,1,2,4,5,6 (6 of them); ascending: 5(-0.2),6(0.0),1(0.1),2(0.1),0,4
    assert list(np.flatnonzero(bits == 5)) == [1, 5, 6] and list(np.flatnonzero(bits == 3)) == [0, 2, 4]
    assert bits[3] == 4 and np.all(bits[7:] == 4)  # NaN key and unrepresented -> 4 bits
    bits_odd = P5.allocation_bits(np.arange(5.0), np.ones(5, bool), descending=True)
    assert list(bits_odd) == [3, 3, 4, 5, 5]
    for b_ in (bits, bits_odd):
        assert b_.mean() == pytest.approx(4.0)


def test_random_bits_reproducible_and_budget():
    rep = np.ones(100, bool)
    a = P5.random_bits(rep, 13, 3, 0)
    assert np.array_equal(a, P5.random_bits(rep, 13, 3, 0)) and not np.array_equal(a, P5.random_bits(rep, 13, 3, 1))
    assert a.mean() == 4.0 and (a == 5).sum() == 50


def test_calibration_set_rate_and_near_thr_nan_for_always_active():
    p = np.array([1.0, 0.5, 0.05])
    X = P5.calibration_set(3, p, 123, N=20000)
    rate = (X > 0).double().mean(0).numpy()
    assert rate[0] == 1.0 and abs(rate[1] - 0.5) < 0.02 and abs(rate[2] - 0.05) < 0.01
    W = torch.eye(3)[:2]
    nt = P5.near_thr_cal(W, torch.zeros(3), X, 0.05, 0.05)
    assert np.isnan(nt[0]) and np.isfinite(nt[1:]).all()


@pytest.mark.parametrize("objective", ["leak", "loss"])
def test_rank1_flip_updates_match_brute_force(objective):
    W, b = _model(seed=3)
    X = P5.calibration_set(W.shape[1], 0.3, 7, N=500).double()
    imp = torch.ones(W.shape[1], dtype=torch.float64)
    Q = (P5.rtn_mixed(W, np.full(W.shape[1], 4))[0]).double()
    H = X @ Q.T
    PRE = H @ Q + b.double()
    rng = np.random.default_rng(0)
    for _ in range(25):
        i, k = int(rng.integers(0, W.shape[1] - 1)), int(rng.integers(0, W.shape[0]))
        delta = torch.as_tensor(rng.normal(0, 0.05, W.shape[0]))
        d = P5._flip_deltas(PRE, H, Q, X, imp, i, delta, objective)
        for kk in range(W.shape[0]):  # predicted objective change of each candidate == brute force
            Qn = Q.clone()
            Qn[kk, i] += delta[kk]
            brute = (
                P5.objective_value(Qn, b.double(), X, imp, objective)
                - P5.objective_value(Q, b.double(), X, imp, objective)
            ) * X.shape[0]
            assert float(d[kk]) == pytest.approx(brute, rel=1e-9, abs=1e-9)
        P5._apply_flip(PRE, H, Q, X, i, k, float(delta[k]))  # incremental state == recomputation
        assert torch.allclose(H, X @ Q.T, atol=1e-12) and torch.allclose(PRE, (X @ Q.T) @ Q + b.double(), atol=1e-12)


@pytest.mark.parametrize("objective", ["leak", "loss"])
def test_greedy_round_invariants(objective):
    W, b = _model(seed=5)
    X = P5.calibration_set(W.shape[1], 0.3, 11, N=1000)
    cols = np.arange(W.shape[1])
    Q, info = P5.greedy_round(W, b, X, np.ones(W.shape[1]), objective, cols, bits=4)
    Q0, codes0, s = P5.rtn_mixed(W, np.full(W.shape[1], 4))
    assert info["J_final"] <= info["J0"] + 1e-12 and info["n_flips"] > 0
    assert info["J_tracked"] == pytest.approx(info["J_final"], rel=1e-6, abs=1e-10)
    codes = info["codes"]
    u = W / s
    assert torch.all((codes - u).abs() < 1.0 + 1e-6) and torch.all(codes.abs() <= 7)  # one grid step, in range
    assert torch.equal(Q, s * codes)  # on the grid
    Qz, infoz = P5.greedy_round(W, b, X, np.ones(W.shape[1]), objective, cols, bits=4, passes=0)
    assert torch.equal(Qz, Q0) and infoz["n_flips"] == 0  # zero passes = RTN
