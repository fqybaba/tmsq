import numpy as np
import torch

from tmsq.analysis import EvalParams, build_eval_set, evaluate_instance
from tmsq.metrics import (
    d_plus,
    leak_mean,
    margin_tau,
    mse_from_strata,
    peak_ratio,
    pearson_columns,
    perturbation_covariates,
    r2_from_strata,
    var_x,
)
from tmsq.quantize import QuantConfig


def test_var_and_mse_decomposition_against_direct_estimate():
    rng = np.random.default_rng(0)
    p = 0.3
    N = 400_000
    x = (rng.random(N) < p) * rng.random(N)
    xhat = 0.8 * x + 0.02  # a leaky, under-gained reconstruction
    assert abs(var_x(p) - x.var()) < 2e-3
    act, inact = x > 0, x == 0
    cond_mse = ((x - xhat) ** 2)[act].mean()
    leak = (xhat**2)[inact].mean()
    mse_direct = ((x - xhat) ** 2).mean()
    assert abs(mse_from_strata(p, cond_mse, leak) - mse_direct) < 2e-4
    r2_direct = 1 - mse_direct / x.var()
    assert abs(r2_from_strata(p, cond_mse, leak) - r2_direct) < 5e-3


def test_p_one_and_p_zero_edge_cases():
    assert mse_from_strata(1.0, 0.1, np.nan) == 0.1
    assert np.isnan(r2_from_strata(0.0, 0.1, 0.0))
    assert np.isclose(var_x(1.0), 1 / 12)


def test_pearson_pathologies():
    a = np.random.default_rng(1).random((500, 3))
    b = a.copy()
    b[:, 0] *= 0.6  # pure gain change -> correlation 1 (scale-blind)
    b[:, 1] = 0.5  # constant -> NaN
    r = pearson_columns(a, b)
    assert np.isclose(r[0], 1.0) and np.isnan(r[1]) and np.isclose(r[2], 1.0)


def test_sign_aware_dimensionality_and_peak_ratio():
    antipodal = torch.tensor([[[1.0, -1.0]]])  # D = 0.5 but no positive overlap -> D_plus = 1
    assert np.allclose(d_plus(antipodal), 1.0)
    ang = 2 * np.pi * np.arange(5) / 5
    pent = torch.tensor(np.stack([np.cos(ang), np.sin(ang)])[None])
    assert np.allclose(d_plus(pent), 1 / (1 + 2 * np.cos(2 * np.pi / 5) ** 2))
    W = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    pr = peak_ratio(W)
    assert np.isclose(pr[0, 0], 1.0) and np.isnan(pr[0, 2])
    assert np.allclose(margin_tau(pent, torch.full((1, 5), -0.25)), 0.05 + 0.25 - np.cos(2 * np.pi / 5))
    lm = leak_mean(pent, 0.2)
    assert np.allclose(lm, 0.2 * 2 * np.cos(2 * np.pi / 5) / 2)


def test_perturbation_covariates_on_scaled_matrix():
    W = torch.tensor([[[1.0, 0.0, 0.3], [0.0, 1.0, 0.0]]])
    b = torch.zeros(1, 3)
    Wq = W * torch.tensor([1.1, 1.0, 0.0])  # gain up, unchanged, killed
    c = perturbation_covariates(W, Wq, b, b)
    assert c["killed"].tolist() == [[False, False, True]]
    assert np.allclose(c["d_norm2"][0], [1.1**2 - 1, 0.0, -0.09])
    assert np.allclose(c["dW"][0], [0.1, 0.0, 0.3])
    assert np.isclose(c["d_overlap"][0, 0], 0.3)  # overlap with column 2 went from 0.3 to 0


def test_eval_set_and_instance_evaluation_end_to_end():
    torch.manual_seed(0)
    n, m = 6, 3
    params = EvalParams(n_active=300, n_inactive=1000, n_nat_min=4000, n_nat_max=8000)
    es = build_eval_set(n, p=0.3, seed=5, params=params)
    assert es.idx_act.shape == (300, n) and es.idx_inact.shape[1] == n
    assert es.n_inactive_eval == 1000 and es.n_forced_active == 0
    ar = torch.arange(n)
    assert (es.X[es.idx_act, ar] > 0).all() and (es.X[es.idx_inact, ar] == 0).all()
    es_sparse = build_eval_set(n, p=0.02, seed=5, params=params)
    assert es_sparse.n_forced_active == n  # natural active rows < 300 -> forced batches
    assert (es_sparse.X[es_sparse.idx_act, ar] > 0).all()
    es_dense = build_eval_set(n, p=1.0, seed=5, params=params)
    assert es_dense.idx_inact is None and es_dense.n_inactive_eval == 0

    W = torch.randn(m, n) * 0.5
    b = torch.zeros(n) - 0.1
    cfgs = [QuantConfig(8, "column"), QuantConfig(2, "column")]
    rows, models = evaluate_instance(W, b, es, params, cfgs, importance=np.ones(n), p=0.3)
    assert len(rows) == 2 * n and len(models) == 2
    r8 = rows[rows.bits == 8]
    assert (r8.dR2.abs() < 0.05).all()  # random untrained model: 8-bit perturbs R2 only slightly
    assert (r8.surv_corr > 0.99).all()
    assert r8.n_active_eval.iloc[0] == 300 and r8.n_inactive_eval.iloc[0] == 1000
    r2 = rows[rows.bits == 2]
    assert r2.dR2.mean() > r8.dR2.mean()
    assert models.loc[models.bits == 8, "loss_increase_rel"].iloc[0] < 0.05
    assert models.loc[models.bits == 2, "loss_increase"].iloc[0] > models.loc[models.bits == 8, "loss_increase"].iloc[0]
