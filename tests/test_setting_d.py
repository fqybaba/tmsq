"""Setting D: power-law frequencies and the Liu, Liu & Gore weight-decay step (CLAUDE.md test list)."""

import numpy as np
import pytest
import torch

from tmsq.data import FeatureSampler, power_law_frequencies
from tmsq.model import ToyModel
from tmsq.train import LiuDecayAdam, TrainConfig, importance_weighted_loss, train


@pytest.mark.parametrize("alpha,k", [(1.0, 2), (1.0, 5), (1.5, 2), (1.5, 5)])
def test_frequencies_sum_to_k(alpha, k):
    p, c = power_law_frequencies(200, alpha, k)
    assert p.shape == (200,) and abs(p.sum() - k) < 1e-9
    assert (np.diff(p) <= 0).all() and p.max() <= 1.0
    closed = k / ((np.arange(200) + 1.0) ** (-alpha)).sum()
    if (alpha, k) == (1.5, 5):  # the clip binds only here (CLAUDE.md: c = 2.719 instead of 2.023)
        assert p[0] == 1.0 and abs(c - 2.719) < 1e-3 and abs(closed - 2.0234) < 1e-3
    else:
        assert p[0] < 1.0 and abs(c - closed) < 1e-9


def _run(opt_factory, steps=25, seed=0):
    torch.manual_seed(seed)
    model = ToyModel(3, 8, 3, seed=seed)
    sampler = FeatureSampler(8, 0.3, n_instances=3, seed=seed + 1, n_chunks=1)
    opt = opt_factory(model)
    imp = torch.ones(3, 8)
    for step in range(steps):
        lr = 1e-2 * (1 + step % 3)  # varying lr, as under a schedule
        for g in opt.param_groups:
            g["lr"] = lr
        x = sampler.sample(64)
        loss = importance_weighted_loss(x, model(x), imp).sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return model.W.detach().clone(), model.b.detach().clone()


def test_gamma_zero_reproduces_adam_exactly():
    W1, b1 = _run(lambda m: LiuDecayAdam(m, 1e-2, torch.zeros(3)))
    W2, b2 = _run(lambda m: torch.optim.Adam([{"params": [m.W]}, {"params": [m.b]}], lr=1e-2))
    assert torch.equal(W1, W2) and torch.equal(b1, b2)


def test_positive_gamma_reproduces_adamw_on_W_only():
    W1, b1 = _run(lambda m: LiuDecayAdam(m, 1e-2, torch.full((3,), 0.01)))
    W2, b2 = _run(
        lambda m: torch.optim.AdamW(
            [{"params": [m.W], "weight_decay": 0.01}, {"params": [m.b], "weight_decay": 0.0}], lr=1e-2
        )
    )
    assert torch.allclose(W1, W2, atol=1e-6) and torch.allclose(b1, b2, atol=1e-6)


def test_negative_gamma_targets_unit_norm():
    model = ToyModel(1, 4, 3, seed=0)
    with torch.no_grad():
        model.W.zero_()
        model.W[0, :, 0] = torch.tensor([1.0, 0.0, 0.0])  # norm 1: fixed point of the decay term
        model.W[0, :, 1] = torch.tensor([3.0, 0.0, 0.0])  # norm 3: must shrink
        model.W[0, :, 2] = torch.tensor([0.0, 0.2, 0.0])  # norm 0.2: must grow
        model.W[0, :, 3] = torch.tensor([0.0, 0.0, 1.0])
    opt = LiuDecayAdam(model, 0.1, torch.tensor([-1.0]))
    before = model.W.detach().norm(dim=1)[0].clone()
    model.W.grad = torch.zeros_like(model.W)  # isolate the decay term: Adam with zero gradient does nothing
    model.b.grad = torch.zeros_like(model.b)
    opt.step()
    after = model.W.detach().norm(dim=1)[0]
    assert abs(after[0] - 1.0) < 1e-6 and after[1] < before[1] and after[2] > before[2]
    assert after[1] > 1.0 and after[2] < 1.0  # moves toward 1 from both sides without overshooting
    assert torch.allclose(after[1], torch.tensor(3.0 - 0.1 * 1.0 * 3.0 * (1 - 1 / 3)), atol=1e-5)


def test_vector_eval_set_strata():
    from tmsq.analysis import EvalParams, build_eval_set_vec, evaluate_instance
    from tmsq.quantize import QuantConfig

    p, _ = power_law_frequencies(12, 1.5, 5)  # p_0 = 1 (clipped), a few frequent head features, a long tail
    params = EvalParams(n_active=300, n_inactive=1500, n_nat_min=2000)
    es = build_eval_set_vec(12, p, seed=3, params=params)
    ar = torch.arange(12)
    assert (es.X[es.idx_act, ar] > 0).all()
    assert es.has_inactive.tolist() == (p < 1).tolist() and not es.has_inactive[0]
    ok = torch.as_tensor(es.has_inactive)
    assert (es.X[es.idx_inact[:, ok], ar[ok]] == 0).all()
    assert es.n_forced_inactive >= 1 and es.n_forced_active >= 1  # head needs forced-inactive, tail forced-active
    W = torch.randn(4, 12) * 0.5
    rows, _ = evaluate_instance(W, torch.zeros(12) - 0.05, es, params, [QuantConfig(8, "row")], np.ones(12), p)
    clipped = p >= 1.0  # n = 12 clips two features (n = 200 clips one)
    assert (np.isnan(rows.false_act_fp.to_numpy()) == clipped).all()
    assert (np.isnan(rows.near_thr.to_numpy()) == clipped).all() and np.allclose(rows.p, p)
    assert np.isfinite(rows.R2_fp).all()


def test_mixed_gamma_per_instance_and_train_entry():
    model = ToyModel(2, 6, 2, seed=0)
    sampler = FeatureSampler(6, torch.tensor([[0.5] * 6, [0.2] * 6]), n_instances=2, seed=1)
    cfg = TrainConfig(steps=50, batch=64, lr=1e-2, lr_schedule="warmup_cosine", warmup_steps=10, min_lr_frac=0.05,
                      decay_mode="liu", progress=False)  # fmt: skip
    res = train(model, sampler, torch.ones(6), cfg, gamma=torch.tensor([0.1, -1.0]))
    assert res.loss_history.shape == (50, 2) and np.isfinite(res.loss_history).all()
    with pytest.raises(ValueError):
        train(ToyModel(2, 6, 2), sampler, torch.ones(6), cfg)  # gamma missing
