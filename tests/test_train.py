import numpy as np
import torch

from tmsq.data import FeatureSampler
from tmsq.model import ToyModel
from tmsq.train import TrainConfig, importance_weighted_loss, lr_at, plateau_ratio, train


def test_loss_definition_mean_over_batch_sum_over_features():
    x = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])  # K=1, B=2, n=2
    out = torch.zeros_like(x)
    imp = torch.tensor([[1.0, 0.5]])
    loss = importance_weighted_loss(x, out, imp)
    # sample 1: 1*1 + 0.5*0 = 1 ; sample 2: 1*0 + 0.5*1 = 0.5 ; mean = 0.75
    assert torch.allclose(loss, torch.tensor([0.75]))


def test_training_reduces_loss_and_is_deterministic():
    def run():
        model = ToyModel(2, 5, 2, seed=0)
        sampler = FeatureSampler(5, torch.tensor([[1.0], [0.2]]), n_instances=2, seed=1)
        imp = torch.tensor(0.9 ** np.arange(5), dtype=torch.float32)
        cfg = TrainConfig(steps=300, batch=256, progress=False, snapshot_steps=(150,))
        res = train(model, sampler, imp, cfg)
        return model.W.detach().clone(), res

    W1, r1 = run()
    W2, r2 = run()
    assert torch.equal(W1, W2)
    assert r1.loss_history.shape == (300, 2)
    assert (r1.loss_history[-20:].mean(axis=0) < r1.loss_history[:20].mean(axis=0)).all()
    assert set(r1.snapshots) == {150}
    assert r1.snapshots[150][0].shape == (2, 2, 5)


def test_lr_schedules():
    c = TrainConfig(steps=100, lr=1e-2, lr_schedule="constant")
    assert lr_at(0, c) == lr_at(99, c) == 1e-2
    w = TrainConfig(steps=100, lr=1e-2, lr_schedule="warmup_cosine", warmup_steps=10, min_lr_frac=0.05)
    assert lr_at(0, w) == 0.0 and abs(lr_at(10, w) - 1e-2) < 1e-12
    assert abs(lr_at(100, w) - 0.05e-2) < 1e-12


def test_plateau_ratio():
    hist = np.ones((1000, 3))
    hist[:, 1] = np.linspace(1, 0.5, 1000)
    pr = plateau_ratio(hist, frac=0.1)
    assert np.isclose(pr[0], 1.0) and pr[1] < 1.0 and np.isclose(pr[2], 1.0)
