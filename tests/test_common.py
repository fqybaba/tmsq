import numpy as np
import pytest
import torch

from tmsq.experiments.common import expand_sweep, importance_curve, load_weights, save_weights, train_config_from


def test_save_and_load_weights_round_trip(tmp_path):
    W, b = torch.randn(3, 2, 5), torch.randn(3, 5)
    density = np.array([1.0, 0.5, 0.1])
    save_weights(tmp_path, "final", W, b, density=density, seed_idx=np.arange(3), importance=0.9 ** np.arange(5))
    d = load_weights(tmp_path / "final.pt")
    assert torch.equal(d["W"], W) and torch.equal(d["b"], b)
    assert np.array_equal(d["density"], density) and d["seed_idx"].tolist() == [0, 1, 2]


def test_steps_override_and_validation():
    cfg = {"train": {"steps": 2000, "batch": 8, "lr": 1e-3, "snapshot_half": True}}
    assert train_config_from(cfg).steps == 2000
    assert train_config_from(cfg, 50).steps == 50
    assert train_config_from(cfg, 50).snapshot_steps == (25,)
    with pytest.raises(ValueError):
        train_config_from(cfg, 0)


def test_expand_sweep_and_importance():
    dens, sweep, seed = expand_sweep(np.array([1.0, 0.1]), 3)
    assert dens.tolist() == [1.0, 1.0, 1.0, 0.1, 0.1, 0.1]
    assert sweep.tolist() == [0, 0, 0, 1, 1, 1] and seed.tolist() == [0, 1, 2, 0, 1, 2]
    assert np.allclose(importance_curve(4, 0.5), [1, 0.5, 0.25, 0.125])
    assert np.allclose(importance_curve(3, 1.0), 1.0)
