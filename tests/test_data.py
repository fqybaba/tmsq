import torch

from tmsq.data import FeatureSampler


def test_sparsity_rate_matches_feature_prob():
    p = torch.tensor([[1.0, 0.5, 0.1, 0.01, 0.0]])
    s = FeatureSampler(5, p, n_instances=1, seed=0)
    x = s.sample(200_000)
    rate = (x > 0).float().mean(dim=1)[0]
    assert torch.allclose(rate, p[0], atol=0.005), rate


def test_values_uniform_on_unit_interval():
    s = FeatureSampler(3, 1.0, n_instances=2, seed=1)
    x = s.sample(100_000)
    assert x.min() >= 0 and x.max() < 1
    assert abs(x.mean().item() - 0.5) < 0.01
    assert abs((x**2).mean().item() - 1 / 3) < 0.01


def test_shape_and_per_instance_density():
    p = torch.tensor([[1.0], [0.2]])
    s = FeatureSampler(4, p, n_instances=2, seed=3)
    x = s.sample(50_000)
    assert x.shape == (2, 50_000, 4)
    assert (x[0] > 0).float().mean() > 0.999
    assert abs((x[1] > 0).float().mean().item() - 0.2) < 0.01


def test_determinism_under_seed():
    a = FeatureSampler(6, 0.3, n_instances=3, seed=42).sample(1000)
    b = FeatureSampler(6, 0.3, n_instances=3, seed=42).sample(1000)
    c = FeatureSampler(6, 0.3, n_instances=3, seed=43).sample(1000)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_conditional_sampling():
    s = FeatureSampler(4, 0.1, n_instances=1, seed=0)
    on = s.sample_conditional(20_000, feature=2, active=True)
    off = s.sample_conditional(20_000, feature=2, active=False)
    assert (on[:, :, 2] > 0).all()
    assert (off[:, :, 2] == 0).all()
    assert abs((on[:, :, 0] > 0).float().mean().item() - 0.1) < 0.02
