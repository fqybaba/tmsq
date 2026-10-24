import math

import torch

from tmsq.model import ToyModel


def test_shapes_and_forward_pass():
    m = ToyModel(3, 5, 2, seed=0)
    x = torch.rand(3, 7, 5)
    out = m(x)
    assert m.W.shape == (3, 2, 5) and m.b.shape == (3, 5)
    assert out.shape == (3, 7, 5)
    assert (out >= 0).all()


def test_forward_matches_definition():
    m = ToyModel(2, 6, 3, seed=1)
    with torch.no_grad():
        m.b.uniform_(-0.5, 0.5)
    x = torch.rand(2, 11, 6)
    out = m(x)
    for k in range(2):
        W = m.W[k]
        ref = torch.relu(W.T @ (W @ x[k].T) + m.b[k][:, None]).T
        assert torch.allclose(out[k], ref, atol=1e-6)


def test_init_is_per_instance_xavier_and_bias_zero():
    m = ToyModel(200, 100, 20, seed=0)
    expected = math.sqrt(2 / (100 + 20))
    assert abs(m.W.std().item() - expected) < 0.01
    assert torch.equal(m.b, torch.zeros_like(m.b))
    m2 = ToyModel(200, 100, 20, seed=0, init_std=0.03)
    assert abs(m2.W.std().item() - 0.03) < 0.003


def test_init_determinism_and_instance_independence():
    a = ToyModel(4, 5, 2, seed=7).W.detach()
    b = ToyModel(4, 5, 2, seed=7).W.detach()
    c = ToyModel(4, 5, 2, seed=8).W.detach()
    assert torch.equal(a, b)
    assert not torch.equal(a, c)
    assert not torch.equal(a[0], a[1])
    # instance k depends only on (seed, k): the first 2 instances of a 4-instance model equal a 2-instance model
    d = ToyModel(2, 5, 2, seed=7).W.detach()
    assert torch.equal(a[:2], d)


def test_run_with_separate_encoder_decoder():
    m = ToyModel(1, 4, 2, seed=0)
    x = torch.rand(1, 5, 4)
    assert torch.allclose(ToyModel.run(m.W, m.W, m.b, x), m(x))
    W2 = m.W.detach() * 0.5
    out = ToyModel.run(m.W, W2, m.b, x)
    assert out.shape == (1, 5, 4)
