import pytest
import torch

from tmsq.quantize import (
    QuantConfig,
    apply_quant,
    build_configs,
    noise_control,
    qmax_for,
    quantize_bias,
    rtn,
    rtn_scales,
)


@pytest.mark.parametrize("bits", [2, 3, 4, 5, 6, 8])
def test_codebook_size_on_dense_input(bits):
    W = torch.linspace(-1, 1, 100_001)[None, None, :]  # [1, 1, N] spans the range densely
    Q, s = rtn(W, bits, "tensor")
    assert torch.unique(Q).numel() == 2**bits - 1
    assert bits != 2 or set(torch.unique(Q).tolist()) == {-1.0, 0.0, 1.0}


@pytest.mark.parametrize("granularity", ["tensor", "column", "row"])
def test_error_bound_and_idempotence(granularity):
    W = torch.randn(3, 4, 7)
    for bits in (2, 4, 8):
        Q, s = rtn(W, bits, granularity)
        assert ((Q - W).abs() <= s / 2 + 1e-7).all()
        Q2, _ = rtn(Q, bits, granularity)
        assert torch.allclose(Q2, Q)


def test_scale_shapes():
    W = torch.randn(3, 4, 7)
    assert rtn_scales(W, 4, "tensor").shape == (3, 1, 1)
    assert rtn_scales(W, 4, "column").shape == (3, 1, 7)
    assert rtn_scales(W, 4, "row").shape == (3, 4, 1)
    assert torch.allclose(rtn_scales(W, 4, "tensor")[:, 0, 0], W.abs().amax(dim=(1, 2)) / qmax_for(4))


def test_zero_column_and_row_quantize_to_zero_without_nan():
    W = torch.randn(2, 4, 6)
    W[:, :, 2] = 0.0
    W[:, 1, :] = 0.0
    for g in ("column", "row", "tensor"):
        Q, _ = rtn(W, 3, g)
        assert torch.isfinite(Q).all()
        assert (Q[:, :, 2] == 0).all() and (Q[:, 1, :] == 0).all()
    Qz, _ = rtn(torch.zeros(1, 3, 3), 4, "column")
    assert (Qz == 0).all()


def test_noise_control_matches_per_column_error_norm():
    W = torch.randn(2, 5, 9)
    Q, _ = rtn(W, 3, "column")
    N = noise_control(W, Q, seed=0)
    assert torch.allclose((N - W).norm(dim=1), (Q - W).norm(dim=1), atol=1e-6)
    assert torch.allclose((N - W).norm(), (Q - W).norm(), atol=1e-5)  # Frobenius too
    assert torch.equal(noise_control(W, Q, 0), N) and not torch.equal(noise_control(W, Q, 1), N)


def test_quantize_bias_levels_and_bound():
    b = torch.linspace(-0.3, 0.3, 1001)[None, :]
    Qb = quantize_bias(b, 3)
    assert torch.unique(Qb).numel() == 7
    assert ((Qb - b).abs() <= 0.3 / 3 / 2 + 1e-7).all()


def test_apply_quant_ablations():
    W, b = torch.randn(2, 3, 5), torch.randn(2, 5)
    r = apply_quant(QuantConfig(4, "column"), W, b)
    assert torch.equal(r.W_enc, r.W_dec) and torch.equal(r.W_enc, r.Wq) and torch.equal(r.b, b)
    e = apply_quant(QuantConfig(4, "column", ablation="enc_only"), W, b)
    assert torch.equal(e.W_enc, r.Wq) and torch.equal(e.W_dec, W)
    d = apply_quant(QuantConfig(4, "column", ablation="dec_only"), W, b)
    assert torch.equal(d.W_enc, W) and torch.equal(d.W_dec, r.Wq)
    qb = apply_quant(QuantConfig(4, "column", ablation="quant_b"), W, b)
    assert not torch.equal(qb.b, b) and torch.equal(qb.W_enc, r.Wq)
    n = apply_quant(QuantConfig(4, "column", "noise", noise_seed=3), W, b)
    assert torch.allclose((n.Wq - W).norm(dim=1), (r.Wq - W).norm(dim=1), atol=1e-6)
    with pytest.raises(ValueError):
        QuantConfig(4, "column", "noise", ablation="quant_b")


def test_build_configs_count():
    cfgs = build_configs([8, 4], ["tensor", "column", "row"], 5, ["quant_b", "enc_only", "dec_only"], ["column", "row"])
    assert len(cfgs) == 2 * (3 * 6 + 2 * 3)
    assert len({c.label for c in cfgs}) == len(cfgs)
