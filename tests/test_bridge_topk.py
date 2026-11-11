"""Tests for ``tmsq.bridge_topk`` (Lawson TopK SAEs) and the pure helpers of the branch-A smoke script.

The reference forward is built from ``torch.nn.Linear`` modules in the safetensors layout and follows
``mlsae/model/autoencoders/topk.py`` (commit f26303f) line by line, including its sparse decoder
(``decode_cuda``: ``sparse_coo @ weight.t()``), so it shares no code path with the dense or sparse path under test.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from tmsq import bridge_topk as T
from tmsq.experiments import bridge_scale_smoke as S


def _layout(d_in=8, n_lat=32, seed=0, norm_jitter=0.0):
    """Tensors in the safetensors layout: encoder.weight [n_lat, d_in], decoder.weight [d_in, n_lat]."""
    g = torch.Generator().manual_seed(seed)
    enc = torch.randn(n_lat, d_in, generator=g)
    dec = torch.randn(d_in, n_lat, generator=g)
    dec = dec / dec.norm(dim=0, keepdim=True)
    if norm_jitter:
        dec = dec * (1 + norm_jitter * (2 * torch.rand(n_lat, generator=g) - 1))
    return {
        "encoder.weight": enc,
        "decoder.weight": dec.contiguous(),
        "pre_encoder_bias": 0.1 * torch.randn(d_in, generator=g),
    }


def _sae(k=4, standardize=True, **kw):
    w = _layout(**kw)
    return T.TopKSAE(w["encoder.weight"].T.contiguous(), w["decoder.weight"].T.contiguous(), w["pre_encoder_bias"], k,
                     standardize), w  # fmt: skip


def _mlsae_reference(x, w, k, standardize):
    """mlsae TopKSAE.forward: nn.Linear encoder/decoder (no bias), topk, relu, sparse decode, unstandardize."""
    n_lat, d_in = w["encoder.weight"].shape
    encoder = torch.nn.Linear(d_in, n_lat, bias=False)
    decoder = torch.nn.Linear(n_lat, d_in, bias=False)
    with torch.no_grad():
        encoder.weight.copy_(w["encoder.weight"])
        decoder.weight.copy_(w["decoder.weight"])
        inputs, stats = x, None
        if standardize:  # utils.standardize
            mu = inputs.mean(dim=-1, keepdim=True)
            inputs = inputs - mu
            std = inputs.std(dim=-1, keepdim=True)
            inputs = inputs / (std + 1e-5)
            stats = (mu, std)
        latents = encoder.forward(inputs - w["pre_encoder_bias"])
        values, indices = torch.topk(latents, k=k, sorted=False)
        values = torch.relu(values)
        # decoder.decode_cuda: a sparse [T, n_latents] matrix times weight.t()
        t = indices.shape[0]
        sparse = torch.sparse_coo_tensor(
            torch.stack([torch.arange(t).repeat_interleave(k), indices.flatten()]), values.flatten(), (t, n_lat)
        )
        recons = torch.sparse.mm(sparse, decoder.weight.t()) + w["pre_encoder_bias"]
        if stats is not None:
            recons = recons * stats[1] + stats[0]
    return recons, sparse.to_dense()


@pytest.mark.parametrize("standardize", [True, False])
@pytest.mark.parametrize("k", [1, 4, 12])
def test_dense_and_sparse_paths_match_the_mlsae_reference(standardize, k):
    sae, w = _sae(k=k, standardize=standardize, norm_jitter=2e-3)
    x = 3.0 * torch.randn(50, sae.d_in, generator=torch.Generator().manual_seed(1)) + 1.0
    xh_ref, f_ref = _mlsae_reference(x, w, k, standardize)
    xh, f = T.reconstruct(x, sae)
    assert torch.allclose(f, f_ref, atol=1e-6)
    assert torch.allclose(xh, xh_ref, rtol=1e-6, atol=1e-5)
    code, z = T.encode_sparse(x, sae)
    f_sparse = torch.zeros_like(z).scatter_(-1, code.idx, code.vals)
    assert torch.equal(f_sparse, f)
    assert torch.allclose(T.decode_sparse(code, sae), xh_ref, rtol=1e-6, atol=1e-5)
    assert torch.equal(code.cut, T.encode(x, sae)[2])


def test_topk_then_relu_not_relu_then_topk():
    z = torch.tensor([[5.0, 4.0, 3.0, 2.5, 2.0, -1.0, -2.0, -3.0]])
    f, cut = T.topk_activations(z, k=4)  # the 5th largest (2.0) is positive but must not be promoted
    assert torch.equal(f, torch.tensor([[5.0, 4.0, 3.0, 2.5, 0.0, 0.0, 0.0, 0.0]]))
    assert float(cut) == 2.5
    z3 = torch.tensor([[5.0, -0.5, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0]])
    f3, cut3 = T.topk_activations(z3, k=4)
    assert int((f3 > 0).sum()) == 1 and float(cut3) == -2.0  # ReLU zeroes kept negatives: L0 can be below k


def test_standardize_shift_invariance_3d_input_and_zero_variance_token():
    sae, _ = _sae()
    x = torch.randn(20, sae.d_in, generator=torch.Generator().manual_seed(2))
    shift = 7.0 * torch.randn(20, 1, generator=torch.Generator().manual_seed(3))
    z0 = T.pre_activation(x, sae)
    assert torch.allclose(z0, T.pre_activation(x + shift, sae), atol=1e-5)  # what TL's centering removes
    xs, mu, sd = T.standardize(x)
    assert torch.allclose(xs.mean(-1), torch.zeros(20), atol=1e-6)
    assert torch.allclose(x.std(-1, keepdim=True), sd)  # unbiased, torch default
    x3 = x.reshape(4, 5, sae.d_in)  # [batch, pos, d]
    assert torch.allclose(T.pre_activation(x3, sae).reshape(20, -1), z0, atol=1e-6)
    assert torch.allclose(T.reconstruct(x3, sae)[0].reshape(20, -1), T.reconstruct(x, sae)[0], atol=1e-5)
    const = torch.full((1, sae.d_in), 3.0)  # all-equal residual: sd = 0, standardized input = 0, no NaN
    xh, f = T.reconstruct(const, sae)
    assert torch.isfinite(xh).all() and torch.isfinite(f).all()
    assert torch.allclose(xh, const)  # (f @ W_dec + b) * 0 + mu


def test_decode_needs_stats_for_a_standardizing_sae():
    sae, _ = _sae(standardize=False)
    f = torch.zeros(3, sae.n_latents)
    f[0, 1], f[1, 5], f[2, 7] = 1.0, 2.0, 0.5
    assert torch.allclose(T.decode(f, sae), f @ sae.W_dec + sae.b_pre)
    with pytest.raises(ValueError):
        T.decode(f, _sae(standardize=True)[0])


def test_load_topk_sae_layout_norm_tolerance_and_rejections(tmp_path):
    from safetensors.torch import save_file

    w = _layout(norm_jitter=2e-3)  # real files are about 2e-3 off unit norm and must load, unchanged
    cfg = {"auxk": 256, "dead_steps_threshold": 76, "dead_threshold": 0.001, "k": 4, "n_inputs": 8, "n_latents": 32,
           "standardize": True}  # fmt: skip
    (tmp_path / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    save_file(w, str(tmp_path / "model.safetensors"))
    sae = T.load_topk_sae(tmp_path / "config.json", tmp_path / "model.safetensors")
    assert torch.equal(sae.W_dec, w["decoder.weight"].T) and torch.equal(sae.W_enc, w["encoder.weight"].T)
    assert 1e-4 < sae.dec_norm_max_dev <= 2e-3 + 1e-6 and sae.k == 4 and sae.standardize
    x = torch.randn(9, 8, generator=torch.Generator().manual_seed(5))
    assert torch.allclose(T.reconstruct(x, sae)[0], _mlsae_reference(x, w, 4, True)[0], rtol=1e-6, atol=1e-5)
    (tmp_path / "raw.json").write_text(json.dumps({**cfg, "standardize": False}), encoding="utf-8")
    raw = T.load_topk_sae(tmp_path / "raw.json", tmp_path / "model.safetensors")
    assert not raw.standardize
    assert torch.allclose(T.reconstruct(x, raw)[0], _mlsae_reference(x, w, 4, False)[0], rtol=1e-6, atol=1e-5)
    for name, bad in (
        ("norm", {**w, "decoder.weight": 1.05 * w["decoder.weight"]}),
        ("swapped", {**w, "encoder.weight": w["decoder.weight"], "decoder.weight": w["encoder.weight"]}),
        ("extra", {**w, "encoder.bias": torch.zeros(32)}),
    ):
        save_file(bad, str(tmp_path / f"{name}.safetensors"))
        match = {"norm": "unit-norm", "swapped": "shape mismatch", "extra": "unexpected safetensors keys"}[name]
        with pytest.raises(ValueError, match=match):
            T.load_topk_sae(tmp_path / "config.json", tmp_path / f"{name}.safetensors")


def _dense_reference_stats(xs, pos0s, sae, fracs):
    """Everything TopKFPStats reports, recomputed densely in float64 numpy from concatenated batches."""
    x, pos0 = torch.cat(xs), torch.cat(pos0s).numpy()
    f, z, cut = T.encode(x, sae)
    xh = T.reconstruct(x, sae)[0].double().numpy()
    f, z, cut, xd = f.double().numpy(), z.double().numpy(), cut.double().numpy(), x.double().numpy()
    n, k = len(xd), sae.k
    order = np.argsort(-z, axis=1, kind="stable")
    kept = np.zeros_like(z, dtype=bool)
    np.put_along_axis(kept, order[:, :k], True, axis=1)
    rank2k = np.zeros_like(z, dtype=bool)
    np.put_along_axis(rank2k, order[:, k : 2 * k], True, axis=1)
    on = f > 0
    ref = {
        "fire": on.sum(0), "rate": on.mean(0), "rate_nopos0": on[~pos0].mean(0), "mean": f.mean(0),
        "peak": f.max(0), "rate_pos_z": (z > 0).mean(0), "mean_z": z.mean(0), "sd_z": z.std(0),
        "rate_rank2k": rank2k.mean(0),
    }  # fmt: skip
    for frac in fracs:
        w = frac * np.abs(cut)[:, None]
        ref[f"near_off_{frac:g}"] = (~kept & (z > cut[:, None] - w)).mean(0)
        ref[f"near_on_{frac:g}"] = (on & (z < cut[:, None] + w)).mean(0)

    def fvu(m):
        return ((xd[m] - xh[m]) ** 2).sum() / ((xd[m] - xd[m].mean(0)) ** 2).sum()

    scal = {"fvu": fvu(np.ones(n, bool)), "fvu_nopos0": fvu(~pos0), "fvu_pos0": fvu(pos0), "l0_mean": on.sum(1).mean(),
            "frac_tokens_l0_below_k": (on.sum(1) < k).mean(), "cut_mean": cut.mean(), "cut_min": cut.min(),
            "cut_nonpositive_frac": (cut <= 0).mean()}  # fmt: skip
    return ref, scal


@pytest.mark.parametrize("k,standardize", [(4, True), (20, True), (4, False)])
def test_topk_fp_stats_match_a_dense_reference(k, standardize):
    # k = 20 of 32 latents: some kept pre-activations are negative (L0 < k, cut < 0), which the sparse path must handle
    sae, _ = _sae(k=k, standardize=standardize, d_in=8, n_lat=48 if k == 20 else 32)
    g = torch.Generator().manual_seed(7)
    xs = [2.0 * torch.randn(24, sae.d_in, generator=g) + 0.5 for _ in range(3)]
    pos0 = torch.zeros(24, dtype=torch.bool)
    pos0[::6] = True  # 4 "blocks" of 6 tokens per batch
    xs = [torch.where(pos0[:, None], 5.0 * x, x) for x in xs]  # position-0 tokens have large norms
    st = T.TopKFPStats(sae)
    for x in xs:
        st.update(x, pos0)
    ref, scal = _dense_reference_stats(xs, [pos0] * 3, sae, st.fracs)
    tab = st.table()
    for key, want in ref.items():
        np.testing.assert_allclose(tab[key], want, rtol=1e-5, atol=1e-6, err_msg=key)
    fired = ref["fire"] > 0
    np.testing.assert_allclose(tab["mu_plus"][fired], (ref["mean"] * 72 / np.maximum(ref["fire"], 1))[fired], rtol=1e-5)
    assert np.isnan(tab["mu_plus"][~fired]).all() and np.isnan(tab["pos0_share"][~fired]).all()
    summ = st.summary(active_rate=0.1)
    for key, want in scal.items():
        assert summ[key] == pytest.approx(want, rel=1e-5, abs=1e-7), key
    assert summ["tokens"] == 72 and summ["tokens_pos0"] == 12
    assert summ["n_above_rate"]["1x"] == int((ref["rate"] > 0.1).sum())
    assert summ["n_above_rate_nopos0"]["1x"] == int((ref["rate_nopos0"] > 0.1).sum())
    assert summ["n_above_rate_pos_z"]["0.5x"] == int((ref["rate_pos_z"] > 0.05).sum())
    if k == 20:
        assert summ["frac_tokens_l0_below_k"] > 0 and summ["cut_nonpositive_frac"] > 0  # the case is exercised


def test_smoke_config_and_manifest_helpers(tmp_path):
    cfg = {
        "sizes": {"70m": {"model": "m70", "layers": [4]}}, "sae_repo": "org/sae-{size}-L{layer}",
        "sae_revisions": {"70m-4": "abc"}, "dataset": "d", "dataset_file": "f", "dataset_revision": "r",
        "n_blocks": 2, "block_len": 3, "batch_blocks": 1,
    }  # fmt: skip
    c = S.size_cfg(cfg, "70m")
    assert c["model"] == "m70" and c["n_blocks"] == 2 and "sizes" not in c
    assert S.sae_repo(cfg, "70m", 4) == ("org/sae-70m-L4", "abc")
    with pytest.raises(KeyError):
        S.sae_repo(cfg, "70m", 5)
    blob = tmp_path / "w.bin"
    blob.write_bytes(b"weights")
    rows = [{"repo": "org/sae-70m-L4", "file": "model.safetensors", "revision": "abc", "sha256": S.sha256(blob)},
            {"repo": "org/sae-70m-L4", "file": "config.json", "revision": "abc", "sha256": "x"}]  # fmt: skip
    (tmp_path / "manifest.json").write_text(json.dumps(rows), encoding="utf-8")
    man = S.load_manifest(tmp_path / "manifest.json")
    assert "match" in S.check_manifest(man, "org/sae-70m-L4", "model.safetensors", "abc", blob)
    assert S.check_manifest(None, "org/sae-70m-L4", "model.safetensors", "abc", blob) == "no manifest given"
    with pytest.raises(RuntimeError, match="config pins"):
        S.check_manifest(man, "org/sae-70m-L4", "model.safetensors", "other", blob)
    blob.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="SHA-256"):
        S.check_manifest(man, "org/sae-70m-L4", "model.safetensors", "abc", blob)
    with pytest.raises(RuntimeError, match="not in the fetch manifest"):
        S.check_manifest(man, "org/other", "model.safetensors", "abc", blob)
    rows.append({"repo": "org/sae-70m-L4", "file": "x", "revision": "zzz", "sha256": "y"})
    (tmp_path / "bad.json").write_text(json.dumps(rows), encoding="utf-8")
    with pytest.raises(RuntimeError, match="two revisions"):
        S.load_manifest(tmp_path / "bad.json")


def test_compare_conventions_gates_on_loss_and_relative_z():
    sae, _ = _sae(k=4, d_in=8, n_lat=32)
    g = torch.Generator().manual_seed(11)
    r = torch.randn(2, 16, sae.d_in, generator=g)
    same = S.compare_conventions(r, r - r.mean(-1, keepdim=True), sae, 1.0, 1.0, 1e-3, 0.02)  # TL-style centering
    assert same["pass"] and same["tokens_topk_set_differs"] == 0 and same["z_max_abs_diff"] < 1e-4
    assert same["resid_max_abs_diff_up_to_token_shift"] < 1e-5 and same["tokens"] == 32
    other = S.compare_conventions(r, torch.randn(2, 16, sae.d_in, generator=g), sae, 1.0, 1.0, 1e-3, 0.02)
    assert other["tokens_topk_set_differs"] > 0 and other["z_rel_diff"] > 0.02 and not other["pass"]
    assert not S.compare_conventions(r, r, sae, 1.0, 1.1, 1e-3, 0.02)["pass"]  # a loss mismatch alone fails
    # identity encoder on 6 latents, k = 2: z = first six coordinates, so the numbers can be read off
    sae1 = T.TopKSAE(torch.eye(8)[:, :6].contiguous(), torch.eye(8)[:6].contiguous(), torch.zeros(8), 2, False)
    a = torch.tensor([[[5.0, 3.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]])  # top-2 {0, 1}, gap z_(2) - z_(3) = 2
    b = a.clone()
    b[0, 0, 1], b[0, 0, 2] = 1.9, 2.0  # top-2 becomes {0, 2}; max |dz| = 1.1, relative 1.1 / 5 = 0.22
    res = S.compare_conventions(a, b, sae1, 1.0, 1.0, 1e-3, 0.02)
    assert res["tokens_topk_set_differs"] == 1 and res["z_max_abs_diff"] == pytest.approx(1.1)
    assert res["z_rel_diff"] == pytest.approx(0.22) and res["gap_q001_q01_q50"][2] == pytest.approx(2.0)
    assert not res["pass"] and S.compare_conventions(a, b, sae1, 1.0, 1.0, 1e-3, 0.25)["pass"]


def test_hf_tap_one_forward_feeds_all_layers_and_removes_its_hooks():
    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(6, 6)

        def forward(self, h):
            return (h + self.lin(h),)

    class Body(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([Layer() for _ in range(3)])

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(0)
            self.emb = torch.nn.Embedding(10, 6)
            self.gpt_neox = Body()
            self.out = torch.nn.Linear(6, 10)

        def forward(self, tokens):
            h = self.emb(tokens)
            self.hidden = []
            for layer in self.gpt_neox.layers:
                h = layer(h)[0]
                self.hidden.append(h)
            return type("O", (), {"logits": self.out(h)})()

    hf = Toy()
    tokens = torch.randint(0, 10, (2, 5), generator=torch.Generator().manual_seed(1))
    tap = S.HFTap(hf, [0, 2])
    res, loss = tap(tokens)
    assert set(res) == {0, 2} and torch.equal(res[0], hf.hidden[0]) and torch.equal(res[2], hf.hidden[2])
    logits = hf(tokens).logits[:, :-1]
    want = torch.nn.functional.cross_entropy(logits.reshape(-1, 10), tokens[:, 1:].reshape(-1))
    assert loss == pytest.approx(float(want.detach())) and not res[0].requires_grad
    tap.close()
    assert all(len(layer._forward_hooks) == 0 for layer in hf.gpt_neox.layers)
    # pairing check: an SAE whose dictionary spans only block 1's output directions must pick block 1
    with torch.no_grad():
        hidden = [h.reshape(-1, 6) for h in hf.hidden]
    u = torch.linalg.svd(hidden[1] - hidden[1].mean(0), full_matrices=False).Vh[:2]  # top-2 directions of block 1
    dirs = torch.cat([u, -u])  # 4 latents: +-u1, +-u2
    sae = T.TopKSAE(dirs.T.contiguous(), dirs.contiguous(), hidden[1].mean(0), 2, False)
    pair = S.pairing_check(hf, {1: sae}, tokens, 3)["1"]
    want = []
    for h in hidden:
        xh = T.reconstruct(h, sae)[0]
        want.append(float(((h - xh) ** 2).sum() / ((h - h.mean(0)) ** 2).sum()))
    assert pair["fvu_by_block"] == pytest.approx(want, rel=1e-5)
    assert pair["best_block"] == int(np.argmin(want)) and pair["pass"] == (pair["best_block"] == 1)
    assert all(len(layer._forward_hooks) == 0 for layer in hf.gpt_neox.layers)  # its own tap is closed too
