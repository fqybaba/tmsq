"""Branch A (scale), step 1: load one Pythia size with its Lawson TopK SAEs, check conventions, time and run the
FP pass.

``.venv-bridge\\Scripts\\python -m tmsq.experiments.bridge_scale_smoke --config configs/bridge_scale.yaml --size 410m
[--layers 19] [--manifest results/bridge/<fetch-scale>/manifest.json]``

FP only: no quantized model is built and nothing from ``tmsq.experiments.bridge``'s FP/quant stages runs (their
gates and ReLU-SAE ``encode`` are 70M/Duan-specific). Steps:

1. token stream (Duan's); the size's tokenizer must serialize identically to the 70M tokenizer (same stream);
2. each SAE loaded at the revision pinned in the config, with the layout checks of ``tmsq.bridge_topk``; with
   ``--manifest``, revisions and SHA-256 of the weight files are checked against the fetch manifest;
3. convention check on one batch, all layers from one forward of each model: the HF hidden state after block ``N``
   (raw) vs TransformerLens ``blocks.N.hook_resid_post`` (default processing, built from the same HF weights).
   Passes if the losses agree to ``conv_loss_tol`` and ``max |dz| / max |z| <= conv_z_rel_tol``. The number of
   tokens whose top-k set differs is reported, not gated: a set can only flip where the cut is nearly tied
   (``z_(k) - z_(k+1) <= 2 max_j |dz_j|`` always holds for such a token), so it measures the float32 noise floor of
   top-k membership (70m layer 4: 48 of 2048 tokens between the two float32 models, none in float64);
   Pairing check on the same batch: each SAE must reconstruct the output of its own block better (lower FVU) than
   the output of any other block, which the convention check cannot see;
4. the FP pass, ONE HF forward per batch feeding every configured layer: next-token loss (FP perplexity) and
   ``TopKFPStats`` per layer. The first ``timing_batches`` batches give the ETA; the pass stops there if the ETA
   exceeds ``max_minutes`` (rerun with ``--layers`` to split, or ``--allow-long``).

Output: ``results/bridge_scale/<ts>-smoke-<size>/summary.json`` and ``latents_fp_L<layer>.parquet`` (one row per
latent; gitignored, shipped with the Release).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .. import bridge_topk as TK
from .bridge import batches, load_hf, load_tokenizer, load_tokens
from .common import PROJECT_ROOT, apply_runtime, load_config, make_run_dir, save_provenance, write_json

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

STREAM_KEYS = ("dataset", "dataset_file", "dataset_revision", "n_blocks", "block_len", "batch_blocks")


def size_cfg(cfg: dict[str, Any], size: str) -> dict[str, Any]:
    """The per-size model entry merged with the shared token-stream keys (the shape ``bridge.load_*`` expect)."""
    c = dict(cfg["sizes"][size])
    c.update({k: cfg[k] for k in STREAM_KEYS})
    return c


def sae_repo(cfg: dict[str, Any], size: str, layer: int) -> tuple[str, str]:
    """Repo id and pinned revision of the SAE for ``(size, layer)``."""
    key = f"{size}-{layer}"
    if key not in cfg["sae_revisions"]:
        raise KeyError(f"no pinned SAE revision for {key} in the config (sae_revisions)")
    return cfg["sae_repo"].format(size=size, layer=layer), cfg["sae_revisions"][key]


def load_manifest(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """Fetch manifest as ``{(repo, file): row}``; a repo must have a single revision."""
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    revs: dict[str, str] = {}
    out = {}
    for r in rows:
        if revs.setdefault(r["repo"], r["revision"]) != r["revision"]:
            raise RuntimeError(f"manifest has two revisions for {r['repo']}")
        out[(r["repo"], r["file"])] = r
    return out


def rel_to_project(path: Path) -> str:
    """Path as recorded in the provenance: relative to the project root when inside it (results are public)."""
    p = Path(path).resolve()
    return p.relative_to(PROJECT_ROOT).as_posix() if p.is_relative_to(PROJECT_ROOT) else p.name


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_manifest(manifest: dict | None, repo: str, fname: str, revision: str, path: Path) -> str:
    """Compare a loaded file with the fetch manifest (revision and SHA-256); returns a status string."""
    if manifest is None:
        return "no manifest given"
    row = manifest.get((repo, fname))
    if row is None:
        raise RuntimeError(f"{repo}/{fname} is not in the fetch manifest")
    if row["revision"] != revision:
        raise RuntimeError(f"{repo}: config pins {revision}, manifest has {row['revision']}")
    if sha256(path) != row["sha256"]:
        raise RuntimeError(f"{repo}/{fname}: SHA-256 differs from the fetch manifest")
    return "revision and sha256 match the manifest"


def load_sae(cfg: dict[str, Any], size: str, layer: int, manifest: dict | None) -> tuple[TK.TopKSAE, dict[str, Any]]:
    from huggingface_hub import hf_hub_download

    repo, rev = sae_repo(cfg, size, layer)
    paths = {f: Path(hf_hub_download(repo, f, revision=rev)) for f in cfg["sae_files"]}
    status = [check_manifest(manifest, repo, f, rev, p) for f, p in paths.items()][0]
    sae = TK.load_topk_sae(paths["config.json"], paths["model.safetensors"])
    info = {
        "repo": repo, "revision": rev, "manifest_check": status,
        "config": json.loads(paths["config.json"].read_text(encoding="utf-8")),
        "d_in": sae.d_in, "n_latents": sae.n_latents, "k": sae.k, "standardize": sae.standardize,
        "decoder_norm_max_abs_dev": sae.dec_norm_max_dev, "b_pre_norm": float(sae.b_pre.norm()),
        "W_enc_col_norm_mean": float(sae.W_enc.norm(dim=0).mean()),
    }  # fmt: skip
    return sae, info


class HFTap:
    """One HF forward per batch that returns the hidden state after every tapped block, and the next-token loss."""

    def __init__(self, hf: Any, layers: list[int]) -> None:
        self.hf, self.layers = hf, list(layers)
        self._store: dict[int, torch.Tensor] = {}
        self._handles = []
        for layer in self.layers:

            def hook(_m, _i, o, layer=layer):  # transformers 5: the layer returns the post-block residual
                self._store[layer] = o[0] if isinstance(o, tuple) else o

            self._handles.append(hf.gpt_neox.layers[layer].register_forward_hook(hook))

    @torch.no_grad()
    def __call__(self, tokens: torch.Tensor) -> tuple[dict[int, torch.Tensor], float]:
        logits = self.hf(tokens).logits[:, :-1]
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens[:, 1:].reshape(-1))
        return {layer: self._store.pop(layer) for layer in self.layers}, float(loss)

    def close(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []


def load_tl_from(c: dict[str, Any], hf: Any) -> Any:
    """TransformerLens model (default processing) built from the already loaded HF weights, at the pinned revision.

    The tokenizer is passed with ``add_bos_token=True`` so that TransformerLens does not reload it (and the model
    config) from ``main``; tokens are fed as tensors, so it is never used to tokenize.
    """
    from transformer_lens import HookedTransformer
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(c["model"], revision=c["model_revision"], add_bos_token=True)
    return HookedTransformer.from_pretrained(
        c["tl_name"], device="cpu", hf_model=hf, tokenizer=tok, revision=c["model_revision"]
    ).eval()


@torch.no_grad()
def compare_conventions(
    r_hf: torch.Tensor, r_tl: torch.Tensor, sae: TK.TopKSAE, loss_hf: float, loss_tl: float, loss_tol: float,
    z_rel_tol: float,
) -> dict[str, Any]:  # fmt: skip
    """HF raw hidden state vs TransformerLens resid_post for one layer: SAE pre-activations and top-k sets."""
    d, k = r_hf.shape[-1], sae.k
    z_hf = TK.pre_activation(r_hf.reshape(-1, d), sae)
    z_tl = TK.pre_activation(r_tl.reshape(-1, d), sae)
    v_hf, i_hf = torch.topk(z_hf, k + 1, dim=-1, sorted=True)
    gap = v_hf[:, k - 1] - v_hf[:, k]  # z_(k) - z_(k+1): how close the cut is to a tie
    set_hf = i_hf[:, :k].sort(-1).values
    set_tl = torch.topk(z_tl, k, dim=-1).indices.sort(-1).values
    differ = ~(set_hf == set_tl).all(-1)
    dz_tok = (z_hf - z_tl).abs().amax(-1)
    shift = (r_hf - r_tl).mean(-1, keepdim=True)  # TL centers the residual: undo the per-token shift
    out = {
        "loss_hf": loss_hf, "loss_tl": loss_tl, "loss_abs_diff": abs(loss_hf - loss_tl),
        "resid_max_abs_diff_up_to_token_shift": float((r_hf - r_tl - shift).abs().max()),
        "resid_max_abs": float(r_hf.abs().max()),
        "z_max_abs_diff": float(dz_tok.max()), "z_max_abs": float(z_hf.abs().max()),
        "tokens": int(differ.numel()), "tokens_topk_set_differs": int(differ.sum()),
        "gap_q001_q01_q50": [float(v) for v in torch.quantile(gap, torch.tensor([0.001, 0.01, 0.5]))],
    }  # fmt: skip
    out["z_rel_diff"] = out["z_max_abs_diff"] / out["z_max_abs"]
    out["pass"] = bool(out["loss_abs_diff"] <= loss_tol and out["z_rel_diff"] <= z_rel_tol)
    return out


def convention_check(
    c: dict[str, Any], hf: Any, tap: HFTap, saes: dict[int, TK.TopKSAE], tokens: torch.Tensor, loss_tol: float,
    z_rel_tol: float,
) -> dict[str, Any]:  # fmt: skip
    """Step 3 for every tapped layer, from one HF forward and one TransformerLens forward of the same batch."""
    probe = hf.gpt_neox.layers[0].attention.query_key_value.weight
    before = probe.detach().clone()
    t0 = time.perf_counter()
    tl = load_tl_from(c, hf)
    build_s = time.perf_counter() - t0
    if not torch.equal(before, probe):
        raise RuntimeError("building TransformerLens changed the HF weights")
    res_hf, loss_hf = tap(tokens)
    store: dict[int, torch.Tensor] = {}
    hooks = [(f"blocks.{L}.hook_resid_post", lambda act, hook, L=L: store.__setitem__(L, act.detach())) for L in saes]
    with torch.no_grad():
        loss_tl = float(tl.run_with_hooks(tokens, return_type="loss", fwd_hooks=hooks))
    out = {
        str(L): compare_conventions(res_hf[L], store[L], saes[L], loss_hf, loss_tl, loss_tol, z_rel_tol) for L in saes
    }
    out["tl_build_s"] = build_s
    return out


@torch.no_grad()
def pairing_check(hf: Any, saes: dict[int, TK.TopKSAE], tokens: torch.Tensor, n_layers: int) -> dict[str, Any]:
    """Is each SAE paired with the block it was trained on? FVU of the SAE on the output of EVERY block, one batch.

    The convention check cannot see a wrong pairing (HF and TransformerLens agree on any block). The SAE named
    ``layers-N`` must reconstruct block N's output better than any other block's (70m layer 4: FVU 1.23, 1.11,
    0.68, 0.53, 0.25, 12.96 on blocks 0..5).
    """
    tap = HFTap(hf, list(range(n_layers)))
    try:
        res, _ = tap(tokens)
    finally:
        tap.close()
    out = {}
    for layer, sae in saes.items():
        fvu = []
        for b in range(n_layers):
            x = res[b].reshape(-1, res[b].shape[-1])
            xh = TK.reconstruct(x, sae)[0]
            sst = (x - x.mean(0)).double().pow(2).sum()
            fvu.append(float((x - xh).double().pow(2).sum() / sst))
        best = int(np.argmin(fvu))
        out[str(layer)] = {"fvu_by_block": fvu, "best_block": best, "pass": best == layer}
    return out


def run_size(cfg: dict[str, Any], size: str, args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    c = size_cfg(cfg, size)
    layers = sorted(set(args.layers)) if args.layers else list(c["layers"])
    if not set(layers) <= set(c["layers"]):
        raise ValueError(f"--layers {layers} not among the configured layers {c['layers']}")
    manifest = load_manifest(args.manifest) if args.manifest else None
    tok = load_tokenizer(c)
    tokens = load_tokens(c, tok)
    same_tok = tok.backend_tokenizer.to_str() == load_tokenizer(size_cfg(cfg, "70m")).backend_tokenizer.to_str()
    if not same_tok:
        raise RuntimeError(f"the {size} tokenizer differs from the 70m tokenizer: the token stream is not shared")
    special = sorted({i for i in (tok.eos_token_id, tok.bos_token_id, tok.pad_token_id) if i is not None})
    n_special = int(torch.isin(tokens, torch.tensor(special)).sum())
    print(f"[{size}] tokens {tuple(tokens.shape)}; tokenizer identical to 70m; special-token ids {special} "
          f"occur {n_special} times", flush=True)  # fmt: skip
    t0 = time.perf_counter()
    hf = load_hf(c)
    load_s = time.perf_counter() - t0
    if (hf.config.num_hidden_layers, hf.config.hidden_size) != (c["n_layers"], c["d_model"]):
        raise RuntimeError(f"model shape {hf.config.num_hidden_layers} x {hf.config.hidden_size} != config")
    out: dict[str, Any] = {
        "size": size, "model": c["model"], "model_revision": c["model_revision"],
        "n_params": sum(p.numel() for p in hf.parameters()), "n_layers": c["n_layers"], "d_model": c["d_model"],
        "load_hf_s": load_s, "tokens": list(tokens.shape), "tokenizer_identical_to_70m": same_tok,
        "special_token_ids": special, "special_tokens_in_stream": n_special,
        "threads": torch.get_num_threads(), "near_fracs": list(TK.NEAR_FRACS), "layers": {},
    }  # fmt: skip
    if manifest is not None:
        from huggingface_hub import hf_hub_download

        path = Path(hf_hub_download(c["model"], "model.safetensors", revision=c["model_revision"]))
        out["model_manifest_check"] = check_manifest(
            manifest, c["model"], "model.safetensors", c["model_revision"], path
        )
    saes: dict[int, TK.TopKSAE] = {}
    for layer in layers:
        saes[layer], info = load_sae(cfg, size, layer, manifest)
        out["layers"][str(layer)] = {"sae": info}
        print(f"[{size} L{layer}] SAE {info['repo']}@{info['revision'][:8]}: d_in {info['d_in']}, n_latents "
              f"{info['n_latents']}, k {info['k']}, max |decoder norm - 1| {info['decoder_norm_max_abs_dev']:.2e}; "
              f"{info['manifest_check']}", flush=True)  # fmt: skip
    bl = batches(c)
    tap = HFTap(hf, layers)
    try:
        conv = convention_check(c, hf, tap, saes, tokens[bl[0]], cfg["conv_loss_tol"], cfg["conv_z_rel_tol"])
        out["tl_build_s"] = conv.pop("tl_build_s")
        pair = pairing_check(hf, saes, tokens[bl[0]], c["n_layers"])
        for layer in layers:
            out["layers"][str(layer)]["convention_check"] = conv[str(layer)]
            out["layers"][str(layer)]["pairing_check"] = pair[str(layer)]
            print(f"[{size} L{layer}] convention: {json.dumps(conv[str(layer)])}", flush=True)
            print(f"[{size} L{layer}] pairing: {json.dumps(pair[str(layer)])}", flush=True)
        stats = {layer: TK.TopKFPStats(saes[layer]) for layer in layers}
        pos0 = torch.zeros(tokens.shape[1], dtype=torch.bool)
        pos0[0] = True
        losses, n_blocks, t_fwd, t_upd, stopped = [], [], [], [], None
        t_start = time.perf_counter()
        for i, s in enumerate(bl, 1):
            a = time.perf_counter()
            res, loss = tap(tokens[s])
            b = time.perf_counter()
            mask = pos0.repeat(tokens[s].shape[0])
            for layer in layers:
                r = res[layer]
                stats[layer].update(r.reshape(-1, r.shape[-1]), mask)
            t_fwd.append(b - a)
            t_upd.append(time.perf_counter() - b)
            losses.append(loss)
            n_blocks.append(tokens[s].shape[0])
            el = time.perf_counter() - t_start
            if i == cfg["timing_batches"]:
                eta_min = el / i * len(bl) / 60
                out["timing"] = {
                    "batches_timed": i, "fwd_s_per_batch_median": float(np.median(t_fwd)),
                    "update_s_per_batch_median_all_layers": float(np.median(t_upd)), "n_layers_tapped": len(layers),
                    "n_batches": len(bl), "full_pass_eta_min": eta_min,
                    "note": "a quantized variant costs one more model forward per batch, plus its own SAE statistics",
                }  # fmt: skip
                print(f"[{size}] timing after {i} batches: fwd {np.median(t_fwd):.2f}s + update {np.median(t_upd):.2f}s"
                      f" per batch ({len(layers)} layers) -> full pass ETA {eta_min:.1f} min", flush=True)  # fmt: skip
                if eta_min > cfg["max_minutes"] and not args.allow_long:
                    stopped = f"stopped after {i} batches: ETA {eta_min:.1f} min > max_minutes {cfg['max_minutes']}"
                    print(f"[{size}] {stopped}", flush=True)
                    break
            if i % 20 == 0 or i == len(bl):
                print(f"  [{size}] FP {i}/{len(bl)} batches, {el:.0f}s, ETA {el / i * (len(bl) - i):.0f}s", flush=True)
    finally:
        tap.close()
    out["full_pass"] = stopped or {
        "wall_s": time.perf_counter() - t_start,
        # ppl_fp: exp of the unweighted mean of the per-batch losses, the convention of tmsq.experiments.bridge
        # (Duan's batches of 4 blocks; the last batch has 2). The token-weighted value is given next to it.
        "ppl_fp": float(np.exp(np.mean(losses))),
        "ppl_fp_token_weighted": float(np.exp(np.average(losses, weights=n_blocks))),
        "fwd_s_per_batch_median": float(np.median(t_fwd)),
        "update_s_per_batch_median_all_layers": float(np.median(t_upd)),
    }  # fmt: skip
    checks_ok = all(rec["convention_check"]["pass"] and rec["pairing_check"]["pass"] for rec in out["layers"].values())
    out["status"] = "checks FAILED" if not checks_ok else "stopped by the ETA gate" if stopped else "complete"
    if stopped is None:
        for layer in layers:
            st = stats[layer]
            rec = out["layers"][str(layer)]
            rec["fp"] = st.summary(cfg["active_rate"])
            pd.DataFrame({"latent": np.arange(st.n_lat), **st.table()}).to_parquet(
                out_dir / f"latents_fp_L{layer}.parquet", index=False
            )
            print(f"[{size} L{layer}] FP: {json.dumps(rec['fp'])}", flush=True)
        print(f"[{size}] full pass: {json.dumps(out['full_pass'])}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Branch A step 1: convention check + FP pass (no quantization)")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--size", required=True)
    ap.add_argument("--layers", type=int, nargs="+", default=None, help="subset of the configured layers")
    ap.add_argument("--manifest", type=Path, default=None, help="manifest.json of bridge_fetch --set scale")
    ap.add_argument("--allow-long", action="store_true", help="finish the pass even if the ETA exceeds max_minutes")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    cfg = load_config(args.config)
    apply_runtime(cfg)
    out_dir = make_run_dir("bridge_scale", args.tag or f"smoke-{args.size}")
    save_provenance(out_dir, cfg, {"stage": "smoke", "size": args.size, "layers": args.layers,
                                   "manifest": rel_to_project(args.manifest) if args.manifest else None,
                                   "allow_long": args.allow_long})  # fmt: skip
    res = run_size(cfg, args.size, args, out_dir)
    write_json(out_dir / "summary.json", res)
    print(f"written {out_dir}; status: {res['status']}", flush=True)
    if res["status"] != "complete":
        raise SystemExit(f"smoke not complete ({res['status']}); see {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
