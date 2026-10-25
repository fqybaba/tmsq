"""Phase 6b bridge runner (``.venv-bridge`` only): FP stage, quantized runs, registered analysis.

``python -m tmsq.experiments.bridge fp --config configs/bridge.yaml``
``python -m tmsq.experiments.bridge quant --config configs/bridge.yaml --arm code --bits 6``
``python -m tmsq.experiments.bridge quant --config configs/bridge.yaml --arm code --bits 6 --noise-seed 0``
``python -m tmsq.experiments.bridge analyze --config configs/bridge.yaml --primary <quant run> [--extra <runs>]``

Plan: ``notes/LOG.md`` Phase 6b (D1–D7). Code arm = Duan's released pipeline (TransformerLens default
processing, scales over all dims but the last, centered residual into the SAE); stated arm = HF raw weights,
one scale per output row, residual convention chosen by gate G1. Only block linear weights are quantized.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .. import bridge as B
from .common import apply_runtime, load_config, make_run_dir, save_provenance

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

DUAN = {"active": 5688, "ppl_fp": 67.84, "survived_int6": 62.395}
G2_TOL = {"active": 30, "ppl_fp": 1.0, "survived_int6": 3.0}
PRIMARY_MARGIN = "margin"  # registered primary M column (see the pre-registration); the other is secondary
TL_QUANT = ("attn.W_Q", "attn.W_K", "attn.W_V", "attn.W_O", "mlp.W_in", "mlp.W_out")
HF_QUANT = ("attention.query_key_value", "attention.dense", "mlp.dense_h_to_4h", "mlp.dense_4h_to_h")


# ------------------------------------------------------------------------------------------------ loading


def load_tokens(cfg: dict[str, Any], tok: Any) -> torch.Tensor:
    """Duan's token stream: train split, non-empty lines joined by blank lines, no special tokens, first blocks."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(cfg["dataset"], cfg["dataset_file"], repo_type="dataset", revision=cfg["dataset_revision"])
    text = "\n\n".join(x for x in pd.read_parquet(path)["text"] if x.strip())
    ids = tok.encode(text, add_special_tokens=False)
    n = cfg["n_blocks"] * cfg["block_len"]
    return torch.tensor(ids[:n], dtype=torch.long).view(cfg["n_blocks"], cfg["block_len"])


def load_tokenizer(cfg: dict[str, Any]) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(cfg["model"], revision=cfg["model_revision"])


def load_hf(cfg: dict[str, Any]) -> Any:
    from transformers import GPTNeoXForCausalLM

    return GPTNeoXForCausalLM.from_pretrained(cfg["model"], revision=cfg["model_revision"], dtype=torch.float32).eval()


def load_tl(cfg: dict[str, Any], tok: Any) -> Any:
    """TransformerLens with its default processing (fold_ln, center_writing_weights, center_unembed,
    fold_value_biases, float32), as in Duan's script, built from the pinned HF checkpoint."""
    from transformer_lens import HookedTransformer

    return HookedTransformer.from_pretrained(cfg["tl_name"], device="cpu", hf_model=load_hf(cfg), tokenizer=tok).eval()


def load_sae(cfg: dict[str, Any]) -> dict[str, torch.Tensor]:
    """SAE weights via SAELens, checked against the pinned safetensors file."""
    from huggingface_hub import hf_hub_download
    from sae_lens import SAE
    from safetensors.torch import load_file

    sae = SAE.from_pretrained(cfg["sae_release"], cfg["sae_id"], device="cpu", dtype="float32")
    pinned = load_file(
        hf_hub_download(cfg["sae_repo"], f"{cfg['layer']}-res-sm/sae_weights.safetensors", revision=cfg["sae_revision"])
    )
    w = {k: getattr(sae, k).detach().float() for k in ("W_enc", "W_dec", "b_enc", "b_dec")}
    for k, v in w.items():
        if not torch.equal(v, pinned[k].float()):
            raise RuntimeError(f"SAELens {k} differs from the pinned revision")
    if not sae.cfg.apply_b_dec_to_input:
        raise RuntimeError("expected apply_b_dec_to_input = True")
    return w


def encode(r: torch.Tensor, sae: dict[str, torch.Tensor], cols: np.ndarray | None = None) -> torch.Tensor:
    """Pre-activation ``z = (r - b_dec) @ W_enc + b_enc`` (optionally only for feature columns ``cols``)."""
    W, b = (sae["W_enc"], sae["b_enc"]) if cols is None else (sae["W_enc"][:, cols], sae["b_enc"][cols])
    return (r - sae["b_dec"]) @ W + b


class Arm:
    """Forward pass of one model variant: block-4 residual (as the SAE sees it) and the next-token loss."""

    def __init__(self, arm: str, model: Any, layer: int, centered: bool) -> None:
        self.arm, self.model, self.layer, self.centered = arm, model, layer, centered
        self._store: dict[str, torch.Tensor] = {}
        if arm == "stated":

            def hook(_m, _i, o):  # transformers 5: the layer returns the post-block residual as a tensor
                self._store["r"] = o[0] if isinstance(o, tuple) else o

            model.gpt_neox.layers[layer].register_forward_hook(hook)

    @torch.no_grad()
    def __call__(self, tokens: torch.Tensor) -> tuple[torch.Tensor, float]:
        if self.arm == "code":

            def hook(act, hook):  # noqa: ARG001
                self._store["r"] = act.detach()

            loss = self.model.run_with_hooks(
                tokens, return_type="loss", fwd_hooks=[(f"blocks.{self.layer}.hook_resid_post", hook)]
            )
            r = self._store.pop("r")
        else:
            logits = self.model(tokens).logits[:, :-1]
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens[:, 1:].reshape(-1))
            r = self._store.pop("r")
            if self.centered:
                r = r - r.mean(-1, keepdim=True)
        return r, float(loss)


def quantize_model(base: Any, arm: str, bits: int, noise_seed: int | None) -> Any:
    """Copy of ``base`` with its block linear weights RTN-quantized (or replaced by the matched-noise control)."""
    m = copy.deepcopy(base)
    with torch.no_grad():
        if arm == "code":
            params = [(i, n, blk.get_parameter(n)) for i, blk in enumerate(m.blocks) for n in TL_QUANT]
            q_fn = B.rtn_lastdim
        else:
            params = [
                (i, n, layer.get_submodule(n).weight) for i, layer in enumerate(m.gpt_neox.layers) for n in HF_QUANT
            ]
            q_fn = B.rtn_rows
        for j, (i, _n, p) in enumerate(params):
            q = q_fn(p.data, bits)
            if noise_seed is not None:
                gen = torch.Generator().manual_seed(1_000_003 * (noise_seed + 1) + 1009 * i + j)
                q = B.noise_matched(p.data, q, arm, gen)
            p.data.copy_(q)
    return m


def batches(cfg: dict[str, Any]) -> list[slice]:
    nb, bb = cfg["n_blocks"], cfg["batch_blocks"]
    return [slice(i, min(i + bb, nb)) for i in range(0, nb, bb)]


def eta(t0: float, done: int, total: int, label: str) -> None:
    el = time.time() - t0
    print(f"  {label}: {done}/{total} batches, {el:.0f}s elapsed, ETA {el / done * (total - done):.0f}s", flush=True)


# ------------------------------------------------------------------------------------------------- FP stage


def gate_g1(cfg: dict[str, Any], tokens: torch.Tensor, sae: dict[str, torch.Tensor]) -> dict[str, Any]:
    """G1 (FP only): SAE FVU and L0 on raw vs per-token-centered block-4 residuals, positions < pos_split."""
    hf = load_hf(cfg)
    arm = Arm("stated", hf, cfg["layer"], centered=False)
    acc = {c: {"n": 0, "sx": torch.zeros(512, dtype=torch.float64), "sxx": 0.0, "sse": 0.0, "l0": 0.0}
           for c in ("raw", "centered")}  # fmt: skip
    short = tokens[:, : cfg["pos_split"]]
    for i in range(0, len(short), 16):
        r, _ = arm(short[i : i + 16])
        r = r.reshape(-1, r.shape[-1])
        for c, x in (("raw", r), ("centered", r - r.mean(-1, keepdim=True))):
            f = torch.relu(encode(x, sae))
            xh = f @ sae["W_dec"] + sae["b_dec"]
            a = acc[c]
            a["n"] += x.shape[0]
            a["sx"] += x.sum(0, dtype=torch.float64)
            a["sxx"] += float(x.double().pow(2).sum())
            a["sse"] += float((x - xh).double().pow(2).sum())
            a["l0"] += float((f > 0).sum())
    out = {}
    for c, a in acc.items():
        sst = a["sxx"] - float((a["sx"] ** 2).sum()) / a["n"]
        out[c] = {"fvu": a["sse"] / sst, "l0": a["l0"] / a["n"], "tokens": a["n"]}
    diff = out["centered"]["fvu"] - out["raw"]["fvu"]
    out["choice"] = "centered" if diff < -cfg["g1_fvu_tol"] else "raw"
    out["rule"] = f"lower FVU wins; within {cfg['g1_fvu_tol']} FVU use raw (Marks et al.: 88% VE, L0 61 on Pile)"
    return out


def fp_arm(
    cfg: dict[str, Any], arm_name: str, arm: Arm, tokens: torch.Tensor, sae: dict[str, torch.Tensor]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """FP pass for one arm: Duan's statistics for all features, near-threshold density, predictors, perplexity."""
    d = sae["W_enc"].shape[1]
    fp = B.FPStats(d)
    resid = torch.empty(cfg["n_blocks"], cfg["block_len"], sae["W_enc"].shape[0])
    losses, t0, bl = [], time.time(), batches(cfg)
    for k, s in enumerate(bl, 1):
        r, loss = arm(tokens[s])
        resid[s] = r
        losses.append(loss)
        fp.update(torch.relu(encode(r.reshape(-1, r.shape[-1]), sae)))
        if k % 20 == 0:
            eta(t0, k, len(bl), f"FP {arm_name}")
    t = fp.table()
    active = t["rate"] > B.ACTIVE_RATE
    cols = np.flatnonzero(active)
    near = B.NearThrStats(B.NEAR_FRAC * t["mu_plus"][cols])
    flat = resid.view(-1, resid.shape[-1])
    for i in range(0, len(flat), 8192):
        near.update(encode(flat[i : i + 8192], sae, cols))
    df = pd.DataFrame(
        {"feature": np.arange(d), "active": active, **t, **B.duan_predictors(t["rate"], t["mean"], t["peak"])}
    )
    df["near_thr"] = np.nan
    df.loc[cols, "near_thr"] = near.value()
    m = B.margin_analog(sae["W_enc"], sae["W_dec"], sae["b_enc"], t["mu_plus"], active)
    beta = sae["b_enc"] - sae["b_dec"] @ sae["W_enc"]
    mb = B.margin_analog(sae["W_enc"], sae["W_dec"], sae["b_enc"], t["mu_plus"], active, bias=beta)
    df["interf"], df["margin"], df["margin_beta"] = m["interf"], m["margin"], mb["margin"]
    df["b_enc"], df["beta"] = sae["b_enc"].double().numpy(), beta.double().numpy()
    df["bias_only"] = np.where(active, -df["b_enc"] / df["mu_plus"], np.nan)
    df["bias_only_beta"] = np.where(active, -df["beta"] / df["mu_plus"], np.nan)
    df["interf_sum"] = B.interference_sum(sae["W_enc"], sae["W_dec"], t["mu_plus"], t["rate"], active)
    summary = {
        "arm": arm_name, "centered": arm.centered, "n_active": int(active.sum()), "tokens": fp.n,
        "ppl_fp": float(np.exp(np.mean(losses))), "l0_mean": float(t["rate"].sum()),
    }  # fmt: skip
    return df, summary


def stage_fp(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    out = make_run_dir("bridge", args.tag or "fp")
    save_provenance(out, cfg, {"stage": "fp"})
    tok = load_tokenizer(cfg)
    tokens = load_tokens(cfg, tok)
    sae = load_sae(cfg)
    print(f"tokens {tuple(tokens.shape)}; SAE {tuple(sae['W_enc'].shape)}", flush=True)
    g1 = gate_g1(cfg, tokens, sae)
    print("G1:", json.dumps(g1), flush=True)
    summ: dict[str, Any] = {"g1": g1}
    arms = {
        "code": Arm("code", load_tl(cfg, tok), cfg["layer"], centered=True),
        "stated": Arm("stated", load_hf(cfg), cfg["layer"], centered=g1["choice"] == "centered"),
    }
    for name, arm in arms.items():
        df, s = fp_arm(cfg, name, arm, tokens, sae)
        df.to_parquet(out / f"features_fp_{name}.parquet", index=False)
        summ[name] = s
        print(name, json.dumps(s), flush=True)
    c = summ["code"]
    summ["g2_fp"] = {
        "active": {"value": c["n_active"], "target": DUAN["active"], "tol": G2_TOL["active"],
                   "pass": abs(c["n_active"] - DUAN["active"]) <= G2_TOL["active"]},
        "ppl_fp": {"value": c["ppl_fp"], "target": DUAN["ppl_fp"], "tol": G2_TOL["ppl_fp"],
                   "pass": abs(c["ppl_fp"] - DUAN["ppl_fp"]) <= G2_TOL["ppl_fp"]},
    }  # fmt: skip
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summ, f, indent=2)
    print("G2 (FP part):", json.dumps(summ["g2_fp"]), f"\nwritten {out}", flush=True)


# ------------------------------------------------------------------------------------------ quantized runs


def stage_quant(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    fp_dir = Path(cfg["fp_run"])
    fp_summary = json.loads((fp_dir / "summary.json").read_text(encoding="utf-8"))
    fp = pd.read_parquet(fp_dir / f"features_fp_{args.arm}.parquet")
    cols = np.flatnonzero(fp["active"].to_numpy())
    leak_thr = B.NEAR_FRAC * fp["mu_plus"].to_numpy()[cols]
    tag = f"{args.arm}-b{'-'.join(map(str, args.bits))}" + (
        "" if args.noise_seed is None else f"-noise{args.noise_seed}"
    )
    out = make_run_dir("bridge", args.tag or tag)
    save_provenance(out, cfg, {"stage": "quant", "arm": args.arm, "bits": args.bits, "noise_seed": args.noise_seed,
                               "fp_run": str(fp_dir)})  # fmt: skip
    tok = load_tokenizer(cfg)
    tokens = load_tokens(cfg, tok)
    sae = load_sae(cfg)
    centered = fp_summary[args.arm]["centered"]
    base = load_tl(cfg, tok) if args.arm == "code" else load_hf(cfg)
    variants = {b: quantize_model(base, args.arm, b, args.noise_seed) for b in args.bits}
    fp_arm_ = Arm(args.arm, base, cfg["layer"], centered)
    q_arms = {b: Arm(args.arm, m, cfg["layer"], centered) for b, m in variants.items()}
    stats = {b: B.PairStats(4, leak_thr) for b in args.bits}
    losses = {"fp": [], **{b: [] for b in args.bits}}
    ps, t0, bl = cfg["pos_split"], time.time(), batches(cfg)
    for k, s in enumerate(bl, 1):
        half = 0 if s.start < cfg["half_split_block"] else 1
        r, loss = fp_arm_(tokens[s])
        losses["fp"].append(loss)
        x = torch.relu(encode(r, sae, cols))  # [B, L, n_active]
        for b, qa in q_arms.items():
            rq, lq = qa(tokens[s])
            losses[b].append(lq)
            y = torch.relu(encode(rq, sae, cols))
            for pg, sl in ((0, slice(None, ps)), (1, slice(ps, None))):
                stats[b].update(2 * half + pg, x[:, sl].reshape(-1, len(cols)), y[:, sl].reshape(-1, len(cols)))
        if k % 10 == 0:
            eta(t0, k, len(bl), tag)
    ppl_fp = float(np.exp(np.mean(losses["fp"])))
    ppl_ref = fp_summary[args.arm]["ppl_fp"]
    rows, summ = [], {"fp_run": str(fp_dir), "arm": args.arm, "noise_seed": args.noise_seed, "ppl_fp": ppl_fp,
                      "fp_sanity": {"ppl_fp_stage": ppl_ref, "pass": abs(ppl_fp - ppl_ref) <= 1e-3},
                      "n_active": len(cols), "variants": {}}  # fmt: skip
    for b in args.bits:
        st = stats[b]
        o = B.outcomes(st.merged())
        h1 = B.outcomes(st.merged([0, 1]))
        h2 = B.outcomes(st.merged([2, 3]))
        p128 = B.outcomes(st.merged([0, 2]))
        rows.append(pd.DataFrame({
            "bits": b, "feature": cols, "corr": o["corr"], "corr_act": o["corr_act"], "dfalse": o["dfalse"],
            "corr_h1": h1["corr"], "corr_h2": h2["corr"], "dfalse_h1": h1["dfalse"], "dfalse_h2": h2["dfalse"],
            "corr_p128": p128["corr"], "dfalse_p128": p128["dfalse"],
        }))  # fmt: skip
        lab = B.label(o["corr"])
        ppl = float(np.exp(np.mean(losses[b])))
        summ["variants"][str(b)] = {
            "survived_pct": 100 * float((lab == 2).mean()), "degraded_pct": 100 * float((lab == 1).mean()),
            "damaged_pct": 100 * float((lab == 0).mean()), "mean_corr": float(np.mean(o["corr"])),
            "median_corr": float(np.median(o["corr"])), "ppl": ppl, "dppl_pct": 100 * (ppl / ppl_fp - 1),
        }  # fmt: skip
    pd.concat(rows).to_parquet(out / "features_q.parquet", index=False)
    if args.arm == "code" and args.noise_seed is None and 6 in args.bits:
        v = summ["variants"]["6"]["survived_pct"]
        summ["g2_int6"] = {"value": v, "target": DUAN["survived_int6"], "tol": G2_TOL["survived_int6"],
                           "pass": abs(v - DUAN["survived_int6"]) <= G2_TOL["survived_int6"]}  # fmt: skip
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summ, f, indent=2)
    print(json.dumps({k: v for k, v in summ.items() if k != "fp_run"}, indent=1), f"\nwritten {out}", flush=True)


# ------------------------------------------------------------------------------------------------ analysis


def merged_table(fp_dir: Path, q_dir: Path, bits: int) -> pd.DataFrame:
    s = json.loads((q_dir / "summary.json").read_text(encoding="utf-8"))
    fp = pd.read_parquet(fp_dir / f"features_fp_{s['arm']}.parquet")
    q = pd.read_parquet(q_dir / "features_q.parquet")
    return q[q.bits == bits].merge(fp, on="feature", validate="one_to_one")


def registered(df: pd.DataFrame, reg: B.Registered) -> dict[str, Any]:
    """The four registered tests (Families A and B for T = near_thr and M = margin), Holm over the four."""
    d = df.dropna(subset=["near_thr", PRIMARY_MARGIN, "log_peak", "corr", "dfalse"])
    y = (d["corr"].to_numpy() > B.SURVIVE).astype(int)
    lp = d["log_peak"].to_numpy()
    res = {
        "A_T": B.family_a(y, lp, d["near_thr"].to_numpy(), reg),
        "A_M": B.family_a(y, lp, d[PRIMARY_MARGIN].to_numpy(), reg),
        "B_T": B.family_b(d["dfalse"].to_numpy(), lp, d["near_thr"].to_numpy(), reg),
        "B_M": B.family_b(d["dfalse"].to_numpy(), lp, d[PRIMARY_MARGIN].to_numpy(), reg),
    }
    ph = B.holm({k: v["p"] for k, v in res.items()})
    for k, v in res.items():
        v["p_holm"] = ph[k]
    res["A_T"]["verdict"] = B.decide_a(res["A_T"], ph["A_T"], reg, -1)
    res["A_M"]["verdict"] = B.decide_a(res["A_M"], ph["A_M"], reg, 0)
    res["B_T"]["verdict"] = B.decide_b(res["B_T"], ph["B_T"], reg, +1)
    res["B_M"]["verdict"] = B.decide_b(res["B_M"], ph["B_M"], reg, 0)
    res["n_features"] = len(d)
    res["n_dropped"] = len(df) - len(d)
    res["survived_frac"] = float(y.mean())
    res["family_a_interpretable"] = bool(min(y.sum(), len(y) - y.sum()) >= 200)  # stop rule
    if not res["family_a_interpretable"]:
        for k in ("A_T", "A_M"):
            res[k]["verdict"] = "not interpretable (fewer than 200 features in a class)"
    # Registered secondary S1 (no Holm): the same tests beyond log peak AND rarity. In the FP stage near_thr has
    # Spearman -0.93 with rarity, so a T effect beyond peak alone may be frequency. Interpretation rule for T in
    # each family: "threshold geometry" only if S1 also meets the minimum effect with unadjusted p < alpha and the
    # same sign; otherwise "beyond peak, not beyond frequency".
    C = d[["log_peak", "rarity"]].to_numpy()
    s1 = {
        "A_T": B.family_a(y, C, d["near_thr"].to_numpy(), reg),
        "A_M": B.family_a(y, C, d[PRIMARY_MARGIN].to_numpy(), reg),
        "B_T": B.family_b(d["dfalse"].to_numpy(), C, d["near_thr"].to_numpy(), reg),
        "B_M": B.family_b(d["dfalse"].to_numpy(), C, d[PRIMARY_MARGIN].to_numpy(), reg),
    }
    res["S1_beyond_peak_and_rarity"] = s1
    for fam, stat, mn, sign in (("A_T", "dauc", reg.dauc_min, None), ("B_T", "rho", reg.rho_min, +1)):
        if res[fam]["verdict"] != "supported":
            continue
        r = s1[fam]
        ok = r[stat] >= mn and r["p"] < reg.alpha
        ok = ok and (np.sign(r["coef_pred"]) == -1 if sign is None else np.sign(r["rho"]) == sign)
        res[fam]["interpretation"] = "threshold geometry" if ok else "beyond peak, not beyond frequency"
    return res


def descriptive(df: pd.DataFrame, reg: B.Registered, outcome: str = "corr") -> dict[str, Any]:
    """Secondary, no permutation: dAUC (spline base on log peak) and cell-consistent partial Spearman with dfalse
    for every predictor, plus peak's univariate AUC and Duan's four-predictor linear CV AUC."""
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score

    preds = ["near_thr", "margin", "margin_beta", "bias_only", "bias_only_beta", "interf_sum", "rarity", "log_mean",
             "log_conc"]  # fmt: skip
    d = df.dropna(subset=["log_peak", outcome, *preds])
    y = (d[outcome].to_numpy() > B.SURVIVE).astype(int)
    lp = d["log_peak"].to_numpy()
    out: dict[str, Any] = {"n": len(d), "survived_frac": float(y.mean())}
    if 0 < y.mean() < 1 and min(y.sum(), len(y) - y.sum()) >= reg.cv_folds:
        out["auc_peak_insample"] = float(roc_auc_score(y, lp))
        splits = B.cv_folds(y, reg.cv_repeats, reg.cv_folds, reg.seed)
        out["auc_peak_cv_linear"] = B.cv_auc(y, lp, splits)
        out["auc_duan4_cv_linear"] = B.cv_auc(y, d[["rarity", "log_mean", "log_peak", "log_conc"]].to_numpy(), splits)
        fd = B.FoldDesign(y, lp, splits)
        out["auc_peak_cv_spline"] = base = fd.auc()
        for p in preds:
            out[f"dauc_{p}"] = fd.auc(d[p].to_numpy()) - base
    cells = B.strata(lp, reg.n_strata)
    for p in [*preds, "log_peak"]:
        out[f"spearman_{p}_corr"] = float(spearmanr(d[p], d[outcome]).statistic)
        if p != "log_peak" and "dfalse" in d:
            out[f"pspearman_{p}_dfalse"] = B.partial_spearman(d[p].to_numpy(), d["dfalse"].to_numpy(), lp, cells)
    return out


def stage_analyze(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    reg = B.Registered()
    fp_dir = Path(cfg["fp_run"])
    out = make_run_dir("bridge", args.tag or "analysis")
    save_provenance(out, cfg, {"stage": "analyze", "primary": str(args.primary), "extra": [str(e) for e in args.extra],
                               "registered": reg.__dict__})  # fmt: skip
    t0 = time.time()
    prim = merged_table(fp_dir, args.primary, reg.bits)
    res: dict[str, Any] = {"primary": registered(prim, reg)}
    print(json.dumps(res["primary"], indent=1), f"({time.time() - t0:.0f}s)", flush=True)
    sec: dict[str, Any] = {"primary_cell": descriptive(prim, reg)}
    sec["primary_cell_corr_act"] = descriptive(prim.assign(corr=prim["corr_act"]), reg)
    sec["halves"] = {
        h: descriptive(prim.assign(corr=prim[f"corr_{h}"], dfalse=prim[f"dfalse_{h}"]), reg) for h in ("h1", "h2")
    }
    sec["pos_lt_128"] = descriptive(prim.assign(corr=prim["corr_p128"], dfalse=prim["dfalse_p128"]), reg)
    for q_dir in [args.primary, *args.extra]:
        s = json.loads((q_dir / "summary.json").read_text(encoding="utf-8"))
        key = f"{s['arm']}{'' if s['noise_seed'] is None else '-noise' + str(s['noise_seed'])}"
        for b in s["variants"]:
            sec[f"{key}-b{b}"] = {"summary": s["variants"][b], **descriptive(merged_table(fp_dir, q_dir, int(b)), reg)}
    res["secondary"] = sec
    with open(out / "results.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, default=float)
    print(f"written {out} ({time.time() - t0:.0f}s)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 6b bridge")
    ap.add_argument("stage", choices=["fp", "quant", "analyze"])
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--arm", choices=["code", "stated"], default="code")
    ap.add_argument("--bits", type=int, nargs="+", default=[6])
    ap.add_argument("--noise-seed", type=int, default=None)
    ap.add_argument("--primary", type=Path, default=None)
    ap.add_argument("--extra", type=Path, nargs="*", default=[])
    ap.add_argument("--fp-run", type=str, default=None, help="override cfg fp_run")
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.fp_run:
        cfg["fp_run"] = args.fp_run
    apply_runtime(cfg)
    {"fp": stage_fp, "quant": stage_quant, "analyze": stage_analyze}[args.stage](cfg, args)


if __name__ == "__main__":
    main()
