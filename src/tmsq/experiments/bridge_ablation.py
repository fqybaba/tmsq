"""Phase 6b, EXPLORATORY (not pre-registered): which difference between Duan's released code and his stated
method drives the INT6 survival gap (code arm 62.4% vs stated arm 85.3%)?

``.venv-bridge\\Scripts\\python -m tmsq.experiments.bridge_ablation --config configs/bridge.yaml --group c1``

Five things differ between the arms. Each variant is a TransformerLens model with explicit processing flags,
quantized at INT6, and differs from one end in exactly one factor:
- F: fold_ln (LayerNorm gains folded into W_Q/W_K/W_V/W_in before quantization);
- C: center_writing_weights (W_O/W_out centered before quantization; the residual becomes centered);
- B: fold_value_biases (b_V folded into b_O, so it bypasses the quantized W_O);
- Q: W_Q/W_K/W_V scales shared across heads per d_head index (Duan's code, "s") or one per output channel
  (per head x d_head, "p"; W_O/W_in/W_out always get per-output-channel scales);
- I: the SAE input, per-token-centered ("c") or raw ("r"; only possible when C = 0).
Code arm A0 = (F1, C1, B1, Qs, Ic); stated arm S = (F0, C0, B0, Qp, Ir). Survival is measured against the FP
model with the same input convention (FP SAE features do not depend on F, C or B: they are reparameterizations).
Reuses the frozen bridge code without modifying it.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .. import bridge as B
from .bridge import TL_QUANT, batches, encode, eta, load_hf, load_sae, load_tokenizer, load_tokens
from .common import apply_runtime, load_config, make_run_dir, save_provenance

BITS = 6
# name: (fold_ln, center_writing, fold_value_biases, qkv scales, sae input)
VARIANTS = {
    "A0": (1, 1, 1, "s", "c"),
    "A0-Q": (1, 1, 1, "p", "c"),
    "A0-F": (0, 1, 1, "s", "c"),
    "A0-B": (1, 1, 0, "s", "c"),
    "A0-C": (1, 0, 1, "s", "c"),
    "A0-CI": (1, 0, 1, "s", "r"),
    "S": (0, 0, 0, "p", "r"),
    "S-Q": (0, 0, 0, "s", "r"),
    "S-F": (1, 0, 0, "p", "r"),
    "S-B": (0, 0, 1, "p", "r"),
    "S-I": (0, 0, 0, "p", "c"),
    "S-CI": (0, 1, 0, "p", "c"),
}
GROUPS = {
    "c1": ["A0", "A0-Q", "A0-F", "A0-B"],
    "c2": ["A0-C", "S-I", "S-CI"],
    "r1": ["S", "S-Q", "S-F"],
    "r2": ["S-B", "A0-CI"],
}


def rtn_reduce(w: torch.Tensor, bits: int, dims: tuple[int, ...]) -> torch.Tensor:
    """RTN with one scale per index of the dims NOT in ``dims`` (Duan's grid and zero-scale rule)."""
    qmin, qmax = B.qrange(bits)
    s = w.abs().amax(dim=dims, keepdim=True) / qmax
    s = torch.where(s == 0, torch.ones_like(s), s)
    return torch.clamp(torch.round(w / s), qmin, qmax) * s


def load_tl_flags(cfg: dict[str, Any], tok: Any, f: int, c: int, b: int) -> Any:
    from transformer_lens import HookedTransformer

    return HookedTransformer.from_pretrained(
        cfg["tl_name"], device="cpu", hf_model=load_hf(cfg), tokenizer=tok, fold_ln=bool(f),
        center_writing_weights=bool(c), center_unembed=bool(c), fold_value_biases=bool(b),
    ).eval()  # fmt: skip


@torch.no_grad()
def quantize_tl(model: Any, bits: int, qkv: str) -> Any:
    """Quantize W_Q/W_K/W_V (shared-across-heads or per-channel scales) and W_O/W_in/W_out (per output channel)."""
    for blk in model.blocks:
        for n in TL_QUANT:
            p = blk.get_parameter(n)
            if n in ("attn.W_Q", "attn.W_K", "attn.W_V") and qkv == "p":
                p.data.copy_(rtn_reduce(p.data, bits, (1,)))  # [h, d_model, d_head]: reduce over d_model only
            else:
                p.data.copy_(B.rtn_lastdim(p.data, bits))
    return model


@torch.no_grad()
def forward(model: Any, tokens: torch.Tensor, layer: int, center: bool) -> tuple[torch.Tensor, float]:
    store: dict[str, torch.Tensor] = {}

    def hook(act, hook):  # noqa: ARG001
        store["r"] = act.detach()

    loss = model.run_with_hooks(tokens, return_type="loss", fwd_hooks=[(f"blocks.{layer}.hook_resid_post", hook)])
    r = store["r"]
    return (r - r.mean(-1, keepdim=True) if center else r), float(loss)


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 6b exploratory ablation of code-vs-stated differences")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--group", choices=list(GROUPS), required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    apply_runtime(cfg)
    names = GROUPS[args.group]
    inp = {VARIANTS[n][4] for n in names}
    assert len(inp) == 1, "a group shares one SAE input convention"
    inp = inp.pop()
    arm = "code" if inp == "c" else "stated"
    fp_dir = Path(cfg["fp_run"])
    fp_summary = json.loads((fp_dir / "summary.json").read_text(encoding="utf-8"))
    fp = pd.read_parquet(fp_dir / f"features_fp_{arm}.parquet")
    cols = np.flatnonzero(fp["active"].to_numpy())
    leak_thr = B.NEAR_FRAC * fp["mu_plus"].to_numpy()[cols]
    out = make_run_dir("bridge_ablation", f"{args.group}-int{BITS}")
    extra = {"stage": "ablation (EXPLORATORY, not pre-registered)", "group": args.group, "bits": BITS,
             "variants": {n: VARIANTS[n] for n in names}, "fp_run": str(fp_dir)}  # fmt: skip
    save_provenance(out, cfg, extra)
    tok = load_tokenizer(cfg)
    tokens = load_tokens(cfg, tok)
    sae = load_sae(cfg)
    # FP reference: default processing gives the centered residual; no processing gives the raw residual
    fp_model = load_tl_flags(cfg, tok, 1, 1, 1) if inp == "c" else load_tl_flags(cfg, tok, 0, 0, 0)
    models = {}
    for n in names:
        f, c, b, q, _ = VARIANTS[n]
        models[n] = (quantize_tl(load_tl_flags(cfg, tok, f, c, b), BITS, q), inp == "c" and c == 0)
    stats = {n: B.PairStats(1, leak_thr) for n in names}
    losses: dict[str, list[float]] = {"fp": [], **{n: [] for n in names}}
    t0, bl = time.time(), batches(cfg)
    for k, s in enumerate(bl, 1):
        r, loss = forward(fp_model, tokens[s], cfg["layer"], center=False)
        losses["fp"].append(loss)
        x = torch.relu(encode(r, sae, cols)).reshape(-1, len(cols))
        for n, (m, center) in models.items():
            rq, lq = forward(m, tokens[s], cfg["layer"], center)
            losses[n].append(lq)
            stats[n].update(0, x, torch.relu(encode(rq, sae, cols)).reshape(-1, len(cols)))
        if k % 10 == 0:
            eta(t0, k, len(bl), args.group)
    ppl_fp = float(np.exp(np.mean(losses["fp"])))
    ref = fp_summary[arm]["ppl_fp"]
    summ: dict[str, Any] = {"exploratory": True, "bits": BITS, "input": inp, "ppl_fp": ppl_fp,
                            "fp_sanity": {"ppl_fp_stage": ref, "pass": abs(ppl_fp - ref) <= 1e-3},
                            "n_active": len(cols), "variants": {}}  # fmt: skip
    rows = []
    for n in names:
        o = B.outcomes(stats[n].merged())
        lab = B.label(o["corr"])
        ppl = float(np.exp(np.mean(losses[n])))
        f, c, b, q, i = VARIANTS[n]
        summ["variants"][n] = {
            "flags": {"fold_ln": f, "center_writing": c, "fold_value_biases": b, "qkv_scales": q, "sae_input": i},
            "survived_pct": 100 * float((lab == 2).mean()), "damaged_pct": 100 * float((lab == 0).mean()),
            "mean_corr": float(np.mean(o["corr"])), "ppl": ppl, "dppl_pct": 100 * (ppl / ppl_fp - 1),
        }  # fmt: skip
        rows.append(pd.DataFrame({"variant": n, "feature": cols, "corr": o["corr"], "dfalse": o["dfalse"]}))
    pd.concat(rows).to_parquet(out / "features_ablation.parquet", index=False)
    with open(out / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summ, fh, indent=2)
    print(json.dumps(summ, indent=1), f"\nwritten {out}", flush=True)


if __name__ == "__main__":
    main()
