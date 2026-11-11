"""Branch A (scale): FP-only facts about candidate predictors, computed BEFORE the pre-registration.

No quantized model exists yet; this reads only the per-latent FP tables of the smoke runs
(``results/bridge_scale/<run>/latents_fp_L<layer>.parquet``). For the eligible latents (FP firing rate > 0.001)
of each (size, layer) it prints the distribution of each candidate predictor, its Spearman correlation with Duan's
four statistics, and the R^2 of its ranks on the ranks of the four statistics (how much of it is already in
Duan's model). Run: ``.venv\\Scripts\\python notes/posthoc/scale_fp_predictor_facts.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[2]
RUNS = {
    ("70m", 4): "20261111-110140-smoke-70m",
    ("160m", 4): "20261111-111133-smoke-160m",
    ("160m", 9): "20261111-111133-smoke-160m",
    ("410m", 4): "20261111-111519-smoke-410m",
    ("410m", 19): "20261111-111519-smoke-410m",
}
ACTIVE = 0.001
DUAN = ["rarity", "log_mean", "log_peak", "log_conc"]


def table(size: str, layer: int) -> tuple[pd.DataFrame, float]:
    run = ROOT / "results" / "bridge_scale" / RUNS[(size, layer)]
    df = pd.read_parquet(run / f"latents_fp_L{layer}.parquet")
    cut_mean = json.loads((run / "summary.json").read_text(encoding="utf-8"))["layers"][str(layer)]["fp"]["cut_mean"]
    d = df[df["rate"] > ACTIVE].copy()
    d["rarity"] = -np.log(d["rate"])
    d["log_mean"] = np.log(d["mean"])
    d["log_peak"] = np.log(d["peak"])
    d["log_conc"] = np.maximum(np.log(d["peak"] / d["mean"]), 0.0)
    d["log_mu_plus"] = np.log(d["mu_plus"])
    d["mu_plus_over_cut"] = d["mu_plus"] / cut_mean
    d["z_margin"] = (cut_mean - d["mean_z"]) / d["sd_z"]  # sd units below the mean cut; larger = further from firing
    for f in ("0.05", "0.1", "0.25"):
        d[f"near_off_{f}_per_off"] = d[f"near_off_{f}"] / (1 - d["rate"])  # P(near cut | off)
        d[f"near_on_{f}_per_on"] = d[f"near_on_{f}"] / d["rate"]  # P(near cut | firing)
    d["rank2k_per_off"] = d["rate_rank2k"] / (1 - d["rate"])
    return d, cut_mean


def r2_on(d: pd.DataFrame, y: str, xs: list[str]) -> float:
    X = np.column_stack([np.ones(len(d))] + [rankdata(d[x]) for x in xs])
    r = rankdata(d[y])
    beta, *_ = np.linalg.lstsq(X, r, rcond=None)
    res = r - X @ beta
    return float(1 - res.var() / r.var())


def main() -> None:
    cands = ["near_off_0.05", "near_off_0.1", "near_off_0.25", "near_off_0.1_per_off", "near_on_0.05_per_on",
             "near_on_0.1_per_on", "near_on_0.25_per_on", "rate_rank2k", "rank2k_per_off", "z_margin",
             "mu_plus_over_cut", "pos0_share"]  # fmt: skip
    for size, layer in RUNS:
        d, cut_mean = table(size, layer)
        print(f"\n=== {size} layer {layer}: {len(d)} eligible latents (rate > {ACTIVE}); mean cut {cut_mean:.3f} ===")
        print("Duan-4 Spearman matrix (rarity, log_mean, log_peak, log_conc):")
        print(np.round(spearmanr(d[DUAN]).statistic, 2))
        print(
            f"{'candidate':24s} {'median':>9s} {'q10':>9s} {'q90':>9s} | rho: rarity log_mean log_peak log_conc "
            f"| R2 on Duan-4 ranks | R2 on rarity+log_peak"
        )
        for c in cands:
            v = d[c]
            rho = [spearmanr(v, d[k]).statistic for k in DUAN]
            print(
                f"{c:24s} {v.median():9.4g} {v.quantile(0.1):9.4g} {v.quantile(0.9):9.4g} | "
                f"{rho[0]:+7.2f} {rho[1]:+8.2f} {rho[2]:+8.2f} {rho[3]:+8.2f} | {r2_on(d, c, DUAN):18.3f} | "
                f"{r2_on(d, c, ['rarity', 'log_peak']):.3f}"
            )
        print(
            "Spearman among candidates (near_off_0.1, near_on_0.1_per_on, rank2k_per_off, z_margin, mu_plus_over_cut):"
        )
        sub = ["near_off_0.1", "near_on_0.1_per_on", "rank2k_per_off", "z_margin", "mu_plus_over_cut"]
        print(np.round(spearmanr(d[sub]).statistic, 2))
        print(
            f"log_mu_plus vs log_mean + rarity identity: max abs diff "
            f"{(d['log_mu_plus'] - d['log_mean'] - d['rarity']).abs().max():.2e}"
        )


if __name__ == "__main__":
    main()
