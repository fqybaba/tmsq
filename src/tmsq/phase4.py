"""Phase 4: head vs tail (H4) in setting D. Specification: the Phase 4 pre-registration in notes/LOG.md.

Reuses the Phase 3 statistics helpers (per-instance Spearman with a validity rule, summaries, Holm, ICC).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import phase3 as P3

PRIMARY_GRAN, PRIMARY_BITS = "row", 3
SECONDARY_BITS = (3, 4, 5)
REGIMES = ("weak", "strong")
MIN_N = 10  # eligible features per instance (weak regime has ~23-35 represented features)
MIN_DISTINCT = 5
MIN_VALID = 16  # of 20 instances per regime
MIN_VALID_SENS = 12  # of the 15 strong-regime instances with phi_half >= 0.9 (sensitivity)
PHI_STRONG = 0.9
EFFECT_MIN = 0.2
ICC_MIN = 0.10
HEAD_FRAC, TAIL_FRAC = 0.10, 0.50
COLUMNS = ["setting", "instance", "feature", "bits", "granularity", "method", "noise_seed", "ablation", "eligible",
           "represented", "p", "freq_rank", "alpha", "k", "gamma", "norm", "D", "D_plus", "bias", "near_thr",
           "margin_tau", "R2_fp", "dR2", "rel_damage", "dLoss", "mse_fp", "dcond_mse", "dleak_energy",
           "dfalse_act", "false_act_fp", "var_x"]  # fmt: skip


def load(path) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=COLUMNS)
    df["neg_rank"] = -df["freq_rank"]  # larger = more frequent (rank 0 = most frequent)
    # split of dR2 into its active and inactive terms: dR2 = [p dcond_mse + (1 - p) dleak_energy] / Var(x)
    with np.errstate(divide="ignore", invalid="ignore"):
        df["dR2_active_term"] = df.p * df.dcond_mse / df.var_x
        df["dR2_inactive_term"] = (1 - df.p) * df.dleak_energy / df.var_x
    return df


def cell(df: pd.DataFrame, gran: str, bits: int, method: str = "rtn") -> pd.DataFrame:
    m = (df.granularity == gran) & (df.bits == bits) & (df.method == method) & (df.ablation == "none") & df.eligible
    return df[m]


def buckets(c: pd.DataFrame) -> pd.Series:
    """Per instance, among eligible features: 'head' = top 10% by p, 'tail' = bottom 50% by p, else 'middle'."""
    out = pd.Series("middle", index=c.index)
    for _, d in c.groupby("instance"):
        r = d.p.rank(ascending=False, method="first")  # 1 = most frequent
        n = len(d)
        n_head = max(1, int(np.floor(HEAD_FRAC * n)))
        n_tail = max(1, int(np.floor(TAIL_FRAC * n)))
        out.loc[d.index[r <= n_head]] = "head"
        out.loc[d.index[r > n - n_tail]] = "tail"
    return out


def h4_test(df: pd.DataFrame, gran: str, bits: int, min_valid: int, x: str = "p", y: str = "rel_damage",
            partial: str | None = None) -> dict:  # fmt: skip
    c = cell(df, gran, bits)
    rh, inv, dropped = P3.instance_rhos(c, x, y, partial=partial, min_n=MIN_N)
    s = P3.summarize(rh, min_valid)
    return {**s, "rhos": rh, "invalid": inv, "dropped": dropped}


def head_tail(df: pd.DataFrame, gran: str, bits: int, min_valid: int, value: str) -> dict:
    """Per-instance mean(value | head) - mean(value | tail), with per-bucket means and eligible counts."""
    c = cell(df, gran, bits)
    c = c.assign(bucket=buckets(c))
    diffs, heads, tails = {}, [], []
    for inst, d in c.groupby("instance"):
        h, t = d[d.bucket == "head"][value].dropna(), d[d.bucket == "tail"][value].dropna()
        if len(h) and len(t):
            diffs[int(inst)] = float(h.mean() - t.mean())
            heads.append(h.mean())
            tails.append(t.mean())
    s = P3.summarize(diffs, min_valid)
    return {
        **s,
        "head_mean": float(np.mean(heads)) if heads else np.nan,
        "tail_mean": float(np.mean(tails)) if tails else np.nan,
    }


def decide(s: dict, p_holm: float, icc: float) -> str:
    """Two-sided exploratory H4 categories (pre-registration)."""
    if not s["summarised"]:
        return "Not testable (too few valid instances)"
    if p_holm >= P3.ALPHA:
        if np.isfinite(icc) and icc < ICC_MIN:
            return "Not testable (outcome dominated by the perturbation draw)"
        return "No detectable frequency effect"
    direction = (
        "head loses a larger relative fraction (Wu-like)" if s["mean"] > 0 else "tail loses a larger relative fraction"
    )
    size = "" if abs(s["mean"]) >= EFFECT_MIN else " [small: |mean rho| < 0.2]"
    return direction + size


def regime_descriptives(df: pd.DataFrame) -> pd.DataFrame:
    """FP-only, per bucket: eligible fraction and geometry covariates (D, bias, near_thr, norm)."""
    c = df[(df.method == "rtn") & (df.bits == 8) & (df.granularity == "row") & (df.ablation == "none")]
    rows = []
    for inst, d in c.groupby("instance"):
        el = d[d.eligible].assign(bucket=buckets(d[d.eligible]))
        rows.append({"instance": inst, "n_eligible": int(d.eligible.sum()), "n_represented": int(d.represented.sum()),
                     **{f"{b}_{v}": float(el[el.bucket == b][v].mean()) if (el.bucket == b).any() else np.nan
                        for b in ("head", "tail") for v in ("p", "D", "bias", "near_thr", "norm", "R2_fp")}})  # fmt: skip
    return pd.DataFrame(rows)
