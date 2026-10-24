"""Phase 3: the pre-registered hypothesis tests.

Specification: notes/LOG.md, "Phase 3 pre-registration (v2...)" (commit f76b49a, approved) and the
implementation clarifications (commit d3b8bfa). Every constant below is taken from those entries; nothing
here may be changed after K results have been looked at without a logged deviation.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------------------------------------------------
# Registered constants
# ------------------------------------------------------------------------------------------------------
PRIMARY_CELL = (0.7488, "row", 3)  # (S rounded to 4 decimals, granularity, bits)
S_GRID = (0.7488, 0.8415, 0.9, 0.9602)
GRANULARITIES = ("tensor", "column", "row")
SECONDARY_BITS = (3, 4, 5)
ALL_BITS = (8, 6, 5, 4, 3, 2)
ALPHA = 0.05
EFFECT_MIN = 0.2  # |mean rho| for Supported
DOWNGRADE_MIN = 0.1  # |mean rho| floor for a downgrade cell
ICC_MIN = 0.10  # testability by reliability
IDENT_MAX = 0.8  # identifiability rule rho(norm, D)
MIN_N = 20  # eligible features per instance
MIN_DISTINCT = 5
VALIDITY_MAX_8BIT = 0.05  # label validity rule
CELL_MIN_CLASS = 30
CELL_MIN_INSTANCES = 3
AUC_MIN_CLASS = 5  # per-instance AUC
BOOT_RHO = 1_000
BOOT_AUC = 10_000
LABEL_TOL = 1e-7  # clarification 6
SURV_CORR, SURV_DR2, SURV_DFALSE = 0.9, 0.1, 0.05


@dataclass(frozen=True)
class Hypothesis:
    name: str
    x: str
    y: str
    sign: int  # direction as stated in CLAUDE.md
    alt: str | None  # registered alternative reading of the opposite sign
    downgrade: tuple[tuple[float, str, int], ...]


HYPOTHESES = (
    Hypothesis(
        "H1",
        "margin_tau",
        "dfalse_act",
        -1,
        "threshold-density reading",
        ((0.7488, "column", 3), (0.7488, "tensor", 3)),
    ),
    Hypothesis(
        "H2",
        "D",
        "dR2",
        -1,
        None,
        ((0.8415, "row", 3), (0.9, "row", 3), (0.7488, "column", 3), (0.7488, "tensor", 3)),
    ),
    Hypothesis("H3", "norm", "dR2", -1, "absolute-error reading", ((0.8415, "row", 3), (0.9, "row", 3))),
)
D_PLUS_COMPANION = Hypothesis("H2_Dplus", "D_plus", "dR2", -1, None, HYPOTHESES[1].downgrade)

# Secondary 1: (predictor, outcome, registered sign or None when the plan registers none, partial-on or None)
SECONDARY_PAIRS = (
    ("margin_tau", "dfalse_act", -1, None),
    ("D", "dR2", -1, None),
    ("norm", "dR2", -1, None),
    ("margin_tau", "dR2", None, None),
    ("near_thr", "abs_dfalse_act", +1, None),
    ("D_plus", "dR2", -1, None),
    ("norm", "dcond_mse", None, None),
    ("p99_out_fp", "dR2", -1, None),
    ("peak_out_fp", "dR2", None, None),
    ("D", "dR2", None, "norm"),
    ("norm", "dR2", None, "D"),
)
EXPECTED_BITS = set(ALL_BITS)
EXPECTED_NOISE_SEEDS = 5
# Secondary 3: predictors and their registered orientation (score = sign * predictor, label 1 = damaged)
AUC_PREDICTORS = {
    "margin_tau": -1,
    "D": -1,
    "D_plus": -1,
    "norm": -1,
    "p99_out_fp": -1,
    "peak_out_fp": -1,
    "near_thr": +1,
}
LOGIT_SETS = {
    "{p99_out_fp}": ["p99_out_fp"],
    "{peak_out_fp}": ["peak_out_fp"],
    "{margin_tau}": ["margin_tau"],
    "{D}": ["D"],
    "{all}": ["margin_tau", "D", "D_plus", "norm", "p99_out_fp"],
}

FEATURE_COLUMNS = [
    "S",
    "instance",
    "feature",
    "bits",
    "granularity",
    "method",
    "noise_seed",
    "ablation",
    "eligible",
    "represented",
    "p",
    "margin_tau",
    "D",
    "D_plus",
    "norm",
    "near_thr",
    "p99_out_fp",
    "peak_out_fp",
    "peak_ratio",
    "false_act_fp",
    "dR2",
    "dfalse_act",
    "dcond_mse",
    "dleak_energy",
    "dW",
    "surv_corr",
    "survived",
    "composite_survived",
]


def load_features(path) -> pd.DataFrame:
    """Load a sweep's features.parquet with the Phase 3 columns plus ``S4`` and ``abs_dfalse_act``."""
    df = pd.read_parquet(path, columns=FEATURE_COLUMNS)
    df["S4"] = df["S"].round(4)
    df["abs_dfalse_act"] = df["dfalse_act"].abs()
    return df


def rows(df: pd.DataFrame, S: float, gran: str, bits: int, method: str = "rtn", eligible: bool = True) -> pd.DataFrame:
    """Rows of one (S, granularity, bits) cell, ablation 'none'."""
    m = (df.S4 == S) & (df.granularity == gran) & (df.bits == bits) & (df.method == method) & (df.ablation == "none")
    if eligible:
        m &= df.eligible
    return df[m]


# ------------------------------------------------------------------------------------------------------
# Statistics helpers
# ------------------------------------------------------------------------------------------------------
def partial_spearman(x, y, z) -> float:
    """Spearman correlation of x and y given z: Pearson of the OLS residuals of the average ranks on rank(z)."""
    rx, ry, rz = (stats.rankdata(np.asarray(v, dtype=float)) for v in (x, y, z))
    A = np.column_stack([np.ones_like(rz), rz])
    ex = rx - A @ np.linalg.lstsq(A, rx, rcond=None)[0]
    ey = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
    if ex.std() == 0 or ey.std() == 0:
        return float("nan")
    return float(np.corrcoef(ex, ey)[0, 1])


def instance_rhos(
    c: pd.DataFrame,
    x: str,
    y: str,
    partial: str | None = None,
    min_n: int = MIN_N,
    min_distinct_x: int = MIN_DISTINCT,
    min_distinct_y: int = MIN_DISTINCT,
) -> tuple[dict[int, float], dict[int, str], int]:
    """Per-instance (partial) Spearman with the registered validity rule; returns (rhos, invalid, dropped rows)."""
    rhos: dict[int, float] = {}
    invalid: dict[int, str] = {}
    dropped = 0
    cols = [x, y] + ([partial] if partial else [])
    for inst, d in c.groupby("instance"):
        dd = d[cols].dropna()
        dropped += len(d) - len(dd)
        if len(dd) < min_n:
            invalid[int(inst)] = f"n={len(dd)} < {min_n}"
            continue
        if dd[x].nunique() < min_distinct_x:
            invalid[int(inst)] = f"{x} has {dd[x].nunique()} distinct values"
            continue
        if dd[y].nunique() < min_distinct_y:
            invalid[int(inst)] = f"{y} has {dd[y].nunique()} distinct values"
            continue
        if partial and dd[partial].nunique() < min_distinct_x:
            invalid[int(inst)] = f"{partial} has {dd[partial].nunique()} distinct values"
            continue
        r = partial_spearman(dd[x], dd[y], dd[partial]) if partial else stats.spearmanr(dd[x], dd[y]).statistic
        if not np.isfinite(r):
            invalid[int(inst)] = "rho undefined"
            continue
        rhos[int(inst)] = float(r)
    return rhos, invalid, dropped


def _signed_sums(v: np.ndarray) -> np.ndarray:
    """All 2^len(v) values of sum(s_i v_i), s_i in {+1, -1}, in itertools.product((1, -1)) order."""
    signs = np.array(list(itertools.product((1.0, -1.0), repeat=len(v))))
    return signs @ v


def sign_flip_p(v: np.ndarray) -> float:
    """Exact two-sided sign-flip permutation p-value of the mean (all 2^n sign patterns).

    Meet-in-the-middle: the 2^n signed sums are the outer sum of the two halves' signed sums, so n = 20
    needs two 2^10 enumerations instead of a 2^20 x 20 sign matrix. Still exact; identical to the brute force.
    """
    v = np.asarray(v, dtype=float)
    n = len(v)
    if n == 0:
        return float("nan")
    obs = abs(v.mean())
    h = n // 2
    total = _signed_sums(v[:h])[:, None] + _signed_sums(v[h:])[None, :] if h else _signed_sums(v)[None, :]
    means = np.abs(total / n)
    return float(np.mean(means >= obs - 1e-12))


def summarize(values: dict[int, float] | np.ndarray, min_valid: int, boot: int = BOOT_RHO, seed: int = 0) -> dict:
    """Mean, std, 95% t-interval, two-sided t-test p, sign-flip p and bootstrap interval of per-instance values."""
    v = np.asarray(list(values.values()) if isinstance(values, dict) else values, dtype=float)
    n = len(v)
    out = {"n_valid": n, "summarised": bool(n >= min_valid), "mean": float("nan"), "std": float("nan")}
    out.update({"ci_lo": float("nan"), "ci_hi": float("nan"), "p": float("nan"), "p_signflip": float("nan")})
    out.update({"boot_lo": float("nan"), "boot_hi": float("nan")})
    if n == 0:
        return out
    out["mean"] = float(v.mean())
    if n >= 2:
        sd = float(v.std(ddof=1))
        out["std"] = sd
        half = stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n)
        out["ci_lo"], out["ci_hi"] = out["mean"] - half, out["mean"] + half
        out["p"] = float(stats.ttest_1samp(v, 0.0).pvalue) if sd > 0 else (0.0 if out["mean"] != 0 else 1.0)
        out["p_signflip"] = sign_flip_p(v)
        rng = np.random.default_rng(seed)
        bm = v[rng.integers(0, n, size=(boot, n))].mean(axis=1)
        out["boot_lo"], out["boot_hi"] = float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))
    if not out["summarised"]:  # "a cell is summarised only if >= min_valid instances are valid"
        for k in ("ci_lo", "ci_hi", "p", "p_signflip", "boot_lo", "boot_hi"):
            out[k] = float("nan")
    return out


def sqrt_icc(icc: float) -> float:
    """Reliability ceiling sqrt(max(ICC, 0)) (clarification 7)."""
    return math.sqrt(max(icc, 0.0)) if np.isfinite(icc) else float("nan")


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values; NaN p (untestable) enters as 1 (clarification 3)."""
    names = list(pvals)
    p = np.array([1.0 if not np.isfinite(pvals[k]) else pvals[k] for k in names])
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[idx]))
        adj[idx] = running
    return {k: float(a) for k, a in zip(names, adj, strict=True)}


def bh(pvals: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg q-values (NaN stays NaN)."""
    p = np.asarray(pvals, dtype=float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    pv = p[ok]
    m = len(pv)
    if m == 0:
        return q
    order = np.argsort(pv)
    ranked = pv[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    qq = np.empty(m)
    qq[order] = np.minimum(ranked, 1.0)
    q[ok] = qq
    return q


def icc1(noise_cell: pd.DataFrame, y: str) -> tuple[float, list[float]]:
    """One-way ICC(1) of ``y`` over noise seeds (features = groups) per instance, averaged (clarification 7)."""
    vals = []
    for _, d in noise_cell.groupby("instance"):
        piv = d.pivot_table(index="feature", columns="noise_seed", values=y).dropna()
        n, k = piv.shape
        if n < 2 or k < 2:
            continue
        grand = piv.values.mean()
        rowm = piv.mean(axis=1)
        msb = k * ((rowm - grand) ** 2).sum() / (n - 1)
        msw = ((piv.sub(rowm, axis=0)) ** 2).values.sum() / (n * (k - 1))
        denom = msb + (k - 1) * msw
        vals.append(float((msb - msw) / denom) if denom > 0 else float("nan"))
    return (float(np.nanmean(vals)) if vals else float("nan")), vals


def concordance_matrix(scores: np.ndarray, labels: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """C[k, l] = #(pos in k, neg in l) with s_pos > s_neg (+0.5 ties); n[k] = (n_pos, n_neg). For fast
    cluster-bootstrap AUCs: a resample with multiplicities w gives AUC = w C w / (w n_pos)(w n_neg)."""
    ug = np.unique(groups)
    K = len(ug)
    C = np.zeros((K, K))
    npos = np.zeros(K)
    nneg = np.zeros(K)
    pos = [scores[(groups == g) & (labels == 1)] for g in ug]
    neg = [scores[(groups == g) & (labels == 0)] for g in ug]
    for a in range(K):
        npos[a] = len(pos[a])
        nneg[a] = len(neg[a])
    for a in range(K):
        if not len(pos[a]):
            continue
        for b in range(K):
            if not len(neg[b]):
                continue
            diff = pos[a][:, None] - neg[b][None, :]
            C[a, b] = (diff > 0).sum() + 0.5 * (diff == 0).sum()
    return C, np.stack([npos, nneg], axis=1)


def pooled_auc_with_bootstrap(scores, labels, groups, boot: int = BOOT_AUC, seed: int = 0) -> dict:
    """Pooled AUC and a percentile cluster bootstrap over groups (instances)."""
    scores, labels, groups = (np.asarray(v) for v in (scores, labels, groups))
    C, nn = concordance_matrix(scores, labels.astype(int), groups)
    K = C.shape[0]
    w1 = np.ones(K)
    auc = float(w1 @ C @ w1 / ((w1 @ nn[:, 0]) * (w1 @ nn[:, 1])))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, K, size=(boot, K))
    W = np.stack([np.bincount(r, minlength=K) for r in idx]).astype(float)
    num = np.einsum("bk,kl,bl->b", W, C, W)
    den = (W @ nn[:, 0]) * (W @ nn[:, 1])
    with np.errstate(invalid="ignore", divide="ignore"):
        bs = num / den
    bs = bs[np.isfinite(bs)]
    return {"auc_pooled": auc, "boot_lo": float(np.percentile(bs, 2.5)), "boot_hi": float(np.percentile(bs, 97.5))}


def auc_rank(score: np.ndarray, label: np.ndarray) -> float:
    """Mann–Whitney AUC with average ranks for ties; label 1 = positive."""
    score = np.asarray(score, dtype=float)
    label = np.asarray(label).astype(int)
    n1 = label.sum()
    n0 = len(label) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = stats.rankdata(score)
    return float((r[label == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# ------------------------------------------------------------------------------------------------------
# Labels, validity and cell rules
# ------------------------------------------------------------------------------------------------------
def composite_label(c: pd.DataFrame, corr: float = SURV_CORR, dr2: float = SURV_DR2, dfalse: float = SURV_DFALSE):
    """Damaged = 1 / survived = 0 from stored columns (clarification 6); NaN when dfalse_act is NaN."""
    surv = (c.surv_corr >= corr - LABEL_TOL) & (c.dR2 <= dr2 + LABEL_TOL) & (c.dfalse_act <= dfalse + LABEL_TOL)
    lab = np.where(c.dfalse_act.isna(), np.nan, (~surv).astype(float))
    return pd.Series(lab, index=c.index)


def damaged(c: pd.DataFrame) -> pd.Series:
    """Registered binary label from the stored composite column: 1 = damaged, 0 = survived, NaN undefined."""
    return 1.0 - c.composite_survived


def label_validity(df: pd.DataFrame, S: float, gran: str) -> dict:
    c8 = rows(df, S, gran, 8)
    lab = damaged(c8).dropna()
    frac = float(lab.mean()) if len(lab) else float("nan")
    return {
        "S": S,
        "granularity": gran,
        "damaged_frac_8bit": frac,
        "valid": bool(len(lab) > 0 and frac <= VALIDITY_MAX_8BIT),
    }


def cell_rule(df: pd.DataFrame, S: float, gran: str, bits: int, valid: bool) -> dict:
    c = rows(df, S, gran, bits)
    lab = damaged(c)
    ok = lab.notna()
    nd = int((lab[ok] == 1).sum())
    nu = int((lab[ok] == 0).sum())
    inst_d = int(c.loc[ok & (lab == 1), "instance"].nunique())
    inst_u = int(c.loc[ok & (lab == 0), "instance"].nunique())
    passes = (
        valid
        and nd >= CELL_MIN_CLASS
        and nu >= CELL_MIN_CLASS
        and inst_d >= CELL_MIN_INSTANCES
        and inst_u >= CELL_MIN_INSTANCES
    )
    pearson_damaged = int((~c.survived.astype(bool)).sum())  # Pearson-only label, reported as secondary
    return {"S": S, "granularity": gran, "bits": bits, "label_valid": valid, "n_damaged": nd, "n_undamaged": nu,
            "inst_damaged": inst_d, "inst_undamaged": inst_u, "passes": bool(passes),
            "pearson_only_damaged": pearson_damaged, "n_eligible": int(len(c))}  # fmt: skip


# ------------------------------------------------------------------------------------------------------
# Primary tests and verdicts
# ------------------------------------------------------------------------------------------------------
@dataclass
class PrimaryResult:
    hypothesis: str
    summary: dict
    rhos: dict
    invalid: dict
    dropped: int
    p_holm: float = float("nan")
    icc: float = float("nan")
    sqrt_icc: float = float("nan")
    p_holm_signflip: float = float("nan")
    holm_decision_disagrees: bool = False
    label: str = ""
    verdict: str = ""
    reasons: list = field(default_factory=list)
    downgrade: list = field(default_factory=list)


def run_test(
    df: pd.DataFrame, h: Hypothesis, cell: tuple[float, str, int], min_valid: int, exclude_leak_free: bool = False
):
    c = rows(df, *cell)
    if exclude_leak_free:
        c = c[c.false_act_fp > 0]
    rh, inv, dropped = instance_rhos(c, h.x, h.y)
    return rh, inv, dropped, summarize(rh, min_valid)


def decide(h: Hypothesis, summ: dict, p_holm: float, icc: float, downgrade: list[dict]) -> tuple[str, list[str]]:
    """Ordered verdict rule (a)–(f) of pre-registration v2."""
    if not summ["summarised"]:
        return "Not testable", [f"only {summ['n_valid']} valid instances"]
    mean = summ["mean"]
    if p_holm < ALPHA and np.sign(mean) == -h.sign:
        alt = f"; consistent with the registered {h.alt}" if h.alt else ""
        return "Not supported (opposite sign)" + alt, []
    if p_holm >= ALPHA:
        if np.isfinite(icc) and icc < ICC_MIN:
            return "Not testable (outcome dominated by the perturbation draw)", [f"ICC_noise = {icc:.2f} < {ICC_MIN}"]
        return "Not supported", []
    reasons = []
    if abs(mean) < EFFECT_MIN:
        reasons.append(f"small: |mean rho| = {abs(mean):.3f} < {EFFECT_MIN}")
    for d in downgrade:
        s = d["summary"]
        if s["summarised"] and np.sign(s["mean"]) == -h.sign and abs(s["mean"]) >= DOWNGRADE_MIN and s["p"] < ALPHA:
            reasons.append(f"inconsistent: {d['cell']} mean rho {s['mean']:+.3f}, p = {s['p']:.3g}")
    return ("Mixed", reasons) if reasons else ("Supported", [])


def primary_analysis(df: pd.DataFrame, min_valid: int, precheck: dict) -> dict:
    """All primary tests, Holm, testability, identifiability, downgrade cells and the D_plus companion."""
    stop = precheck.get("stop", False)
    ident = precheck["rho_norm_D_primary"]
    merged = bool(np.isfinite(ident) and ident >= IDENT_MAX)
    icc = {"dfalse_act": precheck["icc_primary"]["dfalse_act"], "dR2": precheck["icc_primary"]["dR2"]}
    labels = {"H1": "H1 margin_tau vs dfalse_act", "H2": "H2 D vs dR2", "H3": "H3 norm vs dR2",
              "H2_Dplus": "H2 companion (secondary, no Holm): D_plus vs dR2"}  # fmt: skip
    if merged:
        labels["H2"] = "H2/H3 norm/D axis (H2's test; identifiability rule)"
    results: dict[str, PrimaryResult] = {}
    for h in HYPOTHESES + (D_PLUS_COMPANION,):
        rh, inv, dropped, summ = run_test(df, h, PRIMARY_CELL, min_valid)
        res = PrimaryResult(
            h.name, summ, rh, inv, dropped, icc=icc[h.y], sqrt_icc=sqrt_icc(icc[h.y]), label=labels[h.name]
        )
        for cell in h.downgrade:
            r2, i2, dr2, s2 = run_test(df, h, cell, min_valid)
            icc_c = icc1(rows(df, *cell, method="noise"), h.y)[0]
            res.downgrade.append({"cell": f"S={cell[0]} {cell[1]} {cell[2]}b", "summary": s2, "invalid": i2,
                                  "dropped_rows": dr2, "sqrt_icc_noise": sqrt_icc(icc_c),
                                  "status": "summarised" if s2["summarised"] else "not testable (too few valid instances)"})  # fmt: skip
        results[h.name] = res
    family = ["H1", "H2"] if merged else ["H1", "H2", "H3"]
    adj = holm({k: (results[k].summary["p"] if results[k].summary["summarised"] else float("nan")) for k in family})
    adj_sf = holm(
        {k: (results[k].summary["p_signflip"] if results[k].summary["summarised"] else float("nan")) for k in family}
    )
    for k in family:
        results[k].p_holm = adj[k]
        results[k].p_holm_signflip = adj_sf[k]
        results[k].holm_decision_disagrees = bool((adj[k] < ALPHA) != (adj_sf[k] < ALPHA))
    comp = results["H2_Dplus"]
    comp.p_holm = comp.summary["p"] if comp.summary["summarised"] else float("nan")  # no Holm for the companion
    comp.p_holm_signflip = comp.summary["p_signflip"] if comp.summary["summarised"] else float("nan")
    comp.holm_decision_disagrees = bool(
        np.isfinite(comp.p_holm) and (comp.p_holm < ALPHA) != (comp.p_holm_signflip < ALPHA)
    )
    for name, res in results.items():
        h = next(x for x in HYPOTHESES + (D_PLUS_COMPANION,) if x.name == name)
        if stop:
            res.verdict, res.reasons = "STOP (sanity check failed; descriptive only)", precheck.get("stop_reasons", [])
        elif merged and name == "H3":
            res.verdict, res.reasons = (
                "Merged with H2 into the norm/D axis (identifiability rule)",
                [f"rho(norm, D) = {ident:.2f}"],
            )
        else:
            p_eff = res.p_holm if np.isfinite(res.p_holm) else 1.0
            res.verdict, res.reasons = decide(h, res.summary, p_eff, res.icc, res.downgrade)
    if not stop:
        d_cat, dp_cat = results["H2"].verdict, results["H2_Dplus"].verdict
        note = None
        if d_cat != dp_cat:
            note = f"D (paper-comparable): {d_cat}; D_plus (sign-aware, carries the mechanistic claim): {dp_cat}"
    else:
        note = None
    leak_free = {}
    for h in HYPOTHESES:
        rh, inv, dropped, summ = run_test(df, h, PRIMARY_CELL, min_valid, exclude_leak_free=True)
        leak_free[h.name] = {**summ, "sqrt_icc_noise": sqrt_icc(icc[h.y]), "n_invalid": len(inv)}
    prim_cell = next(c for c in precheck["cells"] if (c["S"], c["granularity"], c["bits"]) == PRIMARY_CELL)
    label_status = {"label_valid": prim_cell["label_valid"], "passes_cell_rule": prim_cell["passes"],
                    "note": None if prim_cell["passes"] else
                    "primary cell fails the label validity or count rule: label-based secondaries skipped there; "
                    "primary Spearman verdicts unaffected (pre-registration, stop rule 5)"}  # fmt: skip
    return {"merged_H2_H3": merged, "rho_norm_D_primary": ident, "holm_family": family, "results": results,
            "D_vs_Dplus_note": note, "leak_free_sensitivity": leak_free, "primary_cell_label_status": label_status}  # fmt: skip


# ------------------------------------------------------------------------------------------------------
# Pre-checks (labels, FP quantities and outcome-only statistics; no predictor-outcome association)
# ------------------------------------------------------------------------------------------------------
def completeness(df: pd.DataFrame, instances_per_S: int) -> dict:
    """Check that a sweep is the registered one: every S in the grid, all bit widths, 5 noise seeds, n instances."""
    problems = []
    per_S = {float(S): int(df[df.S4 == S].instance.nunique()) for S in S_GRID}
    for S, n in per_S.items():
        if n != instances_per_S:
            problems.append(f"S={S}: {n} instances, expected {instances_per_S}")
    bits = set(int(b) for b in df.bits.unique())
    if bits != EXPECTED_BITS:
        problems.append(f"bit widths {sorted(bits)}, expected {sorted(EXPECTED_BITS)}")
    seeds = int(df[df.method == "noise"].noise_seed.nunique())
    if seeds != EXPECTED_NOISE_SEEDS:
        problems.append(f"{seeds} noise seeds, expected {EXPECTED_NOISE_SEEDS}")
    return {"instances_per_S": per_S, "bits": sorted(bits), "noise_seeds": seeds, "problems": problems}


def label_reproduction_mismatches(df: pd.DataFrame) -> int:
    """Clarification 6: the recomputed composite label must reproduce the stored one (all RTN 'none' rows)."""
    c = df[(df.method == "rtn") & (df.ablation == "none") & df.eligible & df.S4.isin(S_GRID)]
    reg, stored = composite_label(c), damaged(c)
    both = reg.notna() & stored.notna()
    return int((reg[both] != stored[both]).sum())


def precheck(df: pd.DataFrame, sanity: dict, instances_per_S: int | None = None) -> dict:
    reasons = []
    if not sanity.get("bits8_pass", False):
        reasons.append("bits8_pass is false or missing")
    mono = sanity.get("monotonicity", {})
    for g in GRANULARITIES:
        if g not in mono:
            reasons.append(f"monotonicity flag missing for per-{g} scaling")
        elif not mono[g].get("monotone_nondecreasing_as_bits_decrease", False):
            reasons.append(f"non-monotone damage across bits under per-{g} scaling")
    comp = completeness(df, instances_per_S) if instances_per_S else None
    mism = label_reproduction_mismatches(df)
    validity = [label_validity(df, S, g) for S in S_GRID for g in GRANULARITIES]
    vmap = {(v["S"], v["granularity"]): v["valid"] for v in validity}
    cells = [cell_rule(df, S, g, b, vmap[(S, g)]) for S in S_GRID for g in GRANULARITIES for b in SECONDARY_BITS]
    prim = rows(df, *PRIMARY_CELL)
    eligible_per_instance = {
        f"S={S}": rows(df, S, "row", 3).groupby("instance").size().astype(int).to_dict() for S in S_GRID
    }
    noise = rows(df, *PRIMARY_CELL, method="noise")
    icc = {y: icc1(noise, y)[0] for y in ("dfalse_act", "dR2")}
    rnd, _, _ = instance_rhos(prim, "norm", "D")  # same per-instance validity rule as every test
    return {
        "stop": bool(reasons),
        "stop_reasons": reasons,
        "bits8_pass": sanity.get("bits8_pass"),
        "monotone": {g: mono.get(g, {}).get("monotone_nondecreasing_as_bits_decrease") for g in GRANULARITIES},
        "completeness": comp,
        "label_reproduction_mismatches": mism,
        "eligible_per_instance_row3": eligible_per_instance,
        "label_validity": validity,
        "cells": cells,
        "icc_primary": icc,
        "sqrt_icc_primary": {k: sqrt_icc(v) for k, v in icc.items()},
        "rho_norm_D_primary": float(np.mean(list(rnd.values()))) if rnd else float("nan"),
        "rho_norm_D_n_instances": len(rnd),
    }


# ------------------------------------------------------------------------------------------------------
# Secondary analyses
# ------------------------------------------------------------------------------------------------------
def break_bits_table(df: pd.DataFrame, S: float, gran: str) -> pd.DataFrame:
    """Per eligible feature: break_bits, monotone envelope, non-monotone flag, damaged-at-3-bits label."""
    sub = df[(df.S4 == S) & (df.granularity == gran) & (df.method == "rtn") & (df.ablation == "none") & df.eligible]
    lab = sub.assign(dmg=damaged(sub)).pivot_table(index=["instance", "feature"], columns="bits", values="dmg")
    out = []
    for (inst, feat), r in lab.iterrows():
        dmg = {b: r.get(b, np.nan) for b in ALL_BITS}
        damaged_bits = [b for b in ALL_BITS if dmg[b] == 1]
        bb = max(damaged_bits) if damaged_bits else np.nan
        nonmono = any(dmg[b] == 1 and any(dmg[b2] == 0 for b2 in ALL_BITS if b2 < b) for b in ALL_BITS)
        env = np.nan
        for b in sorted(ALL_BITS, reverse=True):
            if dmg[b] == 1 and all(dmg[b2] == 1 for b2 in ALL_BITS if b2 < b):
                env = b
                break
        out.append({"instance": inst, "feature": feat, "break_bits": bb, "break_bits_env": env, "non_monotone": nonmono,
                    "damaged3": dmg[3]})  # fmt: skip
    t = pd.DataFrame(out)
    t["break_bits_rank"] = t.break_bits.fillna(1.0)  # survives all widths -> ranked below 2
    t["break_bits_env_rank"] = t.break_bits_env.fillna(1.0)
    fp = sub[sub.bits == 3][["instance", "feature", "D", "norm"]]
    return t.merge(fp, on=["instance", "feature"], how="left")


def secondary1(df: pd.DataFrame, min_valid: int, validity: dict, stop: bool = False) -> pd.DataFrame:
    """One family: per-instance Spearman (and partial) summaries, BH q, sqrt(ICC_noise), dMSE split, break_bits."""
    recs = []
    icc_cache: dict = {}
    for S, g, b in itertools.product(S_GRID, GRANULARITIES, SECONDARY_BITS):
        c = rows(df, S, g, b)
        act = float((c.p * c.dcond_mse).sum())
        ina = float(((1 - c.p) * c.dleak_energy).sum())
        noise = rows(df, S, g, b, method="noise")
        for x, y, sign, partial in SECONDARY_PAIRS:
            key = (S, g, b, y)
            if key not in icc_cache:
                icc_cache[key] = icc1(noise, y)[0]
            rh, inv, dropped = instance_rhos(c, x, y, partial=partial)
            s = summarize(rh, min_valid)
            recs.append({"S": S, "granularity": g, "bits": b, "predictor": x, "outcome": y, "partial_on": partial,
                         "predicted_sign": sign if sign is not None else np.nan, **s, "n_invalid": len(inv),
                         "invalid_instances": "; ".join(f"{k}: {v}" for k, v in inv.items()), "dropped_rows": dropped,
                         "sqrt_icc_noise": sqrt_icc(icc_cache[key]),
                         "dMSE_active_term": act, "dMSE_inactive_term": ina,
                         "inactive_share_dMSE": ina / (act + ina) if act + ina > 0 else np.nan,
                         "in_family": True, "statistic": "spearman" if partial is None else "partial_spearman"})  # fmt: skip
    for S, g in itertools.product(S_GRID, GRANULARITIES):
        if not validity[(S, g)]:
            continue
        t = break_bits_table(df, S, g)
        values = sorted(t.break_bits_rank.dropna().unique())
        n_distinct = len(values)
        for x in ("D", "norm"):
            if n_distinct <= 2:
                # clarification 1 + pre-freeze review: registered per-instance rule (>= 20 features, predictor >= 5
                # distinct, binary outcome >= 2 distinct); binary = "break_bits equals the larger of its two values"
                # (= "damaged at 3 bits" whenever the values are {2, 3}, as in every E cell)
                hi = values[-1] if values else np.nan
                tt = t.assign(binary=(t.break_bits_rank == hi).astype(float))
                per, inv = {}, {}
                for inst, d in tt.groupby("instance"):
                    d = d.dropna(subset=["binary", x])
                    if len(d) < MIN_N:
                        inv[int(inst)] = f"n={len(d)} < {MIN_N}"
                    elif d[x].nunique() < MIN_DISTINCT:
                        inv[int(inst)] = f"{x} has {d[x].nunique()} distinct values"
                    elif d.binary.nunique() < 2:
                        inv[int(inst)] = "binary break_bits has 1 distinct value"
                    else:
                        per[int(inst)] = 2 * auc_rank(d[x], d.binary) - 1
                s = summarize(per, min_valid)
                stat, fam = f"rank_biserial(break_bits=={hi:g})", False
            else:
                per, inv, _ = instance_rhos(t, x, "break_bits_rank", min_distinct_y=2)
                s = summarize(per, min_valid)
                stat, fam = "spearman_break_bits", True
            recs.append({"S": S, "granularity": g, "bits": np.nan, "predictor": x, "outcome": "break_bits",
                         "partial_on": None, "predicted_sign": -1, **s, "statistic": stat, "in_family": fam,
                         "n_invalid": len(inv), "invalid_instances": "; ".join(f"{k}: {v}" for k, v in inv.items()),
                         "n_distinct_break_bits": int(n_distinct), "break_bits_values": str([float(v) for v in values]),
                         "n_non_monotone": int(t.non_monotone.sum())})  # fmt: skip
            rh_env, inv_env, _ = instance_rhos(t, x, "break_bits_env_rank", min_distinct_y=2)
            se = summarize(rh_env, min_valid)
            recs.append({"S": S, "granularity": g, "bits": np.nan, "predictor": x, "outcome": "break_bits_envelope",
                         "partial_on": None, "predicted_sign": -1, **se, "statistic": "spearman_sensitivity",
                         "in_family": False, "n_invalid": len(inv_env)})  # fmt: skip
    out = pd.DataFrame(recs)
    fam = out.in_family & out.summarised
    out["q_bh"] = np.nan
    out.loc[fam, "q_bh"] = bh(out.loc[fam, "p"].to_numpy())
    out["significant_q05"] = out.q_bh < 0.05
    if stop:  # stop rule 5: tables are descriptive only
        out["q_bh"] = np.nan
        out["significant_q05"] = False
    out["descriptive_only"] = bool(stop)
    return out


def secondary3_auc(df: pd.DataFrame, cells: list[dict], min_valid: int) -> pd.DataFrame:
    """Per-instance oriented AUC (t-interval) and pooled AUC with a cluster bootstrap, cells passing the rule."""
    recs = []
    for cr in cells:
        if not cr["passes"]:
            continue
        c = rows(df, cr["S"], cr["granularity"], cr["bits"])
        c = c.assign(lab=damaged(c)).dropna(subset=["lab"])
        for pred, orient in AUC_PREDICTORS.items():
            d = c.dropna(subset=[pred])
            per = {}
            excluded = 0
            for inst, di in d.groupby("instance"):
                if (di.lab == 1).sum() >= AUC_MIN_CLASS and (di.lab == 0).sum() >= AUC_MIN_CLASS:
                    per[int(inst)] = auc_rank(orient * di[pred].to_numpy(), di.lab.to_numpy())
                else:
                    excluded += 1
            s = summarize(per, min_valid)
            pooled = pooled_auc_with_bootstrap(orient * d[pred].to_numpy(), d.lab.to_numpy(), d.instance.to_numpy())
            recs.append({"S": cr["S"], "granularity": cr["granularity"], "bits": cr["bits"], "predictor": pred,
                         "orientation": orient, "auc_mean": s["mean"], "auc_ci_lo": s["ci_lo"], "auc_ci_hi": s["ci_hi"],
                         "summarised": s["summarised"], "n_valid_instances": s["n_valid"],
                         "n_excluded_instances": excluded, **pooled, "n_instances_pooled": int(d.instance.nunique()),
                         "pooled_note": "pooled AUC mixes between-instance differences in label rate and predictor level",
                         "bootstrap_note": "cluster bootstrap over instances undercovers (~90% for nominal 95%)"})  # fmt: skip
    return pd.DataFrame(recs)


def vif(X: np.ndarray) -> np.ndarray:
    """Variance inflation factors of the columns of X (standardized internally)."""
    Z = (X - X.mean(0)) / X.std(0)
    out = []
    for j in range(Z.shape[1]):
        others = np.delete(Z, j, axis=1)
        A = np.column_stack([np.ones(len(Z)), others])
        resid = Z[:, j] - A @ np.linalg.lstsq(A, Z[:, j], rcond=None)[0]
        r2 = 1 - resid.var() / Z[:, j].var()
        out.append(1 / (1 - r2) if r2 < 1 else np.inf)
    return np.array(out)


def secondary4_logistic(df: pd.DataFrame, cells: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Unpenalised logistic regression, GroupKFold(5) by instance, pooled OOF AUC + cluster bootstrap."""
    recs, coefs = [], []
    for cr in cells:
        if not cr["passes"]:
            continue
        c = rows(df, cr["S"], cr["granularity"], cr["bits"])
        c = c.assign(lab=damaged(c)).dropna(subset=["lab"])
        for name, cols in LOGIT_SETS.items():
            d = c.dropna(subset=cols)
            X, y, g = d[cols].to_numpy(float), d.lab.to_numpy(int), d.instance.to_numpy()
            n_groups = len(np.unique(g))
            oof = np.full(len(y), np.nan)
            fold_aucs = []
            folds = []
            for tr, te in GroupKFold(n_splits=min(5, n_groups)).split(X, y, g):
                folds.append(sorted(int(v) for v in np.unique(g[te])))
                sc = StandardScaler().fit(X[tr])
                m = LogisticRegression(C=np.inf, max_iter=10000).fit(sc.transform(X[tr]), y[tr])
                oof[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
                if len(np.unique(y[te])) == 2:
                    fold_aucs.append(auc_rank(oof[te], y[te]))
            pooled = pooled_auc_with_bootstrap(oof, y, g)
            rec = {"S": cr["S"], "granularity": cr["granularity"], "bits": cr["bits"], "set": name,
                   "cv_auc_pooled_oof": pooled["auc_pooled"], "cv_auc_boot_lo": pooled["boot_lo"],
                   "cv_auc_boot_hi": pooled["boot_hi"], "cv_auc_mean_fold": float(np.mean(fold_aucs)) if fold_aucs else np.nan,
                   "n_folds": min(5, n_groups), "n": len(y), "test_folds_instances": str(folds),
                   "bootstrap_note": "cluster bootstrap over instances undercovers (~90% for nominal 95%)"}  # fmt: skip
            if len(cols) > 1:
                v = vif(X)
                rec["max_vif"] = float(np.max(v))
                sc = StandardScaler().fit(X)
                m = LogisticRegression(C=np.inf, max_iter=10000).fit(sc.transform(X), y)
                for col, coef, vv in zip(cols, m.coef_[0], v, strict=True):
                    coefs.append({"S": cr["S"], "granularity": cr["granularity"], "bits": cr["bits"], "set": name,
                                  "predictor": col, "std_coef": float(coef), "vif": float(vv),
                                  "interpretable": bool(np.max(v) <= 5)})  # fmt: skip
            recs.append(rec)
    return pd.DataFrame(recs), pd.DataFrame(coefs)


def secondary5_noise(df: pd.DataFrame, primary: dict, min_valid: int) -> pd.DataFrame:
    """Rounding structure vs generic noise for each primary pair in the primary cell (single-draw rho)."""
    recs = []
    for h in HYPOTHESES:
        res = primary["results"][h.name]
        rtn = res.rhos
        nz = rows(df, *PRIMARY_CELL, method="noise")
        per_seed = {}
        for s, d in nz.groupby("noise_seed"):
            rh, _, _ = instance_rhos(d, h.x, h.y)
            per_seed[int(s)] = rh
        noise_k = {}
        for k in rtn:
            vals = [per_seed[s][k] for s in per_seed if k in per_seed[s]]
            if vals:
                noise_k[k] = float(np.mean(vals))
        paired = [k for k in rtn if k in noise_k]
        r_rtn = np.array([rtn[k] for k in paired])
        r_noise = np.array([noise_k[k] for k in paired])
        rec = {"hypothesis": h.name, "n_paired": len(paired), "verdict_primary": res.verdict,
               "sqrt_icc_noise": res.sqrt_icc}  # fmt: skip
        if len(paired):
            sigma = np.sign(r_rtn.mean()) if r_rtn.mean() != 0 else 1.0
            R = float(r_noise.mean() / r_rtn.mean()) if r_rtn.mean() != 0 else np.nan
            delta = sigma * (r_rtn - r_noise)
            sd = summarize(delta, min_valid)
            rec.update({"mean_rho_rtn": float(r_rtn.mean()), "mean_rho_noise": float(r_noise.mean()), "R": R,
                        "delta_mean": sd["mean"], "delta_ci_lo": sd["ci_lo"], "delta_ci_hi": sd["ci_hi"], "delta_p": sd["p"]})  # fmt: skip
            applicable = res.verdict.startswith("Supported") or res.verdict.startswith("Mixed")
            if not applicable:
                reading = "not applicable (primary verdict is neither Supported nor Mixed)"
            else:
                sig = sd["mean"] > 0 and sd["p"] < ALPHA
                if R < 0.5 and sig:
                    reading = "specific to rounding"
                elif R >= 0.8 and not sig:
                    reading = "explained by perturbation size"
                else:
                    reading = "partly specific / unresolved"
            rec["reading"] = reading
        recs.append(rec)
    return pd.DataFrame(recs)


def _zrank(v: np.ndarray) -> np.ndarray:
    r = stats.rankdata(v)
    return (r - r.mean()) / r.std()


def secondary6_dn10(df: pd.DataFrame, min_valid: int) -> pd.DataFrame:
    """Design note 10, within instance, rank-standardized: (1) dR2 ~ margin_tau + D + norm; (2) + resid dW."""
    c = rows(df, *PRIMARY_CELL)
    per = []
    for inst, d in c.groupby("instance"):
        d = d.dropna(subset=["dR2", "margin_tau", "D", "norm", "dW", "peak_ratio"])
        if len(d) < MIN_N:
            continue
        # spec order: dW residualized on norm and peak_ratio by OLS, then ALL variables rank-transformed and
        # standardized within the instance (pre-freeze review)
        A = np.column_stack([np.ones(len(d)), d.norm.to_numpy(float), d.peak_ratio.to_numpy(float)])
        dw = d.dW.to_numpy(float)
        resid_raw = dw - A @ np.linalg.lstsq(A, dw, rcond=None)[0]
        z = {k: _zrank(d[k].to_numpy()) for k in ("dR2", "margin_tau", "D", "norm")}
        res = _zrank(resid_raw)
        X1 = np.column_stack([z["margin_tau"], z["D"], z["norm"]])
        X2 = np.column_stack([X1, res])
        b1 = np.linalg.lstsq(np.column_stack([np.ones(len(d)), X1]), z["dR2"], rcond=None)[0][1:]
        b2 = np.linalg.lstsq(np.column_stack([np.ones(len(d)), X2]), z["dR2"], rcond=None)[0][1:]
        v2 = vif(X2)
        per.append({"instance": int(inst), "b1_margin_tau": b1[0], "b1_D": b1[1], "b1_norm": b1[2],
                    "b2_margin_tau": b2[0], "b2_D": b2[1], "b2_norm": b2[2], "b2_resid_dW": b2[3],
                    "vif_margin_tau": v2[0], "vif_D": v2[1], "vif_norm": v2[2], "vif_resid_dW": v2[3]})  # fmt: skip
    p = pd.DataFrame(per)
    recs = []
    if len(p):
        for col in [c for c in p.columns if c.startswith("b")]:
            s = summarize(p[col].to_numpy(), min_valid)
            recs.append({"term": col, "mean": s["mean"], "ci_lo": s["ci_lo"], "ci_hi": s["ci_hi"], "n": s["n_valid"]})
        for k in ("margin_tau", "D", "norm"):
            s = summarize((p[f"b2_{k}"] - p[f"b1_{k}"]).to_numpy(), min_valid)
            recs.append(
                {"term": f"change_{k}", "mean": s["mean"], "ci_lo": s["ci_lo"], "ci_hi": s["ci_hi"], "n": s["n_valid"]}
            )
        for col in [c for c in p.columns if c.startswith("vif")]:
            recs.append({"term": col, "mean": float(p[col].mean()), "ci_lo": np.nan, "ci_hi": np.nan, "n": len(p)})
    return pd.DataFrame(recs)


def secondary7_sensitivity(df: pd.DataFrame, cells: list[dict], min_valid: int) -> pd.DataFrame:
    """Composite-label threshold grid in every cell that passes the cell rule at the registered thresholds
    (gated like the other label-based analyses): mean per-instance oriented AUC per predictor."""
    recs = []
    for cr in cells:
        if not cr["passes"]:
            continue
        c = rows(df, cr["S"], cr["granularity"], cr["bits"])
        for corr, dr2, dfa in itertools.product((0.8, 0.9, 0.95), (0.05, 0.1, 0.2), (0.025, 0.05, 0.1)):
            lab = composite_label(c, corr, dr2, dfa)
            d = c.assign(lab=lab).dropna(subset=["lab"])
            for pred, orient in AUC_PREDICTORS.items():
                per = {}
                for inst, di in d.dropna(subset=[pred]).groupby("instance"):
                    if (di.lab == 1).sum() >= AUC_MIN_CLASS and (di.lab == 0).sum() >= AUC_MIN_CLASS:
                        per[int(inst)] = auc_rank(orient * di[pred].to_numpy(), di.lab.to_numpy())
                s = summarize(per, min_valid)
                recs.append({"S": cr["S"], "granularity": cr["granularity"], "bits": cr["bits"], "corr": corr,
                             "dR2": dr2, "dfalse": dfa, "registered": (corr, dr2, dfa) == (0.9, 0.1, 0.05),
                             "predictor": pred, "auc_mean": s["mean"], "auc_ci_lo": s["ci_lo"], "auc_ci_hi": s["ci_hi"],
                             "summarised": s["summarised"], "n_valid_instances": s["n_valid"],
                             "damaged_frac": float(d.lab.mean())})  # fmt: skip
    return pd.DataFrame(recs)


def secondary8_descriptive(df: pd.DataFrame, setting: str) -> pd.DataFrame:
    """Settings A/B (clarification 2): mean per-instance rho, instances with >= 5 eligible and >= 3 distinct."""
    recs = []
    for S in sorted(df.S4.unique()):
        for b in SECONDARY_BITS:
            c = rows(df, S, "row", b)
            noise = rows(df, S, "row", b, method="noise")
            for h in HYPOTHESES:
                rh, inv, _ = instance_rhos(c, h.x, h.y, min_n=5, min_distinct_x=3, min_distinct_y=3)
                recs.append({"setting": setting, "S": S, "bits": b, "hypothesis": h.name, "predictor": h.x, "outcome": h.y,
                             "mean_rho": float(np.mean(list(rh.values()))) if rh else np.nan, "n_valid": len(rh),
                             "n_instances": int(c.instance.nunique()),
                             "sqrt_icc_noise": sqrt_icc(icc1(noise, h.y)[0])})  # fmt: skip
    return pd.DataFrame(recs)
