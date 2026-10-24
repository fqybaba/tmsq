"""End-to-end smoke test of the Phase 4 entry point on synthetic sweep directories (all columns, 6 bit widths,
5 noise seeds, a p = 1 feature), so the frozen analysis never meets its first real input on the confirmatory run."""

import json
import sys

import numpy as np
import pandas as pd
import yaml

from tmsq import phase4 as P4
from tmsq.experiments import common
from tmsq.experiments import phase4 as E4


def _sweep(tmp, name, n_inst=20, n_feat=14, phi_low=(), seed=0):
    rng = np.random.default_rng(seed)
    train = tmp / f"train_{name}"
    train.mkdir()
    pd.DataFrame(
        {"instance": range(n_inst), "phi_half": [0.6 if i in phi_low else 1.0 for i in range(n_inst)]}
    ).to_parquet(train / "instances.parquet")
    recs = []
    p = np.minimum(1.0, 1.2 * (np.arange(n_feat) + 1.0) ** -1.0)  # p_0 = 1 (clipped)
    for inst in range(n_inst):
        geo = {"norm": rng.uniform(0.6, 1.2, n_feat), "D": rng.uniform(0.1, 1, n_feat), "bias": rng.normal(-0.1, 0.1, n_feat),
               "near_thr": rng.uniform(0, 1, n_feat), "R2_fp": rng.uniform(0.5, 1, n_feat)}  # fmt: skip
        for b in (8, 6, 5, 4, 3, 2):
            for g in ("tensor", "column", "row"):
                for method, ns in [("rtn", 0)] + [("noise", s) for s in range(5)]:
                    scale = 2.0 ** (-b)
                    for f in range(n_feat):
                        has_inact = p[f] < 1
                        dr2 = abs(rng.normal(scale, scale))
                        recs.append({"setting": name, "instance": inst, "feature": f, "bits": b, "granularity": g,
                                     "method": method, "noise_seed": ns, "ablation": "none", "eligible": f < n_feat - 3,
                                     "represented": True, "p": p[f], "freq_rank": f, "alpha": 1.0, "k": 2.0,
                                     "gamma": 7.0, "norm": geo["norm"][f], "D": geo["D"][f], "D_plus": geo["D"][f],
                                     "bias": geo["bias"][f], "near_thr": geo["near_thr"][f] if has_inact else np.nan,
                                     "margin_tau": -0.05, "R2_fp": geo["R2_fp"][f], "dR2": dr2,
                                     "rel_damage": dr2 / geo["R2_fp"][f], "dLoss": dr2 * p[f], "mse_fp": 0.01,
                                     "dcond_mse": dr2 * 0.1, "dleak_energy": dr2 * 0.01 if has_inact else np.nan,
                                     "dfalse_act": 0.0 if has_inact else np.nan,
                                     "false_act_fp": 0.01 if has_inact else np.nan,
                                     "var_x": p[f] / 3 - p[f] ** 2 / 4})  # fmt: skip
    sweep = tmp / f"sweep_{name}"
    sweep.mkdir()
    pd.DataFrame(recs).to_parquet(sweep / "features.parquet")
    mono = {g: {"monotone_nondecreasing_as_bits_decrease": True} for g in ("tensor", "column", "row")}
    (sweep / "sanity.json").write_text(json.dumps({"bits8_pass": True, "monotonicity": mono}))
    (sweep / "meta.json").write_text(json.dumps({"source_run": str(train)}))
    return sweep


def test_phase4_entry_point_end_to_end(tmp_path, monkeypatch, capsys):
    sw, ss = _sweep(tmp_path, "weak"), _sweep(tmp_path, "strong", phi_low=(5, 6, 7, 8, 9), seed=1)
    cfg = {"sweeps": {"weak": str(sw), "strong": str(ss)}, "instances_per_regime": 20}
    cfg_path = tmp_path / "phase4.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(common, "RESULTS_ROOT", tmp_path / "results")
    monkeypatch.setattr(common, "FIGURES_ROOT", tmp_path / "figures")
    monkeypatch.setattr(sys, "argv", ["phase4", "--config", str(cfg_path), "--tag", "e2e"])
    E4.main()
    run = next((tmp_path / "results" / "phase4").iterdir())
    prim = json.loads((run / "primary.json").read_text())
    assert {"weak", "strong", "strong_sensitivity_phi_ge_0.9", "phase3_mechanism_prediction"} <= set(prim)
    assert prim["strong_sensitivity_phi_ge_0.9"]["n_instances_kept"] == 15
    for r in P4.REGIMES:
        assert prim[r]["n_valid"] == 20 and prim[r]["verdict"]
    sec = pd.read_csv(run / "secondary_spearman.csv")
    assert {"dcond_mse", "dR2_active_term", "dR2_inactive_term"} <= set(sec.y)
    assert (run / "head_tail.csv").exists() and (run / "bucket_geometry.csv").exists()
    assert (run / "figures" / "phase4_reldamage_vs_rank.png").exists() and (
        run / "figures" / "phase4_head_tail.pdf"
    ).exists()
    assert not (tmp_path / "figures").exists()  # tagged run: not published
    assert "Phase 3 mechanism prediction" in capsys.readouterr().out
