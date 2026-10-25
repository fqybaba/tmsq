# TMS-Q: Superposition x Quantization

A replication of key results from *Toy Models of Superposition* (Elhage et al., 2022), plus an original
extension: which features stored in superposition does post-training weight quantization (RTN) break first,
and can superposition geometry predict this in advance?

Status (2026-10-25):
- Phases 0–5 are complete, and so is the pre-registered bridge to a real model (Pythia-70M with a pretrained
  SAE).
- The write-up exists as a draft in `writeup/post.md`. Every number was audited against its source on
  2026-10-25; it awaits the owner's review before publishing.
- Duan was emailed on 2026-10-25 about the implementation note (shared Q/K/V scales); his reply is pending.

`notes/LOG.md` is the lab notebook, with every result, pre-registration and deviation. `CLAUDE.md` and
`notes/DESIGN_NOTES.md` hold the canonical definitions.

## Results in brief

- **Replication** (settings A–C):
  - The pentagon in 5→2: 4/5, 3/5 and 5/5 seeds at S = 0.9, 0.95 and 0.99.
  - Superposition grows with sparsity in 20→5 (Spearman 0.998).
  - The D* = 1/2 plateau in 100→20, on a fine sparsity grid.
  - Three pre-registered criteria failed as written: B's bias criterion, and C's plateaus at 1 and at 1/2 on
    the coarse grid. The 1/2 plateau passed on the registered fine-grid follow-up.
- **The fragile features are the zero-bias antipodal pairs, not the crowded ones** (setting C, S = 0.75).
  - Two of the three pre-registered predictors (smaller margin, smaller D) point the *opposite* way:
    ρ = +0.49 and +0.40 on 10 fresh models. Feature norm predicts nothing.
  - Post hoc, and then confirmed on 20 further fresh models: antipodal-pair features (D ≈ 0.5) learn a bias of
    about +0.01, so their output on inputs where they should be off sits just below the 0.05 cutoff for false
    activations.
  - At 3-bit per-row RTN, all of them are damaged, against 46% of the other features. The per-model
    difference is +0.53 [+0.48, +0.58]. At 4 bits, essentially only they break. Matched Gaussian noise gives
    the same pattern.
  - The binary label depends on that cutoff; the threshold-free outcomes agree in sign with smaller gaps.
  - To our knowledge, this result has not been reported before.
- **Head vs tail is a weak side result** (setting D, strong superposition).
  - The pre-registered test leans toward "rare features lose a larger *relative* fraction of their R²"
    (ρ = −0.21). The effect is borderline and not robust: the sensitivity analysis gives −0.17, and the effect
    shrinks at 4 and 5 bits.
  - It comes from the normalization. The change in R² itself does not depend on frequency, while absolute
    damage sits in the frequent features, and so, post hoc, do new false activations.
  - Frequency is nearly collinear with proximity to the 0.05 false-activation cutoff in these models
    (ρ = 0.93), so frequency and geometry cannot be separated.
  - The weak-superposition regime was stopped by a sanity rule: heavy weight decay makes 2-bit RTN *improve*
    the model.
- **Real model: Duan (2026) replicates almost exactly; post hoc, the toy predictors add almost nothing beyond
  his statistics.**
  Setup: Pythia-70M, the layer-4 residual SAE `pythia-70m-deduped-res-sm`, and RTN, as in Duan.
  - With his released code (mirror `sae-feature-survival-quantization-0141`, 2026-06-08), survival and
    perplexity reproduce his Table 1 to within 0.02 at INT8–INT4. Peak activation alone predicts survival with
    AUC 0.879.
  - Implementation note:
    - The code shares Q/K/V quantization scales across heads. With per-channel scales, 94.3% of features
      survive INT6 instead of 62.4%.
    - The described method gives 85.3%. Folding the LayerNorm gains in before per-channel quantization
      raises that to 94.6%.
    - The ablation is exploratory.
  - Post hoc, threshold density and a margin analog add at most 0.004 AUC for survival beyond Duan's four
    statistics, on both his pipeline and the described method. A small residual association of threshold
    density with new false activations remains (ρ ≈ 0.11), depending on the controls.
    - The margin passed its pre-registered test against peak activation alone (+0.024). Post hoc, Duan's
      statistics explain it; the registered baseline should have been his full model.
    - Threshold density's strong link to new false activations is feature frequency: given peak and rarity it
      falls to about 0.
- **Geometry-aware bit allocation does not beat uniform 4-bit RTN** at an equal budget in setting C.
  - Mixing 3- and 5-bit columns roughly triples the loss increase (2.9×). The pre-registration predicted it
    would be worse, but its 4^-b estimate (2.1×) undershot; the grid's actual steps give about 2.8×. No
    geometric ranking recovers the penalty.
  - In setting D a threshold-density ranking does beat uniform (α = 1.5 only); post hoc, it is frequency
    allocation in disguise.
  - Generic loss-aware rounding halves the loss increase at the primary cell of setting C (and lowers it by
    7–31% at higher sparsities), but does not rescue the fragile features.

## How this project was made

This is a collaboration between a person and AI assistants. The roles are as follows:

- **Shen Yuting**: designed the research question, the overall architecture and the phase plan, and made
  the approval and scope decisions while present, including the project's evidence-first rule. Claude
  (claude.ai) helped
  only to write that design up as the Markdown files `CLAUDE.md` and `PROMPTS.md` and with a few simple
  lookups.
- **Claude Code** (Anthropic): checked the plan against primary sources, implemented the package, ran the
  experiments and analyses, and ran multi-agent AI review workflows on its own code, plans and drafts (not
  human review). It works within the owner's approved plans. While the owner was away, under their standing
  instruction, it also made some scope decisions: it added the Phase 3b confirmation, wrote the
  pre-registrations for Phases 3b–5, and chose the setting-D weak-regime γ under a logged deviation rule.
  Every decision is logged for review.

`notes/SESSION_JOURNAL.md` records who decided what and when.

## Setup (Windows, PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
pytest            # 277 tests, ~45 s
```

## Reproducing everything

Every experiment is a Python entry point driven by a YAML config. Each run writes to
`results/<exp>/<timestamp>/` (config copy, git hash, weights, tables, per-run figures). Untagged runs also copy
their figures to `figures/`. Times are for a desktop CPU.

| Phase | Command | Main figures | Time |
|---|---|---|---|
| 1 A | `python -m tmsq.experiments.rep_5in2 --config configs/rep_5in2.yaml` | `rep_5in2_arrows`, `rep_5in2_loss` | 1 min |
| 1 B | `python -m tmsq.experiments.rep_20in5 --config configs/rep_20in5.yaml` | `rep_20in5_gram_bias`, `_norms`, `_count` | 2 min |
| 1 C | `python -m tmsq.experiments.rep_dims --config configs/rep_dims.yaml` | `rep_dims_dstar`, `rep_dims_feature_D` | 10 min |
| 1 C fine | `python -m tmsq.experiments.rep_dims --config configs/rep_dims_fine.yaml` | `rep_dims_fine_dstar` | 6 min |
| 2 | `python -m tmsq.experiments.quant_sweep --config configs/quant_sweep.yaml --setting A` (likewise `B`, `C`) | `quant_{A,B,C}_*`, `quant_A_arrows` | up to 3 min |
| 3 models | `python -m tmsq.experiments.rep_dims --config configs/rep_dims_confirm.yaml`, then `quant_sweep ... --setting C_confirm` | `rep_dims_confirm_*` | 8 min |
| 3 tests | `python -m tmsq.experiments.phase3 --config configs/phase3.yaml --set K` (also `--set E`, `--compare <E> <K>`) | `phase3_forest_E_vs_K`, `phase3_K_*` | 1 min |
| 3 post hoc | `python -m tmsq.experiments.phase3_posthoc --config configs/phase3.yaml` | (tables) | 1 min |
| 3b | `python -m tmsq.experiments.rep_dims --config configs/rep_dims_confirm2.yaml`; `quant_sweep ... --setting C_confirm2`; `python -m tmsq.experiments.phase3b --sweep-glob "results/quant_sweep/*-C_confirm2"` | — | 5 min |
| 3b figure | `python -m tmsq.experiments.figure_antipodal --sweep results/quant_sweep/<ts>-C_confirm2` | `phase3b_antipodal` | 10 s |
| 4 models | `python -m tmsq.experiments.train_setting_d --config configs/setting_d.yaml --gamma=7.0` (and `--gamma=-1.0`) | `setting_d_g*_norm_vs_rank` | 6 min each |
| 4 sweep | `python -m tmsq.experiments.quant_sweep --config configs/quant_sweep_d.yaml --setting D_weak` (and `D_strong`) | `quant_D_*` | 5 min each |
| 4 tests | `python -m tmsq.experiments.phase4 --config configs/phase4.yaml`; table: `python -m tmsq.experiments.phase4_table --config configs/phase4.yaml` | `phase4_head_tail`, `phase4_reldamage_vs_rank` | 1 min |
| 5 models | `python -m tmsq.experiments.rep_dims --config configs/rep_dims_p5.yaml` | `rep_dims_p5_*` | 5 min |
| 5 arms | `python -m tmsq.experiments.phase5 --config configs/phase5.yaml --setting C_P5` (and `D_strong`) | — | 5 min each |
| 5 tests | `python -m tmsq.experiments.phase5_analysis --config configs/phase5.yaml` | `phase5_arms_vs_u4` | 1 min |
| 6b setup | separate venv: `python -m venv --system-site-packages .venv-bridge`, then install `requirements-bridge.txt` (sae-lens 6.54.12); `.venv-bridge\Scripts\python -m tmsq.experiments.bridge_fetch` (309 MB, manifest) | — | 2 min |
| 6b FP | `.venv-bridge\Scripts\python -m tmsq.experiments.bridge fp --config configs/bridge.yaml` | — | 3 min |
| 6b runs | `... bridge quant --config configs/bridge.yaml --arm code --bits 6` (R1); `--bits 8 7 5 4` (R2); `--arm stated --bits 8 7 6 5 4` (R3); `--bits 6 --noise-seed 0` (R4) | — | 2–6 min each |
| 6b tests | `... bridge analyze --config configs/bridge.yaml --primary <R1> --extra <R2> <R3> <R4>` | — | 18 min |
| 6b ablation | `... bridge_ablation --config configs/bridge.yaml --group r1` (also `c1`, `c2`, `r2`) | — | 5 min each |
| 6b figures | `python -m tmsq.experiments.figure_bridge` (main venv; reads R1–R3 and the ablation runs) | `bridge_replication`, `bridge_ablation` | 10 s |

Configs that consume earlier runs (`quant_sweep*.yaml`, `phase4.yaml`, `phase5.yaml`) name the exact result
directories used in the LOG. On a fresh checkout, point them at your own runs. Post-hoc analyses are scripts in
`notes/posthoc/`. Figures can be regenerated from any finished run without retraining:
`python -m tmsq.experiments.replot --run results/<exp>/<timestamp>`.

**Pre-registration.** Every hypothesis test has a pre-registration in `notes/LOG.md`, committed before the
confirmatory data were read. Each has a frozen analysis module (`tmsq.phase3`, `tmsq.phase4`, `tmsq.phase5`)
and an end-to-end smoke test on synthetic data. Seed independence across all model sets, eval sets, noise
directions and calibration sets is tested in `tests/test_seed_independence.py`.

**Bit-exact reruns** need the same `threads` value as the run's `meta.json` records. Different thread counts
give losses equal to about 8 decimals, but weights that are not bit-identical.

## Layout

```
src/tmsq/{data,model,train,quantize,metrics,analysis,plots,plots_quant}.py   core library
src/tmsq/{phase3,phase4,phase5,bridge}.py                                     frozen analysis / method code
src/tmsq/experiments/                                                         entry points
configs/   tests/   results/ (gitignored)   figures/   writeup/   notes/{LOG,DESIGN_NOTES,SESSION_JOURNAL}.md
```
