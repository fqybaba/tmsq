# TMS-Q: Superposition x Quantization

A replication of key results from *Toy Models of Superposition* (Elhage et al., 2022), plus an original
extension: which features stored in superposition does post-training weight quantization (RTN) break first,
and can superposition geometry predict this in advance?

Status (2026-10-24):
- Phases 0–5 are complete, and so is the pre-registered bridge to a real model (Pythia-70M with a pretrained
  SAE).
- The write-up exists as draft v3 in `writeup/post.md`, awaiting the owner's review before publishing.
`notes/LOG.md` is the lab notebook, with every result, pre-registration and deviation. `CLAUDE.md` and
`notes/DESIGN_NOTES.md` hold the canonical definitions.

## Results in brief

- **Replication** (settings A–C):
  - The pentagon in 5→2: 4/5, 3/5 and 5/5 seeds at S = 0.9, 0.95 and 0.99.
  - Superposition grows with sparsity in 20→5 (Spearman 0.998).
  - The D* = 1/2 plateau in 100→20, on a fine sparsity grid.
  - Two pre-registered criteria failed as written; they are reported as such in the LOG.
- **The fragile features are the zero-bias antipodal pairs, not the crowded ones** (setting C, S = 0.75).
  - The pre-registered predictors (smaller margin, smaller D) point the *opposite* way: ρ = +0.49 and +0.40
    on 10 fresh models.
  - Post hoc, and then confirmed on 20 further fresh models: antipodal-pair features (D ≈ 0.5) learn a bias of
    about 0, so their inactive pre-activation sits at the threshold.
  - At 3-bit per-row RTN, all of them are damaged, against 46% of the other features. The per-model
    difference is +0.53 [+0.48, +0.58]. At 4 bits, essentially only they break. Matched Gaussian noise gives
    the same pattern.
  - To our knowledge, this mechanism has not been reported before.
- **Head vs tail is a weak side result** (setting D, strong superposition).
  - The pre-registered test leans toward "rare features lose a larger *relative* fraction of their R²"
    (ρ = −0.21). The effect is borderline and not robust: the sensitivity analysis gives −0.17, and the effect
    shrinks at 4 and 5 bits.
  - It comes from the normalization. The change in R² itself does not depend on frequency, while absolute
    damage and new false activations sit in the frequent features.
  - Frequency is nearly collinear with threshold proximity in these models (ρ = 0.94), so frequency and
    geometry cannot be separated.
  - The weak-superposition regime was stopped by a sanity rule: heavy weight decay makes 2-bit RTN *improve*
    the model.
- **Bridge to a real model: Duan's results replicate exactly, but the toy mechanism does not transfer.**
  Setup: Pythia-70M, the layer-4 residual SAE `pythia-70m-deduped-res-sm`, and per-channel RTN, as in Duan
  (2026).
  - Survival and perplexity reproduce Duan's Table 1 to within 0.02 at INT8–INT4. Peak activation alone
    predicts survival with AUC 0.879.
  - Threshold density adds nothing to peak activation for survival (dAUC +0.008, registered minimum 0.02).
    Its strong link to new false activations is entirely feature frequency: partial ρ 0.79 → −0.01 with
    rarity controlled.
  - A margin analog clears its registered bar beyond peak (dAUC +0.024, larger margin = lower survival).
    Post hoc, that information is Duan's own activation statistics: it adds 0.001–0.002 beyond his four
    predictors.
  - Duan's released code quantizes TransformerLens-processed weights with Q/K/V scales shared across heads.
    The paper's stated per-channel method damages far less (INT6: 85% vs 62% survive). This is descriptive,
    with three confounded differences.
- **Geometry-aware bit allocation does not beat uniform 4-bit RTN** at an equal budget.
  - Mixing 3- and 5-bit columns costs about 2× the loss increase (1.9× measured, 2.1× predicted), and no
    geometric ranking recovers that.
  - Generic loss-aware rounding halves the loss increase in setting C but does not rescue the fragile
    features.

## How this project was made

This is a collaboration between a person and AI assistants. The roles are as follows:

- **Shen Yuting**: designed the research question, the overall architecture and the phase plan, and made
  every approval and scope decision, including the project's evidence-first rule. Claude (claude.ai) helped
  only to write that design up as the Markdown files `CLAUDE.md` and `PROMPTS.md` and with a few simple
  lookups.
- **Claude Code** (Anthropic): checked the plan against primary sources, implemented the package, ran the
  experiments and analyses, and ran independent review workflows on its own code and plans. It works within
  the owner's approved plans. Under the owner's standing instruction it also wrote the pre-registrations for
  Phases 3b–5 while the owner was away, logging every decision for review.

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
