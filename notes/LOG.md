# TMS-Q research log

Entry template (one per phase or experiment):

```
## <date> — <phase / experiment>
Goal:
Config: (path to results/<exp>/<timestamp>/config.yaml, git hash)
Results: (numbers, one line per acceptance criterion or hypothesis)
Surprises / deviations:
Next steps:
```

---

## 2026-10-24 — Phase 0 + Phase 1 plan (pre-registered before any training)

Goal: build the package, reproduce the three Toy-Models-of-Superposition results (settings A, B, C), and check the
acceptance criteria in PROMPTS.md without tuning anything after seeing results.

Plan:
1. Package `tmsq` (src layout): `data.py` (sparse generator, per-feature `S_i`, seeded `torch.Generator`,
   output `[K, B, n]`), `model.py` (`W [K, m, n]`, `b [K, n]`, tied forward `ReLU(W^T W x + b)`),
   `metrics.py` (geometry: `norm_i`, `D_i`, `feats_per_dim`, `max_overlap_i`, `margin_i`, superposition
   score), `train.py` (AdamW, constant lr, per-instance loss summed over instances, full loss history,
   weight snapshots), `plots.py`, three experiment entry points, YAML configs, pytest tests.
2. Decisions fixed before training (see "Decisions" below).
3. Review pass on the code against CLAUDE.md definitions (independent test author + adversarial checker),
   then run A, B, C and evaluate every acceptance criterion with the tolerances written here.

Decisions (fixed before any training run):
- Init: per-instance Xavier normal, `std = sqrt(2 / (n + m))` for each `[m, n]` matrix, drawn from a
  per-instance generator seeded by `(base_seed, instance index)`. The official notebook calls
  `nn.init.xavier_normal_` on the whole `[K, n, m]` tensor, whose fan computation makes the std depend on K
  (checked in Phase 0, see below); we treat that as a notebook quirk and do not reproduce it.
- Optimizer: AdamW, `weight_decay = 0.01` on `W` and `b` (notebook passes all parameters), constant lr 1e-3,
  batch 1024.
- Seeds: 5 instances per S value inside one batched run, each with its own init generator and independent data
  draws; stored as `seed_idx` 0..4.
- Steps: A and B 20k (cheap; CLAUDE.md asks for more than 10k at S >= 0.99); C decided from a timing smoke
  test to fit in ~10 minutes, recorded in the C entry below. Weight snapshot at half the steps for C to show
  whether the geometry is still moving.
- Acceptance tolerances (numeric, fixed now):
  - A, S = 0: exactly 2 features with `norm > 0.5`, `|cos angle| < 0.1` between them, the other 3 with
    `norm < 0.1`.
  - A, high S: a seed is a "pentagon" iff all five `norm_i > 0.5` and all five `D_i` in `[0.35, 0.45]`
    (CLAUDE.md design note 9). Report the fraction per S; this is a report, not a gate.
  - B: Spearman correlation between S and the number of represented features > 0.9 across the 12 S values
    (mean over seeds); "superposition appears" at an S if any represented feature has `D_i < 0.9`; there the
    mean bias of represented features must be < 0.
  - C: "sticky near 1/2" iff for at least two adjacent S values at least 3/5 seeds have `|D* - 0.5| < 0.05`;
    "sticky near 1" iff the same holds for `|D* - 1| < 0.05` at the dense end.
- Loss normalization reported in plots: ours (mean over batch of the importance-weighted SUM over features).
- Git: the repository has no commits yet (committing was not requested); results record "no commit".

Phase 0 measurements (2026-10-24):
- Notebook-style `xavier_normal_` on the whole `[K, n, m]` tensor, n=5, m=2: std 0.355 (K=1), 0.264 (K=10),
  0.162 (K=30), 0.110 (K=80). Per-instance Xavier for the `[2, 5]` matrix: 0.535. We use the per-instance
  value (`init_std: null`); `init_std` can override it for a control run.
- Sampler: one uniform draw per entry (`active = u < p`, value `u / p`), instances split into 4 chunks with
  their own generators drawn in threads; output verified identical across repeated runs with the same seed.
- Timing for setting C (K = 80, n = 100, m = 20, batch 1024, 24 threads): ~39 ms/step, of which ~13 ms is
  sampling and ~20 ms forward/backward/AdamW. 20k steps would take ~13 min; to respect the ~10 min budget the
  C run uses 15,000 steps with a weight snapshot at 7,500. If the plateau ratio or the snapshot shows the
  sparse end still moving, a longer run is proposed rather than silently extended.
- Tests: 21 unit tests (data sparsity rate, shapes, determinism, loss definition, every CLAUDE.md sanity
  value, the running example with its negative margin) plus 38 black-box tests written by an independent
  reviewer from the CLAUDE.md specification alone (`tests/test_blackbox.py`); all 59 pass.

---

## 2026-10-24 — Phase 1, setting A (rep_5in2)

Goal: visual sanity check, n=5, m=2, I_i = 0.9^i, S in {0, 0.7, 0.8, 0.9, 0.95, 0.99}, 5 seeds, 20k steps.
Config: `results/rep_5in2/20261024-062922/config.yaml` (git: no commit). Train time 63.5 s. Figures:
`figures/rep_5in2_arrows.*`, `figures/rep_5in2_loss.*`.

Results (acceptance tolerances as pre-registered above):
- S = 0: 5/5 seeds PASS (exactly 2 represented features, |cos| < 0.1, other three norms < 0.1).
- Pentagon fraction (all norms > 0.5 and all D_i in [0.35, 0.45]): S=0.9: 0.80, S=0.95: 0.60, S=0.99: 1.00.
  Non-pentagon seeds at S=0.9/0.95 have 4 represented features (two antipodal pairs / a square): the
  5->2 local minimum CLAUDE.md warns about, present in 1 of 5 and 2 of 5 seeds respectively.
- Represented counts per S (5 seeds): S=0: 2,2,2,2,2; S=0.7: 4,3,4,4,4; S=0.8: 4,4,4,4,4; S=0.9: 5,5,5,4,5;
  S=0.95: 5,4,5,5,4; S=0.99: 5,5,5,5,5.
- Plateau ratio (last 10% / previous 10% of steps) within 0.997–1.027 for every instance: converged.

- Angle view (added after the review; `acceptance_corrected.json`): sorted gaps between adjacent represented
  columns, in degrees. S = 0: two features at 90°. S = 0.7 and 0.8: four features at 90° ± 0.4° (two antipodal
  pairs = a square) in 9 of 10 seeds, one seed at S = 0.7 with three features (180°, 90°, 90°). S = 0.9:
  pentagons with gaps 70.1°–74.1° in 4 seeds, a square in 1; S = 0.95: pentagons (70.4°–73.6°) in 3 seeds,
  squares in 2; S = 0.99: pentagons in all 5 seeds, every gap within 71.0°–73.7°.

Surprises / deviations: none. The paper predicts deformed polytopes for non-uniform importance; with
I_i = 0.9^i the deformation is small (gaps within 72° ± 2.1° at S = 0.99), well inside the D_i tolerance.
The alternative solution at S = 0.9–0.95 is always the square (two antipodal pairs, the least important
feature dropped), not a random local minimum.

---

## 2026-10-24 — Phase 1, setting B (rep_20in5)

Goal: the paper's main demonstration, n=20, m=5, I_i = 0.7^i, 1-S log-spaced from 1 to 0.001 (12 values),
5 seeds, 20k steps. Config: `results/rep_20in5/20261024-063030/config.yaml` (git: no commit). Train time
120 s. Figures: `figures/rep_20in5_gram_bias.*`, `rep_20in5_norms.*`, `rep_20in5_count.*`, `rep_20in5_loss.*`.

Results:
- Spearman(S, mean number of represented features) = 0.998 → PASS (criterion > 0.9).
- "Mean bias of represented features < 0 wherever superposition is present" → FAIL as pre-registered
  under both readings of the criterion: per instance (what the script evaluates) 35 of 54 instances with
  superposition have negative mean bias; per S (the LOG wording) the seed-mean bias at S = 0.466–0.919 is
  positive. Per-S means over seeds (represented features only; n_rep = mean count; min_D = smallest
  D_i among represented features):

  | S | 0 | 0.466 | 0.715 | 0.848 | 0.919 | 0.957 | 0.977 | 0.988 | 0.993 | 0.996 | 0.998 | 0.999 |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|
  | n_rep | 5.0 | 5.6 | 8.0 | 9.2 | 9.4 | 10.0 | 10.6 | 10.6 | 11.6 | 12.4 | 13.2 | 13.8 |
  | mean bias | -0.000 | +0.033 | +0.017 | +0.005 | +0.002 | -0.002 | -0.062 | -0.059 | -0.131 | -0.148 | -0.156 | -0.165 |
  | mean norm | 1.000 | 0.984 | 0.990 | 0.991 | 0.985 | 0.996 | 1.026 | 1.029 | 1.061 | 1.057 | 1.044 | 1.037 |
  | min D_i | 1.00 | 0.41 | 0.40 | 0.29 | 0.30 | 0.46 | 0.30 | 0.34 | 0.09 | 0.09 | 0.08 | 0.08 |

  Reading: the detector "any represented feature with D_i < 0.9" fires from S = 0.466 on (antipodal pairs
  appear early, D_i ≈ 0.4–0.5), but in that regime the represented features have norms slightly below 1 and
  a small POSITIVE bias (the ReLU output of a sub-unit-norm feature is lifted toward the target), and the
  unrepresented features carry bias ≈ E[x_i] = (1-S)/2 (0.5 at S = 0, 0.27 at S = 0.466, 0.14 at S = 0.715;
  visible in the bias panels). Negative biases, the interference-suppression mechanism the paper describes,
  appear only in the dense-superposition regime: every instance with S >= 0.957 has mean bias < 0 (30/30),
  and the biases grow more negative with S. The paper's qualitative statement is reproduced; the
  pre-registered operationalisation was wrong about WHERE superposition "appears". Post-hoc, NOT
  pre-registered, reported for information only: Spearman(S, mean bias) = -0.92.
- Geometry in `rep_20in5_gram_bias`: S = 0 five orthogonal features; S = 0.466–0.957 antipodal pairs (off-
  diagonal entries ≈ -1) added one at a time, less important features first; S >= 0.977 dense superposition
  with many |W_i · W_j| in 0.2–0.6.
- Plateau ratio median per S within 0.995–1.011: converged.

Surprises / deviations: the bias criterion failure above (criterion design, not model behaviour). Represented
norms exceed 1 slightly (1.03–1.06) at high S: with a negative bias the slope ||W_i||^2 must exceed 1 to
reconstruct x_i, so this is expected, not a weight-decay artefact.

Note on re-stating the criterion: any replacement formulated now is formulated AFTER seeing these numbers
and cannot count as a pre-registered test of this run ("n_represented > m" would also fail, since
instances with 6–9 represented features at S = 0.466–0.848 still have positive mean bias). For the
write-up this criterion is reported as failed-as-written with the table above; if a bias criterion is
wanted for future runs, it must be fixed before those runs (candidate: Spearman(S, mean bias) < -0.9, a
trend statement rather than a threshold on "where superposition appears").

---

## 2026-10-24 — Phase 1, setting C (rep_dims)

Goal: the paper's "dimensions per feature" curve on the scaled-down configuration n=100, m=20, I_i = 1,
1-S log-spaced from 1 to 0.001 (16 values), 5 seeds, 15k steps (budget decision above), snapshot at 7,500.
Config: `results/rep_dims/20261024-063236/config.yaml` (git: no commit). Train time 593 s.
Figures: `figures/rep_dims_dstar.*`, `rep_dims_feature_D.*`, `rep_dims_loss.*`.

Results (D* = m / ||W||_F^2, 5 seeds agree to 3 decimals at every S):

  | 1/(1-S) | 1 | 1.58 | 2.51 | 3.98 | 6.31 | 10 | 15.8 | 25.1 | 39.8 | 63.1 | 100 | 158 | 251 | 398 | 631 | 1000 |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
  | D* | 0.996 | 0.80 | 0.51 | 0.42 | 0.29 | 0.21 | 0.17 | 0.155 | 0.149 | 0.147 | 0.148 | 0.149 | 0.151 | 0.154 | 0.156 | 0.157 |

- "Sticky near 1" → FAIL as pre-registered (corrected after the code review, see the review entry below:
  the script's gate contained an extra clause `or (0 in one)` that passed on the single densest point; the
  LOG text requires an adjacent pair at the dense end). Only S = 0 is within 0.05 of 1 (D* = 0.996, all
  seeds); the next grid point (S = 0.369) is at 0.80. The S = 0 value itself matches the paper (with uniform
  importance the dense model spreads 20 dimensions over all 100 features with partial norms, the paper's
  remark that the feature basis is not privileged at S = 0; D_i then scatters between 0.1 and 0.75), but
  with a factor-1.58 grid no second point can sit near 1, so the criterion as written cannot pass on this
  grid. `acceptance.json` keeps the originally coded flag; `acceptance_corrected.json` records both.
- "Sticky near 1/2 for at least two ADJACENT grid points" → FAIL as pre-registered: only one grid point
  (1/(1-S) = 2.51, S = 0.602) has D* within 0.05 of 1/2 (all five seeds at 0.509–0.511). The per-feature
  plot shows this point is a clean antipodal-pair solution (every represented D_i = 0.500, 40 features in
  20 dimensions, the rest dead); the next grid point (3.98) is already a mixture of D_i = 0.5 pairs and a
  cluster at 0.40–0.45 (pentagon-like), then 0.25–0.30, then from 1/(1-S) = 10 onward every feature has
  D_i = 0.20 = m/n: all 100 features are represented and the curve saturates at D* ≈ 0.15 (norms ≈ 1.15).
  Diagnosis: (i) the log grid has a factor 1.58 between points, so a plateau narrower than that cannot show
  two adjacent hits; (ii) with n/m = 5 the sparse end saturates at m/n long before the paper's n/m = 13
  would. Neither is a training problem: plateau ratios 0.993–1.000 and |D*(15k) - D*(7.5k)| <= 0.004 at
  every S.
- Follow-up registered before running (configs/rep_dims_fine.yaml): same model and training, 10 densities
  log-spaced from 0.556 to 0.143 (1/(1-S) from 1.8 to 7), same criterion (two adjacent grid points with
  >= 3/5 seeds within +-0.05 of 1/2). Reported in the next entry; the coarse-grid failure above stands as
  written.

Surprises / deviations: the saturation at D* ≈ 0.15 for S >= 0.96 (all 100 features represented) means
setting C has no "unrepresented tail" at high sparsity; Phase 2 should keep this in mind when choosing S
values (the interesting geometric variety lives at S between 0.37 and 0.95 for this n/m).

---

## 2026-10-24 — Phase 1, setting C fine grid (rep_dims_fine; registered follow-up)

Goal: resolve the 1/2 plateau that the coarse grid sampled with one point. Same model and training, 10
densities log-spaced from 0.556 to 0.143, 5 seeds, 15k steps, snapshot at 7,500; criterion unchanged
(two adjacent grid points with >= 3/5 seeds within +-0.05 of 1/2). Config:
`results/rep_dims_fine/20261024-064438/config.yaml`. Train time 352 s. Figures: `figures/rep_dims_fine_*`.

Results:

  | S | 0.444 | 0.522 | 0.589 | 0.646 | 0.696 | 0.739 | 0.775 | 0.807 | 0.834 | 0.857 |
  |---|---|---|---|---|---|---|---|---|---|---|
  | 1/(1-S) | 1.80 | 2.09 | 2.43 | 2.83 | 3.29 | 3.83 | 4.45 | 5.18 | 6.02 | 7.00 |
  | D* (5 seeds) | 0.60–0.61 | 0.533–0.535 | 0.513–0.514 | 0.504–0.506 | 0.469–0.501 | 0.441–0.471 | 0.376–0.388 | 0.334–0.340 | 0.299–0.304 | 0.267–0.268 |

- "Sticky near 1/2" → PASS: four adjacent grid points (S = 0.522, 0.589, 0.646, 0.696; 1/(1-S) from 2.1
  to 3.3) have >= 3/5 seeds within +-0.05 of 1/2, with D* creeping from 0.534 down to 0.50 as the
  antipodal-pair packing fills in. Plateau ratios 0.999–1.000.
- The coarse-grid FAIL above therefore was a resolution artefact: the plateau spans roughly a factor 1.6 in
  1/(1-S), the same as the coarse grid's spacing.

Surprises / deviations: none beyond the resolution point. ("Sticky near 1" is not applicable on this grid,
which does not include S = 0.)

---

## 2026-10-24 — Phase 0/1 code review (independent, after the runs)

Three independent reviewers (definition checker, black-box test author, reproducibility reviewer) plus a
refutation pass on every reported bug. Outcome:
- 1 bug confirmed: the setting C "sticky near 1" gate was looser than the pre-registered text (extra
  `or (0 in one)` clause). Fixed in `rep_dims.py`; the coarse-grid C entry above was re-stated as FAIL as
  pre-registered; `acceptance_corrected.json` written next to the original.
- 38 black-box tests written from the CLAUDE.md specification alone passed without modification; adopted as
  `tests/test_blackbox.py`.
- Minor items fixed: weight files now load through `load_weights` (torch 2.13 defaults to
  `weights_only=True`, which rejects the numpy metadata); figures are written per run under
  `results/<exp>/<run>/figures/` and copied to `figures/` only for untagged full runs (smoke runs no longer
  overwrite them); the W^T W heatmap no longer clips values above 1 (trained |W_i · W_j| reach 1.2–1.5);
  `threads` can be set in the config and bit-exact reruns need the same value as `meta.json` records
  (verified: 24 threads reproduce bit-for-bit, 1 and 4 threads give a different but mutually identical
  result, final loss equal to 8 decimals); `--steps 0` is rejected; meta.json records the loss
  normalization; acceptance JSON keys and float rounding made consistent; axis labels added to the
  arrows and heatmap panels; sorted angular gaps between represented features recorded for setting A.
- Documented, not changed: the single-draw sampler's active values lie on a lattice of p · 2^24 levels
  (1.7e4 at p = 0.001; harmless for MSE); an instance's data stream depends on (seed, K, n_chunks) while its
  init depends only on (seed, k); `is_pentagon` is a looser proxy for the 72° ± 10° angle rule (CLAUDE.md
  design note 9 reworded).

---

## 2026-10-24 — Phase 2 plan (quantization sweep; data only, no hypothesis interpretation)

Goal: quantize the Phase 1 models and produce ONE tidy per-feature table with every geometry predictor,
every damage metric and the realized-perturbation covariates, plus the sanity figures. H1–H4 are not
tested here; the Phase 3 pre-registration is written before any per-feature analysis.

Models reused (from results/): A `rep_5in2/20261024-062922` (all 6 S, 30 instances), B
`rep_20in5/20261024-063030` (all 12 S, 60 instances), C `rep_dims/20261024-063236` restricted to
S ∈ {0, 0.602, 0.749, 0.842, 0.900, 0.960, 0.975, 0.990} (8 × 5 = 40 instances). Deviation from
PROMPTS.md ("for C: S ∈ {0, 0.9, 0.97, 0.99}"): 0.975 is the grid point nearest 0.97, and the four mid-S
values are added because Phase 1 showed the geometric variety of this n/m lives at S ≈ 0.37–0.95 (for
S ≥ 0.96 every feature has D_i = 0.20).

Quantization (CLAUDE.md "Quantization (PTQ) definitions"): symmetric RTN, qmax = 2^(bits-1) - 1, bits
{8, 6, 5, 4, 3, 2}, granularity {per-tensor, per-column, per-row} with the eps guard on zero scales; tied
(same Q(W) as encoder and decoder), b in FP32; `round` is torch.round (round-half-to-even; ties have
measure zero). Noise control exactly as in CLAUDE.md: per column delta_i = ||Q(W)_i - W_i|| g_i / ||g_i||,
tied, 5 noise seeds; the Gaussian direction is drawn from a generator seeded by (noise seed, model
instance, bit width, granularity), so directions are independent across models and configurations
(the first sweep run shared one direction per noise seed; see the review entry). Ablations (RTN only, no noise controls) at
per-column and per-row: quantize b too (symmetric RTN with a per-instance per-tensor scale on b, same
qmax), encoder-only, decoder-only. Per model: 6 bit widths × (3 granularities × (1 RTN + 5 noise seeds) +
2 granularities × 3 ablations) = 144 configurations. For encoder-only / decoder-only the covariates
`killed`, `d_norm2`, `dW`, `d_overlap`, `margin_Q` are computed from the quantized matrix (the side that
was perturbed). `peak_ratio_i` is NaN for dead columns (norm < 0.1), like `D_i`. `tau = 0.05` is a config
value used consistently for `false_act`, `margin_tau` and `near_thr`.

Table size: A 30 × 5 × 144 = 21,600 rows, B 60 × 20 × 144 = 172,800, C 40 × 100 × 144 = 576,000; built
per setting (one invocation and one results directory per setting, float32 metric columns, S stored as
float64, `instance` is the join key to the Phase 1 tables), concatenated only for the figures.

Populations of the model-level columns: `n_eligible`, `survived_frac`, `composite_frac`, `worst_dR2`,
`mean_dR2` are over ELIGIBLE features; `n_killed_represented` counts features represented in FP
(norm > 0.5) whose quantized norm drops below 0.5; `n_switched_on` counts INELIGIBLE features whose
quantized false_act exceeds tau (design note 3). The per-feature `killed` column follows the literal
definition (quantized norm < 0.5) and is therefore True for every feature that was already dead in FP.
`near_thr` uses a CLOSED window |pre - tau| <= 0.05, so a leak-free feature with b_i in [0, 0.1] (inactive
pre-activation identically b_i) gets near_thr = 1; the Phase 3 pre-registration must decide the window and
report the fraction of eligible features with an identically-constant inactive pre-activation.

Eval set (design note 4, fixed per (setting, instance) by seed, identical for FP and every config; the
seed is `10000 + 7919 * training_seed + instance`, distinct from the training stream): a natural batch of
N = clip(ceil(1.1 · 20,000 / S), 40,000, 120,000) rows; per feature the inactive stratum is the first
20,000 natural rows with x_i = 0 (fixed count; if a feature has fewer, the common minimum is used and
recorded as `n_inactive_eval`), and the active stratum is the first 2,000 natural rows with x_i > 0 when
there are at least 2,000, else a forced-active batch of 2,000 rows (feature i ~ Uniform[0,1), others
natural; this happens for S >= 0.95, and at S = 0.95 exactly for about half of the features per instance,
recorded per feature in the `forced_active` column). Fixed counts keep `peak_out` and the survival correlation comparable
across S. For S = 0 (p = 1) the inactive stratum does not exist: false_act, leak energy and near_thr are
NaN, the inactive term of MSE is 0, and the composite label is NaN (not False). Both constructions are
exact conditional samples because features are independent.

Per-feature quantities written to the table (per model × config × feature): FP geometry (norm, D, D_plus,
max_overlap, margin, margin_tau, superposition, bias, peak_ratio, near_thr, leak_mean, p, importance);
damage (cond_mse, false_act(tau = 0.05), leak_energy, MSE via the design-note-4 decomposition, R2, dR2,
dLoss = I_i ΔMSE, rel_damage = dR2 / R2_FP, survival correlation on the active stratum, survived (≥ 0.9),
composite label, eligible = norm > 0.5 & R2_FP ≥ 0.5, peak_out, p99_out); covariates from Q(W) (dW_i,
d_norm2, d_overlap, margin_Q, margin flip, killed). Model-level table: total importance-weighted loss FP vs
quantized, survived fraction, worst-feature dR2.

Sanity checks (reported, not gated): 8-bit damage ≈ 0 (median |dR2| < 0.01, total loss increase < 1%);
mean damage increases monotonically as bits decrease for every (setting, granularity).

Figures: total weighted loss increase vs bits (one line per S, per setting and granularity); survived
fraction vs bits; RTN vs noise control (same axes); setting A arrows FP vs 4/3/2-bit per-column RTN.

Budget estimate: ~7 TMAC of forward passes for C (dominant), a few minutes on CPU; measured in the entry.

---

## 2026-10-24 — Phase 2 results (quant_sweep; data only)

Runs (canonical, produced by the reviewed code; see the review entry): `results/quant_sweep/20261024-092937-A`
(21,600 feature rows, 13 s), `20261024-092954-B` (172,800 rows, 60 s), `20261024-093058-C` (576,000 rows,
144 s); each holds `features.parquet`, `models.parquet`,
`sanity.json`, `meta.json` (with the source Phase 1 run and its meta), `config.yaml`, `figures/`. Git: no
commit. Code: `quantize.py`, Phase 2 part of `metrics.py`, `analysis.py`, `experiments/quant_sweep.py`,
`plots_quant.py`; 83 tests pass (62 from Phase 0/1 incl. the 38 black-box tests, plus 21 Phase 2 tests).

Sanity checks (per-column RTN unless stated; "eligible" = norm > 0.5 and R2_fp >= 0.5):
- 8 bits is lossless in practice: median |dR2| over eligible features 3e-4 (A), 9e-5 (B), 4e-4 (C); max
  relative total-loss increase 0.26% (A), 0.08% (C). PASS for all three settings.
- Mean dR2 is monotone non-decreasing as bits decrease for every setting and granularity. PASS.
  C, per-column, eligible features, mean dR2 by bits 8/6/5/4/3/2: 0.00003 / 0.0006 / 0.0024 / 0.012 /
  0.075 / 1.14; per-tensor is 1.5–3x worse than per-column at 3–6 bits.
- Per S (C, per-column, mean dR2 at 4 / 3 bits): S=0: 0.115 / 0.78; 0.602: 0.022 / 0.114; 0.749:
  0.014 / 0.077; 0.842: 0.009 / 0.055; 0.900: 0.009 / 0.054; 0.960: 0.009 / 0.062; 0.975: 0.009 / 0.056;
  0.990: 0.007 / 0.053. The dense model (S = 0, 20 dimensions spread over 100 features, only 7.8 eligible
  features per instance) is by far the most fragile.
- Noise control (independent directions per instance / bits / granularity; 5 seeds): ratio RTN / noise of
  the mean total loss increase, per-column, for 8/6/5/4/3/2 bits: C 1.00, 0.93, 1.02, 1.02, 1.00, 0.90;
  B 0.99, 0.92, 0.98, 1.06, 0.89, 0.89; A 0.67, 0.62, 0.24, 0.44, 0.23, 0.55. In B and C rounding
  structure is, at the model level, as damaging as isotropic noise of the same per-column norm; in the
  2-dimensional setting A rounding is 2–4x LESS damaging than matched noise. Checked mechanism (per-column
  4-bit, represented features of the A run): the RTN error vector is more tangential to W_i than an
  isotropic direction (mean |cos(error, W_i)| 0.41 vs 0.64 for the noise control; the isotropic 2-D
  expectation is 2/pi = 0.64), so it changes the gain less (median |d_norm2| 0.020 vs 0.037 at 4 bits,
  0.060 vs 0.080 at 3 bits; mean dR2 0.005 vs 0.006 at 4 bits, 0.014 vs 0.041 at 3 bits). With per-column
  scaling the dominant component of a 2-vector maps exactly onto +-qmax, so part of the error is confined
  to the minor axis; in 20 dimensions (C) the RTN error is as isotropic as noise (mean |cos| 0.14 vs the
  isotropic 0.18) and the ratio is ~1. Whether the per-feature pattern differs is Phase 3's question.

Observations the Phase 3 pre-registration must take into account (reported, not interpreted):
1. The Pearson survival label (design note 3's pathology, now observed): at >= 3 bits essentially every
   eligible feature "survives" (fraction 1.00 in A, B, C for 4–8 bits and 0.92–1.00 at 3 bits) even where
   dR2 is large, and in setting A it is 1.00 even at 2 bits with dR2 = 0.3–0.5 (a 2-D model's columns snap
   to a few directions, keeping the active-stratum correlation high while the gain changes). The composite
   label (corr >= 0.9 AND dR2 <= 0.1 AND dfalse_act <= 0.05) is informative where Pearson is not:
   C composite-survived fraction at 4 / 3 bits: S=0.602: 0.33 / 0.17; 0.749: 0.84 / 0.45; 0.842: 1.00 / 0.82;
   0.900: 1.00 / 0.90; 0.960: 1.00 / 0.87; 0.990: 1.00 / 0.92; 0 at 2 bits everywhere. B at 4 / 3 bits:
   0.52–1.00 / 0.08–0.98 depending on S. The useful bit range for a binary label is therefore 3–5 bits;
   2 bits destroys everything and 6–8 bits damages nothing.
2. Eligible features per instance in C: 7.8 (S=0), 38 (0.602), 46 (0.749), 57 (0.842), 32 (0.900), 100
   (S >= 0.960). At S = 0.9 only a third of the represented features reach R2_fp >= 0.5 (heavy
   interference with ~95 features in 20 dimensions); at S >= 0.96 all 100 features are represented AND
   well reconstructed.
3. Margin sign (design note 5 asked for this check before H1): fraction of eligible features with
   margin < 0 in C: 1.00 (S=0), 0.99 (0.602), 1.00 (0.749), 0.93 (0.842), then 0.00 for S = 0.900,
   0.960, 0.975, 0.990. The prediction "most represented features have negative margins" holds only in
   the dense / mid-sparsity regime; at S >= 0.9 the biases are negative enough that no single feature at
   x_j = 1 can leak. Raw `margin_i` therefore has one sign inside most (setting, S) cells and changes sign
   across S; H1's predictor choice (raw margin vs |margin| vs near_thr) must be made per cell with this in
   mind, and pooling across S would confound margin with S.
4. The sparse end of C (S >= 0.96) used forced-active batches for all 100 features (natural active rows
   < 2,000); every other instance used natural rows only. Inactive stratum: 20,000 rows everywhere except
   S = 0 (no inactive stratum, p = 1).
5. A: the square solutions at S = 0.7–0.8 are axis-aligned and survive even 2-bit per-column RTN
   (mean dR2 0.005 at S = 0.8) while the pentagons (S >= 0.9) lose dR2 = 0.3–0.4 at 2 bits.

Figures (`figures/quant_{A,B,C}_loss_vs_bits`, `_survival_vs_bits`, `_rtn_vs_noise`, `quant_A_arrows`):
loss-increase axes are symlog because a quantized model is occasionally slightly better than FP
(7.8% of A's RTN model rows have a negative loss increase, all at 5–8 bits).

Surprises / deviations: none in the pipeline; the two substantive observations are 1 (Pearson survival is
uninformative above 2 bits) and 3 (margin sign flips with S in setting C), both pre-announced as things to
check in CLAUDE.md design notes 3 and 5. Margin-sign fractions for the other settings (eligible features,
margin < 0): A 0.90 (S=0), 1.00 (0.7, 0.8), 0.96 (0.9), 1.00 (0.95, 0.99); B 0.96 (S=0), 0.81 (0.466), 1.00
(0.715, 0.848), 0.98 (0.919), 0.94 (0.957), 0.89–0.98 (S >= 0.977). Only C at S >= 0.9 has all-positive
margins.

---

## 2026-10-24 — Phase 2 code review (independent, after the first sweep) and rerun

Three reviewers (definition checker, black-box test author, data-integrity checker on the produced
tables) plus a refutation pass. Outcome:
- 1 bug confirmed, numerically marginal: the composite label compared a float32-derived false-activation
  rate with the float64 literal 0.05, so a feature with a quantized rate of exactly 1000/20000 (and FP
  rate 0) was labelled not survived (2 rows of 680,400). Fixed: rates computed in float64 and the
  thresholds compared with a 1e-9 tolerance.
- Minor, substantive for Phase 3: the noise control drew ONE Gaussian direction per noise seed and reused
  it for every instance, bit width and granularity (each row was still a correctly matched control, but
  cross-instance means were built on 5 shared directions). Fixed: generator seeded by (noise seed,
  instance, bits, granularity).
- Minor: loss-increase figures used a log axis on a signed quantity (negative increases vanished); now
  symlog with the linear band stated on the axis label. `pearson_columns` now has a dtype-independent
  constant-column guard. `S` is stored as float64. `clean_resp = max(0, ||W_i||^2 + b_i)`, a per-feature
  `forced_active` flag, `composite_frac` and `n_switched_on` were added; `n_killed` was renamed
  `n_killed_represented` to state its population.
- The 131-test black-box suite written from the specification alone (130 passed before the fixes; the
  one failure was the Pearson constant-column case) is adopted as `tests/test_blackbox_p2.py`; the full
  suite now has 214 tests.
- Because the tables on disk were produced before these fixes, the sweep was rerun once from the fixed
  code so tables, figures and code agree. The first-run directories (`20261024-085308-A`,
  `20261024-085325-B`, `20261024-085421-C`) are kept but superseded; RTN rows are identical between the
  two runs (verified: max |difference| 6e-8 over dR2, dfalse_act, surv_corr, mse_q, dW, d_overlap,
  margin_q), the composite label changed in 1 row per setting, and the noise-control rows differ (A:
  mean dR2 over noise rows 0.068 -> 0.075; B 0.081 -> 0.077; C 0.319 -> 0.316). The canonical Phase 2
  tables are the rerun directories named in the results entry.

---
