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

## 2026-10-24 — Phase 3 pre-registration (v2, awaiting the owner's approval)

### Status and provenance

- Written BEFORE any association between a geometry predictor and a damage outcome was computed on any
  data set, by me or by a reviewer. It is committed to git in the commit that adds this section; that hash is
  the pre-registration timestamp and is quoted in the next LOG entry. A v1 draft of this plan was never
  committed. It was reviewed by three independent reviewers (statistics, science, research integrity), and
  v2 replaces it; the changes are listed at the end of this entry.
- What had been computed when v2 was written, all on the exploration set E: label rates per cell; which
  condition of the composite label fails; marginal dispersion of the predictors; correlations between FP
  quantities only (for example norm vs D, margin_tau vs the FP false-activation rate); and one outcome-only
  statistic, the reliability ICC of dfalse_act and dR2 across the 5 noise-control seeds. The choice of the
  primary cell therefore used outcome-only and FP-only information. It used no predictor–outcome
  association.
- The confirmation models K were trained before approval: 2026-10-24 13:07:37, from commit 002c4ba plus
  three files that were uncommitted at the time and were committed in 474b458. Of their outputs only the FP
  geometry summary was opened (D*, represented counts, plateau ratios; reported in "Data"). The K
  quantization sweep has not been run, and no K outcome exists.

### Data

- Exploration set E: `results/quant_sweep/20261024-093058-C` (5 instances per S, training seed 0).
- Confirmation set K: 10 fresh instances per S for S ∈ {0.7488, 0.8415, 0.9000, 0.9602}, trained with
  `configs/rep_dims_confirm.yaml` into `results/rep_dims_confirm/20261024-130737` (300 s). The config is
  identical to `configs/rep_dims.yaml` except for the experiment and figure_prefix names, seed 1, n_seeds 10
  and `density_grid.keep: [3, 4, 5, 7]`. That last key selects the four densities from the same 16-point grid
  by index, so they are bitwise identical to E. Geometry check (FP only): K reproduces E's D*: 0.431 vs
  0.425, 0.290 vs 0.289, 0.207 vs 0.208, 0.155 vs 0.155. Represented counts per instance are 45.0 vs 45.6,
  71.8 vs 72.6, 96.1 vs 95.6 and 100 vs 100. Plateau ratios are 0.999–1.000. The training script's two
  "sticky" flags read FAIL only because a 4-point grid cannot contain the adjacent pairs they look for; they
  are not used for K. No K instance is ever excluded for a training diagnostic.
- K is quantized with `configs/quant_sweep.yaml`, whose only change is the entry `C_confirm` (all 40
  instances). The command is `python -m tmsq.experiments.quant_sweep --config configs/quant_sweep.yaml
  --setting C_confirm`, with no `--bits`, `--noise-seeds` or `--max-instances` override.
- Independence of K from E. Model init, training data and eval sets use disjoint integer seeds. This is
  checked in `tests/test_seed_independence.py`. The noise-control generator seed now includes the training
  seed. For training seed 0 the formula is unchanged, and an E noise row recomputed after the change matches
  the stored row exactly. With the change, every K noise direction comes from a new seed. Before the change,
  30 of the 40 K instances would have reused E's directions bit for bit. One pre-existing coincidence remains
  and is asserted in the test. K's training-data generator for chunk 2 has seed 2,000,043, and so does E's
  noise direction for (instance 0, 5 bits, per-row, noise seed 2). One is a uniform training-data stream and
  the other a single Gaussian direction, so the dependence is immaterial. Both runs already exist.
- Verdicts are decided on K only. E is analysed with the identical frozen code and decision rule (5
  instances, df = 4) and reported next to K, labelled exploratory. "Disagreement" means that E and K get
  different categories or mean ρ of opposite signs; it is reported, and the verdict follows K.
- Settings A and B are descriptive only and get no verdicts. On the Phase 2 data no A or B cell meets
  design note 7's count rule under either label, because A and B have only 5 and 20 features per instance.

### Fixed definitions

- Quantization rows: RTN, tied, ablation `none`, b in FP32. Per-row scaling is the primary granularity
  (design note 6), and per-column and per-tensor are secondary.
- Eligible feature: `norm_i > 0.5` AND `R2_fp >= 0.5`. On K this is the `eligible` column of the K sweep,
  computed on K's own FP eval set and fixed per feature across bits, granularities and methods.
- Binary damage label: the composite label. A feature survives iff `surv_corr >= 0.9` AND `dR2 <= 0.1` AND
  `dfalse_act <= 0.05`, with the thresholds of design note 3 fixed before any data existed and a 1e-9 float
  tolerance. Damaged means not survived. The Pearson-only label is reported as secondary, because on E its
  damaged fraction is ≈ 0 at 3–8 bits.
- Label validity rule: in an (S, granularity) cell the binary label is valid only if the damaged fraction
  among eligible features at 8 bits is ≤ 0.05; 8 bits is the negative control. The 0.05 cut-off was chosen
  after seeing E's 8-bit rates. On E it excludes S = 0.6019, where the 8-bit damaged fraction is 0.073
  per-column, 0.152 per-row and 0.230 per-tensor. Every one of those 8-bit failures comes from the leakage
  condition. The median FP false-activation rate there (0.132) is similar to S = 0.7488 (0.140) and
  S = 0.8415 (0.124), where the 8-bit damaged fraction is 0.000, so no mechanism is claimed. The rule
  excludes no S in K's grid on E. S = 0 has no defined label because p = 1 and there is no inactive stratum.
  The rule is re-evaluated on K.
- Cell rule for label-based analyses (design note 7): the label is valid, the cell has ≥ 30 damaged and
  ≥ 30 undamaged eligible features pooled over its instances, and each class appears in ≥ 3 instances.
  Cells passing on E (label counts only, S ∈ K's grid): per-row 3 bits at all four S and 4 bits at 0.7488;
  per-column 3 bits at 0.7488, 0.8415 and 0.9602 and 4 bits at 0.7488; per-tensor 3 bits at all four S and
  4 bits at 0.7488. No 5-bit cell passes. The K list is written to this LOG from label counts before any
  label-based analysis is fitted.
- tau = 0.05 everywhere. `near_thr` keeps the closed window |pre − tau| ≤ 0.05. The window edges matter only
  through point masses of the inactive pre-activation at exactly 0 or 0.1, and the only point mass is
  pre = b_i (rows in which no overlapping feature is active). On E (setting C, S > 0) no eligible feature has
  b_i equal to 0 or 0.1. None has an identically constant inactive pre-activation either: the smallest
  max_{j≠i} |W_i · W_j| over eligible features is 0.107. No eligible feature in A, B or C has near_thr = 1.
- H1 margin form: `margin_tau = tau − b_i − max(0, max_overlap_i)` (design note 5). It differs from
  `margin_i` by a constant, so every rank statistic is identical, but the sign fractions differ. Eligible
  features with raw margin_i < 0 / margin_tau < 0, per-row, on E: S = 0.7488: 1.00 / 0.79; 0.8415: 0.93 /
  0.30; 0.9000: 0.00 / 0.00; 0.9602: 0.00 / 0.00.

### Primary cell (one cell for H1–H3)

Setting C, S = 0.7488, per-row, 3 bits, RTN.

Reasons, from E, using outcome-only and FP-only information and no predictor–outcome association:
- Outcome reliability. ICC(1) of the primary outcomes over the 5 noise-control seeds (eligible features,
  per instance, averaged over instances), dfalse_act / dR2, per-row 3 bits: 0.55 / 0.12 at S = 0.7488;
  −0.00 / 0.01 at 0.8415; −0.03 / −0.03 at 0.9000; 0.05 / 0.07 at 0.9602. Under the generic-noise model, any
  FP predictor's |ρ| with a single-draw outcome is bounded by about sqrt(ICC). At S ≥ 0.8415 that bound is
  ≈ 0.1–0.25, below the |ρ| ≥ 0.2 effect of interest, so a null there would say nothing about geometry. The
  v1 choice (S = 0.8415) is withdrawn for this reason.
- Predictor validity (FP only, mean within-instance Spearman): ρ(margin_tau, false_act_fp) = −0.69 at 0.7488
  vs −0.17 at 0.8415. Margin measures FP leakage only at 0.7488.
- Separability of H2 and H3 (FP only): ρ(norm, D) = 0.56 at 0.7488 vs 0.95 at 0.8415 and 0.9000.
- No eligibility truncation: every represented feature is eligible at 0.7488 (228 of 228).
- Counts on E (label counts only): 228 eligible features (44–47 per instance); 138 damaged and 90
  undamaged, both classes in all 5 instances (per instance 25/22, 27/18, 27/20, 34/10, 25/20); 8-bit
  damaged fraction 0.000. The cell contains 17 leak-free features (false_act_fp = 0; 1–9 per instance).
- About 57% of the summed MSE increase in this cell is in the inactive stratum (outcome-only). That is a
  mix of leakage and gain damage, not a single mechanism.

### Hypotheses (primary cell, K)

| ID | predictor | primary outcome | direction as stated in CLAUDE.md | registered alternative reading of the opposite sign |
|----|-----------|-----------------|----------------------------------|------------------------------------------------------|
| H1 | `margin_tau` | `dfalse_act` (signed) | negative ρ: smaller margin, larger increase in false activation (level reading) | positive ρ: threshold-density reading. The change in a threshold-crossing rate scales with the FP density of inactive pre-activations at tau. For one leaker with overlap a > tau − b, d false_act / d a = p_j (tau − b) / a², which shrinks as margin_tau becomes more negative. On E at 0.7488, ρ(margin_tau, near_thr) = +0.61 (FP only). |
| H2 | `D` | `dR2` | negative ρ: smaller D (more crowding), larger dR2 | none registered |
| H3 | `norm` | `dR2` | negative ρ: larger norm, smaller dR2 (Duan's relative-error reading) | positive ρ: absolute-error reading. Under per-row scaling every column receives a similar absolute error, so the self-gain error 2 W_i · δ_i grows with norm_i (design note 6). |

All three tests are two-sided. A significant result of the opposite sign is reported as "Not supported
(opposite sign)" and, for H1 and H3, explicitly as consistent with the registered alternative reading.

Identifiability rule. On K, before any association, compute the mean within-instance ρ(norm, D) in the
primary cell from FP quantities. If it is ≥ 0.8, H2 and H3 are reported as ONE finding, the "norm/D axis",
using H2's test; the Holm family then has two tests, and no write-up counts H2 and H3 as separate evidence.
On E this value is 0.56.

### Test and decision rule

- Per-instance statistic: Spearman ρ between predictor and outcome, `scipy.stats.spearmanr` with average ranks
  for ties, over the eligible features of one instance whose predictor and outcome are both non-NaN. Dropped
  rows are counted per cell. An instance is valid for a test only if it has ≥ 20 such features and both
  predictor and outcome take ≥ 5 distinct values among them. Invalid instances are listed with the reason.
- Summary: the unweighted mean of the valid per-instance ρ, with a 95% t-interval, df = n_valid − 1. The
  mean ± std and, as a secondary interval, a 95% percentile interval from 1,000 bootstrap resamples of the
  instances' ρ values (seed 0) are also reported. A cell is summarised only if ≥ 8 of its 10 K instances
  (≥ 4 of 5 on E) are valid.
- Test: a two-sided one-sample t-test of the valid per-instance ρ against 0. Holm correction across the
  primary tests (three, or two under the identifiability rule) at family-wise α = 0.05. The exact sign-flip
  permutation p (all 2^n_valid sign patterns) is reported next to it as a robustness check; the verdict uses
  the t-test, and any disagreement in the Holm decision is reported. The 95% t-interval is unadjusted; the
  Holm-adjusted p decides.
- Testability by reliability. On K, before any association, compute ICC_noise of each primary outcome in
  the primary cell (outcome-only, same definition as above). sqrt(ICC_noise) is reported next to every
  per-instance mean ρ, primary and secondary, as the attainable ceiling under the generic-noise model.
- Verdicts are applied in this order, and the first line that matches decides. Under Mixed every reason
  that applies is listed.
  - (a) **Not testable**: the primary cell has < 8 valid instances.
  - (b) **Not supported (opposite sign)**: Holm-adjusted p < 0.05 and mean ρ has the sign opposite to the
    CLAUDE.md direction. For H1 and H3 the registered alternative reading is named.
  - (c) **Not testable (outcome dominated by the perturbation draw)**: Holm-adjusted p ≥ 0.05 and
    ICC_noise < 0.10. Low reliability can turn a null into "not testable"; it can never block a positive
    result.
  - (d) **Not supported**: Holm-adjusted p ≥ 0.05.
  - (e) **Mixed**: Holm-adjusted p < 0.05 with the CLAUDE.md direction and at least one of: [small]
    |mean ρ| < 0.2; [inconsistent] a downgrade cell shows a mean ρ of the opposite sign with |mean ρ| ≥ 0.1
    and uncorrected p < 0.05, using the same per-instance rule and test.
  - (f) **Supported**: Holm-adjusted p < 0.05, the CLAUDE.md direction, |mean ρ| ≥ 0.2, and no
    [inconsistent] reason.
- Downgrade cells (K, RTN, 3 bits). They can only downgrade a verdict, never upgrade it, and they are
  deliberately not corrected for multiplicity. A downgrade cell that fails the per-instance rule cannot
  trigger a downgrade and is listed as not testable.
  - H1: S = 0.7488 per-column and per-tensor. Other S values are excluded because the composition of
    margin_tau signs differs there (0.79 negative at 0.7488 vs 0.30, 0.00, 0.00), and the level and
    threshold-density readings can legitimately swap.
  - H2: per-row at S = 0.8415 and S = 0.9000; per-column and per-tensor at S = 0.7488.
  - H3: per-row at S = 0.8415 and S = 0.9000 only. Under per-column and per-tensor scaling, design note 6
    predicts that the norm–damage sign may differ, so those cells are reported as mechanistic variants.
  - S = 0.9602 is not a downgrade cell, because D and norm are nearly constant there (within-instance IQR
    0.004 and 0.009 on E).
- H2 companion (CLAUDE.md H2). The per-instance Spearman of `D_plus` with dR2 in the primary cell is
  classified with the same rule, without Holm, and labelled secondary. If D and D_plus fall in different
  categories, the LOG states "D (paper-comparable): <category>; D_plus (sign-aware, carries the mechanistic
  claim): <category>". The H2 verdict remains the D verdict.
- Operating characteristics, from a simulation that uses no project data: latent bivariate-normal features,
  per-instance Spearman, 10 instances, the first Holm step α = 0.0167. With n = 45 features per instance,
  the size of the primary cell, and no between-instance heterogeneity, P(significant) = 0.54 / 0.82 at
  latent r = 0.15 / 0.20, and P(Supported) = 0.36 / 0.77 / 0.96 at r = 0.20 / 0.25 / 0.30. With n = 57 and
  a between-instance SD of 0.2, P(Supported) = 0.34 / 0.59 / 0.78. So the |mean ρ| ≥ 0.2 point criterion,
  not the test, decides between Supported and Mixed. The design is powered for ρ ≥ 0.3, and a true ρ near
  0.2 is called Supported only about a third of the time. |ρ| = 0.2 (about 4% of rank variance) is the
  smallest effect of interest. Next to every verdict the write-up gives the number, and a Supported
  hypothesis with mean ρ < 0.3 is never described as "geometry predicts damage" without it.

### Secondary analyses (K; labelled secondary)

1. Per-instance Spearman table, one family. Cells: S ∈ {0.7488, 0.8415, 0.9000, 0.9602} × 3 granularities
   × bits {3, 4, 5}. Pairs per cell: the three primary pairs; margin_tau–dR2; near_thr–|dfalse_act|
   (mechanism, predicted positive); D_plus–dR2; norm–dcond_mse (the active-stratum outcome on which the gain
   mechanism acts); p99_out_fp–dR2 (the "Duan analog", predicted negative; a distinct predictor, not a
   robustness variant of norm, since on E ρ(norm, p99_out_fp) = 0.40 at 0.7488 and 0.54 at 0.8415);
   peak_out_fp–dR2; and the partial Spearman of dR2 with D given norm and with norm given D. Also per
   (S, granularity): break_bits with D and with norm. Every cell is reported in one table with mean ρ,
   t-interval, uncorrected p, the Benjamini–Hochberg q-value within this family, and sqrt(ICC_noise). A
   secondary result may be called "significant" only if q < 0.05, and any secondary pattern cited in the
   write-up is labelled exploratory. Each cell also reports the split of the MSE increase into
   p · dcond_mse and (1 − p) · dleak_energy. A sensitivity version repeats the three primary tests without
   the leak-free features (false_act_fp = 0; 17 on E).
2. `break_bits` is computed once per (S, granularity), from the composite label at bits {8, 6, 5, 4, 3, 2}.
   It is the largest bit width at which the feature is damaged, with values in {2, 3, 4, 5, 6, 8}. A
   feature that survives all six widths keeps NaN in the table but is ranked below 2 (most robust) in every
   rank statistic; it is never dropped. break_bits = 2 means "damaged only at 2 bits", not "undamaged". A
   feature whose label is non-monotone across bits keeps its largest damaged width, and the count of such
   features is reported per cell (8 of 6,528 feature × granularity rows on E). A sensitivity version uses
   the monotone envelope: the largest b at which the feature is damaged at b and at every lower width.
   break_bits is analysed only in (S, granularity) cells whose label is valid. Where it takes ≤ 2 distinct
   values among the eligible features of a cell, it is a relabelling of the 3-bit label. It is then
   reported as that label's rank-biserial and not counted as a separate result; on E this is the case for
   every S ≥ 0.8415 under per-row and per-column scaling. Predicted sign with D and with norm: negative.
3. Univariate AUC of each predictor for the binary label, in every cell that passes the cell rule. It is
   computed within each instance; instances with < 5 damaged or < 5 undamaged eligible features are excluded
   and counted. Label 1 = damaged. Each score is oriented by the CLAUDE.md direction: −margin_tau, −D,
   −D_plus, −norm, −p99_out_fp, −peak_out_fp, +near_thr. An AUC below 0.5 is reported as is and never
   flipped. Summary: the mean of the per-instance AUCs with a 95% t-interval. The AUC pooled over instances
   is shown next to it, labelled as mixing between-instance differences, with a 95% percentile cluster
   bootstrap over instances (10,000 resamples, seed 0). With 10 clusters this bootstrap undercovers (about
   90% actual coverage for a nominal 95%), and that is stated wherever it is shown.
4. Logistic regression for the binary label, in cells passing the cell rule. Predictor sets: {p99_out_fp},
   {peak_out_fp}, {margin_tau}, {D}, and {margin_tau, D, D_plus, norm, p99_out_fp}. Deviation from
   PROMPTS.md: p99_out_fp replaces peak_out as Duan's predictor, because the sample max is an extreme-value
   statistic; {peak_out_fp} is fitted as well. Model: `LogisticRegression(penalty=None, max_iter=10000)`, a
   `StandardScaler` fitted inside each training fold, no class weights, `GroupKFold(n_splits=5)` by instance
   (2 instances per test fold on K; on E this is leave-one-instance-out). The CV AUC is the AUC of the pooled
   out-of-fold predictions, with the mean per-fold AUC alongside. Its CI is the cluster bootstrap of item 3
   applied to the out-of-fold predictions without refitting. VIFs are computed on the full cell, and joint
   coefficients are not interpreted when any VIF > 5.
5. Rounding structure vs generic noise, for each primary pair in the primary cell. For each instance k and
   noise seed s, ρ_noise,k,s is computed on that single noise realization, matching the single RTN
   realization; ρ_noise,k is the mean over the 5 seeds. The outcome is never averaged over seeds before
   ranking: averaging raises reliability from ICC to 5·ICC/(1 + 4·ICC) and would inflate |ρ_noise|. Let
   σ = sign(mean ρ_RTN) and R = mean ρ_noise / mean ρ_RTN (signed). The paired difference is
   Δ_k = σ · (ρ_RTN,k − ρ_noise,k), tested with a one-sample t-test and a 95% t-interval. Fixed reading:
   "specific to rounding" iff R < 0.5 (including R ≤ 0) AND mean Δ > 0 with p < 0.05; "explained by
   perturbation size" iff R ≥ 0.8 AND NOT (mean Δ > 0 with p < 0.05); otherwise "partly specific /
   unresolved". A reading is given only when the primary verdict is Supported or Mixed. Otherwise R and the
   Δ interval are reported without a reading.
6. Design note 10, in the primary cell and within each instance. dW is residualized on norm and peak_ratio
   by OLS. Model (1) is dR2 ~ margin_tau + D + norm; model (2) adds the residualized dW. All variables are
   rank-transformed and standardized within the instance. Reported: the per-instance coefficients and the
   per-instance change (2) − (1), each averaged with a 95% t-interval, plus VIFs. Model (2) is never reported
   alone.
7. Sensitivity of the composite label: survival correlation ∈ {0.8, 0.9, 0.95} × dR2 ∈ {0.05, 0.1, 0.2} ×
   dfalse ∈ {0.025, 0.05, 0.1}. AUCs only, reported as robustness, never used to select a threshold.
8. Settings A and B: per-instance Spearman tables of the H1–H3 pairs at every S, bits 3–5, per-row.
   Descriptive only.

Secondaries 3, 4, 6, 7 and 8 are estimation only: intervals, no p-values and no verdicts.

### Procedure and stop rules

1. On E, the analysis code is built and debugged and the E tables are produced. The primary specification is
   frozen at approval: predictor, outcome, cell, direction, alternative readings, the |ρ| ≥ 0.2 criterion,
   Holm, the verdict order, the testability and identifiability rules and the downgrade cells. It is not
   changed after any E association has been computed. Any other change made after E associations were seen
   is logged with its date as "deviation after seeing E".
2. The analysis code is committed, and its hash recorded in this LOG, before the K sweep is run.
3. The K sweep is run with the command above.
4. Before any K predictor–outcome association, this LOG records, from labels, FP quantities and outcome-only
   statistics: `sanity.json` `bits8_pass`, `monotone_nondecreasing_as_bits_decrease` per granularity,
   eligible counts per instance, the 8-bit damaged fraction per (S, granularity), the K cells passing the cell
   rule, ICC_noise of the primary outcomes in the primary cell, and ρ(norm, D) in the primary cell.
5. STOP and report, with no H1–H3 verdict and K tables reported as descriptive only, if `bits8_pass` is false
   or any granularity is non-monotone. The label validity and count rules gate only the label-based analyses:
   secondaries 2, 3, 4 and 7. If the primary cell fails either rule, the primary Spearman verdicts are still
   issued, with the failure reported next to them. No substitute primary cell is ever picked.
6. The frozen code is run on K. Any code change after K's `features.parquet` has been read, bug fixes
   included, is reported with the K results computed both before and after the change.

H4 (frequency, setting D) belongs to Phase 4 and is not part of this plan.

### Changes from v1 (uncommitted draft) after the independent review

- Primary cell moved from S = 0.8415 to S = 0.7488, because of outcome reliability, predictor validity and
  H2/H3 separability; all numbers were recomputed independently and match the reviewer's.
- H1 and H3 are two-sided, each with a registered alternative reading of the opposite sign. Added: the
  identifiability rule (H2/H3), the reliability-based "Not testable" category, a per-instance validity rule,
  an ordered verdict list, explicit downgrade cells per hypothesis with a |ρ| ≥ 0.1 floor, and the D_plus
  companion rule.
- The rounding-vs-noise comparison now uses single-draw ρ averaged over seeds instead of a seed-averaged
  outcome. Added: a paired test, and a signed ratio that stays defined when ρ_RTN is near 0.
- break_bits is defined once per (S, granularity), never drops robust features, and is reported as a
  rank-biserial wherever it collapses to the 3-bit label.
- Secondary multiplicity is handled with BH within one stated family and a no-omission rule. AUCs are now
  per instance, with fixed orientation. Logistic-regression details are fixed, and so is the stop-rule
  order (label rules gate only label-based analyses).
- Code freeze and deviation rules were added. The noise-control seed now includes the training seed, so K's
  noise directions are new, and the seed bookkeeping is tested.
- Corrected statements: the margin sign fraction now distinguishes raw margin from margin_tau; the
  validity-rule explanation no longer claims a mechanism; the near_thr window statement is now justified by
  point masses.
- Operating characteristics added, from a simulation that uses no project data.

---

## 2026-10-24 — Phase 3 pre-registration v2 committed

Committed as `f76b49a` (git log -1 --format=%cI f76b49a gives its timestamp). No predictor–outcome association
had been computed on E or K at that commit; the K quantization sweep had not been run. Awaiting the owner's
approval before any analysis.

---

## 2026-10-24 — Phase 3 pre-registration v2 APPROVED; implementation clarifications

The owner approved pre-registration v2 (`f76b49a`) unchanged. The clarifications below resolve points the
plan leaves open. They are written down before any predictor–outcome association has been computed on any
set. None of them changes a primary specification.

1. `break_bits` tests. The per-instance rule asks for ≥ 5 distinct outcome values, but break_bits has at most
   6 possible values and on E usually 2–4. For break_bits the outcome requirement is therefore ≥ 2 distinct
   values; the predictor still needs ≥ 5. Where break_bits takes ≤ 2 distinct values among the eligible
   features of a cell, the reported statistic is the per-instance rank-biserial
   r_rb = 2·AUC(predictor, label "damaged at 3 bits") − 1. It has the same sign convention as the Spearman
   ρ with break_bits and is kept outside the BH family, as the plan says ("not counted as a separate
   result").
2. Settings A and B (secondary 8, descriptive). The ≥ 20-feature rule cannot be met: A has 5 features per
   instance and B has 20. The A/B tables therefore include an instance if it has ≥ 5 eligible features and
   both variables take ≥ 3 distinct values. They report the mean per-instance ρ and n_valid, with no
   p-values and no intervals.
3. Holm with an untestable primary test. A primary test with < 8 valid instances enters the Holm step-down
   with p = 1, and the family size is unchanged (3, or 2 under the identifiability rule).
4. A downgrade cell can trigger a downgrade only if it has ≥ 8 valid K instances (≥ 4 of 5 on E), the same
   bar as summarising a cell.
5. `LogisticRegression(penalty=None)` is deprecated in scikit-learn 1.9 (FutureWarning, removal in 1.10). The
   identical unpenalised model is fitted as `LogisticRegression(C=np.inf, max_iter=10000)`.
6. The composite label for the sensitivity grid (secondary 7) is recomputed from the stored float32 columns
   with a 1e-7 tolerance. At the registered thresholds (0.9, 0.1, 0.05) the recomputed label must reproduce
   the stored `composite_survived` exactly, and the code asserts it.
7. sqrt(ICC_noise) is reported as sqrt(max(ICC, 0)). ICC is the one-way ICC(1) over the 5 noise-control
   seeds, with features as groups, computed within each instance over eligible features and averaged over
   instances.

---

## 2026-10-24 — Phase 3 analysis code: exploration run, pre-freeze review, freeze

Code: `src/tmsq/phase3.py`, `src/tmsq/experiments/phase3.py`, `src/tmsq/plots_phase3.py`,
`configs/phase3.yaml`, `tests/test_phase3.py` (16 tests on synthetic data). It was run on E
(`results/phase3/20261024-134227-E-dev`, then `...-140808-E-dev2` after the fixes below). From this point the
E predictor–outcome associations had been seen.

Pre-freeze review (three independent reviewers, plus refutation of each bug; no K data existed):
- Independent replication. A reviewer re-implemented the primary analysis and the pre-checks from the
  pre-registration text alone, without reading the code. All 497 compared numbers agree to 1e-6, including
  verdicts, cell lists and invalid-instance lists.
- 5 confirmed bugs, all in secondary or reporting parts, none affecting a primary number. All fixed:
  1. The break_bits rank-biserial used the AUC instance rule instead of the registered per-instance rule
     (clarification 1). This changed n_valid and the means at S = 0.9 on E.
  2. The registered "Holm decision on the sign-flip p, and any disagreement reported" was missing.
  3. sqrt(ICC_noise) was not reported next to every per-instance mean ρ: downgrade cells, the D_plus
     companion, the leak-free sensitivity, secondary 5 and secondary 8.
  4. Secondary 7 was not gated by the label rules.
  5. The primary cell's label-rule status was not reported next to the primary verdicts.
- Smaller fixes:
  - Non-summarised cells no longer report an interval or p.
  - The dMSE split now gives both terms.
  - Downgrade cells carry dropped rows and an explicit "not testable" status.
  - Invalid instances are listed per secondary row.
  - Under STOP, secondary tables are flagged descriptive-only and BH significance is suppressed.
  - The STOP check now requires all three monotonicity flags.
  - Pearson-only damaged counts are reported per cell.
  - An E-vs-K comparison mode (`--compare`) implements the registered disagreement rule.
  - Pooled AUCs are labelled, and cluster bootstraps carry the undercoverage note.
  - GroupKFold test-fold composition is recorded.
  - A completeness check (instances per S, all bit widths, 5 noise seeds) and the clarification-6 label
    reproduction check run before any analysis and abort it on failure.
  - plot_auc no longer crashes when no cell passes.
  - Pairs without a registered sign show none.

Implementation readings made AFTER the E associations had been seen. Per the plan, they are logged as
"deviation after seeing E". None touches a primary specification:
- 8. A break_bits relabelling uses the binary "break_bits equals the larger of its two values". This equals
  clarification 1's "damaged at 3 bits" whenever the values are {2, 3}, which holds for every E cell. It
  avoids a constant label if K had other values.
- 9. Secondary 6 now follows the registered order literally: OLS residualization of raw dW on raw norm and
  peak_ratio, then rank-standardization of every variable. The first version rank-transformed before
  residualizing.
- 10. Secondary 7 runs in every cell that passes the cell rule, not only in the primary cell. The plan names
  no cell, and secondary 3, which it re-labels, covers every passing cell.
- 11. The identifiability statistic ρ(norm, D) uses the registered per-instance validity rule and a NaN-safe
  mean.

E results, exploratory only (5 instances, df = 4; the verdicts are decided on K):
- H1: mean ρ +0.275 [−0.063, +0.613], Holm p 0.261, Not supported. The point estimate has the sign of the
  registered threshold-density reading.
- H2: +0.197 [−0.060, +0.453], Holm p 0.261, Not supported. Its downgrade cells point the same way
  (+0.19 to +0.31).
- H3: +0.026, Not supported.
- D_plus companion: +0.192.
- Sign-flip Holm decisions agree with the t-test everywhere. sqrt(ICC_noise): 0.74 for dfalse_act and 0.35
  for dR2.

The primary specification is unchanged. The analysis code is frozen in the commit that adds this entry, and
its hash is recorded in the next entry before the K sweep is run.

---

## 2026-10-24 — Phase 3 analysis code frozen

Frozen at `268a66e` (2026-10-24T14:09:12-04:00), before the K quantization sweep was run. Next, per the
registered procedure: K sweep (`quant_sweep --setting C_confirm`, no overrides), then the pre-check written here
before any K association, then the frozen analysis on K.

---

## 2026-10-24 — Phase 3 K pre-check (written BEFORE any K predictor–outcome association)

K sweep: `results/quant_sweep/20261024-140931-C_confirm` (registered command, no overrides; 576,000 rows,
124 s). Pre-check run: `results/phase3/20261024-141155-K-precheck` (frozen code `268a66e`). Labels, FP quantities and
outcome-only statistics only:

- sanity: bits8_pass = True; monotone per granularity = {'tensor': True, 'column': True, 'row': True}. No STOP.
- completeness: instances per S = {'0.7488': 10, '0.8415': 10, '0.9': 10, '0.9602': 10}, bits = [2, 3, 4, 5, 6, 8], noise seeds = 5; no problems.
- clarification 6: recomputed composite label reproduces the stored one (0 mismatches).
- 8-bit damaged fraction: 0.000 in all 12 (S, granularity) cells, so the label is valid everywhere in K's grid.
- ICC_noise in the primary cell: dfalse_act 0.65, dR2 0.18 (E: 0.55, 0.12); both ≥ 0.10.
- identifiability: mean within-instance ρ(norm, D) in the primary cell = 0.422 over 10 instances (< 0.8), so H2 and H3 stay separate; Holm family = H1, H2, H3.
- eligible features per instance, per-row 3 bits: S=0.7488 43–47; S=0.8415 45–63; S=0.9 13–55 (several instances < 20, so some S = 0.9 tests may have < 8 valid instances); S=0.9602 100.
- cells passing the cell rule (S, granularity, bits: damaged / undamaged):
  0.7488 tensor 3b: 356/94; 0.7488 tensor 4b: 118/332; 0.7488 tensor 5b: 69/381; 0.7488 column 3b: 244/206; 0.7488 column 4b: 91/359; 0.7488 row 3b: 245/205; 0.7488 row 4b: 97/353; 0.8415 tensor 3b: 373/193; 0.8415 column 3b: 125/441; 0.8415 row 3b: 208/358; 0.9 tensor 3b: 158/118; 0.9 row 3b: 69/207; 0.9602 tensor 3b: 665/335; 0.9602 column 3b: 123/877; 0.9602 row 3b: 322/678.
- primary cell (0.7488, row, 3 bits) passes: 245 damaged / 205 undamaged.

Next: the frozen analysis on K.

---

## 2026-10-24 — Phase 3 RESULTS (confirmatory, set K)

Runs:
- K analysis: `results/phase3/20261024-141233-K` (frozen code `268a66e`, unchanged; 35 s).
- E-vs-K comparison: `results/phase3/20261024-141338-compare`.
- Post-hoc exploration: `results/phase3/20261024-141422-posthoc`.

Figures:
- `figures/phase3_forest_E_vs_K.*`
- `results/phase3/20261024-141233-K/figures/phase3_K_{forest,damage_vs_margin,auc}.*`

### Primary verdicts (K, primary cell C, S = 0.7488, per-row, 3 bits; Holm over H1–H3)

| test | mean ρ (10 instances) | 95% t-interval | Holm p | sign-flip Holm p | sqrt(ICC_noise) | verdict |
|---|---|---|---|---|---|---|
| H1 margin_tau vs dfalse_act | +0.491 | [+0.341, +0.641] | 8.1e-5 | 0.006 | 0.81 | **Not supported (opposite sign); consistent with the registered threshold-density reading** |
| H2 D vs dR2 | +0.399 | [+0.293, +0.506] | 4.2e-5 | 0.006 | 0.43 | **Not supported (opposite sign)** |
| H3 norm vs dR2 | +0.002 | [−0.169, +0.173] | 0.98 | 0.99 | 0.43 | **Not supported** |
| H2 companion D_plus vs dR2 (no Holm) | +0.397 | [+0.287, +0.508] | 1.9e-5 | 0.002 | 0.43 | Not supported (opposite sign) |

- H3 is a genuine null. ICC_noise(dR2) = 0.18 ≥ 0.10, so the reliability rule does not apply.
- The sign-flip Holm decisions agree with the t-test for every test.
- Downgrade cells have the same sign as the primary cell:
  - H1: per-column +0.49, per-tensor +0.54.
  - H2: per-row S = 0.8415 +0.08; per-column +0.36 and per-tensor +0.41 at S = 0.7488; S = 0.9 not testable (7 valid instances).
- Leak-free sensitivity (features with false_act_fp > 0 only): H1 +0.30, H2 +0.30, H3 +0.09.
- E vs K: no disagreement under the registered rule. The signs agree; E's point estimates (+0.28, +0.20, +0.03) were not significant with 5 instances.
- In plain words: in the cell where per-feature damage is reliable, the geometry predictors do predict damage
  (|ρ| ≈ 0.4–0.5), but in the direction OPPOSITE to H1 and H2 as stated. Features with larger margins and
  larger D (less crowded) are damaged MORE. Norm predicts nothing.

### Secondary analyses (K; descriptive, BH within the 309-test family)

- Secondary 1. 106 of 309 family tests have q < 0.05, almost all at S = 0.7488. At S = 0.8415 and 0.96 the
  effects are near 0 (|ρ| ≤ 0.17), as the near-zero reliability there predicts.
- Mechanism pair near_thr–|dfalse_act|: +0.51 [+0.37, +0.66], as predicted.
- Duan analogs in the predicted direction: p99_out_fp–dR2 −0.24, peak_out_fp–dR2 −0.36. A higher FP peak
  output means less damage, while norm itself is null.
- Partial correlations: D | norm +0.41; norm | D −0.19.
- break_bits at S = 0.7488: ρ with D = +0.54 to +0.62 in all three granularities. A larger D breaks at a
  higher bit width.
- Secondary 3 (primary cell, oriented by the CLAUDE.md direction; below 0.5 means the opposite direction):
  margin_tau 0.30, D 0.26, D_plus 0.28, norm 0.50, p99_out 0.63, peak_out 0.70, near_thr 0.73.
- Secondary 4 cross-validated AUC: {D} 0.72, {margin_tau} 0.70, {peak_out} 0.70, {p99_out} 0.64, {all} 0.74.
  The {all} model's VIF is 309 (margin_tau, D and D_plus are nearly collinear), so its coefficients are not
  interpreted.
- Secondary 5, rounding vs noise. ρ_RTN vs ρ_noise: H1 0.491 vs 0.486 (R = 0.99), H2 0.399 vs 0.323
  (R = 0.81); neither paired difference is significant. No reading is given, because the verdicts are not
  Supported or Mixed. Descriptively, the same pattern appears under isotropic Gaussian noise of equal
  per-column norm: it reflects sensitivity to perturbation, not something specific to rounding.
- Secondary 6 (design note 10). The D coefficient (+0.39) and the norm coefficient (−0.19) are unchanged when
  residualized dW is added (changes ≤ 0.012). Residual dW has its own effect (+0.21).
- Secondary 8 (descriptive). Setting B shows the same H1 sign (+0.4 to +0.7) at S ≥ 0.977. A has 5 features
  per instance and is uninformative.

### POST-HOC exploration (decided after seeing K; NOT pre-registered; does not change any verdict)

1. The cross-feature correlations come from discrete geometric classes. In the primary cell, features fall
   into three classes by margin_tau, and within each class the correlations are small and inconsistent in
   sign. K means per class:

   | class | n | D | bias | max_overlap | FP false_act | near_thr | R2_fp | dfalse_act | dR2 | damaged at 3 bits |
   |---|---|---|---|---|---|---|---|---|---|---|
   | margin_tau < −0.1 | 108 | 0.395 | −0.087 | 0.31 | 0.164 | 0.10 | 0.66 | 0.037 | 0.056 | 37% |
   | −0.1 … 0.015 | 249 | 0.445 | −0.075 | 0.14 | 0.137 | 0.15 | 0.75 | 0.046 | 0.066 | 45% |
   | > 0.015 (antipodal pairs, D ≈ 0.5) | 93 | 0.499 | +0.010 | 0.01 | 0.001 | 0.65 | 0.84 | 0.274 | 0.107 | 99% |

   E shows the same three classes and the same pattern (antipodal class 100% damaged, dfalse 0.28).
2. A candidate mechanism. Antipodal-pair features have no positive interference to suppress, so they learn
   a bias of about 0. Their inactive pre-activations therefore sit at the 0.05 threshold (near_thr 0.65), and
   any weight perturbation turns them into false activations. Crowded features learn negative biases that act
   as a buffer. This is the registered threshold-density reading made concrete. It is not only a threshold
   artefact: their dR2, which includes the threshold-free leak energy, is also the largest.
3. The H2 sign is largely a ceiling effect. D correlates with R2_fp (ρ = 0.87), and the partial ρ(D, dR2 | R2_fp)
   is +0.08 [−0.02, +0.18]. Relative damage dR2 / R2_fp still increases with D (+0.28).
4. The H1 sign is carried by threshold density. ρ(margin_tau, near_thr) = +0.78 (FP only), and the partial
   ρ(margin_tau, dfalse_act | near_thr) falls from 0.49 to +0.17.
5. Scope: everything above holds at S = 0.7488, where antipodal pairs and denser polytopes coexist. At
   S ≥ 0.8415, per-feature damage is dominated by the perturbation draw.

### Surprises, limitations, next steps

- Surprise: the direction. The pre-registered intuition ("crowded, low-margin features break first") is
  rejected in the one cell where it is testable. The fragile features are the least crowded ones.
- Limitations:
  - One setting (C, n = 100, m = 20) and one sparsity carry the result.
  - The class structure means the effective sample is closer to "three classes × 10 instances" than to 450
    features.
  - false_act depends on the 0.05 detection threshold; the dR2 result does not.
- Next step, recorded here before any new data: a confirmatory test of the post-hoc class finding on a
  fresh, pre-registered set of models (Phase 3b; plan in the next entry).

---

## 2026-10-24 — Phase 3b pre-registration: confirm the post-hoc "antipodal class" finding on fresh models

Status: written and committed BEFORE the new models are trained. The finding it tests was discovered post
hoc on E and K (Phase 3 results entry). Decided by Claude Code while the owner was away, under the owner's
standing instruction to continue autonomously. The owner can veto it.

Data: a fresh set K2 of 20 instances at S = 0.7488 only, training seed 7. Config
`configs/rep_dims_confirm2.yaml` is identical to `configs/rep_dims.yaml` except seed 7, 20 seeds and
`density_grid.keep: [3]` (bitwise the same density). Seed 7 shares no integer seed with E or K (checked with
the seed-bookkeeping functions; added to `tests/test_seed_independence.py`). Quantized with the unchanged
sweep config as setting `C_confirm2` (no overrides). Analysed by `src/tmsq/experiments/phase3b.py`,
committed before the K2 sweep is run.

Class definition (FP geometry only; fixed now): a feature is "antipodal" iff D_i ≥ 0.48. On E and K at
S = 0.7488, eligible D values have a gap between 0.470 and 0.493. Every other eligible feature is "other".
Eligibility, the composite label and every damage metric are as in Phase 3.

Primary test H5 (cell C, S = 0.7488, per-row, 3 bits, RTN):
- Per instance k: Δ_k = (damaged fraction among eligible antipodal features) − (damaged fraction among
  eligible other features). An instance is valid iff it has ≥ 2 antipodal and ≥ 10 other eligible features.
- Two-sided one-sample t-test of the valid Δ_k against 0; 95% t-interval; exact sign-flip p reported
  alongside.
- **Supported**: p < 0.05, mean Δ > 0, and mean Δ ≥ 0.20 (20 percentage points).
- **Mixed**: p < 0.05 and 0 < mean Δ < 0.20.
- **Not supported**: p ≥ 0.05, or a significant negative Δ (reported as such).
- **Not testable**: fewer than 16 of the 20 instances are valid.
- E and K values of the same statistic are reported next to it (post hoc there).

Secondary (estimation only, no verdicts): the same per-instance difference for
- mean dR2 (threshold-free) and mean dleak_energy (threshold-free leakage);
- mean dfalse_act;
- per-column and per-tensor at 3 bits, and per-row at 4 bits;
- the matched-noise control (per-instance Δ averaged over the 5 noise seeds).

Mechanism (descriptive): class means of bias, near_thr, false_act_fp and R2_fp.

Stop rules: if the K2 sweep's `bits8_pass` is false, or any granularity is non-monotone, or the sweep does not
have 20 instances, 6 bit widths and 5 noise seeds, STOP and report with no verdict.

---

## 2026-10-24 — Phase 3b RESULTS: the antipodal-class finding is confirmed on fresh models

Runs:
- K2 training: `results/rep_dims_confirm2/20261024-141902` (20 instances, seed 7, 137 s; D* 0.404–0.456,
  plateau 1.000).
- K2 sweep: `results/quant_sweep/20261024-142123-C_confirm2`. bits8_pass true; monotone under all three
  granularities.
- Analysis: `results/phase3b/20261024-142225-K2`, frozen code `f45a856`, unchanged.

**H5 verdict: Supported.**
- Primary statistic (C, S = 0.7488, per-row, 3 bits): mean Δ(damaged fraction, antipodal − other) =
  +0.529, 95% t-interval [+0.483, +0.575]. 18 of 20 instances are valid; instances 0 and 4 have no antipodal
  feature. t-test p = 1.4e-14 and sign-flip p = 7.6e-6.
- E and K, post hoc: +0.448 [+0.375, +0.521] and +0.574 [+0.553, +0.594].

Secondary (estimation only):

| statistic (antipodal − other, per instance) | K2 mean | 95% t-interval |
|---|---|---|
| dR2, row 3 bits (threshold-free) | +0.039 | [+0.027, +0.051] |
| dleak_energy, row 3 bits (threshold-free) | +0.0020 | [+0.0015, +0.0024] |
| dfalse_act, row 3 bits | +0.228 | [+0.221, +0.236] |
| damaged fraction, per-column 3 bits | +0.559 | [+0.501, +0.616] |
| damaged fraction, per-tensor 3 bits | +0.282 | [+0.241, +0.323] |
| damaged fraction, per-row 4 bits | +0.984 | [+0.975, +0.993] |
| damaged fraction under matched noise, row 3 bits | +0.513 | [+0.486, +0.541] |

Class means in K2 (antipodal vs other):

| | n | D | bias | near_thr | FP false_act | R2_fp | damaged | dR2 | dfalse_act |
|---|---|---|---|---|---|---|---|---|---|
| antipodal | 166 | 0.500 | +0.010 | 0.65 | 0.000 | 0.85 | 1.00 | 0.102 | 0.273 |
| other | 741 | 0.427 | −0.080 | 0.13 | 0.146 | 0.71 | 0.46 | 0.072 | 0.047 |

Reading:
- In this setting, features that sit in antipodal pairs (D ≈ 0.5) are the ones that break first under RTN.
  At 4 bits essentially only they break. The mechanism is the one sketched post hoc in Phase 3 and
  registered in advance as H1's threshold-density reading:
  - They have no positive interference, so they learn a bias of about 0 and never false-activate in FP.
  - Their inactive pre-activation sits at the 0.05 detection threshold.
  - Any weight perturbation of the matched size, whether rounding or Gaussian, pushes it over.
- Crowded features carry a negative bias of about −0.08 that buffers them.
- The binary-label gap is driven mostly by the leakage condition. The threshold-free damage (dR2, leak energy)
  is also larger for antipodal features, but by a smaller margin: 0.102 vs 0.072 in R².

Limitations:
- One setting (n = 100, m = 20) at one sparsity, the only one where antipodal pairs and denser polytopes
  coexist.
- The detection threshold tau = 0.05 matters for the label.
- Per-tensor scaling shows a smaller gap (+0.28).

To our knowledge, this toy-model mechanism (zero-bias antipodal features are the most quantization-fragile) has
not been reported. That claim should be checked against the literature before the write-up.

---

## 2026-10-24 — Addendum to the Phase 3 results (wording for the write-up)

- H3 is a null for `norm`, but Duan's own predictor is confirmed. In the primary cell, peak_out_fp predicts
  less damage (Spearman with dR2 −0.36; oriented AUC 0.70), and so does p99_out_fp (−0.24; AUC 0.63).
  `norm` is not a stand-in for peak activation in this model: within-instance ρ(norm, p99_out_fp) is about
  0.4 on E. The write-up should say "Duan's peak-activation predictor holds; feature norm does not", not just
  "H3 null".
- The Phase 3b note "to our knowledge, not reported" still requires a literature check (CLAUDE.md) before it
  appears in any write-up or the README. Queued.

---

## 2026-10-24 — Phase 4 plan (head vs tail; owner away, proceeding under the standing instruction)

Goal: setting D (n = 200, m = 20, uniform importance, p_i = min(1, c (i+1)^(−alpha))). Does quantization
damage frequent (head) or rare (tail) features more, absolutely and relatively, in weak vs strong
superposition? The comparison is with Chang et al. (2025) and Wu et al. (2026). This is H4, exploratory and
two-sided (CLAUDE.md).

Infrastructure (done before this entry, tested; 242 tests pass):
- `LiuDecayAdam`. Liu, Liu & Gore's piecewise decay applied to W only, with a per-instance gamma. Tests
  confirm three things: gamma = 0 is Adam bit for bit; gamma = 0.01 equals torch.optim.AdamW on W; gamma < 0
  has a fixed point at unit norm and moves norms toward 1 from both sides.
- `power_law_frequencies`. A root-finder that asserts Σp = k. The clip binds only for alpha 1.5, k 5
  (c = 2.719).
- `build_eval_set_vec`. Per-feature frequencies. The inactive stratum is filled per feature, using
  forced-inactive batches where needed, with no common minimum. p_i = 1 features have no inactive stratum.
  The scalar path for A–C still reproduces stored Phase 2 rows exactly (max |diff| 0.0).
- The sweep handles vector frequencies.

Recipe (CLAUDE.md "Setting D recipe"):
- Training: alpha ∈ {1.0, 1.5}, k ∈ {2, 5}, 5 seeds; lr 1e-2, 2,000 warm-up steps, cosine to 0.05·lr,
  20k steps.
- TMS keeps: batch 1024, Xavier init, b = 0, U[0, 1].
- Deviations from Liu et al., logged here: activations U[0, 1] (theirs U[0, 2]); importance-weighted loss
  with I = 1 (theirs nn.MSELoss); Xavier init and b = 0 (theirs randn/sqrt(m) and random b); batch 1024
  (theirs 2048); density k ∈ {2, 5} (theirs E = 1).

Regime verification:
- Pilot at 4,000 steps, one seed per (alpha, k). gamma = +0.1 gave phi_half 0.40–0.65, not the weak
  regime's ≈ m/n = 0.10. gamma = −1.0 gave 0.62–1.00.
- Per CLAUDE.md ("sweep gamma and report the sweep, do not relabel"), a full-length FP-only sweep is
  running: gamma ∈ {−1.0, +0.1, +0.3, +1.0, +3.0}, one seed per (alpha, k), base seed 900.
- Selection rule, fixed BEFORE the sweep result was read:
  - WEAK gamma = the smallest gamma in the sweep whose phi_half ≤ 0.20 in all four (alpha, k) configs. If
    none qualifies, the sweep is extended upward (×3 steps, up to 30) before anything else.
  - STRONG gamma = −1.0, provided phi_half ≥ 0.90 in every config. Otherwise −1.0 is still used and every
    config with phi_half < 0.90 is reported as not strong, with no relabelling.
  - Each main run then trains 4 configs × 5 seeds = 20 instances per gamma (≈ 6–7 min each, within budget).

Then:
1. FP-only facts per regime (eligible counts, phi_half, D/bias/near_thr by frequency bucket).
2. H4 pre-registration, committed before the quantization sweep.
3. The sweep, without ablations: 6 bits × 3 granularities × (RTN + 5 noise seeds).
4. Pre-check written to the LOG.
5. Frozen analysis.

---

## 2026-10-24 — Phase 4 regime sweep result and a logged DEVIATION from the gamma rule

FP-only sweep (20k steps, one seed per (alpha, k); `results/setting_d/20261024-142930-gsweep-regime` and
`...-143621-gsweep-regime-ext`). phi_half for (alpha, k) = (1.0, 2), (1.0, 5), (1.5, 2), (1.5, 5):

| gamma | phi_half per config | represented features |
|---|---|---|
| −1.0 | 1.000, 0.595, 1.000, 0.980 | 200, 119, 200, 196 |
| +0.1 | 0.630, 0.440, 0.490, 0.380 | 126, 88, 98, 76 |
| +0.3 | 0.650, 0.435, 0.465, 0.355 | 130, 87, 93, 71 |
| +1.0 | 0.565, 0.410, 0.385, 0.320 | 113, 82, 77, 64 |
| +3.0 | 0.415, 0.310, 0.250, 0.240 | 83, 62, 50, 48 |
| +9 | 0.005, 0.000, 0.000, 0.005 | 1, 0, 0, 1 |
| +27 | 0.000, 0.000, 0.000, 0.000 | 0 |

- STRONG: gamma = −1.0, per the rule. The (alpha 1.0, k 5) config reaches only phi_half 0.595 and is reported
  as NOT strong; the other three are ≥ 0.98.
- WEAK: the rule ("smallest gamma with phi_half ≤ 0.20 in all configs") selects gamma = 9. But gamma = 9
  collapses the model (0–1 represented features). That is not the weak regime CLAUDE.md defines
  (phi_half ≈ m/n = 0.10), and it would leave nothing to test.
- **DEVIATION**, logged before the refinement sweep below was run: refine between 3 and 9 with gamma ∈ {4, 5,
  6, 7} (FP only, same protocol). WEAK = the gamma in {3, 4, 5, 6, 7, 9} whose phi_half, averaged over the
  four configs, is closest to 0.10, subject to phi_half ≥ 0.05 (≥ 10 represented features) in every config;
  ties go to the smaller gamma.
- A possible reason why our gammas must be much larger than Liu et al.'s 0.1 (a hypothesis, not tested):
  our loss sums over the 200 features while theirs (nn.MSELoss) averages over them, so for the same gamma
  their decay is about n times stronger relative to the loss gradient.

---

## 2026-10-24 — Phase 4 weak-gamma selection (applying the deviation rule)

Refinement sweep: `results/setting_d/20261024-143917-gsweep-regime-refine`, FP only. Mean phi_half over the
four configs, with the minimum in brackets: gamma 3: 0.304 (0.240); 4: 0.241 (0.190); 5: 0.199 (0.180);
6: 0.191 (0.170); 7: 0.156 (0.115); 9: 0.003 (0.000, fails the ≥ 0.05 floor).

**WEAK gamma = 7**, the closest to 0.10 among the gammas that pass the floor; 23–35 represented features per
config. The transition from gamma 7 (about 0.16) to gamma 9 (collapse) is steep. Our "weak" regime is
therefore 1.5× m/n rather than ≈ m/n, and that is reported next to every number. Main runs: gamma 7 (seed 11)
and gamma −1.0 (seed 111), 4 configs × 5 seeds each. `configs/setting_d.yaml` now lists gammas [7.0, −1.0].

---

## 2026-10-24 — Phase 4 pre-registration: H4 head vs tail (written BEFORE the quantization sweep)

Status: written after setting-D FP training and an FP-only evaluation (eligibility, geometry). No setting-D
quantization outcome exists, apart from an 8-bit timing smoke of one instance whose numbers were not read.
Decided by Claude Code under the owner's standing instruction; the owner can veto.

Models (FP facts):
- WEAK: gamma = 7, `results/setting_d/20261024-144457-g+7`, phi_half 0.107–0.197.
- STRONG: gamma = −1, `results/setting_d/20261024-145113-g-1`, phi_half 1.00 / 0.62 / 1.00 / 0.99 for (alpha, k)
  = (1, 2), (1, 5), (1.5, 2), (1.5, 5); the (1, 5) config is not strong.
- 20 instances per regime: (alpha, k) ∈ {1.0, 1.5} × {2, 5}, 5 seeds each.

Sweep: `configs/quant_sweep_d.yaml`, settings D_weak and D_strong; RTN plus 5 noise seeds × 3 granularities
× 6 bit widths; no ablations. Each regime takes about 12–13 min, over the ~10 min budget. The owner is away,
so this is logged here instead of asked.

FP-only facts that shape the plan (mean within-instance Spearman over eligible features):
- STRONG: 97–200 eligible features per instance. ρ(p, near_thr) = +0.94, ρ(p, bias) = +0.86,
  ρ(p, R2_fp) = +0.96, ρ(p, D) = −0.16. Head (top 10% of eligible): bias −0.13, near_thr 0.13. Tail (bottom
  50%): bias −0.55, near_thr 0.004.
- WEAK: 14–26 eligible features per instance, 90% of them with frequency rank < 20, so the "tail" of the
  eligible set is mid-frequency, not the true tail (selection effect, design note 8c). ρ(p, D) = +0.89,
  ρ(p, bias) = +0.94, ρ(p, norm) = +0.88, ρ(p, near_thr) = +0.50. Head: bias +0.20, near_thr 0.20. Tail:
  bias +0.01, near_thr 0.81.
- Frequency and geometry are confounded by construction, and in the strong regime frequency and threshold
  proximity are nearly collinear (0.94). A frequency effect is therefore reported as "frequency, or the
  geometry it determines here", never as frequency per se.

Primary test H4 (per regime; exploratory and two-sided, per CLAUDE.md):
- Cell: per-row, 3 bits, RTN, eligible features (norm > 0.5 and R2_fp ≥ 0.5). Same cell definition as
  Phase 3's primary.
- Statistic: within each instance, the Spearman ρ between frequency p_i and relative damage
  rel_damage_i = dR2_i / R2_fp_i (threshold-free). An instance is valid iff it has ≥ 10 eligible features and
  both variables take ≥ 5 distinct values. Summary: the mean of the valid ρ with a 95% t-interval. Two-sided
  one-sample t-test; Holm over the two regimes. A regime is summarised only if ≥ 16 of 20 instances are valid.
- Categories (the code is `tmsq.phase4.decide`):
  - "head loses a larger relative fraction": Holm p < 0.05 and mean ρ > 0.
  - "tail loses a larger relative fraction": Holm p < 0.05 and mean ρ < 0.
  - Either category is tagged "[small]" if |mean ρ| < 0.2.
  - "No detectable frequency effect": Holm p ≥ 0.05.
  - "Not testable (outcome dominated by the perturbation draw)": Holm p ≥ 0.05 and ICC_noise(rel_damage) <
    0.10.
  - "Not testable (too few valid instances)".

Out-of-sample test of the Phase 3 mechanism (registered now): Phase 3/3b found that features whose inactive
pre-activation sits near the threshold (high near_thr, bias ≈ 0) are the fragile ones. The FP facts above
then PREDICT the signs:
- STRONG: mean ρ > 0. The head is near the threshold; the tail sits far below it with bias −0.55.
- WEAK: mean ρ < 0. The tail of the eligible set is near the threshold; the head carries bias +0.20.
- The mechanism is "confirmed out of sample" iff both regimes reach Holm p < 0.05 with the predicted signs;
  "partly" iff exactly one does; otherwise "not confirmed". The primary H4 categories above do not depend on
  this prediction.

Secondary (estimation, BH within one family; none changes a category):
- Spearman of p with dLoss (absolute damage, mechanically ∝ p) and with dR2, at bits {3, 4, 5} × the three
  granularities.
- Partial ρ(p, rel_damage | D) and (… | bias).
- ρ of D, bias and near_thr with rel_damage.
- The noise control: single-draw ρ averaged over the 5 seeds.
- Head − tail per-instance differences (head = top 10%, tail = bottom 50% of eligible) of rel_damage and dLoss,
  with t-intervals.
- An FP geometry table per bucket.
- phi_half and the eligible fraction are reported next to every number.

Amendments after the advisor review of this draft (before committing; still before any outcome):
- Mechanical tilts of the two outcomes, in opposite directions:
  - dLoss tilts toward the head, because it is ∝ p.
  - rel_damage tilts toward the TAIL. With Var(x_i) ≈ p_i/3 and MSE_i = p_i·cond_mse_i + (1−p_i)·leak_i, a
    leak-energy change enters dR2 as about 3·Δleak/p_i. So a given leakage change is amplified about 19× for
    a weak-regime tail feature (p ≈ 0.035) relative to its head (p ≈ 0.67).
  - Consequence for the registered mechanism test: in the WEAK regime the Phase 3 prediction (ρ < 0) points
    the same way as the tilt, so a negative ρ there is overdetermined and not a clean test. In the STRONG
    regime the prediction (ρ > 0) runs AGAINST the tilt, so that half is the discriminating one. The
    write-up says this next to the mechanism result.
  - Wu et al.'s "relative retention" is accuracy-based and has no such normalization.
- Added secondaries:
  - ρ(p, dcond_mse): active-stratum error, not weighted by p.
  - The per-feature split of dR2 into its active term p·dcond_mse/Var and inactive term
    (1−p)·dleak_energy/Var, each correlated with p.
  - Partial ρ(p, rel_damage | near_thr).
- Strong-regime sensitivity: both summaries of the strong primary are repeated without the instances with
  phi_half < 0.9; on the FP facts these are the 5 instances of (alpha 1.0, k 5). The cell is summarised if
  ≥ 12 of the 15 are valid. If the two disagree, the primary decides and the sensitivity is reported beside it.
- 8-bit contingency: if a regime's bits8_pass is false, recompute the 8-bit median |dR2| over eligible
  features with p ≥ 0.05. If that is < 0.01, the regime is NOT stopped (the failure is the 1/p effect on rare
  features) and this is reported.
- In the weak regime the head bucket holds about 2 eligible features per instance. The head-minus-tail
  interval is a 2-feature comparison and is labelled as such.
- Implementation note: `sign_flip_p` (in `src/tmsq/phase3.py`) was rewritten as an exact meet-in-the-middle
  enumeration. n = 20 now takes 9 ms instead of about 1 s. It gives identical p-values on random inputs
  (max |diff| 0) and reproduces every stored Phase 3 E/K and Phase 3b sign-flip p exactly. This is a
  performance change to frozen code with verified-identical output.
- An end-to-end smoke test on synthetic sweep directories (`tests/test_phase4_e2e.py`) runs the frozen Phase 4
  entry point completely before any real input. 246 tests pass.

Stop rules: if a regime's `bits8_pass` is false or a granularity is non-monotone, that regime is STOP and
descriptive only. An incomplete sweep (20 instances, 6 bit widths, 5 noise seeds) aborts the analysis.
The analysis code (`src/tmsq/phase4.py`, `src/tmsq/experiments/phase4.py`) is committed before the sweep and
is not changed after the sweep's features are read. Any later change is reported with results before and
after it.

Comparison with the literature (for the write-up): Wu et al. find that head knowledge loses a larger relative
fraction (positive ρ here); Chang et al. find "long-tail" examples less affected (also positive ρ, with the
caveats in CLAUDE.md). The toy model can at most show which regime, or which geometry, reproduces which
pattern.

---

## 2026-10-24 — Phase 4 pre-registration committed

Committed as `a11e490` (2026-10-24T15:17:23-04:00), together with the frozen analysis code. No setting-D
quantization outcome had been read. The two sweeps (D_weak, then D_strong) were started right after, with the
registered config and no overrides.

---

## 2026-10-24 — Phase 4 sweeps and pre-check (logged BEFORE the H4 analysis is run)

Sweeps, run with `configs/quant_sweep_d.yaml` and no overrides; each has 432,000 feature rows and evaluation
took 266 s:
- D_weak: `results/quant_sweep/20261024-151741-D_weak`
- D_strong: `results/quant_sweep/20261024-152211-D_strong`
- The 12–13 min per regime estimated in the pre-registration was wrong. The real time was about 4.5 min,
  within budget.

Incident:
- Both sweep processes crashed AFTER `features.parquet`, `models.parquet` and `sanity.json` were written.
  The crash was in the Phase 2 figure code (`plots_quant.plot_metric_vs_bits` colours lines by S, and S is
  NaN for setting D: `KeyError: nan`). The saved data are complete; the completeness check below confirms it.
- Fix: `plots_quant._with_groups` labels setting-D lines by (alpha, k) instead of S, and the figures were
  regenerated from the saved `models.parquet` with `tmsq.experiments.replot`.
- This is figure code, not the frozen Phase 4 analysis code, and no number changed.

Pre-check (`python -m tmsq.experiments.phase4 --config configs/phase4.yaml --precheck-only`):
- Completeness: no problems in either regime (20 instances, 6 bit widths, 5 noise seeds).
- STRONG (gamma −1): passes.
  - bits8_pass true: median |dR2| at 8 bits 0.00056; max relative loss increase 0.0019.
  - Monotone in all three granularities. Mean dR2 (eligible) at 8/6/5/4/3/2 bits, per-row: 0.00003, 0.00065,
    0.0027, 0.0145, 0.112, 6.34.
  - ICC_noise at the primary cell: rel_damage 0.31 (sqrt 0.56), dR2 0.29, dLoss 0.62.
  - Eligible features per instance: 97–106 for the (·, k 5) configs, 194–200 for the (·, k 2) configs.
- WEAK (gamma 7): **STOP by the registered rule.**
  - bits8_pass is true (median |dR2| 0.00046), but the regime is NON-MONOTONE in all three granularities.
  - Mean dR2 (eligible) at 8/6/5/4/3/2 bits:

    | Granularity | 8 | 6 | 5 | 4 | 3 | 2 |
    |---|---|---|---|---|---|---|
    | per-row | 0.00003 | 0.00014 | −0.00043 | −0.00004 | 0.00225 | **−0.148** |
    | per-column | −0.00001 | −0.00015 | −0.00002 | −0.0017 | −0.0032 | −0.152 |
    | per-tensor | 0.0 | 0.0003 | 0.0001 | −0.0007 | −0.017 | −0.122 |

  - At 2 bits, quantization IMPROVES the mean R² of eligible features by about 0.15.
  - ICC_noise(rel_damage) = 0.047, below the 0.10 testability floor. Eligible features per instance: 14–26.
- Consequences, under the rules written before this was seen:
  - The weak regime is descriptive only; its H4 primary reads "STOP (sanity failed; descriptive only)".
  - H4 is testable only in the strong regime. Holm stays over the two regimes as registered (conservative for
    strong).
  - The out-of-sample test of the Phase 3 mechanism can reach at most "partly", because its weak half cannot
    be evaluated. The strong half was registered as the discriminating one.
- Surprise, to be reported and diagnosed post hoc AFTER the frozen analysis: "quantization improves the
  weak-regime model". A candidate explanation, not yet checked: with gamma = 7 the FP weights minimise loss +
  decay, not the reconstruction loss, so features are shrunk (head norm ≈ 0.63). Any perturbation adds
  E||δ_i||² > 0 to the self-gain ||W_i + δ_i||², which partly undoes the shrinkage. If so, the matched-noise
  control should show the same improvement.

---

## 2026-10-24 — Phase 4 results: H4 head vs tail (setting D)

Goal: test H4 under the pre-registration committed as `a11e490`, using the frozen analysis code.
- Analysis run: `results/phase4/20261024-152919-H4`.
- Re-run after a cosmetic figure fix: `results/phase4/20261024-153307-H4`. primary.json, precheck.json and all
  three CSVs are equal between the two runs.

Config:
- Setting D, n = 200, m = 20, uniform importance; p_i = min(1, c (i+1)^-alpha) for alpha in {1.0, 1.5} and
  k in {2, 5}, 5 seeds each.
- WEAK: gamma = 7, phi_half 0.107-0.197. STRONG: gamma = -1, phi_half 1.00 / 0.62 / 1.00 / 0.99.
- RTN plus 5 matched-noise seeds x 3 granularities x 6 bit widths.
- The per-feature stratified eval set: 2,000 active and 20,000 inactive rows per feature.

### Primary (registered): Spearman(p_i, rel_damage_i) within instance, per-row, 3 bits, RTN, eligible features

| Regime | Mean rho [95% t] | n | p (t) | p (sign-flip) | Cluster bootstrap | Holm p | sqrt(ICC) | Category |
|---|---|---|---|---|---|---|---|---|
| strong | -0.212 [-0.273, -0.151] | 20 | 6.6e-7 | 1.3e-5 | [-0.267, -0.158] | 1.3e-6 | 0.56 | tail loses a larger relative fraction |
| weak | +0.002 [-0.146, +0.150] | 20 | 0.98 | - | - | - | 0.22 | STOP (pre-check), descriptive only |

- The strong effect size is BORDERLINE. |rho| = 0.212 only just clears the registered 0.2 "[small]" tag.
- The registered strong-regime sensitivity (phi_half >= 0.9, 15 instances) gives -0.174 [-0.243, -0.105],
  p = 9.1e-5. The sign agrees, but the sensitivity alone would carry the "[small]" tag. The primary decides.
- Holm stayed over both regimes after the weak STOP, as registered; this is conservative for strong.
- Registered out-of-sample test of the Phase 3 mechanism: **not confirmed**. The strong half predicted
  rho > 0 and the observed value is -0.21; the weak half cannot be evaluated (STOP).

### Registered secondaries (strong, per-row, 3 bits unless stated; BH within the family)

- rho(p, dLoss) = +0.72: absolute damage sits in the head (mechanically proportional to p).
- **rho(p, dR2) = -0.03 [-0.09, +0.03], q = 0.42: no frequency dependence in dR2 itself.**
- Split of dR2:
  - Active term: rho(p, dR2_active_term) = +0.38, with rho(p, dcond_mse) = +0.37. Toward the head.
  - Inactive term: rho(p, dR2_inactive_term) = -0.20. Toward the tail, via the 1/Var(x) ~ 3/p factor.
- Partial rho(p, rel_damage | .): near_thr -0.18, D -0.21, bias -0.13.
- Single-variable rho with rel_damage: near_thr -0.16, bias -0.17, D +0.04 (q = 0.08).
- Noise control (single-draw rho averaged over 5 seeds): -0.235 [-0.27, -0.20]. Same as RTN, so the effect is
  not specific to rounding.
- By granularity, at 3 / 4 / 5 bits:
  - tensor: -0.25 / -0.12 / -0.02
  - column: -0.24 / -0.08 / -0.01
  - row: -0.21 / -0.08 / -0.06
- Head minus tail (top 10% / bottom 50% of eligible features), per-row 3 bits:
  - rel_damage: head 0.102, tail 0.212, difference -0.110 [-0.146, -0.073].
  - dLoss: head 0.0041, tail 0.0002, difference +0.0040 [+0.0025, +0.0055].
  - Weak regime: the head bucket holds about 2 features per instance.

**Reading (the decomposition belongs with the headline).** Under the pre-registered relative-retention
outcome the tail loses a larger fraction. This is a property of the normalization, not of the perturbation
response:
- dR2 is null with frequency, because its active term (toward the head) and its inactive term (toward the
  tail through 1/Var) cancel.
- The net negative sign of rel_damage = dR2 / R2_fp comes from the denominator: R2_fp is 0.61 for the tail and
  0.94 for the head, with rho(p, R2_fp) = +0.96.
- Absolute damage (dLoss) and raw leakage (post hoc, below) both sit in the head.
- The perturbation itself does not depend on frequency (post hoc: rho(p, dW) = +0.03,
  rho(p, d_norm2) = -0.01).
- No clean comparison with Wu et al. is possible: their retention is accuracy-based, with no normalization by
  feature variance, and their head/tail are entity-popularity groups. Chang et al.'s "long-tail" is not a
  frequency. What the toy model does show: which bucket "loses more" flips with the outcome normalization.

### Post hoc (NOT registered; `results/phase4_posthoc/diagnostics_20261024.txt`)

(B) Raw leakage in the strong regime (per-row, 3 bits):
- rho(p, dleak_energy) = +0.52 and rho(p, dfalse_act) = +0.52.
- rho(near_thr, dfalse_act) = +0.55.
- dfalse_act is 0.080 in the head vs 0.003 in the tail.
- The tail's dR2 of 0.125 comes from a leak energy of only 0.0002 divided by Var(x) ~ 0.0012.

(D) The median version (per-instance median(head) - median(tail) of rel_damage) is robust to the rel_damage
outliers, which reach about 6: per-row -0.067 [-0.083, -0.051], tensor -0.121, column -0.048.

**Error in the pre-registration (owned).** The mechanism test used the wrong outcome:
- The Phase 3 mechanism is a leakage mechanism, and design note 12 maps leakage to dfalse_act. The
  out-of-sample mechanism test should have used dfalse_act, not rel_damage.
- The pre-registration argued that the strong half was "discriminating" because it ran against the 1/p tilt.
  That argument was wrong: a test on rel_damage discriminates only if the mechanism effect is larger than the
  normalization tilt, and it was not.
- The registered verdict "not confirmed" STANDS. The post-hoc leakage reading is consistent with the mechanism
  but is NOT a confirmation.
- near_thr -> dfalse_act is also close to definitional: a feature whose FP inactive pre-activation sits within
  0.05 of tau flips under almost any perturbation. The only informative part is that this transfers to
  setting D, where frequency and near_thr are collinear (rho = 0.94).

### Weak regime: diagnosis of "quantization improves the model" (post hoc; corrects the pre-check entry)

- The candidate explanation logged in the pre-check (E||delta||^2 undoes the shrinkage, so matched noise should
  improve the model too) is **falsified by its own stated test**.
- At 2 bits, matched noise RAISES the loss and RTN LOWERS it (mean relative loss change):

  | Granularity | Noise | RTN |
  |---|---|---|
  | column | +1.0% | -16.7% |
  | row | +1.9% | -20.0% |
  | tensor | +7.6% | -10.7% |

- d_norm2 of eligible features is +0.25 under RTN and +0.06 under noise.
- Corrected mechanism (verified):
  - With gamma = 7 the decay shrinks W. Eligible features have FP self-gain ||W_i||^2 = 0.36 and bias +0.04,
    so the clean response at x = 1 is 0.40, not about 1.
  - 2-bit RTN is ternary. Per-row, 17.9 of the 20 entries of an eligible column get the code +-s.
  - Those entries contribute +0.262 to d_norm2; the entries rounded to 0 contribute -0.011. The net +0.251
    matches the stored d_norm2, and all 392 eligible columns have d_norm2 > 0.
  - A larger self-gain means better reconstruction: per-instance rho(d_norm2, -dR2) = +0.49.
- **Limitation of the setup, not a head-vs-tail finding:**
  - gamma = 7 was forced, because gamma = 0.1 is not weak at n = 200, m = 20.
  - The resulting model is far from the reconstruction optimum, so "damage relative to FP" is ill-defined in
    this regime.
  - Phase 4 therefore answers H4 for the strong regime only.

### Literature check (antipodal fragility, Phase 3b)

The check was a workflow with 3 search angles plus a synthesis. It found no source that reports the finding;
its premise is known:
- TMS: negative bias "can convert small positive interferences into essentially being negative interferences".
- Chen et al. 2023 (arXiv 2310.06301), Table A.1: b* = 0 for the 4-gon and increasingly negative values for
  the 5- and 6-gon.
- Tlaie 2024: "the bias vector is not negative until larger sparsities (when the antipodal structure is lost)".

The closest robustness results are model-level and input-space:
- TMS: adversarial vulnerability rises with superposition.
- Stevinson et al. 2025 (arXiv 2510.11709).
- Gorton & Lewis report a vulnerability "dip" at antipodal superposition that they leave unexplained. This goes
  the other way for input attacks, in a different perturbation space.

Sentence suggested for the write-up, still to be checked by the owner: "It is known that antipodal features
need no negative bias while crowded polytopes learn increasingly negative ones (Elhage et al. 2022; Chen et
al. 2023). To our knowledge, this work is the first to show that this makes antipodal features the most
fragile under post-training weight quantization and matched weight noise. Prior robustness results that link
superposition to fragility concern input-space adversarial attacks at the model level."

To check by hand before publishing:
- The agents' quotes from Chen et al. 2023 and Stevinson et al. 2025.
- The title of Duan (2026, arXiv 2606.03002). The agent reports the v1 title "How Quantization Changes
  Interpretable Features: A Sparse Autoencoder Analysis of Language Models", while CLAUDE.md uses "Perplexity
  Can Miss SAE Feature Damage Under Quantization", perhaps a later version. CLAUDE.md is not changed without
  checking.

### Post-freeze changes (cosmetic only)

- `src/tmsq/experiments/phase4.py`: the figures now label the stopped regime "STOP (descriptive only)", and the
  head-tail figure has wider panel spacing (its y-labels overlapped).
- Before and after: all numerical outputs are identical (see above).
- The setting-D sweep figures were regenerated from the saved data (`quant_D_*`).

### Surprises

1. The weak regime improves under 2-bit RTN, explained above.
2. The registered relative outcome and raw leakage point in opposite directions.
3. The mechanism-test outcome was chosen wrongly in the pre-registration.

### Next

Phase 4 is closed. Next per PROMPTS.md: Phase 5. The owner should review the H4 framing and the weak-regime
limitation before the write-up.

---

## 2026-10-24 — Phase 4 descriptive table (PROMPTS.md: every bit width, per config, mean ± std over seeds)

Command: `python -m tmsq.experiments.phase4_table --config configs/phase4.yaml`. Output:
`results/phase4/20261024-153800-table/head_tail_by_config.csv`, with a per-instance version alongside. It covers
all 6 bit widths × 3 granularities × 2 bucket definitions and is descriptive only; it is not a test.

STRONG regime, per-row, eligible-feature buckets. Mean ± std over 5 seeds of rel_damage, head vs tail:

| alpha, k | phi_half | 4 bits head | 4 bits tail | 3 bits head | 3 bits tail | 2 bits head | 2 bits tail |
|---|---|---|---|---|---|---|---|
| 1.0, 2 | 1.00 | 0.011 ± 0.002 | 0.018 ± 0.003 | 0.084 ± 0.014 | 0.145 ± 0.023 | 1.85 ± 0.21 | 9.5 ± 0.9 |
| 1.0, 5 | 0.62 | 0.016 ± 0.003 | 0.022 ± 0.004 | 0.088 ± 0.016 | 0.199 ± 0.031 | 1.93 ± 0.39 | 8.5 ± 1.4 |
| 1.5, 2 | 1.00 | 0.012 ± 0.003 | 0.027 ± 0.009 | 0.091 ± 0.016 | 0.219 ± 0.066 | 2.65 ± 0.79 | 19.8 ± 2.3 |
| 1.5, 5 | 0.99 | 0.022 ± 0.006 | 0.034 ± 0.012 | 0.147 ± 0.040 | 0.283 ± 0.094 | 2.30 ± 0.84 | 12.4 ± 2.6 |

- At 2 bits, rel_damage > 1, which means R2 goes negative (worse than predicting the mean).
- dLoss runs the other way in every config and at every bit width ≤ 4. Example at 3 bits, (1.0, 2): head
  0.0013, tail 0.0001.
- The direction is the same in all four configs and at every bit width ≤ 4: the tail has larger relative
  damage and the head has larger absolute damage. At 6 and 8 bits both are ≈ 0.

Selection effect (bucket over ALL 200 features, 3 bits, per-row; eligible fraction of the bucket):
- STRONG, k = 2: the head is 100% eligible and the tail 97–98%. The eligible "tail" is the true tail.
- STRONG, k = 5: the head is 100% eligible but only 4.0% (alpha 1.0) and 5.6% (alpha 1.5) of the true tail is
  eligible. The H4 "tail of eligible features" there is mostly mid-frequency features.
- WEAK: the head is 76–95% eligible and 0% of the true tail is eligible.

The write-up must state these fractions next to the H4 result, per design note 8c.

---

## 2026-10-24 — Phase 5 plan and pre-registration: geometry-aware quantization at a fixed bit budget

Status: written BEFORE any Phase 5 code or outcome exists. Owner away; decided by Claude Code under the
standing instruction and logged for review (the owner can veto).

Question (PROMPTS.md Phase 5): at the same average bits per weight (4.0), can a geometry-aware bit
allocation or rounding rule reduce damage relative to uniform 4-bit RTN?

### Models
- Setting C, a NEW set "P5": `configs/rep_dims_p5.yaml`. It is identical to `configs/rep_dims.yaml` except
  for seed 13, 10 seeds and the grid indices [3, 4, 5, 7], i.e. S = 0.7488, 0.8415, 0.9000, 0.9602. The set is
  new so that the arms motivated by Phase 3/3b are not scored on the models that motivated them.
- Setting D, strong regime (gamma = −1), the 20 Phase 4 models (`results/setting_d/20261024-145113-g-1`).
  The weak regime is excluded: Phase 4 showed its FP model is far from the reconstruction optimum, so 2-bit
  RTN can even improve it.

### Arms

All arms use per-column scales s_i = max|W_i| / qmax(b_i) with the eps guard of design note 2, symmetric RTN,
and tied weights. b stays in FP32.
- (a) U4: every column at 4 bits.
- Allocation arms (b) to (d) share one split. The represented columns (norm_i > 0.5), R of them, are
  ranked by the arm's key. The first floor(R/2) get 5 bits and the last floor(R/2) get 3 bits. If R is odd,
  the middle column gets 4 bits. Unrepresented columns get 4 bits. The average is exactly 4.0 bits per
  weight, and the scale overhead (one FP scale per column) is identical across arms.
  - (b) MARGIN-LOW, as PROMPTS.md specifies: smallest margin_i first. margin_i = −b_i − max(0,
    max_overlap_i); margin_tau has the same ranking.
  - (b2) MARGIN-HIGH: largest margin_i first. Registered because Phase 3 found the opposite sign of H1
    (ρ(margin_tau, dfalse_act) = +0.49 on K).
  - (b3) NEAR-THR: highest near_thr_i first, the threshold density behind the Phase 3/3b mechanism.
    near_thr is computed on the CALIBRATION set (inactive natural samples), never on the eval set.
  - (c) NORM: largest norm_i first (magnitude heuristic).
  - (d) RANDOM: 20 seeded random rankings per instance. The arm's value is the mean over the 20 draws;
    their spread is reported.
- Rounding arms, at 4 bits everywhere (same codebook as U4):
  - (e) LEAK-ROUND: per column, greedily choose round-down or round-up for single entries, minimising the
    calibration leak energy sum_j I_j · mean_s[ReLU(pre_sj)² · 1(x_sj = 0)].
  - (e2) LOSS-ROUND (secondary): the same algorithm minimising the calibration weighted MSE. It separates "leak
    specific" from generic loss-aware rounding.
  - Greedy algorithm, for both (e) and (e2):
    - Start from the RTN codes.
    - Passes over the represented columns, in index order.
    - In each column, evaluate every single-entry flip to the other neighbouring grid point in parallel.
      Accept the best flip only if it strictly lowers the objective. Repeat up to 5 times per column.
    - 2 passes.
    - Every code stays within one grid step of W/s and inside ±qmax.
    - Pre-activation updates are exact rank-1 updates (tied weights: changing W_ki by δ changes h_k by δ x_i
      and pre_i by δ h_k + δ x_i (W_ki + δ)).
- Reference only (not compared): uniform 3 and 5 bits.

### Data
- Calibration set: 8,192 natural-distribution samples, with a seed disjoint from every eval-set seed. It is
  used only by (b3), (e) and (e2).
- Evaluation: THE stratified eval set of design note 4, with the same seeds as Phase 2/4, so FP and every arm
  see identical inputs.

### Outcomes (per model)
- Primary: the relative loss increase r = loss_q / loss_fp − 1, with the loss computed from the strata as in
  Phase 2.
- Secondary: the composite survival fraction among eligible features, the worst-feature dR2 (maximum over
  eligible features) and the mean dR2.

### Primary test
- Cell: setting C, S = 0.7488 (the Phase 3 primary cell), 10 P5 instances.
- For each of (b), (b3) and (e), the per-instance paired difference d_k = r_arm − r_U4. Summary: the mean
  with a 95% t-interval; two-sided one-sample t-test; Holm over the 3 arms.
- Categories:
  - "better than uniform": Holm p < 0.05 and mean d < 0.
  - "worse than uniform": Holm p < 0.05 and mean d > 0.
  - "no detectable difference": otherwise.
- The relative size, mean d / mean r_U4, is reported next to each category.

### Secondary (estimation only; BH within one family; none changes a category)
- All arms vs U4 and the allocation arms vs RANDOM, at every S of setting C and on setting D strong.
- The three secondary outcomes.
- The noise-free FP sanity check: U4 here must reproduce the Phase 2 per-column 4-bit RTN loss to float
  precision on the same model type. On P5 models this is checked by recomputing through the Phase 2
  `apply_quant` path.

### Predictions registered now (not tests)
- Random 3/5 mixing should be WORSE than U4. With comparable column scales, the mean squared rounding error
  of a 3/5 mix is about (4^-3 + 4^-5)/2 ÷ 4^-4 ≈ 2.1× that of U4, so a gain over U4 needs strongly
  heterogeneous column sensitivity.
- Given the Phase 3 H1 result, (b) MARGIN-LOW protects the crowded, buffered columns, so it is expected to
  do no better than RANDOM. (b2) and (b3) are expected to do better than (b).
- (e) may trade active-stratum accuracy for leakage and lose on total loss; (e2) is the control for that.

### Stop rules
- If U4 fails the reproduction check, STOP and report.
- If the greedy objective ever increases on the calibration set, that is a bug: STOP.
- An incomplete run (instances or arms missing) aborts the analysis.

### Budget
- P5 training: about 5–8 min.
- Evaluation: about 30 quantized models per instance, plus greedy rounding (rank-1 updates; estimated
  < 30 s per instance).
- Each entry point should stay under 10 min. Timing is measured on one instance first.

### Limitation stated in advance
Per-column (per-feature) bit allocation has no direct analog in real models, where features are not
axis-aligned (design note 6). Phase 5 is a mechanism demonstration in the toy model, not a deployable method.

### Amendments after the advisor review of this draft (before committing; still before any outcome)

1. Ranking edge cases:
   - A NaN key, which happens only for near_thr_cal of p = 1 features in setting D (no inactive samples), takes
     that column out of the ranked pool, and the column gets 4 bits.
   - Ties are broken by a stable sort on the feature index.
   - In the greedy rounding, an entry already on the grid, such as the column maximum with code ±qmax, is
     never flipped. Every other entry moves only between floor(W/s) and ceil(W/s), both inside ±qmax.
2. Seeds:
   - The calibration generator seed is `cal_seed(train_seed, instance) = 7,000,000,000,003 + 1,000,033 ·
     train_seed + 7919 · instance`.
   - `tests/test_seed_independence.py::test_phase5_streams_are_new_and_calibration_is_disjoint` asserts that
     these seeds are disjoint from every init, training, eval and noise stream of E, K, K2, P5 and D-strong.
     It also asserts that the P5 training and eval streams are new.
   - The random arm's draw d uses `numpy.random.default_rng([5150005, train_seed, instance, d])`.
3. Two registered families, each with its own claim:
   - **Family 1** (arms vs U4: (b), (b3), (e); Holm over 3) answers the deployable question: does a
     geometry-aware rule beat uniform 4-bit at the same budget?
   - **Family 2** (allocation vs RANDOM at the same 3/5 mix: (b), (b2), (b3), (c); Holm over 4) answers the
     mechanistic question: does geometry tell us which columns to protect, net of the penalty for mixing
     3 and 5 bits?
   - Both families are tested in the primary cell, with the same categories, worded "better/worse than random"
     for Family 2.
   - The write-up attributes each claim only to its family.
4. Registered mechanism checks (descriptive; t-intervals over instances):
   - Allocation arms: per instance, the mean dR2 of eligible features in 5-bit columns minus that in 3-bit
     columns. For RANDOM, this is computed per draw and then averaged.
   - Rounding arms: the accepted-flip count, and the calibration objective J_final / J0.
   - Rounding arms: on the eval set, the changes relative to U4 of the active component sum_i p_i·dcond_mse_i
     and of the inactive component sum_i (1 − p_i)·dleak_energy_i, reported separately.
5. The exact sign-flip p is reported beside every t-test p.
6. Code freeze:
   - Frozen with this commit: `src/tmsq/phase5.py`, `src/tmsq/experiments/phase5.py` (runner) and
     `src/tmsq/experiments/phase5_analysis.py`.
   - Tests: `tests/test_phase5.py` (13 tests) checks four things. Mixed RTN with uniform bits equals the
     Phase 2 per-column RTN bit for bit. The rank-1 flip deltas and the incremental state equal brute-force
     recomputation, for both objectives. The greedy objective never increases, and its tracked value equals
     the recomputed one. Codes stay within one grid step and inside ±qmax, and zero passes reproduce RTN.
   - `tests/test_phase5_e2e.py` runs the analysis end to end on synthetic runner output.
   - 261 tests pass.
7. Facts gathered before freezing. None of them is an arm outcome.
   - P5 training (`results/rep_dims_p5/20261024-154441`, 277 s): D* = m/||W||_F² ≈ 0.43 / 0.29 / 0.21 / 0.155
     at the four S values; plateau ratios 1.000 / 1.000 / 0.999 / 0.998.
   - Timing smoke on 1 setting-D instance (`results/phase5/20261024-154755-D_strong-timing`; its arm outcomes
     were not read):
     - 36.6 s per instance, of which greedy rounding takes 16 s (leak) and 14 s (loss).
     - The D-strong runner is therefore about 12 min, over the 10-min budget. The owner is away, so this is
       logged instead of asked.
     - Sanity: U4 is bitwise equal to the Phase 2 `apply_quant` per-column 4-bit result; U4's loss equals the
       Phase 4 sweep's stored per-column 4-bit loss (relative difference 0.0); greedy tracking error 4e-11.
     - The greedy accepted 1,656 (leak) and 1,434 (loss) flips over about 200 columns. Many columns reach the
       registered cap of 5 flips per pass, so the result is a budget-limited greedy, not a converged one; this
       is stated with the results.

---

## 2026-10-24 — Phase 5 runs and pre-check (logged BEFORE the analysis is run)

Runs, made with the frozen runner (`d6abce7`) and `configs/phase5.yaml`, with no overrides:

| Run | Directory | Instances | Time |
|---|---|---|---|
| C_P5 | `results/phase5/20261024-155157-C_P5` | 40 | 283 s |
| D_strong | `results/phase5/20261024-155642-D_strong` | 20 | 276 s |

The timing smoke had estimated about 12 min for D; the real run took 4.6 min, within budget.

Pre-check (`python -m tmsq.experiments.phase5_analysis --config configs/phase5.yaml --precheck-only`):
- Completeness: no problems in either run. Every instance has U4, U3, U5, the 4 allocation arms,
  20 random draws and the 2 rounding arms, and C_P5 has 10 instances per S.
- Sanity: both runs pass.
  - U4 is bitwise equal to the Phase 2 per-column 4-bit RTN.
  - D: U4's loss equals the Phase 4 sweep's stored loss (relative difference 0.0).
  - The greedy objective never increased. The maximum tracking error is 4.9e-9 for C and 7.1e-10 for D.
  - stop = false in both runs.
- Greedy flips, mean per instance:

  | Setting | leak_round | loss_round |
  |---|---|---|
  | C | 608 | 372 |
  | D | 1,464 | 1,216 |

- No stop rule fires, so the analysis runs as frozen.

---

## 2026-10-24 — Phase 5 results: geometry-aware allocation and rounding at 4.0 bits per weight

Analysis: `results/phase5/20261024-160152-analysis`, from the frozen code `d6abce7`; the pre-check is
`fec1998`.
- Setting C P5: 40 instances, 10 per S.
- Setting D strong (gamma −1): 20 instances; phi_half 1.00 / 0.62 / 1.00 / 0.99 per (alpha, k).
- r = loss_q / loss_fp − 1 on the stratified eval set. "d" is the per-instance paired difference.
- At n = 10, the sign-flip p of 0.00195 is the floor of the exact test (2 of 1,024 patterns). It means the t-test
  and the sign-flip test agree; it does not mean a tinier p.

### Headline (setting C)

At the same 4.0-bit budget, **no geometry-aware allocation and no leak-preserving rounding beats uniform
4-bit per-column RTN**. In the primary cell, geometry does not beat random allocation either (Family 2 is null
after Holm). The registered prediction held almost exactly: random 3/5 mixing raises r by +0.0173, 1.9× U4's
own loss increase, against the predicted ≈ 2.1×.

The one arm better than uniform is **loss_round**. It is the registered SECONDARY control (e2), a generic
loss-aware greedy rounding, not the PROMPTS.md method.

### Family 1 (registered): arm vs U4

Primary cell: C, S = 0.7488. Mean r(U4) = 0.0089 (U3 0.0526, U5 0.0020).

| Arm | Mean d [95% t] | Holm p | d / r(U4) | Category |
|---|---|---|---|---|
| (b) margin_low | +0.0187 [+0.0165, +0.0210] | 4.9e-8 | +2.1× | worse than uniform |
| (b3) near_thr | +0.0156 [+0.0137, +0.0175] | 4.9e-8 | +1.75× | worse than uniform |
| (e) leak_round | +0.0149 [+0.0101, +0.0197] | 6.0e-5 | +1.7× | worse than uniform |

### Family 2 (registered): allocation vs RANDOM (primary cell, Holm over 4)

| Arm | Mean d [95% t] | Raw p | Holm p | Category |
|---|---|---|---|---|
| margin_low | +0.0014 [−0.0003, +0.0032] | 0.09 | 0.28 | no detectable difference |
| margin_high | −0.0013 [−0.0031, +0.0006] | 0.16 | 0.28 | no detectable difference |
| near_thr | −0.0017 [−0.0033, −0.0002] | 0.035 | 0.14 | no detectable difference |
| norm | −0.0011 [−0.0024, +0.0002] | 0.10 | 0.28 | no detectable difference |

### Secondaries (BH over the full family)

loss_round vs U4:
- Setting C: −0.0048, −0.0030, −0.0019 and −0.0023 at S = 0.7488, 0.8415, 0.9 and 0.9602 (all q ≤ 0.005).
- At the primary cell this puts r(loss_round) = 0.0041, halfway between U4 (0.0089) and U5 (0.0020).
- Generalization check: on the calibration set it removed about 64% of the quantization-induced loss
  increase (J_final / J0 = 0.994); on the eval set it removed about 54%.
- It improves the mean dR2 (−0.0071) but NOT composite_frac (+0.013, q = 0.11) or worst_dR2 (+0.002,
  q = 0.49). The gain is spread over many features; it does not rescue the fragile ones.
- Setting D: the pooled effect is null (−0.004 [−0.015, +0.007]), and it does not transfer by config (post hoc,
  below). The loss_round result is therefore stated for setting C only.

leak_round:
- Worse than uniform everywhere, and far worse as S grows: +0.054, +0.105 and +0.250 at S = 0.84, 0.9 and
  0.96; +0.39 in D.
- The registered trade-off prediction is confirmed by the registered split, relative to U4 at 0.7488: the
  active component rises by +0.12 while the inactive component falls by −0.056.
- It cuts calibration leakage strongly (J_final / J0 = 0.93 / 0.60 / 0.35 / 0.24 across S) by sacrificing
  active-stratum accuracy.

Allocation vs random at the non-primary C cells:
- Several small advantages are BH-significant: norm −0.0036 at S = 0.8415 and at 0.9602; near_thr −0.0057 at
  0.9602; margin_high −0.0021 at 0.8415.
- All are ≲ 0.006, against a mix penalty of 0.017–0.069. Geometry carries a little information about which
  columns to protect, never enough to pay for the mix.
- margin_low is never better than random. It is worse at 0.9602 (+0.0041, q = 0.03), as predicted from
  Phase 3's opposite-sign H1. The one exception to "margin_high ≥ margin_low": at S = 0.9 margin_high is the
  worse of the two.

Mechanism check (registered), dR2 of the 5-bit half minus the 3-bit half:
- Negative for every arm, so protected columns are protected.
- At 0.7488: near_thr −0.018 vs random −0.029. near_thr's 3-bit half (the buffered columns) is damaged less,
  consistent with the Phase 3 mechanism, but still not enough to beat the mix penalty.

### Setting D: the allocation results are frequency allocation (post hoc)

Post hoc, NOT registered: `results/phase5_posthoc/d_frequency_confound_20261024.txt` and the script beside it.
- Pooled over D strong, near_thr beats U4 (−0.0186 [−0.034, −0.003], q = 0.03). Against random: near_thr
  −0.19, margin_low −0.08, margin_high +0.07, norm +0.18.
- FP-only Spearman of each arm's ranking with frequency p over the represented columns:

  | Arm | ρ(ranking, p) | Range over instances |
  |---|---|---|
  | near_thr | +0.92 | 0.89–0.98 |
  | margin_low | +0.49 | |
  | margin_high | −0.49 | |
  | norm | −0.02 | −0.44 to +0.54 |

- A frequency-ranked arm (highest p gets 5 bits), run through the frozen `rtn_mixed` + `evaluate_q`:
  - vs U4: −0.0200 [−0.035, −0.005]
  - vs near_thr: −0.0014 [−0.003, +0.001], not distinguishable.
- So **the D near_thr "win" is frequency allocation**. Protecting frequent features reduces the total loss
  because dLoss ∝ p; it cannot be attributed to threshold geometry. Its sign is also heterogeneous:

  | alpha, k | near_thr r | U4 r |
  |---|---|---|
  | 1.5, 2 | 0.076 | 0.129 |
  | 1.5, 5 | 0.039 | 0.076 |
  | 1.0, 2 | 0.049 | 0.038 |
  | 1.0, 5 | 0.029 | 0.024 |

  It wins only in the alpha = 1.5 configs. The frequency arm shows the same pattern. Not a result to lead with.
- loss_round in D: in (alpha 1.0, k 2) all 5 instances are WORSE than U4 (+0.016 to +0.027), even though the
  calibration objective fell 12%. This is systematic, not one outlier. The likely cause is calibration
  overfitting with rare features (8,192 natural samples, about 1,200 accepted flips). In (1.5, 5) all 5 are
  better (−0.016 to −0.039).

### Answer to the PROMPTS.md question

In the toy model, at an equal budget, uniform per-column 4-bit RTN is hard to beat by bit allocation. The
penalty for mixing 3- and 5-bit columns (about 2× the loss increase) is larger than anything that geometry
ranking or frequency ranking recovers, except where frequency is very skewed (setting D, alpha = 1.5).
- Leak-specific rounding fails because it trades active accuracy for leakage.
- Generic loss-aware rounding helps in setting C (about half of the quantization loss removed), but it does
  not rescue the most fragile features, and in setting D it can overfit the calibration set.
- Per-column allocation has no direct analog in real models (design note 6), so none of this is a deployable
  method.

Figure: `figures/phase5_arms_vs_u4.png` (paired differences vs U4, all cells).

### Surprises / notes

- The 1.9× mix penalty matched the a-priori prediction, so the allocation question was effectively decided by
  convexity.
- The D allocation result came from the frequency confound. It was caught post hoc and is stated as such.
- The greedy rounding is budget-limited (cap of 5 flips per column per pass), not converged.

Next: Phase 6 (write-up). The owner should review the Phase 4 framing and the Phase 5 D confound before the
blog post is drafted.

---

## 2026-10-24 — Post-hoc scripts saved for traceability

- `notes/posthoc/phase4_diagnostics.py` reproduces every number in `results/phase4_posthoc/diagnostics_20261024.txt`.
  The rerun output (`diagnostics_rerun_20261024.txt`) is identical. The archived file also contains the traceback
  of a failed first attempt of block (C): the ad hoc script passed `W[k]` instead of `W[k][None]` to `rtn`. That
  attempt produced no numbers.
- `notes/posthoc/phase5_d_frequency_confound.py` produced
  `results/phase5_posthoc/d_frequency_confound_20261024.txt`.
- Both scripts are post hoc and labelled as such; neither is part of a frozen analysis.

---

## 2026-10-24 — Phase 6 (draft): blog post and README, for the owner's review

Done:
- `writeup/post.md`: a LessWrong-style draft. Every number traces to a `results/` directory listed in the
  provenance table. "To our knowledge" is used for the antipodal-fragility claim.
  - Sections: TL;DR, motivation, setup, replication, sweep, the antipodal finding (Phase 3/3b), head vs tail
    (Phase 4), geometry-aware allocation (Phase 5), limitations, related work, next steps.
  - After an advisor review, four corrections went in:
    1. Section 6 states phi_half per (alpha, k) and the eligible fraction of the true tail, as CLAUDE.md and
       design note 8c require. It notes that setting D used Liu et al.'s schedule.
    2. Section 4's "every setting" claims are restricted to settings A–C.
    3. In the TL;DR, the post-hoc part of the head/tail claim is labelled.
    4. The literature-check output is archived as `notes/litcheck_2026-10-24.md`: the synthesis plus 30
       candidates with URLs and the agents' quotes, which are data and not yet checked by a human.
- `README.md` rewritten: status, results summary, a table with one command per figure, layout.
- `src/tmsq/experiments/figure_antipodal.py` produces `figures/phase3b_antipodal.*`, a write-up figure of the
  registered Phase 3b result. It reads the K2 sweep and changes no number.

Not done, on purpose:
- The optional real-model bridge (PROMPTS.md Phase 6 step 3). It downloads a model, and PROMPTS.md says to
  ask the owner first.
- Nothing has been pushed or published.

Review points for the owner, also at the end of the draft:
1. The Phase 4 framing ("it depends on what you divide by"; the effect is borderline).
2. The Phase 5 setting-D result, presented as frequency allocation.
3. Citations to check by hand: Chen et al. 2023, Stevinson et al. 2025, Tlaie 2024, and Duan's current title
   (the arXiv v1 HTML and the abs page show different titles).
4. The novelty sentence.
5. Whether to run the bridge before publishing.
6. Voice and tone.

---

## 2026-10-24 — Phase 6: owner's review of draft v1, draft v2, citations re-checked

Owner's decisions (Shen, back from being away):
1. Phase 4 framing: acceptable but not a headline. The order is: the registered result, flagged as weak and not
   robust; then the decomposition as content; then the frequency–geometry confound. The section is cut to a
   short side section, and the main line of the post is the Phase 3b antipodal finding.
2. Phase 5 setting D presented as frequency allocation: agreed.
3. Voice: first person ("I") plus an explicit statement that Claude was used. Claude Code agreed, with one
   condition. "I" covers design, decisions and interpretation. Actions Claude Code took on its own while the
   owner was away keep Claude Code as the explicit actor: the Phase 3 test run, the pre-registrations for
   Phases 3b–5, the wrong-outcome Phase 4 mechanism test, and the post-hoc analyses.
4. The real-model bridge (Pythia-70M + SAE) runs BEFORE publication. Its plan comes to the owner for approval
   before any download (PROMPTS.md).
5. Citations: the owner had Claude chat verify the four flagged citations. Claude Code re-checked every claim
   against the primary pages before editing.
   - Duan, arXiv 2606.03002. The abs page shows the title "Perplexity Can Miss SAE Feature Damage Under
     Quantization", with v1 on 2 Jun 2026 and v2 on 4 Jun 2026.
     - v2 §4.4: "non-survival rises with firing rate—high-firing features are more vulnerable" (INT6).
     - v2 §5.4: "controlling for the other variables, rarer features are not more fragile and if anything
       survive better."
     - v2 cites Borobia et al.
     - Setup, for the bridge: SAE pythia-70m-deduped-res-sm at blocks.4.hook_resid_post; 200k WikiText-2-raw
       tokens; per-output-channel RTN with q in [-2^(b-1), 2^(b-1)-1]; transformer-block linear weights only.
   - Chen, Beyer, Zhu & Chen (2026), arXiv 2609.11716: the authors are confirmed, so "Chen, Beyer et al." is
     correct.
   - Borobia, Seguí-Mas & Tormo-Carbó (2026), arXiv 2603.25325, "How Pruning Reshapes Features: Sparse
     Autoencoder Analysis of Weight-Pruned Language Models": the abs page confirms that rare features survive
     pruning better.
   - Stevinson, Prieto, Barsbey & Birdal: the ICML 2026 poster page (icml.cc/virtual/2026/poster/62594) is
     titled "Adversarial Vulnerability from Interference Between Features in Superposition". The arXiv
     2510.11709 title differs. The PMLR volume and pages are not known yet and were not guessed.
   - Chen, Lau, Mendel, Wei & Murfet (2023), arXiv 2310.06301. Table A.1 gives the 4-gon l* = 1, b* = 0; the
     5-gon b* = -0.28230; the 6-gon b* = -0.61814. The paper works in the high-sparsity limit ("This is the
     high sparsity limit of the TMS input distribution") with r = 2 hidden dimensions, and its loss has no
     importance weights. The post now states this scope.

Changes:
- `writeup/post.md` is now draft v2:
  - "How this was made" moved under the byline.
  - First-person voice, with the attribution condition above.
  - TL;DR reordered so the antipodal finding leads; head/tail is a short side bullet.
  - Section 6 rewritten in the owner's order (a)/(b)/(c). It uses only existing numbers: the FP-only facts in
    the Phase 4 pre-registration (rho(p, near_thr) = +0.94, rho(p, bias) = +0.86) and the H4 secondaries. No
    new analysis was run.
  - Duan and Borobia added as "consistent direction, not confirmation" (different metric).
  - Section 9 rewritten: the two Chens are disambiguated (Chen, Beyer et al. 2026 is now cited); Stevinson is
    cited as ICML 2026; Borobia added.
  - New References section; the review points are pruned to the four still open.
- `README.md`: the status line and the head/tail bullet now match the demoted framing.
- `CLAUDE.md`: the Duan entry now notes v2, the frequency statements and Borobia, quoted above.

Next: the bridge plan, for the owner's approval. No download before approval.

---

## 2026-10-24 — Phase 6b (real-model bridge): PLAN, awaiting the owner's approval

Goal (PROMPTS.md Phase 6 step 3): use Pythia-70M-deduped with a pretrained SAE (SAELens), quantize the model
with RTN, and test whether a margin analog of each SAE feature predicts its survival beyond peak activation,
mirroring Duan (2026). The owner decided the bridge runs before publication. Nothing has been downloaded or
installed. The facts below come from a research workflow (`bridge-research`, 5 agents, with a cross-checker
that re-read the critical sources).

### Verified facts the plan rests on

- SAE: SAELens release `pythia-70m-deduped-res-sm`, sae_id `blocks.4.hook_resid_post`. It points to the HF
  repo `ctigges/pythia-70m-deduped__res-sm_processed`, folder `4-res-sm`. This is Duan's exact choice; his
  script uses the same release and id.
  - cfg: standard ReLU architecture; d_in 512, d_sae 32768; apply_b_dec_to_input true; normalize_activations
    none; context_size 128; prepend_bos false. The training data was the Pile.
  - Encode: f = ReLU((x - b_dec) W_enc + b_enc).
- SAELens 6.54.12 (2026-10-20, Python >= 3.10). `SAE.from_pretrained(release, sae_id, device, dtype)` returns
  ONE SAE object. The tuple form is `from_pretrained_with_cfg_and_sparsity`; the docs misname it
  `load_from_pretrained_with_cfg_and_sparsity`. SAELens pins transformer-lens < 4.0, so 3.9.0 would install,
  and `import sae_lens` requires transformer-lens.
- Duan's method, from the paper v2 plus his released code (anonymized mirror, Appendix M):
  - Survival c_j is the Pearson correlation over ALL 199,680 token positions, zeros included, with float64
    sums. A feature is active if its FP firing rate over all tokens is > 0.001. Survived means c_j > 0.9;
    damaged means c_j < 0.5.
  - Data: WikiText-2-raw TRAIN split. Non-empty lines are joined with "\n\n" and tokenized without special
    tokens. The first 390 x 512 tokens are used, with no BOS and position 0 kept.
  - Quantized: the transformer-block linear weights only (Q, K, V, O, MLP in/out). Embeddings, unembedding,
    LayerNorm, biases and the SAE are untouched. Bit widths 8 to 4.
  - His code runs TransformerLens `HookedTransformer.from_pretrained` with the defaults: fold_ln,
    center_writing_weights, center_unembed and fold_value_biases, in float32. So:
    - the quantized weights are LN-folded and centered;
    - the SAE sees per-token mean-centered residuals;
    - the scale is taken over all dimensions but the last. For W_Q/W_K/W_V ([8, 512, 64]) that means 64 scales
      shared across heads, which is NOT the per-output-channel method the paper describes.
  - His "FP16" baseline is nominal: fp16-stored weights computed in fp32.
  - Pythia-70M results in Table 1, as survived% and perplexity change. INT8 98.3 / +1.0%. INT7 86.3 / +2.3%.
    INT6 62.4 / +14.7%. INT5 38.0 / +71%. INT4 14.5 / +371%. FP perplexity is 67.84, and there are 5,688 active
    features. Table H.1 says 5,686 (two runs).
  - The four predictors are rarity, log mean (over all tokens), log peak (max over tokens) and log
    concentration. INT6 logistic CV AUC is 0.924. **No univariate AUC is reported for peak**: "strongest
    marginal predictor" rests on Figure 3's quartiles.
  - No geometry, bias, pre-activation or overlap analysis was tried.
- HF transformers 5.20.2: `hidden_states[5]` (or a forward hook on `gpt_neox.layers[4]`) is the raw block-4
  output. `from_pretrained` defaults to fp16 for Pythia, so pass `dtype=torch.float32`.
- The Sparse Feature Circuits paper (Marks et al.) reports for this layer-4 residual SAE: 88% variance
  explained and L0 61, on Pile at context 128. Whether the SAE expects raw or centered residuals is NOT stated
  anywhere; that is why there is a gate below.
- Measured on this machine (synthetic GEMM): about 1 TFLOP/s. The estimate is 1–2 min per model variant, with
  streamed sufficient statistics: dense activations would need 26 GB.

### Decisions, with recommendations (the owner chooses; the pre-registration fixes them before any quantized run)

D1. Two arms.
- **Code arm (primary, recommended):** Duan's released pipeline: TL default processing, scales over all dims
  but the last, centered residual into the SAE. It is the only arm that can be checked against Duan's
  Table 1. The shared Q/K/V scales are logged as an inherited deviation, not fixed.
- **Stated arm (secondary):** HF raw weights with true per-row scales (one per output row of
  query_key_value, dense, dense_h_to_4h and dense_4h_to_h), and the raw block-4 output, or the convention
  chosen by gate G1, into the SAE.

D2. Gates, with their rules registered before the numbers are seen.
- G1, FP only (decides only the stated arm's SAE input; the code arm feeds centered residuals as Duan did):
  - Compute the SAE's FVU and L0 on raw vs per-token-centered layer-4 residuals, positions < 128.
  - Pick the convention with the lower FVU. If the two are within 0.01 FVU, use raw.
- G2, replication gate: the first quantized run, code arm, aggregates only.
  - Tolerances: active features 5,688 ± 30; INT6 survived 62.4 ± 3.0 points; FP perplexity 67.84 ± 1.0.
  - Fail means STOP and report. No tuning.
  - The per-feature file is not opened before the pre-registration hash is in this LOG.

D3. Predictors, all computed on the FP model only. Notation: e_j = W_enc[:, j], d_k = W_dec[k], the
effective bias beta_j = b_enc_j - e_j . b_dec, and mu_j+ = the mean of f_j over the tokens where it is active.
- **M (PROMPTS's margin analog, from encoder bias and decoder overlaps):**
  `margin_j = ( -beta_j - max(0, max_{k != j, k active} (e_j . d_k) * mu_k+) ) / mu_j+`
  This mirrors the toy-model margin `-b_i - max(0, max_j W_i.W_j)` with x_j <= 1: interference from feature
  k at its typical active size, in units of feature j's own typical response. It needs all 32768^2 overlaps,
  computed in chunks (about 1 TFLOP).
- **T (the confirmed toy-model mechanism, threshold density):**
  `near_thr_j = P_t( -0.05 * mu_j+ < z_j <= 0 )`, where z_j is the pre-activation. This is the fraction of
  tokens where feature j is off but within 5% of its typical activation of switching on, the analog of the
  toy model's 0.05.
- Secondary: bias only, `-beta_j / mu_j+` (the analog of "antipodal features have b ≈ 0"); the
  density-weighted interference `sum_k p_k (e_j . d_k)+ mu_k+`; Duan's four predictors.
- Peak's univariate AUC is computed here as the baseline, since Duan reports none.

D4. Outcomes: two families, Holm over all four tests. This applies the lesson of design note 12 and the
Phase 4 mistake: test a leakage mechanism on leakage.
- **Family A, survival (comparable to Duan):** INT6, all-token Pearson c_j > 0.9, active features.
  - Statistic: dAUC = repeated (20x) 5-fold stratified CV AUC of the logistic model (log peak + predictor)
    minus that of (log peak).
  - Minimum effect: dAUC >= 0.02.
  - p-value: permutation of the predictor within log-peak deciles, 1,000 permutations.
- **Family B, leakage:** INT6, new false activations
  `dfalse_j = P_t( f_j^Q > 0.05 * mu_j+ | f_j^FP = 0 )`.
  - Statistic: partial Spearman of the predictor with dfalse_j given log peak.
  - Same permutation scheme. Minimum |rho| 0.1.
- Directions:
  - T is predicted: more near-threshold mass gives lower survival and more leakage.
  - M is two-sided. In the toy model the margin hypothesis came out with the opposite sign, so no direction is
    claimed.
- Secondary outcomes: active-stratum Pearson (never mixed with Family A); other bit widths; the stated arm;
  disjoint token halves (stability, not independent evidence); positions < 128 only (the SAE was trained at
  context 128, Duan uses 512).
- Honest limit: this is one model, and its roughly 5,700 features are not independent. There are no
  instances to cluster on, and CLAUDE.md forbids bootstrapping features. Permutation within peak strata is the
  substitute, and feature-level p-values here are weaker evidence than the toy-model ones.

D5. Runs, each under 10 min with an ETA printed. The FP pass is reused across variants. The FP pass is the
same for every bit width within an arm, so it runs once per arm; the 1–2 min per variant is from a synthetic
benchmark, and real overhead may be 2–3x higher.
- Run 1: FP passes (G1, predictors); code arm INT6 (G2).
- Run 2: code arm INT8/7/5/4.
- Run 3: stated arm FP + INT8 to INT4.
- Run 4: matched-noise control at INT6, 1 seed, code arm (each output row's rounding error replaced by a
  Gaussian direction of the same norm).

D6. Install route.
- **Recommended:** a separate `.venv-bridge`, gitignored, with sae-lens==6.54.12 (which pulls
  transformer-lens 3.9.0 and transformers >= 5.9) and torch 2.13.0 CPU. The main venv and its 261 tests stay
  untouched. Cost: about 400–500 MB of packages, including torch again.
- **Alternative:** the main venv, with a constraints file pinning torch and numpy, after a
  `pip install --dry-run`. This would upgrade huggingface_hub and tokenizers and add wandb, plotly, nltk and
  others.

D7. Downloads, about 310 MB of data. The HF cache stays outside OneDrive (default ~/.cache/huggingface).
- EleutherAI/pythia-70m-deduped: model.safetensors (166 MB) plus tokenizer and config (2.1 MB). The duplicate
  pytorch_model.bin is skipped.
- ctigges/pythia-70m-deduped__res-sm_processed, folder 4-res-sm only (134.5 MB).
- Salesforce/wikitext wikitext-2-raw-v1, train parquet only (6.4 MB). It is read with pandas, without
  `datasets`.

Deviations from Duan, logged up front:
- FP32 compute rather than a nominal "FP16".
- Library versions: his are unpinned, ours are pinned.
- The paper's 5,686 vs 5,688 inconsistency is noted.
- Run 4, the noise control, is ours.

CLAUDE.md correction in this commit, with the evidence: Duan's grid is nominally [-2^(b-1), 2^(b-1)-1].
With s = max|W|/qmax we have |W/s| <= qmax, so the lower clip never binds in fp32, and the effective grid is
the same symmetric 2^b - 1 levels as ours. The old CLAUDE.md text said "2^b values, unlike our symmetric
grid".

Next: the owner approves (or edits) D1–D7. After that: install, download, FP-only work and gate G1, then the
pre-registration is committed. Only then is the first quantized run made.

---

## 2026-10-24 — Phase 6b (bridge): PRE-REGISTRATION, written before any quantized run

The owner approved plan D1–D7 ("同意 D1–D7"). This pre-registration fixes every definition and decision
rule before the first quantized forward pass.
- Frozen code, at the commit that adds this entry:
  - `src/tmsq/bridge.py`, with tests in `tests/test_bridge.py` (16 tests);
  - `src/tmsq/experiments/bridge.py`.
- Inputs: pinned in `configs/bridge.yaml`, including `fp_run`. Download manifest:
  `results/bridge/20261024-220300-fetch`.
- Environment: `.venv-bridge`, with system torch 2.13.0+cpu, sae-lens 6.54.12, transformer-lens 3.9.0 and
  transformers 5.20.2.

### FP-only facts, seen before registration (`results/bridge/20261024-224808-fp`)

That run is the FP stage re-run with the reviewed code. Every predictor is identical to the first FP run,
`20261024-220859-fp`, except interf_sum, whose normalization changed (see the corrections below).
- G1, positions < 128 (49,920 tokens):
  - Raw residual: FVU 0.193, L0 73.9.
  - Per-token-centered residual: FVU 0.232, L0 69.0.
  - Raw is lower by 0.039, more than the 0.01 tolerance, so the stated arm uses the raw residual.
  - Side observation, not a test: the SAE fits the raw residual better. Duan's pipeline (the code arm) feeds
    it the convention it fits worse.
- Code arm: 5,688 active features, exactly Duan's Table 2 count. FP perplexity 67.839 (Duan 67.84). Mean L0
  71.8.
- Stated arm: 5,717 active features, FP perplexity 67.839, mean L0 76.1.
- Predictors, code arm, active features (Spearman):
  - near_thr: median 0.0004, 90th percentile 0.0016. Its correlation with log peak is -0.002, **but with
    rarity it is -0.93**. Near-threshold mass is almost a frequency proxy, hence secondary S1 below.
  - margin (b_enc): positive for 94% of features (median 1.12). Correlations: log peak -0.58, bias_only +0.87,
    near_thr +0.10.
  - margin_beta: positive for 29%. Spearman with margin 0.60 (stated arm 0.35). The e_j . b_dec term has a
    median of 1.6 x |b_enc|.
  - All 5,688 encoder biases are negative.
  - interf_sum vs log peak: -0.32.

### Corrections to the approved plan, all made before any quantized run

1. **Margin bias (D3).** D3 wrote the "effective bias" beta_j = b_enc_j - e_j . b_dec. That is the bias with
   respect to the raw origin x = 0, which is not a "no features present" point of the residual stream.
   - The SAE subtracts b_dec before encoding (apply_b_dec_to_input = true), so its zero-feature input is
     x = b_dec. There, z_j = b_enc_j.
   - With x - b_dec ≈ sum_k f_k d_k, the pre-activation is z_j = sum_k f_k (e_j . d_k) + b_enc_j, the analog of
     the toy model's sum_k (W_j . W_k) x_k + b_j. This holds for whatever residual the SAE is given, centered
     or raw.
   - `test_margin_analog_reproduces_toy_margin` checks that the analog reduces to the toy-model margin (the
     pentagon example, -0.059) when encoder = decoder, mu+ = 1 and b_enc = b.
   - **The owner chose b_enc as the primary** (question answered about 22:50, before any quantized run).
     margin_beta is a labelled secondary, and so is the matching bias_only_beta = -beta/mu+.
2. **interf_sum** now follows D3 exactly: sum_k rate_k (e_j . d_k)+ mu_k+. A pre-review draft also divided by
   mu_j+.
3. **Statistics, after the independent review** (workflow `bridge-prereg-review`: 3 reviewers, one skeptic per
   finding; 9 findings confirmed, 6 refuted).
   - Family B: the first draft residualized on one global linear fit but permuted within deciles. That is
     anti-conservative when the predictor and dfalse both depend nonlinearly on the control. The reviewer
     showed false "threshold geometry" verdicts in S1 under a frequency-only null built on the real near_thr.
     **Fixed:** the residualization now uses cell fixed effects plus within-cell slopes on the control ranks,
     with the same cells as the permutation.
   - Family A: the base model was linear in the controls. **Fixed:** the base is now an additive cubic spline
     (6 knots per control), fit within each training fold.
   - Under the U-shaped confounder of `test_nonlinear_shared_confounder_is_not_a_finding`, both fixed
     statistics stay null. The old Family B statistic gives rho > 0.3 on the same data.
4. **Killed features.** A quantized feature that never fires now gets c_j = 0 exactly, as in Duan's code. The
   first draft's fp32 shifted sums could give ±1. Sums are now unshifted fp64.
   `test_killed_features_get_zero_correlation` covers this.
5. **Token-half secondary.** It now uses each half's own dfalse.

### Definitions (all on the arm's FP pass, over the 199,680 tokens)

- Active feature: FP firing rate > 0.001 (Duan). Only active features are analysed.
- mu_j+: the mean of f_j over the tokens where it is active. z_j: the pre-activation.
- **T (threshold density):** near_thr_j = P_t(-0.05 mu_j+ < z_j <= 0).
- **M (margin analog):** margin_j = (-b_enc_j - max(0, max_{k != j, k active} (e_j . d_k) mu_k+)) / mu_j+.
- Duan's predictors, as in his code:
  - rarity = -log(rate);
  - log_mean;
  - log_peak (max over tokens);
  - log_conc = max(0, log(peak/mean)).
  - Inputs are floored at 1e-8.
- Survival c_j: Duan's all-token Pearson correlation of FP and quantized f_j, from fp64 sums.
  - Survived means c_j > 0.9; degraded means 0.5 <= c_j <= 0.9; damaged means c_j < 0.5.
  - A feature that never fires after quantization gets c_j = 0.
- New false activations: dfalse_j = P_t(f_j^Q > 0.05 mu_j+ | f_j^FP = 0).

### Gate G2 (replication), checked first on run R1, aggregates only

- The FP part (active count 5,688 ± 30; FP perplexity 67.84 ± 1.0) has already passed: 5,688 and 67.839.
- INT6 survived must be 62.4 ± 3.0 percent of active features (Duan, Table 1).
- If it fails: STOP and report. No tuning, and the per-feature analysis is not run.

### Primary cell and tests (`registered()` in `tmsq.experiments.bridge`)

Cell: code arm, INT6, layer 4, all token positions, active features.

| Test | Outcome | Predictor | Statistic | Minimum effect | Predicted direction |
|---|---|---|---|---|---|
| A_T | survived (c_j > 0.9) | near_thr | dAUC over log peak | 0.02 | negative coefficient |
| A_M | survived | margin (b_enc) | dAUC over log peak | 0.02 | none (two-sided) |
| B_T | dfalse | near_thr | cell-consistent partial Spearman given log peak | abs 0.10 | positive |
| B_M | dfalse | margin (b_enc) | cell-consistent partial Spearman given log peak | abs 0.10 | none (two-sided) |

- **dAUC:** CV AUC of logistic(spline(log peak) + predictor) minus that of logistic(spline(log peak)).
  - The spline is cubic with 6 knots.
  - The predictor is standardized within the training fold.
  - L2 penalty, C = 1.
  - 20 repeats of stratified 5-fold CV, with fold seeds 6060 + r.
- **p for dAUC:** 1,000 permutations of the predictor within log-peak deciles, with generator seed 6060. The
  statistic is the dAUC on the first 5 repeats, observed vs permuted. p = (1 + #(null >= obs)) / 1001.
- **Coefficient sign:** from the logistic fit of [spline basis, z(predictor)] on all features.
- **Partial Spearman:** the Pearson correlation of the rank residuals.
  - Residuals come from a fit with log-peak-decile fixed effects plus within-decile slopes on rank(log peak).
  - Two-sided p from 10,000 permutations of rank(predictor) within the same deciles, with seed 6061.
- **Holm** over the four tests, alpha 0.05.
- **Verdict strings** (`decide_a`, `decide_b`, `registered()`):
  - "supported" (T only): the minimum effect is met, Holm p < 0.05, and the sign is as predicted.
  - "adds information, opposite to prediction" (A_T) or "associated, opposite to prediction" (B_T): the
    minimum effect and Holm p < 0.05 are met with the opposite sign. **The registered direction is
    refuted.** This is not a null.
  - "adds information beyond peak (coefficient ±)" or "associated beyond peak (±)": M, two-sided.
  - "no added information beyond peak" or "no association beyond peak": null.
  - "not interpretable (fewer than 200 features in a class)": the Family A stop rule.
- **Interpretation limit:** this is one model, and its roughly 5,700 features are not independent. With
  n = 5,688, any effect that clears the minimum will sit at the p floor. The minimum effect sizes carry the
  decision, and the p-values are weaker evidence than the toy-model instance-level tests.

### Registered secondary S1: beyond frequency (no Holm)

The same four tests are run with controls log peak AND rarity:
- the dAUC base is the additive spline in both controls;
- the partial Spearman uses fixed effects for the 5 x 5 quantile cells plus within-cell slopes on both ranks;
- permutation is within the same cells.

Interpretation rule for T, applied only where the primary T test is "supported":
- The support is attributed to **threshold geometry** only if S1 meets the same minimum effect, with
  unadjusted p < 0.05 and the same sign.
- Otherwise it is reported as **"beyond peak, not beyond frequency"**.

This applies the Phase 4–5 lesson (frequency and geometry confounded) before the data, not after.

### Secondary (no Holm; labelled secondary in every report; `descriptive()`)

- Baselines:
  - peak's univariate AUC, in-sample and as a linear CV AUC;
  - the spline-base CV AUC;
  - Duan's four-predictor linear CV AUC (his 0.924).
- dAUC (spline base) and cell-consistent partial Spearman with dfalse for margin_beta, bias_only,
  bias_only_beta, interf_sum, rarity, log_mean and log_conc. Plain Spearman of each predictor with c_j.
- The same on:
  - the active-stratum Pearson;
  - the two token halves (blocks [0, 196) and [196, 390)), each with its own dfalse;
  - positions < 128 (the SAE's training context);
  - INT8, INT7, INT5 and INT4 (R2);
  - the stated arm (R3);
  - the matched-noise control at INT6 (R4).
- Duan's Table 1 aggregates per bit width (survived %, perplexity change) next to ours.

### Runs (each under 10 min, ETA printed)

- R1: code arm, INT6 (gate G2).
- R2: code arm, INT8/7/5/4.
- R3: stated arm, INT8 to INT4.
- R4: code arm, INT6, matched noise, seed 0.

The registered analysis is
`.venv-bridge\Scripts\python -m tmsq.experiments.bridge analyze --config configs/bridge.yaml --primary <R1> --extra <R2> <R3> <R4>`.
Its numbers are reported as they come out, nulls included.

### Stop rules

- G2 fails.
- Fewer than 200 survived or fewer than 200 non-survived features in the primary cell. Family A is then
  reported as not interpretable.
- An FP sanity check fails: R1's FP perplexity must equal the FP stage's to 1e-3 (`fp_sanity` in R1's
  summary).

---

## 2026-10-24 — Phase 6b (bridge): RESULTS

Pre-registration and frozen code: commit `6f3836c` (above). Runs, all with `fp_sanity` passing (each run's FP
perplexity equals the FP stage's):
- R1 `results/bridge/20261024-230636-code-b6`
- R2 `results/bridge/20261024-230918-code-b8-7-5-4`
- R3 `results/bridge/20261024-231452-stated-b8-7-6-5-4`
- R4 `results/bridge/20261024-232022-code-b6-noise0`
- Analysis `results/bridge/20261024-232316-analysis` (1,067 s)

### Gate G2: passed. The code arm reproduces Duan's Table 1 at every bit width

| bits | survived (ours / Duan) | perplexity (ours / Duan) |
|---|---|---|
| INT8 | 98.3 / 98.312 | 68.53 / 68.532 |
| INT7 | 86.3 / 86.322 | 69.37 / 69.372 |
| INT6 | 62.41 / 62.395 | 77.83 / 77.828 |
| INT5 | 38.0 / 37.957 | 116.15 / 116.148 |
| INT4 | 14.5 / 14.504 | 319.59 / 319.602 |

- At INT6 the damaged fraction is 1.00% vs 1.020%, and the mean correlation 0.888 vs 0.888.
- Duan's four-predictor linear CV AUC: 0.924 (his 0.924 ± 0.007).
- **Peak activation's univariate AUC, not reported by Duan: 0.879.**

### Registered tests (primary cell: code arm, INT6, 5,688 active features, 62.4% survived)

| Test | Statistic | Holm p | Verdict |
|---|---|---|---|
| A_T near_thr → survival | dAUC +0.008 (0.879 → 0.887), coefficient − | 0.002 | **no added information beyond peak** (below the 0.02 minimum) |
| A_M margin → survival | dAUC **+0.024** (0.879 → 0.903), coefficient **−** | 0.002 | **adds information beyond peak (coefficient −)** |
| B_T near_thr → dfalse | partial rho **+0.79** | 4e-4 | **supported**; S1 interpretation: **"beyond peak, not beyond frequency"** |
| B_M margin → dfalse | partial rho +0.26 | 4e-4 | **associated beyond peak (+)** |

S1, beyond log peak AND rarity (registered secondary, no Holm):
- A_T: dAUC -0.0001, p 0.77.
- A_M: dAUC +0.017, p 0.001.
- B_T: rho -0.009, p 0.52.
- B_M: rho +0.17, p 1e-4.

Reading, stated plainly:
- **The toy-model mechanism (threshold density) does not add to peak activation for survival.**
- Its very strong association with new false activations is **entirely frequency**: rho drops from 0.79 to
  -0.01 once rarity is controlled for. The frequent features leak more (partial rho(rarity, dfalse) = -0.82).
  This is the same direction as Duan's "non-survival rises with firing rate" and as the head's larger raw
  leakage in Phase 4 (post hoc there).
- **The margin analog adds a modest amount beyond peak** (+0.024, just over the registered 0.02), and less
  beyond peak and rarity (+0.017).
- Its direction is the **opposite of the protective reading**: a larger margin, meaning a more negative
  encoder bias relative to the feature's typical activation, goes with LOWER survival and MORE leakage. This
  parallels the toy model's H1 result, where margin also came out with the opposite sign. The mechanism is not
  established here.

### Secondary (descriptive, no Holm)

- **Robustness of A_M** (dAUC of margin):
  - token halves: 0.026 / 0.024;
  - positions < 128: 0.031;
  - active-stratum Pearson: 0.016;
  - code arm by bit width: INT8 0.022, INT7 0.029, INT5 0.014, INT4 0.018;
  - stated arm: INT6 0.035, INT5 0.026, INT4 0.033;
  - matched noise at INT6: 0.024.

  The sign is the same everywhere it is estimable. Stated INT8 and INT7 have 0 and about 11 non-survivors, so
  their AUCs are not interpretable.
- **What carries M.** bias_only = -b_enc/mu+ alone gives dAUC 0.027. The beta versions (the originally
  approved D3 formula) give 0.006–0.007, and margin_beta's partial rho with dfalse is 0.46. Under the b_enc
  definition the information sits in the encoder bias relative to activation size. The b_enc choice was made
  before the data.
- **interf_sum:** dAUC 0.014. **rarity alone:** dAUC 0.012.
- **Stated arm** (true per-output-channel scales on raw weights, raw residual) is much less damaged than the
  code arm:

  | bits | survived (stated / code) | perplexity change (stated / code) |
  |---|---|---|
  | INT6 | 85.3% / 62.4% | +4.8% / +14.7% |
  | INT5 | 59.2% / 38.0% | +27% / +71% |
  | INT4 | 22.6% / 14.5% | +117% / +371% |

  The arms differ in three ways: shared Q/K/V scales, LN-folded and centered weights, and the residual
  convention. This run cannot say which one drives the difference. Descriptive only.
- **Matched noise at INT6:** survived 63.8% vs RTN 62.4%, and perplexity +13.5% vs +14.7%. Generic perturbation
  of the same size does about the same damage, as in the toy model.

### Surprises and caveats

- The replication is essentially exact (survival to within 0.02 points, perplexity to within 0.02), which validates
  the pipeline. Duan's released pipeline feeds the SAE the residual convention it fits worse (G1), and
  quantizes with Q/K/V scales shared across heads. Our numbers inherit both, by design.
- Single model and dependent features. Every registered p sits at the permutation floor, so the minimum effect
  sizes carry the decisions. A_M clears its minimum by 0.004.

Next: write-up (post Section "Bridge"), then the owner's review before publishing.

---

## 2026-10-24 — Phase 6b: POST-HOC check of the margin result (not registered)

Script `notes/posthoc/bridge_margin_decomposition.py`, output
`results/bridge_posthoc/margin_decomposition_20261024.txt`. Primary cell only; dAUC uses the registered CV
design, with no permutations. The advisor's review prompted it: mu+ = mean / rate, so log mu+ =
log_mean + rarity (checked to 1e-15). The margin's denominator is therefore built from two of Duan's own
statistics.

**(1) What carries the margin's +0.024 beyond peak: the mu+ normalization, not the encoder geometry.**

- Beyond peak (spline base, AUC 0.879): log mu+ alone gives **+0.044**; margin +0.024; bias_only +0.027;
  interf +0.021; **-b_enc alone +0.003**.
- Beyond peak + log mu+ (AUC 0.924): margin **+0.003**; bias_only +0.002; -b_enc +0.001; near_thr +0.006.
- Beyond Duan's four statistics:
  - spline base, AUC 0.934: margin **+0.002**, bias_only +0.004, -b_enc +0.000, near_thr +0.000;
  - linear base, Duan's model, AUC 0.924: margin +0.001, bias_only +0.003.
- **Reading:** the registered verdict "adds information beyond peak" stands as registered. Post hoc, that
  information is Duan's own activation statistics entering through mu+.
  - Neither the encoder bias nor the decoder-overlap interference adds to Duan's four statistics. In this
    model, no geometric quantity tested (margin, bias, interference, threshold density) improves on activation
    statistics for predicting survival.

**(2) B_M's positive sign is partly the shared mu+ normalization.**

The leak threshold is 0.05 mu+, and margin is proportional to 1/mu+. Cell-consistent partial Spearman with
dfalse:

| Predictor | given peak | given peak + log mu+ | given peak + rarity |
|---|---|---|---|
| margin | +0.256 | +0.156 | +0.174 |
| -b_enc | -0.246 | -0.151 | -0.130 |
| near_thr | +0.787 | +0.812 | -0.009 |

- The raw encoder bias has the protective sign: a deeper threshold means fewer new false activations.
- The margin's positive sign comes from the normalization and the interference term. The mechanism is not
  established.
- near_thr's leakage link survives mu+ but not rarity. near_thr's band (0.05 mu+) and the leak threshold move
  in opposite directions with mu+, so mu+ is not the confound for T. Frequency is.

Bridge conclusion for the write-up:
- Duan's results replicate exactly. Peak alone has AUC 0.879.
- The toy-model mechanism does not transfer as a survival predictor.
- Leakage in the real model is a frequency effect.
- The registered "margin adds beyond peak" is, post hoc, Duan's statistics in another form.

---

## 2026-10-25 — Phase 6b: EXPLORATORY ablation of Duan's code vs his stated method (not pre-registered)

Why: the owner's review asked what drives the INT6 survival gap, code arm 62.4% vs stated arm 85.3%, before
anything is published about it. Module `src/tmsq/experiments/bridge_ablation.py` (it reuses the frozen bridge
code unchanged). Runs: `results/bridge_ablation/20261025-024313-r1-int6`, `-024807-c1-int6`, `-025332-c2-int6`,
`-025755-r2-int6`. Every run passes `fp_sanity`.

Five factors differ between the arms:
- F: fold_ln;
- C: center_writing_weights;
- B: fold_value_biases;
- Q: W_Q/W_K/W_V scales shared across heads (s) or per output channel (p);
- I: SAE input centered (c) or raw (r).

Each variant is INT6 in TransformerLens with explicit flags and differs from one end in exactly one factor.
Code arm A0 = F1 C1 B1 Qs Ic; stated arm S = F0 C0 B0 Qp Ir.

| Variant | F C B Q I | Survived | Damaged | Mean corr | Perplexity change |
|---|---|---|---|---|---|
| A0 (Duan's code) | 1 1 1 s c | 62.41% | 1.00% | 0.888 | +14.7% |
| A0-Q | 1 1 1 **p** c | **94.30%** | 0.00% | 0.968 | +2.2% |
| A0-F | **0** 1 1 s c | 59.86% | 3.76% | 0.868 | +22.5% |
| A0-B | 1 1 **0** s c | 62.66% | 0.98% | 0.889 | +14.6% |
| A0-C | 1 **0** 1 s c | 62.48% | 0.98% | 0.888 | +14.9% |
| A0-CI | 1 **0** 1 s **r** | 62.18% | 1.05% | 0.887 | +14.9% |
| S (stated method) | 0 0 0 p r | 85.31% | 0.02% | 0.954 | +4.8% |
| S-Q | 0 0 0 **s** r | **59.56%** | 3.94% | 0.867 | +22.2% |
| S-F | **1** 0 0 p r | **94.60%** | 0.00% | 0.968 | +2.0% |
| S-B | 0 0 **1** p r | 85.22% | 0.02% | 0.954 | +4.8% |
| S-I | 0 0 0 p **c** | 85.57% | 0.02% | 0.955 | +4.8% |
| S-CI | 0 **1** 0 p **c** | 85.69% | 0.02% | 0.955 | +5.0% |

Checks: TL S reproduces the HF stated arm R3 (85.31% vs 85.3%), and A0 reproduces R1 (62.41%).

Reading:
- **The shared Q/K/V scales explain most of the gap.** Changing only that factor moves survival from 62.4% to
  94.3% at the code end, and from 85.3% to 59.6% at the stated end.
- **fold_ln is second and interacts with Q.**
  - With per-channel scales, folding the LayerNorm gains into the weights before quantizing raises survival
    from 85.3% to 94.6%.
  - With shared scales, folding changes it only from 59.9% to 62.4%.
- **Weight centering, value-bias folding and the SAE input convention each move survival by ≤ 0.5 points.**
  The input mismatch found by gate G1 (FVU 0.232 vs 0.193) barely affects survival.
- Implication, descriptive: Duan's headline INT6 number depends mainly on one implementation choice. With
  per-channel Q/K/V scales, 94% of features survive at INT6, not 62%.
- The exact Table 1 match (above) shows that the released code produced the paper's numbers. The mismatch is
  between that code and the method as described in §3.2 / Appendix L.

Code version: anonymous mirror `sae-feature-survival-quantization-0141`. Its API reports
sourceCommitDate 2026-06-08T22:05Z and anonymizedAt 2026-06-09. The README links "Current paper version" to
arXiv 2606.03002, titled "Perplexity Can Miss SAE Feature Damage Under Quantization", which is v2 of
4 June 2026. The script is `scripts/02_pythia_phase2b_streaming_final.py`.

---

## 2026-10-25 — Phase 6b: POST-HOC re-run of the registered tests on the described method, and Duan-4 baselines

Script `notes/posthoc/bridge_stated_arm_tests.py`, output `results/bridge_posthoc/stated_arm_tests_20261025.{txt,json}`.
The first background run exited with code 4 after finishing the code arm and printed no traceback; the cause is
unknown. A rerun with faulthandler completed normally (exit 0).

Registered procedure, frozen `registered()`, on the stated arm R3 (INT6, 5,717 features, 85.3% survived):
- A_T: dAUC +0.000 -> no added information beyond peak.
- A_M: dAUC +0.035 -> adds information beyond peak (coefficient -).
- B_T: rho +0.744 -> "supported", but beyond peak, not beyond frequency (S1 rho -0.027).
- B_M: rho +0.151 -> associated beyond peak (+). S1, given peak + rarity: +0.302.

These are the same four verdicts as on the code arm.

Beyond Duan's four statistics (dAUC with a spline / linear base; partial rho with dfalse given all four):

| | code arm | stated arm |
|---|---|---|
| Duan-4 AUC | 0.934 / 0.924 | 0.902 / 0.891 |
| near_thr | +0.0002 / +0.0018; rho +0.11 | +0.0006 / +0.0004; rho +0.11 |
| margin | +0.0021 / +0.0014; rho +0.03 | +0.0037 / +0.0041; rho +0.07 |
| bias_only | +0.0036 / +0.0032; rho +0.01 | +0.0028 / +0.0027; rho +0.07 |

Reading:
- No proxy adds more than 0.004 AUC beyond Duan's statistics on either pipeline.
- near_thr keeps a small leakage association, rho ≈ 0.11. It depends on the specification: it is about 0 given
  peak + rarity.
- The owner's review asked whether test 4 (B_M) was frequency-controlled. It was, in S1: 0.174 given
  peak + rarity. Given Duan-4 it falls to 0.03 (0.07 on the stated arm).
- Design lesson, owned: the registered baseline for "beyond Duan" should have been his four-statistic model,
  not peak alone.

Post draft v4:
- Section 8, replication with the implementation note and ablation table.
- Section 9, the transfer tests with both pipelines, the margin paragraph and the scope limit.
- TL;DR net result marked post hoc.

The email to Duan is in the owner's Gmail drafts (unsent, no recipient). It now gives the one-factor
attribution of the gap between 85.3% and about 94% to LayerNorm folding, at the owner's request.

---
