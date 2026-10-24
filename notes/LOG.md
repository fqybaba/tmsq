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
