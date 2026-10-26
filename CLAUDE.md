# CLAUDE.md — Superposition x Quantization (TMS-Q)

## What this project is

A research project with two parts:

1. **Replication** of key results from Anthropic's *Toy Models of Superposition* (Elhage et al., 2022):
   https://transformer-circuits.pub/2022/toy_model/index.html
2. **Original extension**: study how **post-training weight quantization (PTQ)** damages features stored in
   superposition, in a toy model where the ground-truth features are known.

Central research question:

> In a model with known ground-truth features, which features does weight quantization break first,
> and can superposition geometry predict this in advance?

The owner (Shen) is an applied-math undergraduate building this as a portfolio project (GitHub repo + blog post).
Rigor and honesty matter more than speed or impressive-looking plots.

## Related work (why this is probably open)

- **Duan (2026), arXiv 2606.03002, "Perplexity Can Miss SAE Feature Damage Under Quantization"** — In Pythia-70M
  and Gemma-2-2B, SAE features degrade under per-output-channel RTN (INT8 → INT4; signed range
  `[-2^(b-1), 2^(b-1)-1]` nominally; but the scale is `s = max|W| / qmax` with `qmax = 2^(b-1)-1`, so
  `|W/s| <= qmax` and the lower clip never binds in fp32: the effective grid is our symmetric `2^b - 1`-level
  grid [corrected 2026-10-24]; his released code shares one scale per d_head index across heads for
  W_Q/W_K/W_V in TransformerLens layout, see `notes/LOG.md`, bridge plan) while perplexity can stay flat or even
  improve. Survival = Pearson correlation between FP16 and quantized SAE activations over identical tokens,
  survived if `c_j > 0.9`, damaged if `c_j < 0.5` (the provenance of our 0.9 threshold). Survival is predictable
  from full-precision statistics (cross-validated AUC 0.92–0.97, peak activation = strongest *marginal*
  predictor). Real models, no ground-truth features; the only mechanism offered is an untested heuristic
  ("quantization perturbations compete with each feature's activation signal"); no geometry or superposition
  quantity is measured. → We aim to supply and test the mechanism with known geometry. The current version is
  v2 (4 June 2026); v1 had a different title. v2 also reports that "non-survival rises with firing rate"
  (§4.4, INT6) and that rarer features "are not more fragile and if anything survive better" once the other
  statistics are controlled for (§5.4). It extends Borobia, Seguí-Mas & Tormo-Carbó (2026, arXiv 2603.25325,
  "How Pruning Reshapes Features"), who find that rare SAE features survive pruning better. (Checked against
  arXiv on 2026-10-24.)
- **Tlaie (LessWrong, Aug 2024), "Toy Models of Superposition: what about BitNets?"** — trains BitLinear networks
  in TMS (n=20, m=5; quantization-aware training from scratch), compares geometry qualitatively, and the author
  warns it is "too little work to extract strong conclusions". Not PTQ, no per-feature damage.
- **Chen et al. (2026), arXiv 2609.11716, "Why Does Post-Training Quantization Work?"** — error propagation at
  the hidden-state level (counteraction between layers, LM-head geometry). Not feature-level.
- **Liu, Liu & Gore (NeurIPS 2025), arXiv 2505.10465, "Superposition Yields Robust Neural Scaling"** — power-law
  feature frequencies in TMS; a weight-decay knob `gamma` controls the degree of superposition (positive =
  standard decoupled decay → weak superposition; negative = a norm-targeting term that pushes feature norms
  toward 1 → strong superposition). We borrow only the frequency distribution and the knob; their exact recipe
  and our deviations are in "Setting D recipe" below (code: github.com/liuyz0/SuperpositionScaling).
- **Gorton & Lewis (2025), arXiv 2508.17456, "Adversarial Examples Are Not Bugs, They Are Superposition"** —
  adversarial vulnerability vs superposition in TMS with n=100, m=20, uniform importance, density 1-S swept
  log-spaced ("exponentially") from 1.0 to 0.1 (30 models). The paper gives NO optimizer, lr, batch size or step
  count for that standard sweep; the only stated budget (150k steps, lr 1e-3) is for the ADVERSARIALLY trained
  models of Sec. 3.1.4. They report only MODEL-level vulnerability and model-level superposition (written
  `||W||_F^2 / n` in their Eq. 1, sic; their text implies `||W||_F^2 / m`, which is what we use) plus mean
  off-diagonal interference; there is no per-feature metric. Template for the setting, not for the per-feature
  analysis.
- **Head vs tail evidence in real LLMs is MIXED, and neither paper is a clean, monotone frequency result**:
  Chang et al. (EMNLP 2025, arXiv 2506.12044, "Why Do Some Inputs Break Low-Bit LLM Quantization?") summarize
  "long-tail examples are relatively less affected by quantization on both FineWeb and PopQA", but on FineWeb
  "long-tail" means high-NLL examples under full-precision Llama3-70B (not token frequency), many low-NLL
  ("head") examples take large errors, and on PopQA (entity popularity) their Mistral-12B 3-bit result is
  non-monotonic (lowest- and highest-popularity buckets degrade least, middle buckets most); for other LLMs they
  only say low-popularity examples "usually have milder degradation". Wu et al. (2026, arXiv 2608.19670, "The
  Asymmetric Harms of LLM Compression") find head knowledge (PopQA; head/middle/tail = equal-size groups by
  Wikipedia page views of the subject entity) loses a *larger relative fraction*: head loss exceeds tail loss in
  93 of 105 PopQA model–method combinations but in only 48 of 105 on the Head-to-Tail benchmark, and standard
  4-bit GPTQ/AWQ/OmniQuant shift retention by at most 5.3 points (the large shifts come from pruning, 2-bit and
  AQLM). Their 11 methods are 4 PTQ methods (GPTQ, AWQ, OmniQuant, AQLM; no RTN) plus pruning, so the result is
  not quantization-specific. **Do NOT assume rare features break first, and do not cite Chang et al. as a clean
  frequency result.**

Never claim novelty in any write-up without the caveat "to our knowledge".

## Model and data (canonical definitions — never change silently)

- Features `x ∈ R^n`, independent. `x_i = 0` with probability `S_i`, else `Uniform[0, 1]`.
  Frequency `p_i = 1 - S_i`.
- Importance `I_i = r**i`, i = 0..n-1, with `r` set per setting (A: 0.9, B: 0.7, C and D: r = 1, i.e. uniform
  `I_i = 1`).
- Model (tied weights): `h = W x` with `W ∈ R^{m×n}`; `x_hat = ReLU(W^T h + b)`.
- Loss: mean over batch of `sum_i I_i * (x_i - x_hat_i)**2` (the paper's loss read as an expectation over x;
  the paper writes `sum_x` in one place and `integral dp(x)` in another).
  Note: the official notebook averages over the feature dimension too (`loss = einops.reduce(error,
  'b i f -> i', 'mean').sum()`: mean over batch AND features, then sum over instances), so its optimized
  per-instance loss equals ours divided by n, and the value it prints is additionally divided by the number of
  instances. Adam and decoupled-weight-decay AdamW are invariant to a constant factor on the loss up to the
  `eps` term (scaling the loss by c equals replacing eps by eps/c; negligible unless per-weight gradients are
  ~1e-7, i.e. very rare, low-importance features). Coupled-L2 `torch.optim.Adam(weight_decay > 0)` is NOT
  invariant. The constant matters when comparing loss values with the paper/notebook: say which normalization a
  plot uses.
- Init: `W ~ Xavier normal`, `b = 0`. Optimizer for settings A–C: **AdamW with `weight_decay = 0.01` on all
  parameters, `b` included** (the official notebook calls `torch.optim.AdamW(list(model.parameters()), lr=lr)`
  and leaves PyTorch's default weight decay; this is NOT plain Adam, and weight decay changes the degree of
  superposition, see Liu, Liu & Gore), constant lr 1e-3, batch 1024, 10k steps. The paper itself states no
  training hyperparameters at all; every value here comes from the notebook, so write-ups must say "the
  official notebook's setup", never "the paper's". Setting D overrides optimizer, weight decay, lr schedule and
  step count as in "Setting D recipe" below (batch 1024 unless the recipe says otherwise). Verify the loss has
  plateaued; use more steps when S >= 0.99. Gorton & Lewis state no training budget for their standard sweep
  (their adversarially trained models use 150k steps at lr 1e-3), so no source says how many steps setting C
  needs at high sparsity: verify the plateau empirically, report the step count used, and decide it before
  Phase 1c because Phase 2 reuses those models.
- Layout note: the official notebook stores `W` transposed, as `[K, n, m]` (feature i is `W[k, i, :]` there and
  `W[k, :, i]` here). Do not copy notebook snippets such as `torch.linalg.norm(W, 2, dim=-1)` or its
  `compute_dimensionality` without transposing.
- Train many model instances in parallel with an instance dimension: `W` has shape `[K, m, n]`.
- At least **5 seeds** per config. Local minima are common (e.g. 5→2 sometimes fails to form a pentagon).
  Report all seeds; never silently keep only the best one.

Standard settings:

| ID | n   | m  | importance | sparsity                         | purpose                                                                     |
|----|-----|----|------------|----------------------------------|-----------------------------------------------------------------------------|
| A  | 5   | 2  | 0.9^i      | sweep S                          | visual sanity check, pentagon (matches the official notebook's intro figure) |
| B  | 20  | 5  | 0.7^i      | sweep S                          | main replication; the paper's n=20, m=5 figures use `I_i = 0.7^i`           |
| C  | 100 | 20 | uniform    | sweep S                          | main quantization testbed (Gorton & Lewis setting; the notebook's own 100/20 cell uses importance `100^(-i/99)`, not uniform) |
| D  | 200 | 20 | uniform    | power law `p_i = min(1, c (i+1)^(-alpha))` | head vs tail (Phase 4); recipe below                              |

The paper's "dimensions per feature" figure (`D* = m / ||W||_F^2` vs `1/(1-S)`, sticky at 1 and 1/2) uses
**n=400, m=30, I_i = 1**; the paper adds that n and m "don't matter very much" as long as n >> m, and the
official notebook reproduces the figure with **n=200, m=20, constant importance, feature probability from 1 to
1/20** (20 instances). Phase 1c runs it on setting C (n=100, m=20): label that figure a scaled-down version and
state both reference configurations. If the 1/2 plateau is unclear at 100/20, run the notebook's 200/20
configuration once as a check before concluding anything. Three normalizations of the same quantity appear in
this file (`D* = m / ||W||_F^2`, `feats_per_dim = ||W||_F^2 / m`, Gorton & Lewis's `||W||_F^2 / n` sic): every
figure and LOG entry must name which one it plots.

### Setting D recipe (Phase 4): what we borrow from Liu, Liu & Gore and what we keep from TMS

Their toy model (arXiv 2505.10465, Sec. 3 + Appendix B; code `exp/exp-10.py`, `exp/adamw.py`): Anthropic's tied
ReLU autoencoder with `W` stored as `[n, m]` (rows = features); init `W = randn(n, m) / sqrt(m)`, `b = randn(n)`
(random bias, no Xavier); data `x_i = Bernoulli(p_i) * Uniform(0, 2)`; loss `nn.MSELoss()` (unweighted, mean over
batch and features); `p_i ∝ 1 / i^alpha`, i = 1..n, with the PAPER's alpha swept 0..2 (the code's `alpha`
variable is paper-alpha minus 1, `1 / i ** (1 + alpha)`; do not copy code ranges literally); density
`E = sum_i p_i = 1` in the main experiments. Small models: n = 1000, m in {10, 15, 25, 39, 63, 100}, batch 2048,
20k steps, lr 1e-2 with 2000 linear warm-up steps then cosine decay to a floor of `0.05 * lr`. Large runs:
n = 10240, lr 0.02 scaled by `(8/m)^0.25` for `W`, bias lr `2.0/m`, 5% warm-up.

The superposition knob is a weight-decay term `gamma` in a CUSTOM AdamW class (`exp/adamw.py`, lines 104–111),
applied to `W` only (`b` has `weight_decay = 0.0`), before the Adam update, per feature vector `W_i` (their
Eq. 2):

- `gamma >= 0`: standard decoupled decay, `W_i <- W_i - lr * gamma * W_i` (code `p.mul_(1 - wd * lr)`).
- `gamma < 0`: a NORM-TARGETING term, `W_i <- W_i - lr * gamma * W_i * (1 / ||W_i|| - 1)`, i.e. gradient descent
  on `(||W_i|| - 1)^2`, pushing every feature norm toward 1 (code `row_norms = p.norm(dim=1) + eps;
  p.add_(wd * p * (1 - 1/row_norms), alpha=lr)`). This is NOT unbounded weight growth and NOT
  `W.mul_(1 - lr * gamma)` with negative gamma.

Small-model sweep: `gamma in linspace(-1, 1, 10)` (step ≈ 0.22). In their large runs `gamma = -1.0` gives strong
superposition and `gamma = 0.1` gives weak superposition (0.1 is not on the small-run grid; nearest is 0.111).
Regime check: `phi_half` = fraction of features with `norm_i > 1/2`; strong superposition has
`phi_half ≈ 1 >> m/n`, weak has `phi_half ≈ m/n`. Neither source establishes which gamma gives which regime at
n = 200, m = 20: the regime must be VERIFIED with `phi_half`, never assumed.

What we do:

- Keep from TMS: `Uniform[0, 1]` activations, importance-weighted loss with `I_i = 1`, tied ReLU model,
  `W ~ Xavier normal`, `b = 0`, batch 1024.
- Borrow: `p_i = min(1, c * (i+1)^(-alpha))`, i = 0..n-1, alpha in {1.0, 1.5} (paper convention); solve `c` with
  a root-finder so that `sum_i p_i = k` for k in {2, 5}. At n = 200 the `min(1, ·)` clip binds only for
  alpha = 1.5, k = 5 (`p_0` saturates at 1; c = 2.719 instead of the closed-form 2.023); for the other three
  configs the root-finder reproduces the closed form. Use the root-finder uniformly and assert `sum_i p_i = k`
  in a test. Their main experiments use E = 1, so k in {2, 5} is our deviation.
- Borrow the knob: implement their piecewise step exactly (standard decay for gamma >= 0, norm-targeting for
  gamma < 0), on `W` only, `b` undecayed. Start from `gamma = +0.1` (weak) and `gamma = -1.0` (strong) and
  confirm the regime with `phi_half`; if a value does not land in the intended regime at 200/20, sweep gamma and
  report the sweep, do not relabel. Warm-up + cosine lr from their small-model schedule (lr 1e-2, 2000 warm-up
  steps, floor 0.05 lr, 20k steps) and verify the plateau. Record `phi_half` for every trained model and report it
  next to every head-vs-tail number.
- Engineering: `torch.optim.AdamW` and `torch.optim.Adam` raise `ValueError: Invalid weight_decay value` for
  negative `weight_decay` (verified on torch 2.13), so write the step by hand or port `exp/adamw.py`. Tests:
  gamma = 0 reproduces plain Adam exactly; gamma = 0.01 reproduces `torch.optim.AdamW`; for gamma < 0 a feature
  vector of norm 1 is a fixed point of the decay term and norms move toward 1 from both sides.
- Record every deviation from their recipe (activation range, init, loss normalization, batch size, density,
  schedule) in `notes/LOG.md`.

## Per-feature geometry metrics (`W_i` = column i of `W`)

- `norm_i = ||W_i||`; feature is **represented** if `norm_i > 0.5`.
- Feature dimensionality `D_i = ||W_i||^4 / sum_j (W_i · W_j)^2` (sum includes j = i); NaN if `norm_i < 0.1`
  (deliberate deviation: the paper assigns unlearned features `D_i = 0` and the notebook clamps the norm at
  1e-6; any "sum of D_i ≈ m" check must treat NaN as 0).
- `feats_per_dim = ||W||_F^2 / m`.
- `max_overlap_i = max_{j≠i} (W_i · W_j)`.
- `margin_i = -b_i - max(0, max_overlap_i)` — worst-case single-feature leak with `x_j <= 1`.
  Positive margin = no other single active feature can leak into feature i's output. A sparse-regime,
  single-leaker quantity; design note 5 gives the dense-regime analogs and the threshold convention.
- `peak_out_i` = max of `x_hat_i` over the `x_i > 0` stratum of the eval set (analog of Duan's "peak activation").
  Also store `p99_out_i` (99th percentile; the sample max is an extreme-value statistic that grows with sample
  size) and the analytic clean response `max(0, ||W_i||^2 + b_i)`.

Sanity values (put these in tests):

- Regular pentagon of unit vectors → `D_i = 0.4`; antipodal pair → `0.5`; orthogonal → `1.0`.
- Running example from our discussion: regular pentagon (unit norms, adjacent dot = cos 72° ≈ 0.309,
  non-adjacent dot ≈ -0.809), `b_i = -0.25`, input `x = (0, 0.7, 0, 0, 0)` (the active feature is index 1 in
  our 0-based indexing, i.e. the second column of `W`) → all four other outputs are exactly 0; the binding case
  is the two adjacent features, pre-activation `cos(72°) * 0.7 - 0.25 ≈ -0.034 < 0` (non-adjacent:
  `-0.809 * 0.7 - 0.25 ≈ -0.816`), and `x_hat_1 = 0.7 - 0.25 = 0.45`. This configuration has a NEGATIVE margin,
  `margin_i = 0.25 - cos(72°) ≈ -0.059` for every i (an adjacent feature at `x_j = 1` would leak with
  pre-activation `+0.059`): the test must assert the four zero outputs, `x_hat_1 = 0.45` and
  `margin_i ≈ -0.059`, not `margin_i > 0`.

## Quantization (PTQ) definitions

- Quantize `W` only by default; keep `b` in FP32 (ablation: quantize `b` too).
- Symmetric uniform round-to-nearest (RTN): `qmax = 2**(bits-1) - 1`,
  `Q(W) = s * clamp(round(W / s), -qmax, qmax)`.
- Scale granularity: per-tensor (`s = max|W| / qmax`), per-column (one scale per feature),
  per-row (one scale per hidden dimension). Duan's "per-output-channel" RTN corresponds to per-row for the
  encoder `W` and to per-column for the tied decoder `W^T`; say which one a comparison uses.
- Bit widths: 8, 6, 5, 4, 3, 2.
- Tied weights: the same `Q(W)` is used as encoder and decoder. Ablation: quantize encoder only / decoder only.
- **Noise control** (all granularities), to separate rounding structure from generic weight noise: for each
  quantized model draw `g_i ~ N(0, I_m)` independently per column and set
  `delta_i = ||Q(W)_i - W_i|| * g_i / ||g_i||`, so every column carries exactly its realized rounding-error norm
  (this also matches the total Frobenius error `||Q(W) - W||_F`). Apply `W + delta` tied (same perturbed matrix
  as encoder and decoder), keep `b` in FP32, use >= 5 noise seeds per quantized model and report mean ± std
  across seeds. Optional second control: iid `Uniform[-s_i/2, s_i/2]` entries, which matches the marginal
  rounding-error distribution and isolates the deterministic dependence of rounding on `W` (e.g. entries with
  `|W_ki| < s_i/2` being zeroed).

## Per-feature damage metrics

Use the same fixed eval set for FP and every quantized model (same seed). THE eval set is the per-feature
stratified set of design note 4; do not also build a natural-distribution set, or `R2_i` and `cond_mse_i` will
disagree. All changes below are SIGNED (quantization can improve a feature); report them signed and use the
pre-registered one-sided thresholds only for binary labels.

- `R2_i = 1 - MSE_i / Var(x_i)` with `MSE_i = p_i * cond_mse_i + (1 - p_i) * E[x_hat_i^2 | x_i = 0]` and
  `Var(x_i) = p_i/3 - p_i^2/4`; `dR2_i = R2_i(FP) - R2_i(Q)`.
- `cond_mse_i = E[(x_i - x_hat_i)^2 | x_i > 0]`; report the change.
- `false_act_i = P(x_hat_i > 0.05 | x_i = 0)` (leakage); report the change `dfalse_act_i`. Also store the
  threshold-free leak energy `E[x_hat_i^2 | x_i = 0]`. Features with `p_i = 1` (clipped head of setting D) have
  no inactive stratum: `false_act_i = NaN`, excluded from H1, and their inactive term in `MSE_i` is 0.
- Survival: Pearson correlation between FP `x_hat_i` and quantized `x_hat_i` over the ACTIVE stratum
  (`x_i > 0`), never over the pooled stratified set (Pearson on a mixture depends on the stratum weights);
  **survived** if correlation >= 0.9 (Duan's `c_j > 0.9`; configurable, but fixed in `notes/LOG.md` before
  Phase 3). Known limitation [verified]: Pearson is scale-blind, so a pure gain change
  (`||Q(W)_i||^2 != ||W_i||^2`) gives correlation 1 with a large `dR2_i`; always report survival next to `dR2_i`
  and `dfalse_act_i`, and see design note 3 for eligibility and the composite label.
- Absolute damage: `dLoss_i = I_i * (MSE_i(Q) - MSE_i(FP))`, feature i's contribution to the loss increase
  (mechanically ∝ `p_i`). `dR2_i` is already normalized by `Var(x_i) ≈ p_i/3`, so a fixed leak gives rare
  features a larger `dR2_i`: never present `dR2_i` as the absolute metric in H4.
- Relative damage (head-vs-tail relative retention): `rel_damage_i = dR2_i / R2_i(FP)`, retention
  `1 - rel_damage_i`, on eligible features only (design note 3). Not `change / FP loss` [verified]: an
  unrepresented feature has FP loss `I_i * p_i / 3` (it predicts 0), not 0, so that ratio is ≈ 0 for dead tail
  features and explodes for well-reconstructed head features whose FP loss ≈ 0.
- Geometric damage indicators from `Q(W)` directly: `killed_i = (||Q(W)_i|| < 0.5)`, signed self-gain change
  `d_norm2_i = ||Q(W)_i||^2 - ||W_i||^2`, and `margin_i(Q)` (from `Q(W)`, and from `Q(b)` in the
  bias-quantization ablation).

## Hypotheses (H1–H3 tested in Phase 3 on settings A–C; H4 in Phase 4 on setting D, where `p_i` varies within a model; pre-register each before looking at results)

Items marked (rec.) are recommendations from the 2026-10-24 review, not source facts; decide and record them in
the pre-registration.

- **H1 margin**: smaller `margin_i` → more damage. Primary outcome `dfalse_act_i` (leakage); `dR2_i` secondary
  (rec.). Pre-register the predictor form (design note 5).
- **H2 crowding**: smaller `D_i` → more damage. Primary outcome `dR2_i` (rec.). `D_i` is sign-blind, but under
  ReLU only positive overlaps leak (an antipodal partner gives `D_i = 0.5` yet zero single-feature leakage), so
  also pre-register the sign-aware companion `D_i^+ = ||W_i||^4 / (||W_i||^4 + sum_{j≠i} max(0, W_i · W_j)^2)`
  (rec.); if they disagree, `D_i^+` carries the mechanistic claim and `D_i` the paper-comparable one.
- **H3 strength**: larger `norm_i` → less damage (Duan's predictor). `peak_out_i ≈ ReLU(||W_i||^2 + b_i + leak)`
  is nearly collinear with `norm_i` in this model: report it as a robustness variant, not a second hypothesis.
  Test H3 separately per scaling granularity (design note 6).
- **H4 frequency** (exploratory, two-sided; the literature is mixed, see Related work): on setting D, head =
  top 10% of eligible features by `p_i`, tail = bottom 50% of eligible features. Report absolute damage
  `dLoss_i` (mechanically ∝ `p_i`, expected larger for the head) and relative damage `rel_damage_i`; the
  pre-registered test is the sign of the Spearman correlation between frequency rank and `rel_damage_i` within
  each (gamma, bits, granularity) cell (rec.).

Statistics: run every analysis within a (setting, S, bits, granularity) cell; never pool across S or bit widths
without instance fixed effects (geometry and damage both shift with S). Pre-register one primary cell and one
primary outcome per hypothesis; everything else is secondary (rec.). Spearman correlations (continuous
outcomes), univariate AUC (binary label), standardized logistic regression with VIFs reported (all predictors
are functions of the same Gram matrix `W^T W` and `b`; do not interpret joint coefficients when VIF > 5);
cross-validate with `GroupKFold` grouped by model instance. Use >= 10 instances per Phase 3 cell (training is
parallel over instances, so this is cheap; a cluster bootstrap over 5 instances has only 126 distinct
resamples) (rec.); report per-instance statistics as mean ± std with a t-interval and a cluster bootstrap over
instances (>= 1,000 resamples) as secondary. Never bootstrap individual features.

## Engineering rules

- Python 3.11+, PyTorch (CPU is enough), numpy, pandas, matplotlib, scikit-learn, pyyaml, tqdm, pytest.
  Use a venv + `requirements.txt`.
- The machine runs **Windows**: use `pathlib`, no bash-only scripts. Every experiment is a Python entry point:
  `python -m tmsq.experiments.<name> --config configs/<name>.yaml`.
- Code is auto-formatted by ruff (hook already configured). Use type hints and short docstrings.
- Layout:
  ```
  src/tmsq/{data,model,train,quantize,metrics,analysis,plots}.py
  src/tmsq/experiments/
  configs/   tests/   results/ (small text outputs tracked; large/binary files gitignored, shipped in the Release)   figures/   notes/LOG.md
  ```
- Reproducibility: seed everything; save `config.yaml`, git commit hash, and metrics (parquet or CSV) in
  `results/<exp>/<timestamp>/`. Never overwrite previous results.
- Each experiment should finish in under ~10 minutes on CPU. Print an ETA; if a run will take longer, ask first.
- Tests must cover: data sparsity rate; model shapes; quantizer (codebook size = `2**bits - 1` for symmetric RTN
  on a dense linspace input, `|Q(W) - W| <= s/2` inside range, idempotence `Q(Q(W)) = Q(W)`, no division by zero
  on an all-zero column); geometry metrics on hand-built pentagon / antipodal / orthogonal cases; the running
  example above (four zero outputs, `x_hat_1 = 0.45`, `margin_i ≈ -0.059`); setting D frequencies
  (`sum_i p_i = k`); the hand-written weight-decay step (gamma = 0 → Adam, gamma = 0.01 → `torch.optim.AdamW`,
  gamma < 0 → norm-1 fixed point).
- Figures: matplotlib, save PNG + PDF in `figures/`, labeled axes, colorblind-safe palette.
- When using library APIs (PyTorch, scikit-learn, SAELens, ...), check current docs (Context7) instead of guessing.

## Design notes (binding; full text in `notes/DESIGN_NOTES.md`)

Read `notes/DESIGN_NOTES.md` before implementing or analysing any phase. Where a note changes a protocol,
the note is the operative definition. Index (numbering used throughout this file and `notes/LOG.md`):

1. Quantizer levels
2. Dead columns and per-column scales
3. Survival eligibility and label
4. Eval set
5. Margin sign and the H1 predictor
6. H3 and scaling granularity
7. Degenerate labels and clustered data
8. Setting D details
9. Phase 1a acceptance
10. Realized-perturbation covariates
11. Importance weighting of damage
12. Which outcome tests which hypothesis

## Working style

- For each phase: propose a short plan first and wait for my approval before writing large amounts of code.
- If a result looks surprising or a definition seems wrong, **STOP and report**. Do not change definitions to get
  nicer results.
- Null or negative results are valid findings. Report them plainly.
- After each phase: append to `notes/LOG.md` (goal, config, results with numbers, surprises, next steps),
  then give me a short summary **in Chinese**.
