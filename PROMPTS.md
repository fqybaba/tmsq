# Phase prompts for Claude Code

How to use:

1. Put `CLAUDE.md` in the project root (Claude Code reads it automatically).
2. Paste **one phase at a time**. Do not paste the whole file at once.
3. Before moving on, check that phase's acceptance criteria yourself.
4. CLAUDE.md governs: its canonical definitions and its "Design notes" apply to every phase. Where a prompt
   below is less specific than CLAUDE.md, CLAUDE.md wins.

---

## Phase 0 — Project setup

```
Read CLAUDE.md first. We are starting Phase 0 (setup). Propose a short plan and wait for my approval.

Tasks:
1. Create the repo layout described in CLAUDE.md: venv instructions, requirements.txt, src/tmsq package,
   tests/, configs/, figures/, results/ (gitignored), notes/LOG.md with a template
   (goal / config / results / surprises / next steps), and a README stub.
2. Implement src/tmsq/data.py: a sparse feature generator that supports a per-feature sparsity vector S_i,
   a fixed seed, and batched sampling on CPU.
3. Implement src/tmsq/model.py: a batched tied-weight toy model with an instance dimension
   (W: [K, m, n], b: [K, n]), forward = ReLU(W^T W x + b), Xavier-normal init, b = 0.
4. Write pytest tests: empirical sparsity rate within tolerance, output shapes, determinism under a fixed seed.

Do not train anything yet. When done, run the tests, update notes/LOG.md, and summarize in Chinese.
```

Acceptance: `pytest` passes; a model with K=3, n=5, m=2 runs a forward pass.

---

## Phase 1 — Replication

```
Read CLAUDE.md. Phase 1: replicate key results of Toy Models of Superposition. Plan first, wait for approval.

Tasks:
1. src/tmsq/train.py: config-driven training (AdamW with weight_decay 0.01 on all parameters as in the official
   notebook, constant lr 1e-3, batch 1024, 10k steps by default with a configurable step count,
   importance-weighted MSE; record which loss normalization the saved curves use),
   many instances in parallel, ≥5 seeds, saves loss curves and final weights to results/.
2. src/tmsq/metrics.py (geometry part): norm_i, D_i, feats_per_dim, max_overlap_i, margin_i, exactly as defined
   in CLAUDE.md. Add the sanity tests (pentagon 0.4, antipodal 0.5, orthogonal 1.0; running example: four zero
   outputs, x_hat_1 = 0.45, margin_i ≈ -0.059, i.e. a NEGATIVE margin).
3. Experiments:
   a. rep_5in2 (setting A): S ∈ {0, 0.7, 0.8, 0.9, 0.95, 0.99}.
      Figure: the 5 columns of W as arrows in 2D, one panel per S, all seeds.
   b. rep_20in5 (setting B): 1-S on a log grid from 1 to 0.001 (about 12 values).
      Figures in the paper's style: W^T W heatmap + bias bar chart per S; feature norms colored by superposition.
   c. rep_dims (setting C, n=100, m=20; a scaled-down version of the paper's n=400, m=30, I_i=1 figure and of the
      official notebook's n=200, m=20 reproduction; label it as such): 1-S on a log grid from 1 to 0.001
      (about 16 values). Decide the step count before running (no source gives a training budget for this
      setting; verify the plateau; Phase 2 reuses these models).
      Figure: dimensions per feature D* = m / ||W||_F^2 vs 1/(1-S), log x-axis.
4. Report what fraction of seeds reach each geometry.

Acceptance criteria (check and report each one explicitly):
- A, S=0: exactly 2 represented features, orthogonal; the other 3 have norm ≈ 0.
- A, high S: report per S the fraction of seeds in which all 5 features are represented with D_i in 0.4 ± 0.05
  (adjacent angles ≈ 72° ± 10°). With I_i = 0.9^i the paper predicts a deformed pentagon (regular only for
  uniform importance), so this is a report, not a gate; do not tune to hit it.
- B: the number of represented features increases with S; biases become negative where superposition appears.
- C: D* is sticky near 1 and shows a plateau near 1/2 (antipodal pairs); if the 1/2 plateau is unclear at
  100/20, run the notebook's 200/20 configuration once before concluding anything.
If any criterion fails, stop and diagnose before tuning anything. Update LOG.md and summarize in Chinese.
```

---

## Phase 2 — Quantization sweep

```
Read CLAUDE.md. Phase 2: post-training quantization of trained toy models. Plan first, wait for approval.
Do NOT interpret the hypotheses yet; this phase only produces clean data.

Tasks:
1. src/tmsq/quantize.py: symmetric RTN as defined in CLAUDE.md, with per-tensor / per-column / per-row scales,
   bits ∈ {8, 6, 5, 4, 3, 2}, the Gaussian noise control (as defined in CLAUDE.md), and the ablations
   (quantize b too; encoder-only; decoder-only).
   Tests: codebook size = 2**bits - 1 on a dense linspace input (2-bit is ternary), error bound
   |Q(W) - W| <= s/2, idempotence, and an all-zero column under per-column / per-row scales quantizes to
   exactly 0 with no NaN.
2. src/tmsq/metrics.py (damage part): dR2_i, change in cond_mse_i, change in false_act_i, survival correlation,
   relative damage, peak_out_i, plus the additional per-feature quantities CLAUDE.md defines, all exactly as
   defined there. Eval set, sample counts and eligibility: as specified in CLAUDE.md ("Per-feature damage
   metrics" and design notes 3–4), same seed for FP and every quantized model.
3. Experiment quant_sweep: reuse the trained models from Phase 1 (settings A, B, C) at several S values
   (for C: S ∈ {0, 0.9, 0.97, 0.99}), all seeds × bits × granularity × {RTN, noise control}.
4. Output ONE tidy table (parquet): one row per (setting, S, seed, instance, bits, granularity, method, feature),
   containing all geometry predictors, all damage metrics, and the additional per-feature columns CLAUDE.md
   defines.
5. Figures: total weighted loss increase vs bits (one line per S); fraction of features surviving vs bits;
   RTN vs noise control; for setting A, overlay FP vs quantized arrows at 4 / 3 / 2 bits.

Sanity checks: 8-bit damage ≈ 0; average damage increases as bits decrease. Report any violation.
Update LOG.md and summarize in Chinese.
```

---

## Phase 3 — Hypothesis tests

```
Read CLAUDE.md. Phase 3: test H1–H3 on the Phase 2 table.

Step 0 (before any analysis): write a pre-registration section in notes/LOG.md with the predicted direction of
H1–H3, the exact criteria we will use to call each one supported / not supported, and the choices CLAUDE.md
leaves to the pre-registration: the H1 predictor form and threshold convention (design note 5), H3's primary
scaling granularity (design note 6), the survival label and eligibility rule (design note 3), the cells
eligible for AUC / logistic regression (design note 7), and the outcome used for each hypothesis (design
note 12). Show it to me and wait.

Then:
1. Within each (S, bits, granularity) cell: Spearman correlation of each predictor (margin_i or the H1
   predictor form fixed in Step 0, D_i, norm_i, peak_out_i) with dR2_i and with the change in false_act_i.
   p_i is constant within a model instance in settings A–C, so it is not a predictor here; it belongs to
   Phase 4.
2. Univariate AUC for predicting "damaged" (the label fixed in Step 0), over eligible features and only in
   the cells that pass the rule fixed in Step 0 (CLAUDE.md design note 7).
3. Standardized logistic regression with GroupKFold by model instance. Compare predictor sets:
   {peak_out} vs {margin} vs {D_i} vs {all}. Report cross-validated AUC with bootstrap CIs computed as
   specified in CLAUDE.md (Statistics paragraph and design note 7).
4. Robustness: split by S, by granularity, and RTN vs noise control. If margin predicts noise-control damage
   just as well, the effect is generic noise rather than rounding structure; say which one it is.
5. Figures: damage vs margin scatter (colored by D_i); AUC comparison bar chart with CIs.

Verdict: for each hypothesis say supported / not supported / mixed, with numbers. Do not overclaim.
Update LOG.md and summarize in Chinese.
```

---

## Phase 4 — Head vs tail

```
Read CLAUDE.md. Phase 4: does quantization hurt frequent (head) or rare (tail) features more?
Plan first, wait for approval.

Setup (setting D): n=200, m=20, uniform importance, power-law frequencies p_i = min(1, c * (i+1)^(-alpha)),
i = 0..n-1, alpha ∈ {1.0, 1.5}, c solved with a root-finder so that sum_i p_i = k ∈ {2, 5} (the clip binds
only for alpha = 1.5, k = 5).
Control the degree of superposition (weak vs strong) with the weight-decay knob gamma exactly as specified in
CLAUDE.md "Setting D recipe": standard decoupled decay for gamma >= 0, Liu, Liu & Gore's norm-targeting term for
gamma < 0, on W only, written by hand because torch rejects negative weight_decay. Start from gamma = +0.1
(weak) and -1.0 (strong), verify the regime with phi_half, and record every deviation from their recipe in
LOG.md. Their paper and code (exp/adamw.py, exp/exp-10.py) remain the reference; do not guess anything the
recipe leaves open.

Define head = top 10% of features by p_i, tail = bottom 50%, over eligible features (eligibility per CLAUDE.md
design note 3). For each bit width report absolute damage and relative damage (relative retention) as defined
in CLAUDE.md for head vs tail, with the eligible fraction per bucket and phi_half next to every number,
mean ± std over ≥5 seeds.
Compare with the two conflicting LLM results described in CLAUDE.md (Chang et al. 2025; Wu et al. 2026):
which pattern does the toy model reproduce, and under which regime (weak vs strong superposition)?
Update LOG.md and summarize in Chinese.
```

---

## Phase 5 — Margin-aware quantization (application)

```
Read CLAUDE.md. Phase 5: can geometry-aware quantization reduce damage at the same bit budget?
Plan first, wait for approval.

Compare at the same AVERAGE bits per weight (e.g. 4.0, as a mix of 3-bit and 5-bit columns):
(a) uniform 4-bit RTN, per-column;
(b) margin-based allocation: 5 bits to the half of the columns with the smallest margin (restricted to
    represented columns, CLAUDE.md design note 3), 3 bits to the rest;
(c) norm-based allocation (magnitude heuristic baseline);
(d) random allocation (baseline);
(e) margin-preserving rounding: per column, greedily choose round-up vs round-down to minimize leakage on a
    calibration set (cap the compute; document the algorithm clearly).

Evaluate on a held-out eval set: total weighted loss, survival fraction, worst-feature damage.
Settings C and D, ≥5 seeds. If there is no gain over the baselines, report that honestly and explain why.
Update LOG.md and summarize in Chinese.
```

---

## Phase 6 — Write-up (+ optional bridge to a real model)

```
Read CLAUDE.md and notes/LOG.md. Phase 6.

1. Draft a blog post (LessWrong style, Markdown) in writeup/post.md: motivation, setup, replication sanity
   checks, main findings with figures, limitations, related work (use "to our knowledge"), next steps.
   Every number must trace back to a results/ folder.
2. Rewrite README.md: what the project is, how to reproduce each figure with one command, the results summary.
3. OPTIONAL bridge (ask me before downloading any model): Pythia-70M with a pretrained SAE (SAELens).
   Compute a margin analog for SAE features from decoder-direction overlaps and encoder biases, quantize the
   model with RTN, and test whether the margin analog predicts feature survival beyond peak activation,
   mirroring Duan (2026). Check the SAELens docs for the current API; do not guess it.
```
