# Design notes (moved from CLAUDE.md on 2026-10-24; numbering unchanged)

These notes are binding for every phase. CLAUDE.md and notes/LOG.md refer to them as "design note N".

Where a note changes a protocol, the note is the operative definition and the canonical sections above point
to it. [verified] = checked against a source or by computation; [rec.] = a recommendation, decide and record it
in `notes/LOG.md`.

1. **Quantizer levels** [verified by enumeration]. Symmetric RTN with `qmax = 2**(bits-1) - 1` has a codebook of
   `2*qmax + 1 = 2**bits - 1` values (2-bit is ternary `{-s, 0, +s}`); Duan's per-channel RTN uses the full
   signed range `[-2^(b-1), 2^(b-1)-1]` (2^b values) instead. Test the codebook on an input that spans the range
   densely (e.g. `linspace(-1, 1, 100001)`): random Gaussian weights do not populate the outer levels at >= 3
   bits, so a test on random data would fail for the wrong reason.
2. **Dead columns and per-column scales** [verified]. Unrepresented features have `W_i ≈ 0`, so a per-column scale
   `s_i = max|W_i| / qmax ≈ 0` divides by zero. Use `s_i = max(s_i, eps)` (eps ≈ 1e-8); such columns quantize
   to exactly 0. Same guard for per-row scales.
3. **Survival eligibility and label** [verified pathology; rec. remedy]. Pearson correlation is NaN when FP
   `x_hat_i` is constant or when the quantized `x_hat_i` is constant, and a column with `norm_i > 0.5` can still
   be dead at the output when `b_i <= -||W_i||^2`. Eligibility for survival, the "damaged" label and relative
   damage: `norm_i > 0.5` AND `R2_i(FP) >= 0.5`, applied identically everywhere (survival tables, head vs tail,
   relative damage). A constant or NaN quantized output counts as not survived. Because Pearson is scale-blind
   (a 0.6x gain change gives correlation 1.0 with `dR2_i ≈ 0.64`; 30% false activations of amplitude 0.1 still
   give correlation ≈ 0.99), the recommended composite label is: survived iff correlation >= 0.9 AND
   `dR2_i <= 0.1` AND `dfalse_act_i <= 0.05` (alternative: Lin's concordance >= 0.9 on the active stratum plus
   the leakage condition). Fix the thresholds before Phase 3 and report a sensitivity sweep as robustness, never
   as a tuning knob. Report per (setting, S, bits): the eligible count, the `killed_i` count, and the number of
   INELIGIBLE features whose quantized `false_act_i` exceeds 0.05 (features switched on by quantization).
4. **Eval set** [verified need; rec. design]. "≥ 2,000 active samples per feature" needs ~200k natural samples at
   S = 0.99 and ~2M at S = 0.999 in expectation only (with N = 200k a p = 0.01 feature falls short of 2,000 about
   half the time), and ~7M for the tail of setting D (`p_199 ≈ 3e-4` at alpha = 1.5, k = 2). Features are
   independent, so stratify per feature: >= 2,000 samples conditioned on `x_i > 0` (other features from their
   natural distribution) for `cond_mse_i`, `peak_out_i` and the survival correlation; >= 20,000 samples
   conditioned on `x_i = 0` for `false_act_i` and `E[x_hat_i^2 | x_i = 0]` (the SE of a 5% rate from 2,000
   samples is ≈ 0.005, too coarse for H1's primary outcome). Then `MSE_i = p_i * cond_mse_i +
   (1 - p_i) * E[x_hat_i^2 | x_i = 0]`, `Var(x_i) = p_i/3 - p_i^2/4`, `R2_i = 1 - MSE_i / Var(x_i)`. Same seed
   for FP and every quantized model. For `p_i = 1` features the inactive term is 0 by construction.
5. **Margin sign and the H1 predictor** [prediction; rec.]. We EXPECT most represented features in superposition
   to have negative margins: a single leaker j fires with probability `p_j * max(0, 1 + b_i / (W_i · W_j))`,
   small mainly because `p_j` is small, so the optimum prefers a mild bias over the reconstruction cost of
   blocking the worst case `x_j = 1`. This is a prediction, not a fact: in Phase 2 report the fraction of
   eligible features with `margin_i < 0` per (setting, S) before testing H1, and STOP and report if it is far
   from "most". `margin_i` is a sparse-regime, single-leaker quantity; at S near 0 many features are active at
   once and it under-summarizes the pre-activation distribution. Pre-register before Phase 3 which predictor H1
   uses: raw `margin_i` (monotone claim), `|margin_i|` (distance-to-threshold claim), or the directly measured
   threshold density `near_thr_i = P(|pre_i - tau| <= 0.05 | x_i = 0)`, where `pre_i = W_i · (W x) + b_i` is the
   FP pre-activation on the inactive stratum (preferred: it is the stated mechanism and it accounts for multiple
   leakers and for `p_j`). Keep thresholds consistent: `false_act` uses `tau = 0.05` while `margin_i` measures
   distance to 0, so either use `margin_i(tau) = tau - b_i - max(0, max_overlap_i)` or set tau = 0; state the
   choice. For setting D also store the expected leak `leak_mean_i = sum_{j≠i} p_j * max(0, W_i · W_j) / 2`.
   Under the bias-quantization ablation use `margin_i(Q)` from `Q(b)` and `Q(W)`.
6. **H3 and scaling granularity** [verified mechanics; rec. design]. Under per-tensor scaling the per-column
   error budget `||delta_i|| ≈ s * sqrt(m/12)` is independent of geometry, so the RELATIVE error ∝ `1/norm_i`
   (small columns can be zeroed entirely) while the self-gain error `2 W_i · delta_i` GROWS with `norm_i`; the
   sign of the norm–damage relation depends on the damage metric. Under per-column scaling
   `||delta_i|| ∝ max|W_i|`, so the absolute error grows with `norm_i` and with column peakiness
   `peak_ratio_i = max|W_i| / ||W_i||` (store it). Per-column (= per-feature) scaling has no analog in real
   models, where features are not axis-aligned; per-row (= per-channel) is the realistic analog and matches
   Duan's per-output-channel RTN on the encoder. Pre-register H3 under per-row scaling as the primary condition,
   report per-tensor and per-column as mechanistic variants, and use the matched-noise control (Quantization
   section) plus the note-10 covariates to separate scale effects from geometry.
7. **Degenerate labels and clustered data** [verified; rec. rule]. At 8 bits almost nothing is damaged and at
   2 bits almost everything is, so a binary label carries no information there. Run AUC / logistic regression
   only in cells with >= 30 damaged AND >= 30 undamaged eligible features, each class present in >= 3 instances;
   list the eligible cells in the LOG before fitting, and fix the survival thresholds BEFORE computing damaged
   fractions. Prefer outcomes that need no window: the per-feature breaking point `break_bits_i` = largest bit
   width at which the feature is NOT survived (NaN if it survives 2 bits), and the continuous `dR2_i`,
   `dfalse_act_i` at every bit width (Spearman). Never bootstrap individual features.
8. **Setting D details** [verified]. (a) Indexing from 0 makes `i^(-alpha)` undefined at i = 0: use
   `(i+1)^(-alpha)`. (b) Solve `c` under the `min(1, ·)` clip with a root-finder (the clip binds only for
   alpha = 1.5, k = 5 at n = 200). (c) In weak superposition tail features are not represented (`W_i ≈ 0`): their
   absolute damage is ≈ 0 while their FP loss is `I_i * p_i / 3`, so a `change / FP loss` ratio would make the
   tail look trivially robust, and the opposite problem (FP loss ≈ 0) hits well-reconstructed head features. Use
   `rel_damage_i` on eligible features (note 3), compare head vs tail only over eligible features, report the
   eligible fraction and `phi_half` next to every number, and state that selecting represented tail features in
   weak superposition is itself a selection effect (they are not typical tail features).
9. **Phase 1a acceptance** [verified context]. The paper shows the regular pentagon for n=5, m=2 with UNIFORM
   importance at 1-S = 0.05 and states that "features varying in importance or sparsity causes smooth
   deformation of polytopes"; the notebook's intro sweep with `0.9^i` stops at feature probability 1/20
   (S = 0.95). So with setting A's `0.9^i` expect a deformed pentagon, and "72° ± 10° at S >= 0.9" has no source.
   Operational definition for reporting: a seed counts as a pentagon iff all five `norm_i > 0.5` and all five
   `D_i` lie in 0.4 ± 0.05 (a looser proxy for adjacent angles within 72° ± 10°; the sorted angle gaps are recorded too). Report the fraction of seeds per S;
   this is a report, not a gate, and nothing is tuned to hit it.
10. **Realized-perturbation covariates** [rec.]. Store per feature `dW_i = ||Q(W)_i - W_i||`, `d_norm2_i`,
    `d_overlap_i = max_{j≠i} |Q(W)_i · Q(W)_j - W_i · W_j|`, `margin_i(Q)` and the flip indicator
    `sign(margin_i) != sign(margin_i(Q))`. These are MEDIATORS of geometry → damage (not colliders), so adding
    them is legitimate but changes the estimand. Pre-register two models per cell: (1) damage ~ geometry only
    (total effect; answers "can geometry predict damage in advance"); (2) damage ~ geometry + residualized
    perturbation (`dW_i` minus its within-cell linear prediction from `norm_i` and `peak_ratio_i`, which removes
    the mechanical collinearity: `dW_i ∝ max|W_i|` under per-column, ≈ constant under per-tensor). Never report
    (2) alone as the predictive value of geometry.
11. **Importance weighting of damage** [rec.]. Settings A and B have non-uniform `I_i`, but every per-feature
    damage metric is unweighted except `dLoss_i`. If Phase 3 includes A or B, state whether "damage" is
    `I_i`-weighted; for C and D (`I_i = 1`) it does not matter.
12. **Which outcome tests which hypothesis** [rec.]. H1: `dfalse_act_i`; H2/H3: `dR2_i` and `break_bits_i`;
    H4: `dLoss_i` and `rel_damage_i`. Fix this mapping in the pre-registration so that hypotheses x outcomes x
    cells does not become a garden of forking paths.
