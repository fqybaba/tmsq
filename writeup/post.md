# Which features does quantization break first? In a toy model of superposition, it is the antipodal pairs, not the crowded ones

> **DRAFT v3, not published.** v2 (2026-10-24) followed Shen's review of v1: the head-vs-tail part became a
> short side section, the voice became first person, and the citations were checked against primary sources.
> v3 adds the real-model bridge (Section 8).
> Claude Code prepared it from `notes/LOG.md`. Every number comes from a `results/` directory listed in the
> provenance table at the end. The open review points are collected in the last section.

*Shen Yuting. Implemented and analysed with Claude Code.*

**How this was made.**
- I designed the research question, the overall architecture and the phase plan, and I made the scope and
  approval decisions.
- I used Claude (claude.ai) to write that design up as Markdown and for a few simple lookups.
- I used Claude Code (Anthropic) to check the plan against primary sources, write all of the code, run every
  experiment and analysis, and draft this post.
- I approved the Phase 3 pre-registration before any test ran. After that I was away, and under my standing
  instruction Claude Code ran the Phase 3 tests, wrote the pre-registrations for Phases 3b–5, and did the
  post-hoc analyses. These are marked where they appear.
- For the real-model bridge (Section 8), I approved Claude Code's plan and chose one of its definitions (the
  encoder bias in the margin analog). Claude Code then wrote the pre-registration, had it reviewed
  independently, and froze it before any quantized run.

"I" below means me. Every decision and deviation, with who made it and when, is logged in `notes/LOG.md` and
`notes/SESSION_JOURNAL.md`.

## TL;DR

- Anthropic's toy model of superposition (a tied ReLU autoencoder) has known ground-truth features. I quantize
  its weights after training with round-to-nearest (RTN) and measure the damage to each feature.
- **The intuitive prediction fails.** I expected crowded, low-margin features in dense superposition to break
  first. The one cell where per-feature damage is reliable is n = 100, m = 20, S = 0.75. There, the
  pre-registered predictors point the *opposite* way.
- **The fragile features are the zero-bias antipodal pairs.**
  - Antipodal-pair features (D_i ≈ 0.5) have no positive interference to suppress, so they learn a bias of
    about +0.01. Their inactive pre-activation sits right at the activation threshold, and any weight
    perturbation of RTN size switches them on.
  - Crowded features learn negative biases (about −0.08), and those biases act as a buffer.
  - The confirmation ran on 20 fresh, pre-registered models. At 3-bit per-row RTN every antipodal feature is
    damaged, against 46% of the other features. At 4 bits, almost only antipodal features break.
  - Isotropic Gaussian noise of the same size gives the same picture, so the cause is sensitivity to any
    perturbation, not something special about rounding.
- **Using geometry to allocate bits does not beat uniform 4-bit RTN.**
  - At a fixed 4.0 bits per weight, mixing 3- and 5-bit columns costs about 2× the loss increase, and no
    geometry-based ranking recovers that.
  - A generic loss-aware rounding removes about half of the quantization loss in one setting, but it does not
    rescue the fragile features.
- **A real-model check replicates Duan exactly, but the mechanism does not transfer.**
  - In Pythia-70M with a pretrained SAE, Duan's survival and perplexity numbers reproduce to within 0.02
    points.
  - Threshold density adds nothing to peak activation for predicting which SAE features survive. Its strong
    link to new false activations is entirely feature frequency.
  - A margin analog clears its pre-registered bar. Post hoc, though, its information turns out to be Duan's
    own activation statistics in another form.
- **Head vs tail is a weak side result.** Under a power-law frequency, the pre-registered test leans toward
  "the tail loses a larger relative fraction". The effect is weak and not robust, it comes from the
  normalization, and frequency is nearly collinear with geometry in those models.
- To my knowledge, the antipodal-fragility mechanism has not been reported before. Its premise, that
  antipodal pairs need no negative bias, is known from Elhage et al. (2022). For two hidden dimensions in the
  high-sparsity limit, Chen, Lau et al. (2023) also show it.

## 1. Motivation

Duan (2026), "Perplexity Can Miss SAE Feature Damage Under Quantization", quantized Pythia-70M and Gemma-2-2B
with per-output-channel RTN and tracked sparse-autoencoder (SAE) features.
- Some features degrade while perplexity stays flat.
- Whether a feature survives can be predicted from full-precision statistics: the cross-validated AUC is
  0.92–0.97, and peak activation is the strongest single predictor.
- The paper offers a heuristic ("quantization perturbations compete with each feature's activation signal")
  but measures no geometry.
- Its released code differs from the method the paper describes, in ways that matter (Section 8).

In a real model the true features are unknown, so the question of *why* a given feature breaks cannot be
asked directly.

The toy models of superposition (Elhage et al., 2022) give exactly that control. The features are known, and
their geometry (norms, overlaps, biases, polytopes) can be read straight off the weights. The model also
trains in seconds. My question:

> In a model with known ground-truth features, which features does weight quantization break first, and can
> superposition geometry predict this in advance?

## 2. Setup

**Model.**
- Features x ∈ R^n are independent: x_i = 0 with probability S, otherwise Uniform[0, 1].
- The model is h = Wx and x̂ = ReLU(Wᵀh + b), with W ∈ R^{m×n} tied as encoder and decoder. The loss is the
  importance-weighted MSE.
- Settings A–C use the official notebook's training setup: AdamW with weight decay 0.01, lr 1e-3, batch 1024.
  The paper itself states no hyperparameters.
- Setting D uses Liu, Liu & Gore's (2025) recipe instead: their weight-decay knob γ on W only, lr 1e-2 with
  2,000 warm-up steps and cosine decay, and 20k steps.
- Every configuration is trained as many instances in parallel, and all seeds are reported.

| Setting | n | m | Importance | Sparsity | Used for |
|---|---|---|---|---|---|
| A | 5 | 2 | 0.9^i | sweep S | sanity check (pentagon) |
| B | 20 | 5 | 0.7^i | sweep S | main replication |
| C | 100 | 20 | uniform | sweep S | main quantization testbed |
| D | 200 | 20 | uniform | power law p_i ∝ (i+1)^−α | head vs tail |

**Quantization.**
- Symmetric RTN with qmax = 2^(b−1) − 1 at 8, 6, 5, 4, 3 and 2 bits.
- Scales per tensor, per column (one per feature) or per row (one per hidden dimension).
- Only W is quantized, tied; b stays in FP32.
- A matched-noise control replaces each column's rounding error with an isotropic Gaussian direction of the
  same norm. This separates the structure of rounding from generic perturbation.

**Damage per feature.** All damage is measured on an eval set stratified by feature: 2,000 rows with the
feature active and 20,000 with it inactive. The FP model and every quantized model use the same set. The
recorded quantities are:
- dR2: the change in the feature's R².
- the change in false activations: P(x̂_i > 0.05 | x_i = 0).
- the change in leak energy: E[x̂_i² | x_i = 0].
- dLoss: the feature's share of the increase in total loss.
- a composite *damaged* label: Pearson survival < 0.9, OR dR2 > 0.1, OR a rise in false activations
  > 0.05.

Pearson survival alone, the criterion Duan uses, turned out to be almost useless in the toy model. At ≥ 4
bits essentially every feature "survives" even when its R² drops a lot (Section 4). A feature is *eligible* if
its norm is > 0.5 and its FP R² is ≥ 0.5.

**Discipline.**
- Every hypothesis test was pre-registered in `notes/LOG.md`, together with its primary cell, outcome,
  decision rule and stop rules. The analysis code was frozen (committed) before the confirmatory data were
  read.
- The setting-C hypotheses were explored on one set of models (E, seed 0) and decided on a fresh set (K,
  seed 1).
- A finding discovered post hoc on K was then confirmed on a third fresh set (K2, seed 7).
- Post-hoc analyses are labelled as post hoc throughout.
- Mistakes are reported where they happened (Sections 6 and 9).

## 3. Replication sanity checks

![setting A](../figures/rep_5in2_arrows.png)

- **Setting A (5 features in 2 dimensions).**
  - At S = 0, all 5 seeds keep exactly 2 orthogonal features.
  - The pentagon (all five D_i ≈ 0.4) forms in 4/5 seeds at S = 0.9, 3/5 at S = 0.95 and 5/5 at S = 0.99,
    with adjacent angles of 70°–74°.
  - The alternative solution is always a square: two antipodal pairs, with the least important feature
    dropped.
- **Setting B (20 in 5).**
  - The number of represented features rises with sparsity (Spearman 0.998).
  - The pre-registered bias criterion, "negative mean bias wherever superposition is present", **failed as
    written**. In the early antipodal regime (S ≈ 0.47–0.92), represented features carry a small *positive*
    bias.
  - Negative, interference-suppressing biases appear only in dense superposition: every instance with
    S ≥ 0.957 has a negative mean bias. This turns out to be the seed of the main finding.
- **Setting C (100 in 20, scaled down from the paper's 400/30).**
  - The "sticky" plateau of D* = m/‖W‖_F² at 1/2 is reproduced on a fine sparsity grid: four adjacent points,
    1/(1−S) from 2.1 to 3.3, lie within ±0.05 of 1/2.
  - The coarse grid missed it. The plateau spans about a factor of 1.6 in 1/(1−S), the same as the coarse
    grid's spacing.
  - The plateau at 1 failed as written, because only S = 0 lies on it at this grid resolution.
  - With n/m = 5, every feature is represented from 1/(1−S) ≈ 10 onward, and D* saturates at about 0.15.

![D* fine grid](../figures/rep_dims_fine_dstar.png)

## 4. The quantization sweep

- **8 bits is lossless in practice**: the median |dR2| is ≤ 4e-4 in settings A–C.
- **Damage rises monotonically as bits fall**, for every granularity in settings A–C. Setting D's weak regime
  is the exception (Section 6).
- **Rounding is about as damaging as matched noise in B and C.**
  - The RTN/noise ratio of the mean loss increase is 0.89–1.06.
  - In the 2-D setting A, rounding is 2–4× *less* damaging. With per-column scales, the dominant coordinate of
    a 2-vector is represented exactly.
- **Pearson survival is uninformative.**
  - At ≥ 4 bits the fraction of eligible features with correlation ≥ 0.9 is 1.00 in every setting, even where
    dR2 is large, so the analysis uses the composite label.
  - In C at S = 0.749, 84% of eligible features survive 4 bits and 45% survive 3 bits. 2 bits destroys
    everything.

## 5. Main result: the antipodal pairs break first

**Pre-registered tests.**
- They ran on K: 10 fresh models in setting C, at S = 0.7488, per-row RTN and 3 bits.
- The statistic is the within-model Spearman ρ, averaged over models, with a 95% t-interval and a Holm
  correction.

| Hypothesis (as stated before the data) | Mean ρ | 95% CI | Holm p | Verdict |
|---|---|---|---|---|
| H1: smaller margin → more new false activations | +0.49 | [+0.34, +0.64] | 8e-5 | not supported, **opposite sign** |
| H2: smaller D (more crowded) → larger dR2 | +0.40 | [+0.29, +0.51] | 4e-5 | not supported, **opposite sign** |
| H3: larger norm → smaller dR2 | +0.00 | [−0.17, +0.17] | 0.98 | not supported (null) |

- H1's opposite sign was registered in advance as an alternative, the "threshold density" reading: features
  whose inactive pre-activation sits close to the threshold should flip most easily.
- Duan's own predictor does hold here. Higher FP peak output means less damage (Spearman with dR2 −0.36; AUC
  0.70), whereas feature norm predicts nothing.

![pre-registered tests, exploration vs confirmation set](../figures/phase3_forest_E_vs_K.png)

**What drives it (post hoc on K, then confirmed).** In this cell the features fall into discrete geometric
classes. The fragile class is the antipodal pairs, D_i ≈ 0.5 (K, per-row 3 bits):

| Class | n | D | Bias | P(pre-activation within 0.05 of threshold) | Damaged at 3 bits |
|---|---|---|---|---|---|
| crowded (margin < −0.1) | 108 | 0.40 | −0.087 | 0.10 | 37% |
| intermediate | 249 | 0.45 | −0.075 | 0.15 | 45% |
| antipodal pairs | 93 | 0.50 | +0.010 | 0.65 | 99% |

**Confirmation on fresh models (Phase 3b, pre-registered, 20 new models).**
- The difference in damaged fraction between antipodal and other features, per model, is **+0.53, 95% CI
  [+0.48, +0.58]**, t-test p = 1e-14 (18 of 20 models had antipodal features).
- Registered verdict: *Supported*.
- At 4 bits the difference is +0.98: essentially only antipodal features break.
- Under the matched-noise control it is +0.51.
- The threshold-free outcomes point the same way, with smaller margins: dR2 +0.04, leak energy +0.002.

![antipodal class](../figures/phase3b_antipodal.png)

**Mechanism.**
- An antipodal pair is the one arrangement with no positive interference. When feature j is active, feature
  i's pre-activation is negative and the ReLU removes it. The model therefore has no reason to pay for a
  negative bias, and learns b_i ≈ 0 (here +0.01).
- Its pre-activation on the inactive stratum then sits at the threshold. Any perturbation, whether rounding
  or Gaussian, pushes a large fraction of it over.
- Crowded features have positive interference, so they learn a negative bias to filter it. That bias is also
  a margin against weight noise.
- In short: **the protection against interference that the model learns is also protection against
  quantization, and the features that needed no protection have none.**

**Two limits of this result.**
- It lives at the one sparsity where antipodal pairs and denser polytopes coexist (S ≈ 0.75 for n/m = 5). At
  S ≥ 0.84, per-feature damage in setting C is dominated by the random draw of the perturbation and is not
  reliably predictable.
- The binary label depends on the 0.05 detection threshold. The threshold-free differences are real but
  smaller.

## 6. Side result: head vs tail is weak, and frequency is tangled with geometry

Real-model results disagree on whether quantization hurts frequent or rare knowledge more (Wu et al., 2026;
Chang et al., 2025, whose "long tail" is not a frequency). Setting D gives each feature a frequency from a
power law (n = 200, m = 20).
- The degree of superposition is set with Liu, Liu & Gore's knob γ.
- The strong regime (γ = −1) is checked with phi_half, the fraction of features with norm > 1/2. It is 1.00 /
  0.62 / 1.00 / 0.99 for (α, k) = (1, 2) / (1, 5) / (1.5, 2) / (1.5, 5), with 5 models each. So (1, 5) is not
  strongly superposed, and the registered sensitivity analysis drops it.
- Head and tail are the top 10% and bottom 50% of the *eligible* features. With k = 2, 97–98% of the true
  bottom-50% tail is eligible. With k = 5 only 4–6% is, so there the eligible "tail" consists mostly of
  mid-frequency features.
- A weak regime could not be built at this size. Liu et al.'s γ = 0.1 is not weak here, and γ = 7 shrinks the
  features so much that 2-bit RTN *improves* the model: ternary rounding re-inflates the norms, which matched
  noise does not do. The registered sanity rule stopped it.

**(a) The registered result is weak and not robust.**
- Strong regime, per-row 3-bit, within-model Spearman of frequency with relative damage dR2/R2_fp:
  **ρ = −0.21 [−0.27, −0.15]**, Holm p = 1e-6.
- The registered category is "the tail loses a larger relative fraction".
- |ρ| only just clears the pre-registered 0.2 size threshold.
- The registered sensitivity analysis (phi_half ≥ 0.9) gives −0.17.
- In the secondaries the effect shrinks with more bits: −0.08 to −0.12 at 4 bits and −0.01 to −0.06 at 5 bits,
  across the three granularities.

**(b) Three damage measures, three answers, one reason.**
- *dR2 (the change in R²)* does not depend on frequency (ρ = −0.03). Its two parts cancel:
  - The error on the active stratum leans toward the head (+0.38).
  - Leakage on the inactive stratum is divided by Var(x_i) ≈ p_i/3, which inflates it for rare features
    (−0.20).
- *Relative damage dR2/R2_fp* is negative only because of the denominator. FP R² is 0.61 in the tail and 0.94
  in the head (ρ(p, R2_fp) = +0.96).
- *Absolute damage dLoss* sits in the head (ρ = +0.72), because it is mechanically proportional to p.
- New false activations also sit in the head: 8% vs 0.3% in the tail, ρ = +0.52 (post hoc).
- The size of each feature's weight error ‖ΔW_i‖ does not depend on frequency (ρ = +0.03, post hoc). So which
  bucket "loses more" is decided by the normalization, not by a different response to the perturbation.

**(c) Frequency and geometry cannot be separated here.**
- In the strong-regime models, frequency is almost collinear with how close a feature's inactive
  pre-activation sits to the threshold (ρ = +0.94), and strongly correlated with its bias (+0.86). Both were
  recorded before the test.
- A frequency effect in this model is therefore "frequency, or the geometry it determines". Partial
  correlations are not trustworthy at this collinearity.
- The same confound explains the setting-D allocation result in Section 7.

**A mistake in the pre-registration.**
- Claude Code's Phase 4 pre-registration, written while I was away, included an out-of-sample test of the
  antipodal mechanism. It predicted that the near-threshold features, which here are the head, would be
  damaged more, so ρ(p, relative damage) > 0.
- The result was negative, and the registered verdict is *not confirmed*.
- The test used the wrong outcome. A leakage mechanism should be tested on leakage, and the project's own
  design note maps leakage to new false activations.
- On that outcome the direction fits the mechanism (ρ = +0.52, above). That is only post hoc, so the
  registered verdict stands.

**Real models.** Two real-model results point the same way as the head's larger raw leakage:
- Duan (2026) reports that "non-survival rises with firing rate" under RTN, and that rarer SAE features "are
  not more fragile and if anything survive better" once other statistics are controlled for.
- Borobia et al. (2026) find that rare SAE features survive *pruning* better.

Both measure correlation in an SAE basis, not leakage. So this is a consistent direction, not a
confirmation.

## 7. Can geometry tell us where to spend bits?

**Setup.**
- The budget was fixed at 4.0 bits per weight with per-column scales. Half of the represented columns got 5
  bits and the other half 3 bits.
- The ranking used one of: smallest margin (the original plan), largest margin, highest threshold density,
  largest norm, or a random order (20 draws).
- Two greedy rounding rules were also tried. One minimizes leakage on a calibration set; the other is a
  generic loss-aware control.
- The comparisons were pre-registered and run on 40 fresh setting-C models plus the 20 strong-regime
  setting-D models.

![arms vs uniform](../figures/phase5_arms_vs_u4.png)

**Results.**
- **Nothing geometric beats uniform 4-bit.**
  - At the primary cell (C, S = 0.7488), uniform 4-bit RTN raises the loss by 0.89%.
  - Random 3/5 mixing adds +1.73 percentage points, 1.9× uniform's own increase. The pre-registration
    predicted about 2.1× from convexity.
  - Every geometric ranking is also worse than uniform: smallest margin +1.87, threshold density +1.56.
  - None is distinguishable from random after Holm correction. Geometry carries a little information: there
    are a few BH-significant gains over random at other sparsities, all ≤ 0.6 points. That is never enough to
    pay the mixing penalty.
- **Leak-minimizing rounding backfires.**
  - It cuts calibration leakage by 7–76% in setting C and by 84% in D.
  - It costs far more in active-stratum accuracy: +1.5 points at S = 0.75 and +25 points at S = 0.96.
- **Generic loss-aware rounding works in setting C, but only on average.**
  - It lowers the loss increase from 0.89% to 0.41% at S = 0.75, halfway to 5-bit, and helps at every
    sparsity.
  - It does *not* reduce the worst-feature damage or the damaged fraction. It spreads small gains rather than
    rescuing the fragile features.
  - In setting D it overfits the calibration set in one configuration.
- **In setting D, ranking by threshold density does beat uniform, but that is frequency allocation in
  disguise.**
  - There, threshold density correlates 0.92 with frequency (Section 6c), and an explicit frequency ranking
    does just as well (post hoc).
  - It wins only for the steeper power law (α = 1.5).
- Per-feature bit allocation has no direct counterpart in real models, where features are not axis-aligned.
  This section is a mechanism demonstration, not a method.

## 8. Bridge to a real model: Pythia-70M with a pretrained SAE

**Question.** Does anything learned in the toy model help predict which SAE features break in a real model?

**Setup: Duan's setting.**
- Model: Pythia-70M-deduped.
- SAE: the SAELens SAE `pythia-70m-deduped-res-sm` on the layer-4 residual stream, 32,768 features, from
  Marks et al. (2024).
- Data: 199,680 WikiText-2 tokens.
- Quantization: RTN on the transformer-block linear weights.
- Survival: the Pearson correlation of each feature's activations before and after quantization, over all
  tokens, must exceed 0.9.
- I approved the plan. The pre-registration was frozen before any quantized run, after an independent review
  that found and fixed 9 problems, including two in the statistics.

**Replication first.**

| Bits | Survived (mine / Duan) | Perplexity (mine / Duan) |
|---|---|---|
| INT8 | 98.3% / 98.3% | 68.53 / 68.53 |
| INT7 | 86.3% / 86.3% | 69.37 / 69.37 |
| INT6 | 62.4% / 62.4% | 77.83 / 77.83 |
| INT5 | 38.0% / 38.0% | 116.15 / 116.15 |
| INT4 | 14.5% / 14.5% | 319.59 / 319.60 |

- Duan's four-statistic predictor gives AUC 0.924 here, as in the paper.
- Peak activation alone gives AUC 0.879. Duan does not report this number.

**Two things about Duan's pipeline, from reading the released code.**
- It quantizes TransformerLens-processed weights, with Q/K/V scales shared across attention heads.
- It feeds the SAE a mean-centered residual, which the SAE reconstructs worse than the raw residual (fraction
  of variance unexplained 0.232 vs 0.193).
- Following the paper's stated per-output-channel method on raw weights damages much less: at INT6, 85% of
  features survive instead of 62%, and perplexity rises 4.8% instead of 14.7%.
- Three things differ between the two pipelines, and this run does not separate them. My registered tests use
  Duan's code, so they can be compared with his paper.

**Registered tests** (INT6, 5,688 active features):

| Test | Result | Verdict (pre-registered rule) |
|---|---|---|
| Threshold density → survival, beyond peak | dAUC +0.008 | no added information (minimum is 0.02) |
| Margin analog → survival, beyond peak | dAUC +0.024, larger margin = *lower* survival | adds information |
| Threshold density → new false activations, beyond peak | partial ρ +0.79 | supported, but frequency (see below) |
| Margin analog → new false activations, beyond peak | partial ρ +0.26 | associated |

**What it means.**
- **The toy-model mechanism does not transfer.** Threshold density does not help predict survival.
  - Its strong link to new false activations disappears once feature frequency is controlled for (ρ = 0.79
    becomes −0.01). That control was registered in advance, because frequency and geometry were confounded in
    Phase 4–5.
  - In this model, frequent features leak more (partial ρ of rarity with new false activations: −0.82). That
    is the same direction as Duan's "non-survival rises with firing rate" and as the head's raw leakage in
    Section 6, though the metrics differ.
- **The margin analog's added information is Duan's own statistics** (post hoc).
  - The analog divides by the feature's mean activation when active, which is mean / firing rate, built from
    two of Duan's statistics.
  - Beyond peak and that one quantity, the margin adds 0.003. Beyond Duan's four statistics it adds
    0.001–0.002.
  - The encoder bias alone adds 0.003 beyond peak.
- So **no geometric quantity I tested improves on activation statistics in this model.**
  - One possible reason, not tested: in the toy model the fragile geometry is the model's own weights and
    biases. An SAE's encoder bias and decoder overlaps belong to a dictionary fitted on top of the model.
- As in the toy model, matched Gaussian noise does about the same damage as rounding: 63.8% survive vs 62.4%.

## 9. Limitations

- **Scope.** One toy architecture, a tied ReLU autoencoder. The main finding rests on one (n, m, S) cell:
  n = 100, m = 20, S ≈ 0.75, where antipodal pairs and denser polytopes coexist.
- **Thresholds.** The damaged label uses a 0.05 detection threshold for false activations. The threshold-free
  outcomes agree in sign but show smaller gaps.
- **Setting D's weak regime** could not be built at this size without heavy weight decay, which breaks the
  premise of a damage study. Section 6 therefore covers strong superposition only.
- **Errors, all in `notes/LOG.md`:**
  - Several pre-registered sanity criteria failed as written (B's bias criterion; C's coarse-grid plateaus).
  - A first pre-registration draft for Phase 3 picked a primary cell whose outcome was unreliable. Review
    caught this before any test ran.
  - The Phase 4 mechanism test used the wrong outcome (Section 6).
  - The Phase 5 setting-D allocation result is confounded with frequency (Section 7).
- **The real-model bridge is one model, one SAE, one layer.**
  - Its roughly 5,700 features are not independent, so permutation p-values sit at their floor, and the
    pre-registered minimum effect sizes carry the decisions.
  - The margin result was reinterpreted post hoc (Section 8).

## 10. Related work

- **The toy model.** Elhage et al. (2022) introduced it, its polytope geometry, and the role of negative
  biases in filtering positive interference.
- **Zero bias for antipodal pairs, analytically.** Chen, Lau et al. (2023) work in the high-sparsity limit
  with two hidden dimensions. They show that the 4-gon critical point (two orthogonal antipodal pairs) has
  b* = 0 and ‖W_i‖ = 1 (their Table A.1). That is consistent with the near-zero biases observed here for
  antipodal pairs in setting C, which is a different regime (n = 100, m = 20, S ≈ 0.75).
- **Quantized TMS.** Tlaie (2024) notes that biases are not negative until the antipodal structure is lost.
  That work studies quantization-aware training of ternary TMS models, not post-training quantization.
- **Superposition and robustness.** This line of work concerns *input-space* adversarial attacks at the model
  level:
  - Elhage et al. (2022).
  - Gorton & Lewis (2025), who also observe an unexplained *dip* in vulnerability at antipodal
    superposition.
  - Stevinson et al. (2026).
- **SAEs in Pythia.** The bridge uses the residual-stream SAEs of Marks et al. (2024).
- **Quantization damage to features in real models.**
  - Duan (2026) studies per-feature quantization damage in real models without geometry. This project supplies
    a geometric mechanism in a setting where the features are known, and agrees with Duan's peak-activation
    predictor.
  - Borobia et al. (2026) do the same kind of SAE analysis for pruning.
- **Why PTQ works at all.** Chen, Beyer et al. (2026) explain it at the level of hidden states (errors
  counteracting across layers), not individual features.
- **Novelty.** To my knowledge, nobody has reported the specific result that zero-bias antipodal features are
  the most fragile under post-training weight quantization, so that more crowded features are damaged *less*.

## 11. Next steps

1. **Separate the three differences between Duan's code and his stated method**: shared Q/K/V scales,
   processed weights, and the residual convention. One of them roughly halves the damage at INT6.
2. **Test the geometry where it lives.** A margin defined on the model's own neurons or attention, rather than
   on an SAE dictionary, in a second model or layer.
3. **Map the result across (n, m, S)**, to see whether antipodal fragility holds wherever antipodal pairs
   coexist with denser structures, including setting B.
4. **Re-test the setting-D mechanism with the right outcome** (new false activations), pre-registered, on
   fresh models.

## References

- Borobia, H., Seguí-Mas, E. & Tormo-Carbó, G. (2026). How Pruning Reshapes Features: Sparse Autoencoder
  Analysis of Weight-Pruned Language Models. arXiv:2603.25325.
- Biderman, S. et al. (2023). Pythia: A Suite for Analyzing Large Language Models Across Training and
  Scaling. ICML 2023.
- Chang et al. (2025). Why Do Some Inputs Break Low-Bit LLM Quantization? EMNLP 2025. arXiv:2506.12044.
- Chen, Y., Beyer, M., Zhu, J. & Chen, J. (2026). Why Does Post-Training Quantization Work? arXiv:2609.11716.
- Chen, Z., Lau, E., Mendel, J., Wei, S. & Murfet, D. (2023). Dynamical versus Bayesian Phase Transitions in a
  Toy Model of Superposition. arXiv:2310.06301.
- Duan, E. (2026). Perplexity Can Miss SAE Feature Damage Under Quantization. arXiv:2606.03002.
- Elhage et al. (2022). Toy Models of Superposition. Transformer Circuits Thread.
- Gorton & Lewis (2025). Adversarial Examples Are Not Bugs, They Are Superposition. arXiv:2508.17456.
- Liu, Liu & Gore (2025). Superposition Yields Robust Neural Scaling. NeurIPS 2025. arXiv:2505.10465.
- Marks, S. et al. (2024). Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs in
  Language Models. arXiv:2403.19647.
- Stevinson, E., Prieto, L., Barsbey, M. & Birdal, T. (2026). Adversarial Vulnerability from Interference
  Between Features in Superposition. ICML 2026. Preprint arXiv:2510.11709, titled "Adversarial Attacks
  Leverage Interference Between Features in Superposition".
- Tlaie, A. (2024). Toy Models of Superposition: what about BitNets? LessWrong, 8 August 2024.
- Wu et al. (2026). The Asymmetric Harms of LLM Compression. arXiv:2608.19670.

## Provenance (every number above)

| Section | Numbers | Source |
|---|---|---|
| 3 | setting A pentagon fractions and angles | `results/rep_5in2/20261024-062922` |
| 3 | setting B Spearman 0.998; bias table | `results/rep_20in5/20261024-063030` |
| 3 | setting C D* (coarse grid; fine-grid plateau) | `results/rep_dims/20261024-063236`, `results/rep_dims_fine/20261024-064438` |
| 4 | 8-bit sanity; RTN/noise ratios; survival fractions | `results/quant_sweep/20261024-092937-A`, `-092954-B`, `-093058-C` |
| 5 | H1–H3 verdicts; peak_out | `results/phase3/20261024-141233-K` |
| 5 | class table | `results/phase3/20261024-141422-posthoc` |
| 5 | Phase 3b verdict and secondaries; figure | `results/phase3b/20261024-142225-K2`, `results/quant_sweep/20261024-142123-C_confirm2` |
| 6 | phi_half; frequency–geometry correlations (0.94, 0.86) | `notes/LOG.md`, Phase 4 pre-registration (FP-only facts), models `results/setting_d/20261024-145113-g-1` |
| 6 | H4 primary, sensitivity and secondaries (incl. 4/5 bits) | `results/phase4/20261024-152919-H4` (rerun `-153307-H4` is identical) |
| 6 | weak-regime diagnosis; raw leakage; ‖ΔW_i‖ (post hoc) | `results/phase4_posthoc/diagnostics_20261024.txt` (script `notes/posthoc/phase4_diagnostics.py`) |
| 6 | Duan's frequency statements | arXiv:2606.03002v2, Sections 4.4 and 5.4 |
| 7 | Phase 5 families and secondaries | `results/phase5/20261024-160152-analysis` |
| 7 | setting-D frequency confound (post hoc) | `results/phase5_posthoc/d_frequency_confound_20261024.txt` (script `notes/posthoc/phase5_d_frequency_confound.py`) |
| 8 | replication, registered tests, secondaries | `results/bridge/20261024-224808-fp` (FP stage), R1–R4 `results/bridge/20261024-230636-code-b6`, `-230918-code-b8-7-5-4`, `-231452-stated-b8-7-6-5-4`, `-232022-code-b6-noise0`; analysis `results/bridge/20261024-232316-analysis` |
| 8 | margin decomposition; B_M check (post hoc) | `results/bridge_posthoc/margin_decomposition_20261024.txt` (script `notes/posthoc/bridge_margin_decomposition.py`) |
| 10 | literature check | `notes/litcheck_2026-10-24.md`; citations re-checked against arXiv, ICML and LessWrong pages on 2026-10-24 (`notes/LOG.md`) |

## Review points still open (delete before publishing)

1. **How to present the margin result** (Section 8). The registered verdict is "adds information beyond
   peak". Post hoc, that information is Duan's own statistics. This draft states both, in that order. Keep
   it this way, or lead with "adds nothing beyond Duan's statistics"?
2. **The code-vs-stated pipeline comparison** (Section 8): 85% vs 62% survival at INT6. It is descriptive and
   has three confounded causes. Keep it as a caveat about Duan's released code, or drop it?
3. **"How this was made."** If Shen has read the Phase 3b–5 pre-registrations, add "and I reviewed them
   afterwards"; otherwise leave the sentence as it is.
4. **Stevinson et al.** The PMLR volume and pages are not out yet. Add them when the proceedings appear; do not
   guess them.
5. **Duan's title.** Version 1 (2 June 2026) was titled "How Quantization Changes Interpretable Features: A
   Sparse Autoencoder Analysis of Language Models". The current version, v2 (4 June 2026), is cited.
