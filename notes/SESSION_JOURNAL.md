# Session journal

A running record of how this project was built: who decided what, what was run, and what was reported at
each point. Scientific results with all numbers live in `notes/LOG.md`. This file records the
collaboration and operations side, so the history can be reconstructed when the project is published.

Roles:
- **Shen Yuting (owner)**: designed the research question, the overall architecture and the phase plan, and
  made every approval and scope decision, including the rule that every change needs evidence. Claude chat only
  helped write the design up as `CLAUDE.md` / `PROMPTS.md` and did a few simple lookups (owner's
  clarification, 2026-10-24 13:50).
- **Claude Code**: implementation, experiments, analysis, independent-review workflows, and this journal. It
  acts autonomously while the owner is away, within the approved plan, and logs every decision here.

Times are local (UTC−4).

---

## 2026-10-24 (morning → 13:47): plan review, Phases 0–2, Phase 3 pre-registration

What the owner asked, what was done, and what was reported back, in order.

1. **Plan review.** The owner brought their plan, written up as Markdown with Claude chat's help
   (`CLAUDE.md`, `PROMPTS.md`), and asked for a check. Claude Code verified every cited paper and the official TMS notebook against primary sources.
   It reported 6 factual corrections (e.g. setting B importance is 0.7^i, the notebook trains with AdamW)
   and 9 design pitfalls.
2. **Owner decision: apply the corrections, but only with evidence.** Corrections went into `CLAUDE.md`, and
   the pitfalls became design notes. A verification workflow, run after the first draft, caught one error of
   Claude Code's own: the negative weight-decay formula. It was corrected from the source code
   (`exp/adamw.py`) and reported to the owner. `PROMPTS.md` was changed only where a source supports it.
3. **Owner decision: "write the plan and start running".** Phase 0 (package, 59 tests) and Phase 1 (replication
   of settings A, B, C) were run. Reported: A and B match the paper. Two pre-registered acceptance criteria
   failed as written (B bias sign, C coarse-grid plateau), and the failures were reported as failures. An
   independent code review found that a C acceptance gate was looser than the pre-registration; it was fixed
   and the result re-stated as FAIL. A registered fine-grid follow-up then passed the 1/2 plateau.
4. **Owner decision: "continue".** Phase 2 quantization sweep: 770k feature rows over settings A, B and C.
   Sanity checks passed. A review found that the noise control reused one random direction across models;
   this was fixed and the whole sweep rerun. Reported: the Pearson survival label saturates above 2 bits, and
   the margin sign flips with sparsity in setting C.
5. **Owner question: "any suggestions?"** Claude Code gave 7 recommendations backed by data (computed without
   looking at any predictor–outcome association). **Owner: "OK"**, so all 7 were executed: first git commits;
   design notes moved to `notes/DESIGN_NOTES.md`; 40 fresh confirmation models trained; Phase 3
   pre-registration drafted.
6. **Pre-registration review.** Three independent reviewers audited draft v1 (41 issues). The main one: in
   the first-choice primary cell, per-feature damage is dominated by the random perturbation draw. Claude Code
   re-derived every reviewer number before changing anything. Pre-registration v2 was committed as `f76b49a`
   at 13:31, with primary cell S = 0.7488, per-row, 3 bits, before any predictor–outcome association had been
   computed.
7. **Owner approved v2 ("ok")** at about 13:35. Seven implementation clarifications were logged as `d3b8bfa`.

Commits so far:

| commit | time | content |
|---|---|---|
| fa65497 | 13:06 | Phase 0–2: package, replication, quantization sweep |
| 002c4ba | 13:06 | design notes moved to notes/DESIGN_NOTES.md |
| 474b458 | 13:13 | confirmation-set config and sweep setting |
| f76b49a | 13:31 | Phase 3 pre-registration v2 |
| ef08535 | 13:31 | pre-registration hash recorded |
| d3b8bfa | 13:36 | approval + implementation clarifications |

## 2026-10-24 13:47: owner leaves for the afternoon

Owner's instruction, verbatim intent: if something comes up, message the phone; if there is no timely reply,
keep going according to my own logic, and archive every operation and every status report here, so the
published project shows it was built together. Owner's clarification at 13:50: the research question, the
architecture and the phase plan are essentially the owner's own design; Claude chat only turned them into
Markdown files and did a few simple lookups.

Standing rules for the autonomous stretch (Claude Code's commitments):
- Follow the approved pre-registration exactly. Any deviation is logged in `notes/LOG.md` with its date
  and reason before results are looked at; anything decided after seeing results is labelled post hoc.
- Phone notifications go out for verdicts, STOP conditions and decisions that need the owner, not for routine
  progress.
- Nothing destructive: no force pushes, no deleted results, no pushes to remote repositories.
- Phases after Phase 3 need a plan. If the owner does not answer, Claude Code writes the plan, logs it,
  notifies, and proceeds, keeping every step reversible and documented. The owner can veto any of it.

Status at 13:47:
- Phase 3 analysis code is written and has been run on the exploration set E. E is labelled exploratory; with
  5 instances none of the three primary tests is significant.
- A pre-freeze code review (fidelity, independent replication, numerics) is running.
- Next: fix the review findings, freeze and commit the analysis code, run the confirmation sweep, record the
  pre-checks, then run the frozen analysis to obtain the verdicts.

---

## 2026-10-24 13:50–14:10 (owner away): pre-freeze review, freeze

- 13:50 Owner clarified the credits: research question, architecture and phase plan are the owner's design;
  Claude chat only formatted them as Markdown and did simple lookups. README and this journal were
  corrected (`5393a34`).
- 13:55–14:08 Pre-freeze code review (3 reviewers + refutation). An independent re-implementation from the
  pre-registration text matched all 497 numbers on E. 5 bugs in secondary or reporting parts were fixed;
  details are in notes/LOG.md.
- 14:09 Analysis code frozen as `268a66e`. Exploratory E results (5 instances): no primary test is
  significant. The H1 and H2 point estimates have the sign opposite to CLAUDE.md's direction.
- Next: K sweep (~4 min), then the pre-check, then the frozen analysis on K.

## 2026-10-24 14:09–14:20 (owner away): Phase 3 verdicts, post-hoc finding, Phase 3b

- 14:09 K quantization sweep run with the registered command (124 s).
- 14:12 K pre-check written to the LOG and committed (`48a9623`) before any K association. All checks passed:
  ICC 0.65 / 0.18; ρ(norm, D) = 0.42, so H2 and H3 stay separate.
- 14:13 Frozen analysis run on K, with the code unchanged since `268a66e`. **Verdicts:**
  - H1: Not supported (opposite sign, ρ = +0.49), consistent with the registered threshold-density reading.
  - H2: Not supported (opposite sign, ρ = +0.40).
  - H3: Not supported (null, +0.00).
  - E-vs-K comparison: no disagreement.
  - Phone notification sent.
- 14:14 Post-hoc exploration, labelled as such in the LOG. The effect is carried by a discrete class: antipodal
  pairs (D ≈ 0.5, bias ≈ 0) sit at the activation threshold and are 99% damaged at 3 bits, versus 37–45% for
  the crowded classes. The same pattern appears under matched Gaussian noise.
- 14:16 Phase 3 results committed (`2a772d2`).
- 14:17 **Decision by Claude Code (owner away):** test the post-hoc finding on fresh models before reporting it
  as a result. The Phase 3b pre-registration was committed (`326fc7c`) before any training. Seed 7 was chosen
  because seed 2 would have repeated one integer seed of E's noise control.
- 14:18 Phase 3b code committed (`f45a856`). K2 training (20 instances) and its sweep were started.

- 14:22 **Phase 3b verdict: Supported.** On 20 fresh models, antipodal-pair features are damaged far more often
  at 3 bits: +0.53, 95% CI [+0.48, +0.58], p = 1e-14. They are 100% damaged vs 46% for the others; at 4 bits
  the gap is +0.98. The threshold-free dR2 gap is smaller (+0.04). The same gap appears under matched noise.
  Phone notification sent.

### 阶段小结（中文，给 owner）

Phase 3 用全新的确认集 K 检验了预注册的 H1–H3：
- H1（margin 越小，泄漏增加越多）：显著，但方向相反，ρ = +0.49。方向与事先登记的"阈值密度"解读一致。
- H2（越拥挤，损伤越大）：显著，但方向相反，ρ = +0.40。事后分析显示，这主要来自天花板效应。
- H3（范数越大，损伤越小）：零结果。不过 Duan 的峰值激活变量方向正确，AUC 为 0.70。

事后分析发现，效应来自一个离散的几何类：反极对特征（D≈0.5）。Phase 3b 在 20 个新模型上做了预注册的确认，
结果为 Supported：3 bit 下反极对 100% 受损，其他特征为 46%。

机制：反极对没有需要抑制的正向干扰，学到的 bias 约为 0，非激活时恰好停在检测阈值上。任何同等大小的扰动，
无论是舍入还是高斯噪声，都会让它们误激活。拥挤的特征则有约 −0.08 的负 bias 作为缓冲。

局限：只在 S=0.7488 一个稀疏度成立（反极对与更密的多面体在这里共存）。二值标签依赖 0.05 的检测阈值；
与阈值无关的 dR2 差距较小（0.102 对 0.072）。

## 2026-10-24 14:24–14:50 (owner away): Phase 3b closed, Phase 4 started

- 14:23 Phase 3b results committed (`091560f`); phone notification sent.
- 14:24 Advisor consulted on Phase 4. Key point taken: frequency and geometry are confounded in setting D, so
  geometry covariates per frequency bucket are mandatory.
- 14:25–14:33 Built and tested the Phase 4 infrastructure (`cdff00b`): Liu et al.'s weight-decay optimizer
  (bit-exact tests), the frequency root-finder, the setting-D trainer, per-feature eval sets. Wrote the
  Phase 4 plan and a gamma-selection rule before reading the regime sweep.
- 14:35 Regime sweep finding: gamma = +0.1 (Liu et al.'s "weak") is not weak in our setup. No gamma up to 3
  reaches phi_half ≤ 0.2, and gamma 9 or 27 collapses the model. The written rule would have picked a
  collapsed model, so a deviation was logged before refining (`bf7daeb`).
- 14:44 Refinement (gamma 4–7): weak = gamma 7 (phi_half 0.12–0.18) selected by the logged rule
  (`dcb4b55`); strong = −1.0 (one config only 0.6, reported as not strong).
- 14:45 Main setting-D training started (2 runs × 20 instances). Next: FP facts → H4 pre-registration
  (advisor review) → sweep → pre-check → frozen analysis.

- 14:52–15:17 Setting-D training done (gamma 7: phi_half 0.11–0.20; gamma −1: 1.0 / 0.62 / 1.0 / 0.99).
  FP-only facts: frequency is strongly confounded with geometry (strong regime ρ(p, near_thr) = 0.94; weak
  regime ρ(p, D) = 0.89).
- The H4 pre-registration includes an OUT-OF-SAMPLE test of the Phase 3 mechanism. Near-threshold features
  should be fragile, which predicts ρ > 0 in the strong regime and ρ < 0 in the weak one.
- Advisor review of the draft added: the mechanical tilts (only the strong-regime half discriminates), extra
  secondaries, a strong-regime sensitivity, an 8-bit contingency, and an end-to-end smoke test before the
  freeze.
- `sign_flip_p` was made exact and fast (meet-in-the-middle), with output verified identical on all stored
  Phase 3/3b inputs. The test suite went from 245 s to 23 s.
- 15:17 Pre-registration and frozen code committed (`a11e490`). Setting-D sweeps started (~25 min).

- 15:22-15:33 Setting-D sweeps finished. A crash in the post-save figure code (S is NaN in setting D) was fixed
  and the figures regenerated from the saved data.
- Pre-check: strong passes; weak STOP. In the weak regime 2-bit RTN IMPROVES the model. The pre-check was
  logged and committed (`7f8483f`) before the analysis ran.
- 15:29 Frozen H4 analysis. Strong: rho(p, rel_damage) = -0.21, so "the tail loses a larger relative fraction"
  (borderline). The out-of-sample test of the Phase 3 mechanism: not confirmed.
- Post-hoc decomposition (advisor-reviewed): the sign comes from the R2_fp normalization. dR2 does not depend on
  frequency, and raw leakage sits in the head, consistent with the Phase 3 mechanism.
- Owned error: the registered mechanism test should have used dfalse_act (design note 12).
- Weak-regime mechanism verified: ternary RTN inflates the norms of a model shrunk by weight decay; matched
  noise does not.
- The literature check found no prior report of the antipodal-fragility result (the premise is known); the
  "to our knowledge" wording is in the LOG.
