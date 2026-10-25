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

### 阶段小结（中文，给 owner）：Phase 4（头部 vs 尾部）

研究问题：在频率服从幂律的 setting D 里，量化对高频特征（头部）和低频特征（尾部）谁的伤害更大？预注册在 `a11e490`，
precheck 在 `7f8483f`，结果在 `3294226`，描述表在 `84f63e7`。

- **Strong 叠加区间（γ = −1，主检验可用）**：按预注册的相对损失 rel_damage = dR2 / R2_fp 衡量，**尾部损失的
  相对比例更大**。ρ(p, rel_damage) = −0.21 [−0.27, −0.15]，Holm p = 1.3e-6。效应量只是勉强越过 0.2 的门槛；
  敏感性分析（只用 phi_half ≥ 0.9 的实例）是 −0.17，方向相同但会被标成 small。
- **这个结论要和分解一起说**：dR2 本身和频率无关（ρ = −0.03）。rel_damage 的负号来自分母 R2_fp：尾部的 FP
  重构本来就差（0.61，头部是 0.94）。绝对损失 dLoss 和原始漏激活都集中在头部。例如 3 bit 时，头部新增
  假激活 8%，尾部只有 0.3%。所以“谁损失更多”取决于用哪种归一化，不能简单对应 Wu 或 Chang 的结论。
- **Phase 3 机制的样本外检验：预注册结论是 not confirmed。** 这是我预注册时的错误：漏激活机制应当用
  dfalse_act 作检验结局（design note 12），我却用了 rel_damage。事后看原始漏激活，方向和机制一致，但这只能
  算事后观察，不能算确认。LOG 里已经写明。
- **Weak 区间（γ = 7）precheck 判为 STOP**：2 bit RTN 反而让模型变好，R² 平均提高约 0.15。机制已验证：γ = 7
  的权重衰减把特征范数压缩到 ‖W_i‖² ≈ 0.36；三值 RTN 把大多数元素量化成 ±s，等于把范数撑大，于是重构
  变好；同等大小的随机噪声没有这个效果。我在 precheck 里写的候选解释被它自己设定的检验否定了，已在 LOG 中
  更正。结论：这是实验设置的局限（γ = 0.1 在 200/20 下不够 weak，只能用很强的衰减），不是关于头尾的发现。
- 选择效应：k = 5 的 strong 配置里，真正尾部只有 4–6% 的特征合格，所以那里的“尾部”大多是中频特征。
- 文献检查（针对 Phase 3b 的反极对发现）：查过的来源里没有人报告过这个发现，但它的前提已知（TMS 原文；
  Chen et al. 2023 给出了 4-gon 的 b* = 0）。写作时用 “to our knowledge”，建议句已写在 LOG 里。发布前需要人工
  核对两条引文和 Duan 论文的标题。

## 2026-10-24 15:35–15:52 (owner away): Phase 5 planned, pre-registered, started

- Phase 5 plan and pre-registration were written before any code or outcome, then amended after the advisor
  review:
  - Two registered families: arm vs uniform (deployable claim) and allocation vs random (mechanistic claim).
  - The ranking of features with NaN keys is specified.
  - The calibration seeds have a test.
  - Mechanism checks are registered.
- Arm (b), "5 bits to the smallest-margin half", is implemented exactly as PROMPTS.md specifies. Phase 3's
  opposite-sign H1 result suggests it protects the wrong half, so (b2) "largest margin" and (b3) "highest
  near-threshold density" were registered in advance as extra arms. Arm (b) was not modified.
- A fresh setting-C set P5 (seed 13, 40 instances) was trained, so that the arms motivated by Phase 3 are not
  scored on the models that motivated them.
- Implementation: `tmsq.phase5` (mixed-precision RTN; greedy rounding with exact rank-1 updates), the runner,
  and the analysis. Tests: rank-1 updates equal brute force, mixed RTN equals the Phase 2 RTN bit for bit,
  seed independence, and an analysis e2e test. 261 tests pass.
- 15:51 Pre-registration and frozen code committed (`d6abce7`). The runs started (C_P5, then D_strong; about
  20 min). The D runner is about 12 min, over the 10-min budget; this is logged.

### 阶段小结（中文，给 owner）：Phase 5（按几何结构分配 bit / 取整）

预注册 `d6abce7`，precheck `fec1998`。研究问题：在同样平均 4.0 bit/权重的预算下，按几何结构分配 bit 或调整
取整，能不能比均匀 4 bit 伤得更少？

- **结论（setting C）：都不能。** 所有预注册的方案，包括 PROMPTS.md 原定的“小 margin 列给 5 bit”、按“靠近
  阈值”分配的方案、以及以减少漏激活为目标的取整，都比均匀 4 bit 差。原因是把 3 bit 和 5 bit 混用本身就要
  付出约 2 倍的损失代价（实测 1.9 倍，和事先预测的 2.1 倍一致），几何排序只能挽回很小一部分（≤ 0.006，而
  混用代价是 0.017–0.069）。在主检验单元，按几何排序也赢不了随机分配（Holm 校正后没有显著差异）。
- **唯一胜过均匀量化的是 loss_round**：一个通用的、以校准集总损失为目标的贪心取整。这是我预注册的次要对照
  臂，不是原定方法。它在 setting C 能消掉约一半的量化损失，但不能挽救最脆弱的特征；在 setting D 的
  (α 1.0, k 2) 配置里，5 个实例全部比均匀量化差，属于校准集过拟合（稀有特征样本太少）。所以只能说在
  setting C 有效。
- **setting D 里 near_thr 方案赢过均匀量化，但这其实是频率分配**（事后检验）：在 D 里 near_thr 和频率的相关
  系数是 0.92，直接按频率分配的效果和它一样（差异 −0.0014，不显著），而且只在 α = 1.5 的两个配置里赢。不能
  当作几何结构的胜利。
- leak_round 的取舍预测得到确认：漏激活确实减少了，但活跃时的重构误差增加得更多，S 越大越糟。

下一步：Phase 6 写作。动笔前建议你先看两处：Phase 4 头尾结论的表述方式，和 Phase 5 setting D 的频率混杂。

## 2026-10-24 16:05–16:30 (owner away): Phase 5 closed, Phase 6 drafted, autonomous run stopped

- 16:01 Frozen Phase 5 analysis run, then post-hoc checks for the setting-D frequency confound (advisor
  review). Results committed (`38ec67b`); phone notification sent.
- The Phase 4 post-hoc diagnostics were saved as a script; it reproduces the archived output exactly.
- Phase 6 draft: `writeup/post.md`, `README.md` and the `phase3b_antipodal` figure. Advisor review
  corrections applied: phi_half and eligible fractions in Section 6, the scope of "every setting", post-hoc
  labelling in the TL;DR, and the literature-check output archived in the repo.
- The optional real-model bridge was NOT started, because it needs the owner's approval to download a model.
- Stopping here. Everything remaining is the owner's decision: framing, citations, the bridge, publishing.
  No push, no deletion, no force operation was done in this session.

### 阶段小结（中文，给 owner）：Phase 6（写作草稿）

- 博客草稿在 `writeup/post.md`（LessWrong 风格）。每个数字都在文末的来源表里对应到 `results/` 目录，新颖性的
  说法都带 “to our knowledge”。advisor 审阅后做了四处修正：
  - 头尾部分补上了 phi_half 和尾部合格比例。
  - “所有设置”的说法限定为 A–C。
  - TL;DR 里的事后结论标上了 post hoc。
  - 文献检查的原始输出存进了 `notes/litcheck_2026-10-24.md`。
- README 已重写：项目状态、结果摘要、每张图一条复现命令。
- 新增一张核心图 `figures/phase3b_antipodal.png`：左边是 bias 和 D 的散点图，右边是每个模型里 antipodal
  特征和其他特征的损坏比例。
- **没做的**：真实模型桥接（Pythia-70M + SAE）。它要下载模型，按 PROMPTS.md 必须先问你。
- **需要你决定的**（草稿末尾也列了）：
  1. Phase 4 的表述方式。
  2. Phase 5 setting D 的结果按“频率分配”来写，你是否同意。
  3. 几条引文要人工核对（Chen 2023、Stevinson 2025、Tlaie 2024，以及 Duan 论文的标题：v1 和当前版本标题
     不同）。
  4. 新颖性那句话。
  5. 是否先做真实模型桥接再发布。
  6. 文章的口吻。

## 2026-10-24 (owner back): review of draft v1, draft v2

- Shen reviewed the draft. Their decisions are recorded in `notes/LOG.md` (Phase 6: owner's review):
  - Phase 4 demoted to a short side section in a fixed order.
  - Phase 5 framing agreed.
  - "I" voice with an explicit statement that Claude was used.
  - The bridge runs before publication.
  - Claude chat's citation check was supplied.
- Claude Code re-checked each citation claim on the primary pages (arXiv abs/HTML, the ICML poster page)
  before editing. All of Claude chat's corrections held. One addition: Chen 2026's co-author "Beyer" was
  confirmed from the arXiv page.
- Draft v2, the README and the CLAUDE.md Duan entry were updated and committed separately from anything
  bridge-related.
- In parallel, a research workflow is collecting the facts the bridge plan needs: the SAELens API, the SAE
  release, the residual-stream processing, Duan's exact method and the install footprint. Nothing is
  downloaded or installed before the owner approves the plan.

## 2026-10-24 (owner back): bridge plan written, waiting for approval

- The research workflow `bridge-research` (5 agents, about 30 min) finished. Its key finding: "mirror Duan" is
  two different pipelines. The paper describes per-output-channel RTN on raw weights, while the released code
  quantizes TransformerLens-processed weights, with Q/K/V scales shared across heads and centered residuals
  fed to the SAE. Only the code version can be checked against his Table 1.
- The plan (decisions D1–D7) is in `notes/LOG.md`. CLAUDE.md's statement about Duan's quantization grid was
  corrected: the effective grid is the same symmetric 2^b-1 levels as ours.
- Nothing has been installed or downloaded. Waiting for the owner's reply.

## 2026-10-24 22:00–22:55: bridge approved, built, reviewed, pre-registered

- 21:58 The owner approved D1–D7 ("同意 D1–D7").
- 22:00 `.venv-bridge` was created with `--system-site-packages`, so the system torch 2.13 is reused and no
  second torch was downloaded. A dry run listed every new package, and two unfamiliar dependencies were
  traced: cwevals comes from wandb, py2store from babe. sae-lens 6.54.12 is installed; the main venv is
  unchanged.
- 22:03 Downloads: 309 MB, only the approved files. The manifest, with revisions and SHA-256, is in
  `results/bridge/20261024-220300-fetch`.
- 22:08 FP-only stage:
  - The code arm reproduces Duan exactly: 5,688 active features and FP perplexity 67.839.
  - G1 picks the raw residual (FVU 0.193 vs 0.232 centered).
  - near_thr turns out to be a frequency proxy (Spearman -0.93 with rarity). Registered secondary S1 was
    added to control for that.
- 22:10–22:45 Independent review (`bridge-prereg-review`, 18 agents): 9 findings confirmed, 6 refuted. All 9
  were fixed:
  - the cell-consistent Family B statistic;
  - the spline base for Family A;
  - exact c_j = 0 for killed features, with fp64 sums;
  - interf_sum back to D3;
  - per-half dfalse;
  - the verdict strings listed in full.
- 22:50 **Owner decision**, via a question: the primary margin uses b_enc, and beta is secondary.
- 22:55 The pre-registration was written into `notes/LOG.md` and the code frozen in the same commit. 277 tests
  pass. Next: R1, the INT6 replication gate.

## 2026-10-24 23:05–23:50: bridge runs, analysis, write-up v3

- 23:06 R1 (code arm, INT6): the replication gate G2 passed. Survived 62.41% vs Duan's 62.395%, perplexity
  77.828 vs 77.828.
- 23:09–23:22 R2–R4. The code arm reproduces Duan's Table 1 at every bit width (survival within 0.02 points,
  perplexity within 0.02). The stated arm (true per-channel RTN on raw weights) damages far less.
- 23:23–23:41 Registered analysis:
  - A_T null.
  - A_M "adds information beyond peak (coefficient -)".
  - B_T "supported", but "beyond peak, not beyond frequency" by the pre-registered S1 rule.
  - B_M "associated beyond peak (+)".
- Advisor review of the interpretation: mu+ = mean / rate is built from Duan's statistics. Post-hoc check
  (`notes/posthoc/bridge_margin_decomposition.py`): the margin adds 0.001–0.002 beyond Duan's four statistics.
  The registered verdict stands; the post-hoc reading is labelled.
- Post draft v3: new Section 8 (bridge); TL;DR, Limitations, Next steps, provenance and review points updated.
  README updated.
- Still the owner's: two framing decisions (how to present the margin result; whether to keep the
  code-vs-stated comparison), the remaining review points, and publishing. Nothing has been pushed.

### 阶段小结（中文，给 owner）：Phase 6b（真实模型桥接）

- **复现几乎完全一致**：INT8–INT4 的存活率和困惑度跟 Duan 的 Table 1 相差都在 0.02 以内；他的四预测量
  AUC 复现为 0.924。另外补了一个他没报告的数：peak 单独的 AUC 是 0.879。
- **toy model 的机制迁移不过来**：
  - 阈值密度对存活没有超出 peak 的预测力（dAUC 0.008，没过 0.02 的门槛）。
  - 它和漏激活的强相关（0.79）一旦控制频率就变成 −0.01，完全是频率效应。这个检验是预先登记的。
- **margin 类比**：按预注册规则“超出 peak 有额外信息”（dAUC +0.024），方向是 margin 越大越容易坏。
  事后检查发现，这部分信息来自分母 μ⁺，而 μ⁺ 由 Duan 自己的两个统计量构成；在他的四个统计量之上，margin
  只多 0.001–0.002。
- **顺带发现**：Duan 公开的代码和论文描述的做法不同（Q/K/V 的 scale 跨 head 共享，量化的是处理过的权重）。
  按论文说的逐通道做法，损伤小得多（INT6：85% vs 62% 存活）。这是描述性结果，三个差异混在一起，没有分开。

## 2026-10-25 (owner present): second review of the bridge, ablation, draft v4

- The owner pasted a review covering five points:
  - replication as its own first section;
  - scope "does not transfer" to the proxies and the pipeline;
  - margin: registered result first, post hoc right after, and admit the baseline choice;
  - 85 vs 62: ablate, re-run the tests on the described method, and email Duan;
  - check B_M's frequency control.
- Ablation (exploratory, 12 variants at INT6): the shared Q/K/V scales drive the gap (62.4% → 94.3%).
  LayerNorm folding explains the rest with per-channel scales (85.3% → 94.6%). Other factors ≤ 0.5 points.
- Re-run of the tests on the described method: the same verdicts. Beyond Duan's four statistics, nothing adds
  more than 0.004 AUC on either pipeline.
- The email to Duan was drafted. **At the owner's request it was put in their Gmail drafts, not sent.** The
  owner then pointed out that the sentence about the 85.3% group did not attribute the 9 points. The draft was
  updated with the one-factor comparisons (S → S-F: fold_ln only; S → S-I: SAE input only).
- Post v4 and README updated. Still the owner's: sending the email and filling in the date, the remaining
  review points, and publishing.
