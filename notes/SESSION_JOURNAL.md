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
