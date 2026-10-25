# Email to Evan Duan (sent by Shen on 2026-10-25 from Gmail; the Gmail version is the sent text)

Status: placed in Shen's Gmail drafts on 2026-10-25 (no recipient filled in) and updated the same day with the
one-factor attribution of the 85.3% vs 94% gap. The Gmail draft is the current text; the copy below may lag it.

Purpose: before the write-up says anything public about the difference between the released code and the
paper's description, ask the author. Neutral tone. Shen sends it from their own address and records the date
in the post's implementation note ("we contacted the author on <date>").

---

**Subject:** Replicating "Perplexity Can Miss SAE Feature Damage Under Quantization": a question about per-channel RTN

Dear Evan Duan,

I am an undergraduate working on a small project about how post-training quantization damages features in
superposition, mostly in toy models. As a real-model check, I replicated the Pythia-70M part of your paper
(arXiv 2606.03002, v2 of 4 June 2026). I used your released code from the anonymous mirror
`sae-feature-survival-quantization-0141`, source commit dated 2026-06-08, whose README links to v2.

First, thank you for releasing the code. With it, your Table 1 reproduces almost exactly:
- INT6 survival 62.41% vs your 62.395%, with perplexity 77.83 in both;
- every bit width from INT8 to INT4 within 0.02 percentage points;
- your four-statistic predictor reaches AUC 0.924, as in Table 2.

While doing this I noticed that the code and the method description in Section 3.2 / Appendix L seem to
differ in a way that matters, and I would like to check my reading with you before writing anything up:

1. The quantizer computes the scale over all dimensions except the last one of each TransformerLens weight.
   For W_Q/W_K/W_V, which have shape [n_heads, d_model, d_head], this gives one scale per d_head index shared
   across all 8 heads (64 scales). It is not one scale per output channel (512).
2. The weights are quantized after TransformerLens's default processing (fold_ln, center_writing_weights,
   fold_value_biases).
3. The SAE then receives the mean-centered residual stream. On positions < 128 that SAE reconstructs the
   centered residual somewhat worse than the raw one (FVU 0.232 vs 0.193).

I ran an ablation at INT6 that changes one factor at a time, starting from either your code or the method as
described. The method as described here means per-channel scales on the raw HuggingFace weights, with the raw
residual fed to the SAE; it gives 85.3% survival. The results:

- Point 1 explains most of the difference.
  - With per-output-channel Q/K/V scales and everything else as in your code, survival is 94.3% instead of
    62.4%, and the perplexity change is +2.2% instead of +14.7%.
  - Conversely, switching only the Q/K/V scales to the shared scheme lowers the described method's 85.3% to
    59.6%.
- The remaining gap between 85.3% and about 94% comes from LayerNorm folding (part of point 2). Starting from
  the described method, folding the LayerNorm gains into the weights before quantization is the only change,
  and it raises survival to 94.6%.
- Point 3 matters little. Feeding the SAE the centered rather than the raw residual changes survival only from
  85.3% to 85.6%.
- Weight centering and value-bias folding each move survival by less than 0.5 points.

Could you tell me:
- whether the mirrored code is the version behind Tables 1 and 2 (the exact match suggests so), and
- whether the shared Q/K/V scales are intended, or whether the per-output-channel description in §3.2 is the
  intended method?

I am happy to share my code and per-feature results. My write-up will describe this neutrally as an
implementation note, and I would like to reflect your answer there correctly.

Best regards,
Shen Yuting

---

Notes for Shen (delete before sending):
- All numbers above are in `notes/LOG.md` (Phase 6b results and ablation entries). Re-check them against the
  final post before sending.
- If he replies, quote or summarize his answer in the post's implementation note. If he does not reply within
  a reasonable time, write "we contacted the author on <date>" and keep the neutral wording.
