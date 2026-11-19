# Review of WRITEUP v3 — fixes for the agent

Rules as before: no numbers from memory; every new number comes from a committed report; analyses not in `PLAN_v2.md` before the run are labelled **post hoc**. Order = priority.

## A. Must fix before release

### A1. One primary training set (section 2.1)
Section 2.1 has two tables with different training sets: T0c (merged2 train) and HUM. The encoder, external sources, seeds and Figure 1 all use HUM, so the paper effectively runs on HUM, but the abstract, and line 78 in particular, quote the T0c numbers.
- Make HUM the primary table in 2.1. Move the T0c table to an appendix with one sentence: "pre-registered first run on the merged2 training set; same conclusion".
- Delete the T0c recap line ("Probe − TF-IDF: −0.088, −0.085, −0.083 ...") from the body.
- Abstract result (1): update the gap to the HUM range (0.086–0.126) or give both ranges explicitly. Check every other in-domain number in the abstract, 1.3, and 3 against the HUM table.

### A2. Steering belongs in the abstract and contributions
Section 2.8 is a main result and is missing from both.
- Abstract: add one sentence, e.g. "Adding the audited probe directions to the residual stream did not change the emotion the model expressed more than random directions did (0 of 7 emotions, Llama-3.2-3B)."
- 1.3: add a fifth contribution, "Reading is not writing", worded the same way.

### A3. Steering section: three gaps
1. **Power.** With 40 prompts, a null of "within ±0.075" is weakly informative. Report the bootstrap CI width, or the minimum detectable difference in judged rate at 80% power, for the probe directions at S*. State in 2.8 and Limitations that small effects cannot be ruled out.
2. **The fluency rule is unclear.** The pilot fixed S* by a perplexity ratio of ≤ 1.14x. The post hoc mean-difference joy result counts as "fluent" at a ratio of 1.40, while love at 1.57 does not. State the fluency threshold that the specificity rule uses, cite where it was pre-registered, and apply it to both direction types the same way.
3. **Norm-matched is not effect-matched.** LDA directions are whitened. A unit-norm LDA direction may move the residual stream along low-variance dimensions that the model largely ignores. Add one post hoc diagnostic from cached states: for each direction, the shift it produces in the projection onto the class mean-difference direction, or in next-token logits, per unit norm. If LDA directions produce much smaller shifts, say so: the null then partly reflects the scale chosen, not only the direction.

### A4. Cite the prior result that the steering finding replicates
Marks & Tegmark (2023; COLM 2024), *The Geometry of Truth*, arXiv:2310.06824: difference-in-means ("mass-mean") probes identify directions that are more causally implicated than other probing techniques, logistic regression included. Section 2.8's post hoc pattern (mean-difference steers, LDA does not) matches it in a new domain. Cite in 2.8 and 1.2, and frame the result as consistent with that finding rather than new. Also add standard steering references, and check them before citing: Turner et al. (2023), activation addition; Panickssery/Rimsky et al. (2024), contrastive activation addition.

### A5. Probe vs encoder is not the same as hidden states vs encoder
H3 tested the single-direction **probe**. In the HUM in-domain table the **trained classifier on the same states** (0.634–0.653) is about 0.07–0.09 above the encoder (0.563); no CI is reported. Compute, **post hoc**, paired CIs for native LR − encoder on the frozen set and the three external sources, for all three models. Then word section 3 accordingly. If native LR beats the encoder, the reading becomes: "the states carry more than the encoder in-domain, but one direction per emotion does not extract it". Do not change the H3 verdict; it stays "not supported" for the probe.

### A6. Encoder caveat
Check the all-mpnet-base-v2 model card for its training data. If it includes Reddit comments, add a caveat that the encoder may have seen text from the same domain as GoEmotions in pre-training. This affects the in-domain comparison more than the external one.

### A7. Align CLAIMS.md and FINDINGS.md with the writeup
- CLAIMS "Supported": move the three hand-written-set lines (probe vs TF-IDF +0.492; estimators indistinguishable; bag-of-embeddings on TEST-all/hand-written) into a new section, "Non-blind / superseded (history)".
- CLAIMS "Supported", cross-domain line: delete the "one model" caveat and add "not beyond a frozen sentence encoder (H3)".
- CLAIMS: say "two models" in the pooling line. Add the Mistral numbers to the recalibration line.
- CLAIMS "Not established": delete "Generalisation across models: one model so far (T6)", which is stale.
- FINDINGS: retitle (three models, not "Llama-3.2-3B-Instruct, MLX"). Rewrite "Defensible claims now" from sections 2–3 of the writeup and delete its hand-written and "one model" bullets. Add the encoder qualifier to the H1 reading under the external table.

## B. Should fix

- **B1.** Section 1.3 item 1: "ties on two sources and loses on two" mixes the in-domain set with external sources. Write: "ties on dair and on the in-domain set; loses on ISEAR and DailyDialog for all three models".
- **B2.** Section 2.6: state that the τ_h macro-F1 loss comes from emotional items moving to neutral. Report the per-class recall change, from the existing report if available.
- **B3.** Figure 1: shade or label the in-domain row apart from the three external rows. The caption already says bag of embeddings is absent in-domain; also add a small "n/a" marker in the plot.
- **B4.** Figure 2 caption: the k = 378 point is a single fit, and the human-only curves dip there for Gemma and Mistral. Say so, so readers don't read the dip as an effect.
- **B5.** Limitations: add the steering limits (one model, one layer, all positions, one zero-shot judge with macro-F1 0.648, 40 prompts that already lean positive, power from A3) and the encoder caveats (A6; one encoder).
- **B6.** Optional title: "Reading is not writing: an audit of linear emotion probes in small LLMs". Decide only after A3–A5 are in.

## C. Optional, pre-register first
- **C1.** Steering replication on Gemma-2-2B. Pre-register a mean-difference vs LDA comparison with the same rule and an effect-matched strength (A3.3). This turns the post hoc mean-difference observation into a test.
- **C2.** T5 audit of one published model-written corpus (unchanged).

## D. Release checklist
A1–A7 done → B items → human read-through → README quickstart for `probekit audit` → arXiv. Check the remaining references and licenses as already listed in section 8.
