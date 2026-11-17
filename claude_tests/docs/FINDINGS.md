# Findings so far (Llama-3.2-3B-Instruct, MLX)

Probes: 8 emotions (joy, sadness, fear, anger, surprise, disgust, calm, love) + neutral; mean-pooled residual states; z-scored vs held-out neutral. All numbers below are 9-way macro-F1 unless stated (chance ~0.11). "Hand-written" = the 92-passage test-only set (47 hand-written + 45 own-voice); it is never used for training, tuning or selection.

## Bug found and fixed
Manual layer loops without a causal mask leak future tokens. Old capture: masked 9.26e-06 vs unmasked 1.74e-01 difference. Capture is now mask-correct (`probekit_capture.py --check`).

## v1 corpus was lexically solvable
Test acc 0.860 / F1 0.857 (layer 6; shuffle-null p95 0.110) but TF-IDF lexical baseline 0.953. So v1 proved nothing about "beyond words".

## Merged corpus (current): synthetic v2 + human GoEmotions (reddit)
Source: `data/eval_merged/<model>/` (report_v3.md, robustness_*/robustness.md, confusion_v3.md). Splits n: train 1128, val-hard 153, test-hard 153, test-all 307, hand-written 92. Selected **lda @ layer 14** by VAL-hard F1 only (tau 1.78). Layer plateau (within 0.03 val-F1): 12, 14, 16, 18.

| | VAL-hard | TEST-hard | TEST-all | hand-written |
|---|---|---|---|---|
| meandiff@14 | 0.616 | 0.540 | 0.627 | 0.669 |
| lda@14 | 0.642 | 0.616 | 0.670 | 0.723 |
| logreg@14 | 0.625 | 0.602 | 0.627 | 0.616 |
| native hidden-state LR | 0.644 | 0.669 | 0.734 | 0.680 |
| lexical TF-IDF | 0.228 | 0.279 | 0.596 | 0.227 |

- Hand-written accuracy, selected probe: 67/92 (95% Wilson 0.63-0.81).
- Paired bootstrap on hand-written F1: probe − lexical +0.492 [+0.370, +0.607] (excludes 0); probe − native LR +0.044 [−0.045, +0.129] (does not exclude 0); lda − meandiff +0.056 [−0.019, +0.128] (does not exclude 0). So estimators are not distinguishable here, and the probe is no longer measurably worse than a trained classifier on the same states on this set.
- Subtypes (selected / native LR / lexical, accuracy): emotion_word_neutral (n=8) 0.88/1.00/0.25; implicit (24) 0.62/0.62/0.12; negation (9) 0.89/0.67/0.33; own_voice (45) 0.71/0.64/0.31; plain_neutral (6) 0.83/1.00/0.67. Subtype n are small; differences between the first two columns are not resolvable.
- Caveat on comparison with the earlier table below: the corpus changed (human reddit rows added), so the VAL/TEST columns are different test sets and are NOT comparable to the pre-merge numbers. Only the hand-written set is the same across runs. Hand-written F1 went up for lda (0.565 → 0.723) and native LR (0.599 → 0.680) after merging, but there was no ablation that isolates human data from the other corpus differences, so the cause is not established.

### Robustness (merged)
- Leave-one-writing-style-out, mean test F1 (worst): meandiff 0.618 (0.284), lda 0.664 (0.235), logreg 0.648 (0.247). The worst fold is always the informal reddit style (n=320 test): F1 0.235-0.284 when it is held out. The synthetic styles hold out at 0.60-0.80.
- Leave-one-writer-out: holding out reddit-goemotions gives test F1 0.235-0.284 (hand-written 0.49-0.60); holding out the synthetic writer (llama-3.2-3b) is a PARTIAL fold (no 'calm' in the remaining training data, those rows excluded) and gives 0.33-0.38. Writer shift is large; training on one writer does not transfer well to the other. There is still only one synthetic writer, so this is a two-writer test, not a general one.
- Learning curve, test-hard F1 at 40% → 100% of train: meandiff 0.570 → 0.540, lda 0.595 → 0.616, logreg 0.592 → 0.602. Flat within noise (±0.02-0.04 over subsamples). More of the same data is not the lever; varied data is.

### Confusion diagnostics (merged, layer 14, hand-written set)
| estimator | own-voice (n=45) | other hand-written (n=47) | coarse 4-way own-voice |
|---|---|---|---|
| meandiff | 29/45 = 0.64 (0.50-0.77) | 31/47 = 0.66 | 0.76 |
| lda | 32/45 = 0.71 (0.57-0.82) | 35/47 = 0.74 | 0.78 |
| logreg | 29/45 = 0.64 (0.50-0.77) | 29/47 = 0.62 | 0.73 |
| native LR | 29/45 = 0.64 (0.50-0.77) | 35/47 = 0.74 | 0.78 |
- Sadness is no longer 0/5: own-voice sadness is 3/5 (meandiff, lda, logreg) and 4/5 (native LR). Earlier 0/5 was on the pre-merge probes; n=5 per class, so treat as suggestive only.
- Remaining weak classes (own-voice, n=5 each): love is 1/5 (meandiff, lda), 0/5 (logreg), 2/5 (native LR), mostly absorbed by neutral or joy; native LR calls 3/5 joy passages neutral; calm→joy appears for meandiff (1) and native LR (2).
- Short text: only 1 own-voice passage is ≤12 words, so short-text reliability cannot be assessed from this set. The earlier observation ("The sky is blue." → sadness z +4.9) stands as an anecdote until length-aware calibration is tested (roadmap 4).

### Layer-0 baselines (PLAN_v2 T1 i; merged corpus, lda@14 probe)
Source: `data/eval_merged/<model>/l0_baseline.md` (`probekit_l0_baseline.py`). `embed` = mean of the INPUT token embeddings (bag of embeddings, no context); `block0` = after decoder block 0; both with the native-LR recipe (C chosen on VAL-hard). Macro-F1:

| | VAL-hard | TEST-hard (n=153) | TEST-all (n=307) | hand-written (n=92) |
|---|---|---|---|---|
| TF-IDF | 0.228 | 0.279 | 0.596 | 0.221 |
| embed (bag of embeddings) | 0.356 | 0.404 | 0.534 | 0.270 |
| block0 | 0.422 | 0.459 | 0.586 | 0.265 |
| native LR @14 | 0.644 | 0.669 | 0.734 | 0.680 |
| probe lda@14 | 0.642 | 0.616 | 0.670 | 0.723 |

Paired bootstrap, probe − embed: VAL-hard +0.285 [+0.185, +0.383], TEST-hard +0.210 [+0.114, +0.305], TEST-all +0.137 [+0.067, +0.206], hand-written +0.451 [+0.331, +0.572]; all exclude 0. Probe − block0 is similar (+0.085 to +0.456, all exclude 0). Probe − native LR@14: TEST-all −0.065 [−0.105, −0.026] (excludes 0), TEST-hard −0.055 [−0.117, +0.005], VAL-hard −0.004, hand-written +0.044 [−0.045, +0.129]. Hand-written accuracy: TF-IDF 25/92, embed 32/92, block0 33/92, native LR 64/92, probe 67/92.
Caveats: these are the current (non-frozen) test sets; the PLAN_v2 decision test is on the T2 frozen human sets, not yet built. TEST-hard rows are selected as hard for TF-IDF, which also lowers the embed baseline; TEST-all is unfiltered and still shows the gap. Hand-written is non-blind (public repo). Wording that is supported: "information beyond bag-of-words baselines (TF-IDF and bag-of-embeddings)", not "beyond topic or semantics".

## Pre-merge results (synthetic corpus only, n=92 hand-written test) — kept for comparison
| | VAL | TEST-hard | TEST-all | hand-written |
|---|---|---|---|---|
| meandiff@14 | 0.798 | 0.683 | 0.758 | 0.499 |
| lda@14 | 0.723 | 0.712 | 0.781 | 0.565 |
| logreg@14 | 0.764 | 0.721 | 0.779 | 0.542 |
| native hidden-state LR | 0.804 | 0.885 | 0.924 | 0.599 |
| lexical TF-IDF | 0.351 | 0.506 | 0.710 | 0.237 |
Hand-written accuracy 49/92 (95% CI 0.43–0.63). Paired bootstrap: probe − lexical +0.257 [+0.132,+0.383]; probe − native LR −0.100 [−0.186,−0.010]. Leave-one-style-out mean test F1 0.697/0.707/0.678, worst 0.600/0.615/0.593. Own-voice 21–24/45; sadness 0/5; attractors fear→surprise, love→joy, calm→joy.
Reading: trained only on synthetic text, the probes were high on in-distribution test (0.68–0.78) and much weaker on hand-written text (0.50–0.57), i.e. a style/writer gap. The merged run narrows that gap on hand-written text but the synthetic-vs-reddit gap shows up in the robustness folds instead.

## Streaming (pre-merge probes; not re-run)
Pooled acc 0.562 / AUC 0.833; max_z AUC 0.927; true-emotion top-token share 0.41. The "onset" metric is INVALID (no neutral baseline) — redesign (roadmap 5).

## Defensible claims now
- Probes carry emotion-concept information beyond lexical cues on hand-written text (probe − lexical +0.492, CI excludes 0; n=92).
- On the merged corpus the best probe (lda@14) is not distinguishable from a trained classifier on the same hidden states on the hand-written set (+0.044, CI includes 0), and is weaker than it on TEST-all (0.670 vs 0.734; −0.065 [−0.105, −0.026]); on TEST-hard the gap (0.616 vs 0.669; −0.055 [−0.117, +0.005]) includes 0.
- The probe beats a bag-of-embeddings baseline (mean input embeddings) on every test set: +0.137 [+0.067, +0.206] on TEST-all (unfiltered, n=307), +0.451 [+0.331, +0.572] on hand-written (n=92, non-blind). Supports "information beyond bag-of-words baselines"; it does not rule out topic or semantic content.
- Estimators (meandiff / lda / logreg) are statistically indistinguishable on this evidence.
- Performance depends strongly on writer/style: informal reddit text held out gives F1 0.24-0.28. Coarse (4-way) own-voice accuracy is 0.73-0.78.
- Short-text reliability is untested here (n=1 short own-voice passage).
- One model (Llama-3.2-3B-Instruct, 4-bit), one layer plateau (12-18), and one synthetic writer. Nothing here is about "internal states" or feelings: these are linear read-outs correlated with emotion-concept text.
