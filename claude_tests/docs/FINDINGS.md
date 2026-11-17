# Findings so far (Llama-3.2-3B-Instruct, MLX)

Probes: 8 emotions (joy, sadness, fear, anger, surprise, disgust, calm, love) + neutral; mean-pooled residual states; z-scored vs held-out neutral. All numbers below are 9-way macro-F1 unless stated (chance ~0.11). "Hand-written" = the 92-passage test-only set (47 hand-written + 45 own-voice); it is never used for training, tuning or selection.

## Bug found and fixed
Manual layer loops without a causal mask leak future tokens. Old capture: masked 9.26e-06 vs unmasked 1.74e-01 difference. Capture is now mask-correct (`probekit_capture.py --check`).

## v1 corpus was lexically solvable
Test acc 0.860 / F1 0.857 (layer 6; shuffle-null p95 0.110) but TF-IDF lexical baseline 0.953. So v1 proved nothing about "beyond words".

## Frozen human test set (PLAN_v2 T0c; current primary evidence) — GoEmotions official test, n=840
Sources: `data/eval_merged2/<model>/frozen_eval.md` (+ `frozen_eval_prereg_run1.md`, the run before post hoc columns were added), `data/testsets/goemotions_test_frozen.manifest.json` (SHA-256 verified on every load). Protocol pre-registered in PLAN_v2 T0c before any result was seen. Test items: official single label, 8 classes (7 emotions + neutral, calm excluded: LLM-only), no length filter, near-duplicates of official train/dev removed; per class n: joy 91, sadness 98, fear 63, anger 128, surprise 84, disgust 74, love 151, neutral 151. Models trained on the pooled merged2 TRAIN (Llama-3.2-3B-written v2, Qwen3.8-27B-written, human GoEmotions-train rows; no calm), layer 16 / lda chosen on VAL-hard only. 8-way macro-F1 [95% bootstrap CI over test items], shuffle-null p95 about 0.14:

| system | macro-F1 | accuracy |
|---|---|---|
| probe lda@16 (selected) | 0.510 [0.476, 0.544] | 0.521 |
| probe meandiff / logreg (not selected) | 0.489 / 0.517 | 0.485 / 0.521 |
| native hidden-state LR @16 | 0.521 [0.486, 0.554] | 0.531 |
| bag of embeddings | 0.403 [0.373, 0.435] | 0.401 |
| block-0 states | 0.432 [0.401, 0.465] | 0.433 |
| TF-IDF | 0.599 [0.564, 0.630] | 0.599 |
| TF-IDF + native LR fusion (post hoc) | 0.617 [0.582, 0.650] | 0.625 |

Paired differences (macro-F1): probe − bag of embeddings +0.107 [+0.065, +0.148]; probe − block-0 +0.078 [+0.039, +0.116]; probe − native LR −0.010 [−0.036, +0.016] (indistinguishable); **probe − TF-IDF −0.088 [−0.128, −0.046]** (TF-IDF is better on human text); fusion − TF-IDF +0.018 [−0.018, +0.053] (post hoc, hidden states add no measurable gain on top of TF-IDF here). Per class (probe recall): love 0.75, fear 0.63, joy 0.58, surprise 0.58, sadness 0.53, anger 0.52, disgust 0.47, **neutral 0.19**. Strata: rater agreement >= 0.67 (n=244) probe 0.594 / TF-IDF 0.749; < 0.67 (n=596) 0.475 / 0.544; <= 12 words (n=431) 0.541 / 0.572; > 12 words (n=409) 0.475 / 0.623. Rule-mined subtypes are too small to score (negation n=10, emotion_word_neutral n=6).

## Cross-domain evaluation on three external human-labelled sources (PLAN_v2 T0d; pre-registered)
Source: `data/eval_merged2/<model>/external_eval.md` (`probekit_external_eval.py`). Sources (hash-manifested, nothing trained or tuned on them): dair-ai/emotion test (tweets; labels machine-generated from hashtags, strongly lexical; license "other", no terms stated), ISEAR (first-person event descriptions; Hub copy, original host unreachable, so provenance unverified), DailyDialog test utterances (CC-BY-NC-SA-4.0; fear n=16 and disgust n=45 too small to read per class). Systems trained on HUM (LLM-written rows + all 3024 human GoEmotions-train pool rows) or LLM (LLM-written rows only), layer 16, lda; predictions restricted to the classes each source has; macro-F1 over those classes [95% CI]:

| source (n, classes, chance) | training | probe | native LR | bag of embeddings | TF-IDF | probe − TF-IDF (H1) | probe − embed (H2) |
|---|---|---|---|---|---|---|---|
| dair (1999, 6, 0.167) | HUM | 0.405 [0.382, 0.427] | 0.405 | 0.329 | 0.254 | +0.151 [+0.122, +0.178]* | +0.076 [+0.049, +0.103]* |
| | LLM | 0.334 | 0.327 | 0.179 | 0.167 | +0.167 [+0.141, +0.192]* | +0.155 [+0.131, +0.178]* |
| ISEAR (1500, 5, 0.200) | HUM | 0.567 [0.539, 0.589] | 0.575 | 0.380 | 0.434 | +0.132 [+0.101, +0.163]* | +0.186 [+0.155, +0.216]* |
| | LLM | 0.486 | 0.511 | 0.274 | 0.224 | +0.262 [+0.230, +0.293]* | +0.212 [+0.180, +0.245]* |
| DailyDialog (2193, 7, 0.143) | HUM | 0.343 [0.315, 0.367] | 0.410 | 0.234 | 0.240 | +0.103 [+0.077, +0.130]* | +0.109 [+0.083, +0.137]* |
| | LLM | 0.280 | 0.326 | 0.178 | 0.154 | +0.125 [+0.097, +0.154]* | +0.102 [+0.074, +0.130]* |

- **H1 holds on 3 of 3 sources for both training sets; H2 holds on 3 of 3.** By the pre-registered rule this supports "hidden-state probes transfer across human domains better than a lexical model trained on the same data", and better than a bag of the model's own input embeddings.
- Probe vs native LR: indistinguishable on dair (+0.000 [−0.020, +0.021]) and ISEAR (−0.008 [−0.028, +0.011]); native LR is better on DailyDialog (−0.067 [−0.089, −0.045], HUM). So a single linear direction per emotion loses little except on short dialogue utterances.
- Combined with the in-domain result above: **TF-IDF wins in-domain on reddit (frozen set); the probe wins when the target domain differs from the training domain.** Both are the same models and the same training data.
- Caveats that apply: dair labels are hashtag-derived and lexical; ISEAR provenance is unverified; sources differ in class sets, so macro-F1 values are not comparable across sources; this is one model.
- Lexicon strata on the GoEmotions frozen set (NRC EmoLex sidecar, exploratory, post hoc; `frozen_subtype_strata.md`): probe − TF-IDF is −0.091 [−0.142, −0.036] on items with an emotion word (n=538) and +0.042 [−0.060, +0.149] on emotion items with no lexicon word (n=151). The direction matches the hypothesis stated beforehand (probe relatively better where words don't give the answer away), but the no-word interval includes 0 and the difference of the two was not tested. Neutral strata are single-class (F1 of the neutral class only) and mostly show the probe's weak neutral recall.

### Human-data learning curve (pre-registered question: data-limited or representation-limited?)
Training = LLM-written rows + k human GoEmotions-TRAIN rows per class (k=378 is the fear-limited maximum), 3 subsampling seeds, frozen-set macro-F1 (mean ± sd). TF-IDF and fusion columns were added post hoc:

| k per class | probe | native LR | TF-IDF | fusion |
|---|---|---|---|---|
| 0 (LLM text only) | 0.305 | 0.271 | 0.146 | 0.262 |
| 20 | 0.439 | 0.448 | 0.451 | 0.535 |
| 80 | 0.512 | 0.548 | 0.583 | 0.638 |
| 160 | 0.542 | 0.588 | 0.620 | 0.674 |
| 320 | 0.566 | 0.632 | 0.659 | 0.704 |
| 378 | 0.576 | 0.617 | 0.666 | 0.696 |

- **D2 (pre-registered): data-limited.** Native LR k=378 − k=80 = +0.069 [+0.039, +0.099] (threshold 0.03).
- Human-only training (no LLM rows) is as good as LLM + human at every k (e.g. k=378: probe 0.596 vs 0.576, native LR 0.628 vs 0.617): once human text is in the training set, the LLM-written corpora add nothing measurable.
- With ZERO human training rows the probe still scores 0.305 [0.272, 0.338] (shuffle-null p95 0.137) while TF-IDF collapses to 0.146: probe − TF-IDF +0.159 [+0.122, +0.197], native LR − TF-IDF +0.125 [+0.090, +0.161]. So probes carry some emotion information that transfers from LLM-written to human text where lexical features do not — but at a low absolute level.
- At equal human data TF-IDF is better than the probe (k=378: −0.090 [−0.126, −0.053]) and than native LR (−0.049 [−0.085, −0.010]); fusion − TF-IDF +0.030 [−0.001, +0.062].

### Pooled merged2 evaluation (LLM-written test splits; hand-written set is non-blind)
`data/eval_merged2/<model>/v3/report_v3.md`: lda@16 chosen. TEST-hard / TEST-all (all LLM-written; the human corpus has no test split any more): meandiff 0.676/0.746, lda 0.807/0.862, logreg 0.781/0.852, native LR 0.828/0.890, TF-IDF 0.354/0.648. Hand-written (n=92, public in git history): lda 0.665 (61/92), native LR 0.622, TF-IDF 0.337; probe − TF-IDF +0.326 [+0.213, +0.438]; probe − native LR +0.046 [−0.049, +0.137]. Leave-one-WRITER-out, 3 writers (test F1 for meandiff / lda / logreg): Llama-3.2-3B-written held out 0.661 / 0.747 / 0.764; Qwen3.8-27B-written held out 0.590 / 0.658 / 0.644; reddit-goemotions held out **0.238 / 0.237 / 0.261**. Leave-one-style-out mean (worst): 0.647 (0.238), 0.734 (0.237), 0.719 (0.261); the worst fold is always reddit. These are in-distribution-friendly numbers and should not be read as human-text performance; the frozen set above is the human-text result.

## Second model family: Gemma-2-2B-it-4bit (PLAN_v2 T6; replication criteria pre-registered)
Sources: `data/eval_merged2/mlx-community__gemma-2-2b-it-4bit/` (same pipeline, corpora, splits, selection on VAL-hard, frozen and external sets). Extractor fixed for Gemma's embedding scale (x sqrt(2304) = 48): causality check masked 2e-4 vs unmasked 8.7e-2, fidelity vs the model's own forward 5.7e-4. Selected lda @ layer 14 of 26 blocks (relative depth 0.54; plateau layers 14-16 = 0.54-0.62; Llama: layer 16 of 28 = 0.57, plateau 14-18 = 0.50-0.64).
- **R1 holds:** probe - bag of embeddings, frozen +0.112 [+0.073, +0.150]; external (HUM) +0.062 [+0.036, +0.089], +0.249 [+0.219, +0.278], +0.105 [+0.079, +0.131] (dair, ISEAR, DailyDialog).
- **R2 same sign as Llama:** in-domain frozen set probe 0.513 [0.478, 0.546] vs TF-IDF 0.599; probe - TF-IDF -0.085 [-0.128, -0.046] (Llama -0.088 [-0.128, -0.046]).
- **R3 holds:** cross-domain H1 (probe > TF-IDF, CI above 0) on 3/3 sources for both training sets: HUM +0.137 [+0.107, +0.166], +0.190 [+0.159, +0.222], +0.095 [+0.069, +0.121]; LLM-only +0.127, +0.264, +0.152.
- Same data-limited picture: native LR k=378 - k=80 = +0.103 [+0.074, +0.132]; LLM-only transfer 0.313 [0.283, 0.344] vs TF-IDF 0.146 (probe - TF-IDF +0.167 [+0.128, +0.206]).
- Differences from Llama: fusion (TF-IDF + native LR) beats TF-IDF in-domain on Gemma, +0.062 [+0.027, +0.095] (post hoc; Llama +0.018 [-0.018, +0.053]); probe - native LR is -0.024 [-0.053, +0.005] on the frozen set.
- By the pre-registered rule (R1 and R3 both hold) the cross-domain claim holds on two model families (Llama-3.2-3B, Gemma-2-2B); one more family would strengthen it, and both models are small and 4-bit.

## Length-aware calibration (PLAN_v2 T3; pre-registered; Llama and Gemma)
Sources: `length_calibration.md` in each model directory; 5,014 held-out neutral GoEmotions-TRAIN comments (reserve A 3,000 + reserve B 2,014; `data/testsets/goemotions_calibration_ids*.json`), buckets by word count, |z| > 3, Clopper-Pearson 95% upper bounds, accepted if <= 2% per emotion and <= 5% per passage in every bucket with n >= 500.
- **RAW calibration fails everywhere.** The probe's z-scores are over-dispersed on human neutral text and more so the shorter it is. Llama: sd of z 2.41 (1-2 words) to 1.19 (20+); share of neutral passages with any |z| > 3: 76.0% (1-2 words), 63.4% (3-4), 47.3% (5-7), 33.4% (8-11), 23.0% (12-19), 15.3% (20+). Gemma is worse: 83.4% to 20.1%. The 1-2 word bucket has only n=416 available, below the n >= 500 needed to certify.
- **Per-bucket recalibration (5-fold cross-validated) repairs most of it but misses the strict bound.** Worst-emotion exceedance falls to 1.0-1.4% (Llama; upper bounds 1.9-3.1%) and per-passage exceedance to 4.4-7.4% (upper bounds 6.0-9.4%). Per-passage fails the 5% bound in every bucket because the z-score tails are heavier than Gaussian. By the fallback rule fixed beforehand, no length qualifies at |z| > 3: scores cannot be called calibrated at that threshold at any length.
- **Post hoc, descriptive:** the alarm threshold on max |z| that keeps false alarms at 5% of neutral passages is 3.8 (20+ words) to 8.4 (1-2 words) with the existing calibration, but a uniform 2.9-3.3 after per-bucket recalibration (99%: 3.9-4.8). This matches the earlier anecdote ("The sky is blue." scored sadness z +4.9: within a short-text false-alarm range). A display should use per-length thresholds or recalibrated z-scores; the unmodified z-scores are not comparable across lengths.

## Geometry and representation experiments (PLAN_v2 T8; cached states; pre-registered; Llama and Gemma)
Sources: `t8_experiments.md` in each model directory. Training = HUM; evaluation = GoEmotions frozen plus three external sources.
- **(a) Hierarchical valence-group -> class LR vs flat LR: no evidence it helps.** Better than flat on 1 of 4 sets for both models (ISEAR, +0.019 [+0.004, +0.033] Llama; +0.017 [+0.004, +0.030] Gemma); the claim needed >= 3 of 4.
- **(e) MLP vs flat LR: no nonlinear information found.** 0 of 4 sets for both models (frozen set Llama -0.031 [-0.058, -0.005], Gemma -0.008 [-0.030, +0.013]): the linear read-out is sufficient at the selected layer.
- **(b) Geometry matches human affect norms.** Spearman correlation between the 21 pairwise cosine dissimilarities of the probe directions and the 21 pairwise distances of the emotion words in NRC valence-arousal-dominance space, exact permutation over 5040 labelings: Llama lda rho +0.53 (p=0.016), meandiff +0.60 (p=0.001); Gemma lda +0.61 (p=0.003), meandiff +0.63 (p=0.001). Both estimators and both models meet the pre-registered rule (p < 0.05). Seven points, pairs not independent: nominal, and a validity note, not a performance result.
- Comparison count: 16 paired comparisons per model in (a)/(e) plus 2 in (b), CIs unadjusted; the 3-of-4 consistency rule guards against chance wins.

## Pooling and prompting variants (PLAN_v2 T9 / T8 c,d; pre-registered; Llama-3.2-3B, replicated on Gemma-2-2B)
Source: `data/eval_merged2/<model>/variants.md` (`probekit_variants.py`). Same HUM training, same flat-LR recipe, layer 16; only the representation changes. Macro-F1 (frozen GoEmotions / dair / ISEAR / DailyDialog): V0 mean-pool 0.641 / 0.405 / 0.575 / 0.410; V1 last token 0.489 / 0.319 / 0.579 / 0.428; V2 max-pool 0.634 / 0.407 / 0.545 / 0.415; V3 mean of last 4 tokens 0.526 / 0.335 / 0.531 / 0.386; **V4 prompted last-token read-out 0.669 / 0.392 / 0.710 / 0.482**; V5 concat of layers 14, 16, 18 0.642 / 0.394 / 0.564 / 0.389.
- **Pre-registered rule (variant - V0 CI above 0 on >= 3 of 4 sets): no variant qualifies.** V1, V2, V3 and V5 do not help (V1 -0.151 [-0.187, -0.116] and V3 -0.114 [-0.149, -0.078] on the frozen set).
- **V4 (prompted read-out) is suggestive, not a finding:** +0.136 [+0.110, +0.164] on ISEAR and +0.072 [+0.039, +0.106] on DailyDialog (CIs exclude 0), +0.028 [-0.003, +0.058] on the frozen set, -0.013 [-0.042, +0.015] on dair: 2 of 4, one short of the rule. With 20 comparisons unadjusted it is nominal; a replication was planned and **did not confirm it**: on Gemma-2-2B V4 is 0 of 4 and slightly negative (frozen -0.023 [-0.056, +0.011], dair -0.037 [-0.065, -0.009], ISEAR -0.029 [-0.057, +0.000], DailyDialog -0.019 [-0.055, +0.016]). On Gemma no variant meets the rule either (V1 last token and V3 last-4 are clearly worse, -0.05 to -0.14; V2 max and V5 concat are equivalent to mean-pooling). Conclusion: mean-pooling is a sound default; the Llama prompted gain is a model-specific observation, not a general result.

## The audit tool (PLAN_v2 T4)
`probekit_audit.py` takes texts + labels (JSONL) and cached activations (npz + `.meta.json` provenance sidecar) and writes a report: per-class n, probe / native hidden-state LR / TF-IDF / bag-of-embeddings with bootstrap CIs and shuffle-null, paired differences, length strata, and skip-and-report for classes missing from train or test. **Acceptance met:** on the Llama merged2 train and the GoEmotions frozen set it reproduces the T0c row exactly (probe 0.510 [0.476, 0.544], native LR 0.521, bag of embeddings 0.403, TF-IDF 0.599, probe - TF-IDF -0.088 [-0.128, -0.046]); `data/eval_merged2/<model>/audit_reproduction.md`. Model-free (numpy/sklearn), tested end to end on synthetic data including missing classes and cache-order guards.

## Merged corpus v1 (superseded: used the old human corpus, which overlaps the frozen set; kept for history): synthetic v2 + human GoEmotions (reddit)
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

## Defensible claims now (see docs/CLAIMS.md for the checked list)
- Cross-domain (three external human sources, pre-registered) the probe BEATS a TF-IDF baseline trained on the same data by +0.10 to +0.26 macro-F1 with CIs excluding 0, and beats a bag-of-embeddings baseline; in-domain on the GoEmotions frozen set it does NOT beat TF-IDF (below).
- On the frozen human test set the probe is NOT better than a TF-IDF baseline trained on the same data (−0.088 [−0.128, −0.046]); it beats a bag-of-embeddings baseline (+0.107 [+0.065, +0.148]) and transfers from LLM-written to human text where TF-IDF does not (0.305 vs 0.146 with no human training rows). The earlier statement below about hand-written text refers to a non-blind set.
- Probes carry emotion-concept information beyond lexical cues on hand-written text (probe − lexical +0.492, CI excludes 0; n=92).
- On the merged corpus the best probe (lda@14) is not distinguishable from a trained classifier on the same hidden states on the hand-written set (+0.044, CI includes 0), and is weaker than it on TEST-all (0.670 vs 0.734; −0.065 [−0.105, −0.026]); on TEST-hard the gap (0.616 vs 0.669; −0.055 [−0.117, +0.005]) includes 0.
- The probe beats a bag-of-embeddings baseline (mean input embeddings) on every test set: +0.137 [+0.067, +0.206] on TEST-all (unfiltered, n=307), +0.451 [+0.331, +0.572] on hand-written (n=92, non-blind). Supports "information beyond bag-of-words baselines"; it does not rule out topic or semantic content.
- Estimators (meandiff / lda / logreg) are statistically indistinguishable on this evidence.
- Performance depends strongly on writer/style: informal reddit text held out gives F1 0.24-0.28. Coarse (4-way) own-voice accuracy is 0.73-0.78.
- Short-text reliability is untested here (n=1 short own-voice passage).
- One model (Llama-3.2-3B-Instruct, 4-bit), one layer plateau (12-18), and one synthetic writer. Nothing here is about "internal states" or feelings: these are linear read-outs correlated with emotion-concept text.
