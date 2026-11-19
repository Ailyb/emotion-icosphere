# Findings so far (three models: Llama-3.2-3B-Instruct, Gemma-2-2B-it, Mistral-7B-Instruct-v0.3; 4-bit, MLX)

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

- **H1 holds on 3 of 3 sources for both training sets; H2 holds on 3 of 3.** By the pre-registered rule this supports "hidden-state probes transfer across human domains better than a lexical model trained on the same data", and better than a bag of the model's own input embeddings. **Qualifier (T11, pre-registered H3, added later): a frozen sentence encoder trained on the same data is as good or better than the probe on these sources (probe - encoder +0.004 to -0.095; H3 not supported), so the advantage is over lexical models, not over contextual encoders in general.**
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

## Third model family and scale: Mistral-7B-Instruct-v0.3-4bit (PLAN_v2 T6b; criteria R1-R5 pre-registered)
Sources: `data/eval_merged2/mlx-community__Mistral-7B-Instruct-v0.3-4bit/` (same protocol; layers searched by relative depth 0.25-0.81). Extractor checks PASS (causal-mask mismatch 3.5e-5 vs unmasked 6.7e-3; fidelity 2.8e-4). Selected **logreg @ layer 10 of 32** (relative depth 0.31; plateau layers 10, 12, 18 = 0.31-0.56), shallower than Llama (0.57) and Gemma (0.54) (R4: the best depth is model-dependent).
- **R1 holds:** probe - bag of embeddings +0.121 [+0.082, +0.160] on the frozen set; external (HUM) +0.101 [+0.072, +0.131], +0.163 [+0.133, +0.194], +0.074 [+0.052, +0.097].
- **R2 same sign:** in-domain frozen set, selected probe 0.515 [0.481, 0.547] vs TF-IDF 0.599; probe - TF-IDF -0.083 [-0.125, -0.042].
- **R3 holds:** cross-domain H1 on 3/3 sources for both training sets (HUM +0.152, +0.127, +0.054; LLM-only +0.135, +0.196, +0.092; all CIs exclude 0). LLM-only transfer to the frozen human set 0.305 [0.273, 0.337] vs TF-IDF 0.146.
- **R5 (scale): the 7B model does not close the in-domain gap** (-0.083 vs -0.088 Llama-3B and -0.085 Gemma-2B). The trained classifier on the same states is better than the single-direction probe on Mistral (native LR 0.559 vs selected probe 0.515, difference -0.044 [-0.070, -0.019]); the unselected lda probe scores 0.547, so here the validation-selected estimator was not the best on the frozen set. Fusion with TF-IDF helps in-domain (+0.058 [+0.023, +0.092], post hoc), as on Gemma.
- Same data-limited curve (native LR k=378 - k=80 = +0.096 [+0.067, +0.126]); T8 hierarchy 0/4 sets and MLP 0/4 sets (linear read-out sufficient); T3 fails the strict length bounds with the same pattern (RAW |z|>3 on 69% of 1-2 word and 15% of 20+ word neutral passages; recalibrated alarm threshold 2.8-3.1 at every length).
- **The geometry result does NOT fully replicate:** direction-vs-VAD correlation is rho +0.31 (p=0.112) for the selected logreg estimator and +0.54 (p=0.005) for meandiff; the pre-registered rule needs both estimators below 0.05, so it is not met on Mistral (met on Llama and Gemma).
- **Overall (three families, 2B-7B, all 4-bit):** the cross-domain advantage over a lexical model, the in-domain lexical win, data-limitedness, linear sufficiency and the length-calibration failure hold on all three; the geometry-VAD correlation holds on two of three.

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

## Steering tests: do the probe directions change what the model writes? (PLAN_v2 T10; pre-registered; Llama-3.2-3B, layer 16)
Sources: `data/eval_merged2/mlx-community__Llama-3.2-3B-Instruct/steering_results.md/.json`, `data/steering/{pilot.json,judge_validation.md,manifest.json}`. Design (fixed before any generation): add `strength x mean token norm (16.2)` times a unit direction to the residual stream after block 16 at every position; 40 neutral prompts ("Write three sentences about {topic}."), greedy, 60 new tokens; 7 emotions x 3 strengths for each of (i) the audited LDA probe direction and (ii) a v1-style mean-difference direction, plus 20 random unit directions at each strength (4,120 generations). Strengths came from a fluency-only pilot: S* = 0.4, so {0.1, 0.2, 0.4}; perplexity of the steered text under the unsteered model stayed within 1.14x for the probe and random directions at S*. The emotion expressed in each text was labelled by a fixed zero-shot judge (Granite-4.2-8B, local) that passed its gate on human text (macro-F1 0.648, accuracy 0.646 on the frozen GoEmotions set). An emotion counts as "specifically steerable" if the steered rate of that emotion exceeds every random direction's and the bootstrap interval over prompts excludes 0, at a fluent strength; the claim needed >= 4 of 7.
- **Pre-registered result: not established. 0 of 7 emotions** for the probe directions. Changes in the judged rate of the target emotion were within about +/- 0.075 at all strengths (e.g. joy +0.025 [-0.075, +0.150] at S*; random-direction maxima +0.00 to +0.075). Generation was fluent, so this is not breakage; judge failures (266/4120 unparsed, 6-7% in every condition) do not differ between conditions.
- **Post hoc (not pre-registered): the v1-style mean-difference directions do steer some emotions.** Same rule: joy +0.375 [+0.200, +0.525] at 0.4 (perplexity ratio 1.40; random maximum 0.000) and anger +0.125 [+0.025, +0.225] (random max +0.025) are specific; love +0.100 [+0.025, +0.200] is not fluent (ratio 1.57); sadness is not fluent (1.68); fear, surprise, disgust show nothing. That is 2 of 7, still below the claim threshold. Probe minus v1 is negative for joy at 0.4 (-0.350 [-0.500, -0.175]) and for several others.
- **Reading:** a direction that reads out emotion well (here: LDA directions whitened to separate classes) is not necessarily a direction that, when added, changes the emotion a model writes; plain difference-of-means directions did better. This is a single model, layer, judge and prompt set; neutral prompts already elicit mostly positive text (unsteered judge labels: joy 15, neutral 19 of 40), which limits headroom for joy and favours judging neutral-to-emotional shifts; the largest fluent strength (0.4 x token norm) is modest. None of this supports claims that the probes are causal handles on expressed emotion; the claim as pre-registered is not established.

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

## Defensible claims now (see docs/CLAIMS.md for the checked list; three models, 4-bit, 2B-7B)
All with test set, n and baseline in the sections above; training set HUM unless stated.
- Cross-domain (three external human sources, pre-registered H1) the probe beats TF-IDF trained on the same data by +0.054 to +0.190 macro-F1 (HUM) and +0.092 to +0.264 (LLM-only), CIs excluding 0, on all three models; it does not beat a frozen sentence encoder (H3 not supported).
- In-domain (GoEmotions frozen set, n=840) TF-IDF beats the probe (-0.086, -0.106, -0.126 HUM; -0.088, -0.085, -0.083 on the first run) and a 7B model does not close the gap; the probe beats a bag of embeddings (+0.080 to +0.098 HUM).
- The trained classifier on the same hidden states is above the probe in-domain and beats the sentence encoder there (+0.071 to +0.091, post hoc); a single direction per emotion does not extract what the states hold.
- Probes trained only on LLM-written text transfer weakly to human text (0.305-0.313 on the frozen set against 0.55-0.59 with human data); human data are the lever; LLM-written rows add nothing once human data are present.
- z-scores are not length-calibrated, 69-86% of held-out neutral human text is labelled emotional at the LLM-chosen threshold, and a human-neutral threshold restores neutral recall at a cost of 0.12-0.17 macro-F1 (pre-registered, not met).
- Linear read-out is sufficient (MLP and hierarchy do not help); mean-pooling is a sound default; probe geometry correlates with VAD norms on two of three models.
- Steering (Llama, layer 16): the probe directions do not shift expressed emotion beyond random controls (0/7; low power, fluency-limited); mean-difference directions shift joy and anger post hoc, matching Marks & Tegmark (2023).
- Nothing here is about "internal states" or feelings: these are linear read-outs correlated with emotion-concept text. Hand-written/own_voice results are non-blind and are history only (see the sections below).

## T11 (2026-11-19, pre-registered in 0d74f80): encoder baseline, EmoLex, human-neutral threshold, seeds
Sources: `data/eval_merged2/<model>/t11.md/.json`, `data/eval_merged2/t11_summary.md`; code `probekit_t11.py`. All systems trained on HUM (LLM-written rows + human GoEmotions-train pool, n=4,192); predictions restricted to each source's classes; paired bootstrap 95% CIs.
- **H3 (frozen sentence encoder; sentence-transformers/all-mpnet-base-v2, revision e8c3b32, 109.5M parameters): not supported.** Probe - encoder on dair / ISEAR / DailyDialog: Llama +0.003 / -0.090 / -0.038; Gemma -0.011 / -0.032 / -0.046; Mistral +0.004 / -0.095 / -0.087 (ISEAR and DailyDialog CIs exclude 0 in the encoder's favour for all three models; frozen set: +0.024, +0.005, -0.015, all include 0). The cross-domain advantage over lexical models is shared by a generic sentence encoder.
- **H4 (EmoLex, no training), descriptive:** EmoLex macro-F1 0.248 on the frozen set and 0.226-0.316 on the external sources; probe - EmoLex +0.300 to +0.339 (frozen) and +0.069 to +0.309 (external), all CIs exclude 0.
- **H5 (human-neutral threshold): rule not met.** tau_h (90th percentile of max z on a human neutral reserve) restores held-out neutral recall to 0.90 (from 0.18 / 0.31 / 0.14) but frozen macro-F1 changes by -0.121 [-0.163, -0.084] (Llama), -0.173 [-0.213, -0.135] (Gemma), -0.134 [-0.173, -0.096] (Mistral).
- **H6 (5 subsampled training seeds):** probe - TF-IDF has a stable sign in all 12 model x set cells (frozen -0.122 to -0.074; external +0.039 to +0.211); probe - encoder is stable in 9 of 12.
- Note: the frozen-set numbers under HUM training (probe 0.549-0.587, TF-IDF 0.674) differ from the T0c numbers (merged2 train; probe 0.510-0.515, TF-IDF 0.599); the in-domain conclusion is unchanged.

## T11 post hoc (2026-11-19; requested in REVIEW_v3; NOT pre-registered)
Sources: `data/eval_merged2/<model>/t11_posthoc.md/.json`, `.../steering_diagnostic.md/.json`; code `probekit_t11_posthoc.py`, `probekit_steer_diag.py`.
- Native LR minus sentence encoder (HUM): frozen +0.078 / +0.071 / +0.091 (CIs exclude 0), dair +0.003 / +0.004 / +0.006 (tie), ISEAR -0.082 / -0.061 / -0.061 (encoder better), DailyDialog +0.029 / +0.053 / +0.018 (only Gemma's CI excludes 0) for Llama / Gemma / Mistral.
- Probe - bag of embeddings, frozen set, HUM: +0.098 [+0.058, +0.139], +0.080 [+0.038, +0.122], +0.090 [+0.048, +0.132].
- Probe - trained classifier: frozen -0.053 / -0.066 / -0.106; DailyDialog -0.067 / -0.098 / -0.105; dair ~0; ISEAR -0.008 / +0.029 / -0.035.
- tau_h per-class recall (frozen): all seven emotion classes lose 0.23-0.43; neutral gains 0.66-0.79; 59-61% of emotional items move to neutral.
- Steering diagnostic (Llama layer 16): cosine of the unit direction with the class mean-difference axis 0.270 (probe) vs 0.274 (mean-difference) vs 0.002 (random); sd of states along the direction 0.21 / 0.59 / 0.10; perplexity ratio at 0.4 is 1.10-1.14 (probe) vs 1.37-1.68 (mean-difference). The null for the probe directions is not explained by a smaller class-axis shift per unit norm.
