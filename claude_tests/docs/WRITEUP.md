# Reading is not writing: an audit of linear emotion probes in small LLMs

*Draft v4, 2026-11-19 (revised after `REVIEW_v3.md`). Every number is traceable to a committed report (paths in brackets) and to a pre-registration in `PLAN_v2.md` written before the run. This is a hobby-scale, non-commercial study of three small 4-bit open models; it makes no claim about model feelings or internal states.*

## Abstract

Linear "emotion probes" read emotion-concept directions out of an open-weight language model's residual stream and are a common basis for visualisations and proposed monitors. We ask a narrow question: what do such probes carry beyond cheap and standard baselines, and where? We audit probes on three small instruction-tuned models (Llama-3.2-3B, Gemma-2-2B, Mistral-7B; all 4-bit). Evaluation uses a frozen, hash-verified human test set (GoEmotions, n=840), three further human-labelled sources (tweets, event descriptions, dialogue), pre-registered hypotheses and decision rules, and paired bootstrap intervals. Baselines are trained on the *same* data: TF-IDF, a zero-training emotion lexicon, a bag of the model's own input embeddings, a frozen sentence encoder, and a trained classifier on the same hidden states. The results hold on all three models. (1) In-domain, TF-IDF beats the probe by 0.09–0.13 macro-F1 (−0.086, −0.106, −0.126), and the 7B model's gap is the largest. (2) Across domains the order reverses: the probe beats TF-IDF on all three external sources (+0.054 to +0.190) but does no better than a frozen 110M-parameter sentence encoder trained on the same data (probe − encoder −0.095 to +0.004 across 9 model × source cells; the encoder is better on ISEAR and DailyDialog for all three models and tied on dair). A trained classifier on the same hidden states does beat that encoder in-domain (+0.071 to +0.091, post hoc), so the states hold more than the encoder offers there, but one direction per emotion does not extract it. (3) Probes trained only on LLM-written emotional text, the recipe used by most emotion-vector studies, reach about 0.31 macro-F1 on human text, against 0.55–0.59 with human training data. Once human data are present, LLM-written data add nothing measurable. (4) The probes' z-scores are not comparable across text length (15–83% of neutral human passages raise some |z|>3, depending on length), and the probe labels 69–86% of neutral human text as emotional when its neutral threshold is set on LLM-written validation text; refitting that threshold on human neutral text restores neutral recall but costs 0.12–0.17 macro-F1, because about 60% of emotional comments then move to neutral. (5) Reading is not writing: adding the audited probe directions to the residual stream did not change the emotion Llama-3.2-3B expressed more than random directions did (0 of 7 emotions, pre-registered), whereas plain mean-difference directions shifted joy and anger (post hoc), the ordering Marks & Tegmark (2023) found for truth directions. A linear read-out is sufficient, mean-pooling is a sound default, and probe geometry correlates with human valence–arousal–dominance norms on two of three models. We release the audit tool, the frozen test-set manifests, and all pre-registrations.

## 1. Question, position and design

### 1.1 Why audit
Emotion probes are usually built from LLM-generated emotional passages and then shown to separate held-out passages from the same generator. That is weak evidence: such corpora can be lexically solvable. Our first corpus was exactly that: TF-IDF reached macro-F1 0.931 [0.906, 0.953] (topic-disjoint, 3 seeds; `data/corpus/corpus_v1_tfidf_gate.md`) against 0.857 for the probe (`docs/FINDINGS.md`; the two numbers come from different splits). The interpretability literature has known since control tasks were introduced (Hewitt & Liang, 2019) that probe accuracy means little without baselines. Emotion-probe studies rarely include them.

### 1.2 Position relative to recent work
Sofroniew et al. (2026) extracted 171 emotion-concept vectors from Claude Sonnet 4.5 using model-written stories and showed that they influence behaviour. Several studies replicated the representational part on open models with model-generated corpora (van der Ben et al., 2026; Hollowell, 2026; Jeong, 2026a,b). Reichman et al. (2026) report an emotional subspace, found from a synthetic rewritten-sentence dataset, whose probes generalise across eight real datasets and five languages. Fang et al. (2026) probe eight 1B–9B open models, including Llama-3.2-3B, on GoEmotions and two other corpora, and test transfer across corpora. Zhang & Zhong (2025) release a probing toolkit and a Reddit emotion dataset. Keeman (2026) controls for keywords differently: clinical vignettes with the emotion words removed, on six Llama and Gemma models, where emotion categorisation drops 1–7% without keywords. For steering, activation addition (Turner et al., 2023) and contrastive activation addition (Panickssery et al., 2023) add mean-difference vectors to the residual stream, and Marks & Tegmark (2023) found for true/false statements that difference-in-means directions are more causally implicated in model outputs than directions from other probing techniques.

None of these compares the probe with a lexical or bag-of-embeddings classifier trained on the same data, separates in-domain from cross-domain evaluation against such baselines, or checks calibration on neutral human text. Those comparisons are our contribution. We do not claim novelty for linear emotion probing, for cross-domain transfer as such, or for the valence geometry.

### 1.3 Contributions
1. **An in-domain/cross-domain reversal, on three model families.** A lexical classifier beats the probe in-domain; the probe beats it across domains. The probe does not beat a frozen sentence encoder trained on the same data (H3 not supported): it ties on dair and on the in-domain set and loses on ISEAR and DailyDialog for all three models, so the cross-domain advantage over lexical models is not specific to LLM probes.
2. **A cost for the common recipe.** Probes built only from LLM-written emotional text transfer weakly to human text, and LLM-written data add nothing once human data are present.
3. **Calibration and neutral detection fail.** Raw z-scores depend strongly on text length, and recalibration does not meet pre-registered bounds. This matters for any live display or monitor.
4. **Reading is not writing.** Adding the audited probe directions to the residual stream does not change the emotion the model expresses more than random directions do (0 of 7 emotions, Llama-3.2-3B, pre-registered); plain mean-difference directions do shift joy and anger (post hoc).
5. **A reusable audit tool** (`probekit audit`) with frozen, hash-verified human test sets and fetch scripts.

### 1.4 Design
**Baselines.** Each comparison trains the baseline on the same data as the probe:
- (a) TF-IDF + logistic regression;
- (b) NRC EmoLex word counts (Mohammad & Turney, 2013), no training. This separates "lexical features fail across domains" from "features fitted to the training domain fail";
- (c) logistic regression on the mean of the model's input token embeddings ("bag of embeddings": vocabulary knowledge, no context);
- (d) a frozen pretrained sentence encoder + logistic regression. `sentence-transformers/all-mpnet-base-v2` (revision e8c3b32edf5434bc2275fc9bab85f82640a19130, 109.5M parameters, 768 dimensions; sentence-transformers library, Reimers & Gurevych, 2019). The encoder is frozen; a logistic regression (StandardScaler, C on validation) is trained on its embeddings with the same recipe as the native-LR baseline. The model card lists Reddit comments (2015–2018; 726M of its more than 1B training pairs) among the encoder's training data, so it may have seen text from the same domain as GoEmotions; this bears more on the in-domain comparison than on the external sources;
- (e) logistic regression on the same hidden states as the probe (the linear ceiling);
- (f) a shuffled-label null.

Comparisons are paired over the same test items and reported with bootstrap 95% intervals. Headline comparisons are also repeated over 5 training seeds or subsamples (section 2.7).

**Pre-registered hypotheses.** H1: probe − TF-IDF > 0 cross-domain (CI above 0 on ≥ 2 of 3 sources). H2: probe − bag of embeddings > 0. H3 (added before the run, `0d74f80`): probe − sentence encoder > 0, CI above 0 on ≥ 2 of 3 external sources per model, on ≥ 2 of 3 models. If H3 fails, we report the encoder as equivalent or better and do not claim LLM-specific information. **H3 failed** (section 2.2). H4 (EmoLex, descriptive), H5 (human-neutral threshold, rule in 2.6) and H6 (five training seeds) were pre-registered in the same commit. D2: native LR at k=378 − k=80 > 0.03 means data-limited.

**Probes.** One direction per emotion (joy, sadness, fear, anger, surprise, disgust, love), fitted by shrinkage LDA, logistic regression or mean difference on mean-pooled mid-layer states (first 4 tokens skipped). Scores are z-scored against held-out neutral text, with a neutral threshold τ. Layer, estimator and τ are chosen on validation only. The extractor uses a causal mask and is checked against each model's own forward pass (relative difference 3e-4 to 6e-4); it handles Gemma's embedding scaling.

**Data.** Training: LLM-written passages from two writers (Llama-3.2-3B and Qwen3.8-27B) plus human GoEmotions comments (up to 378 per class, official train split only). Test:
- (i) a frozen GoEmotions test set: official test split, single-label items, near-duplicates of train/dev removed, n=840, 63–151 per class, SHA-256 manifest; the evaluator refuses a modified file [`data/testsets/*.manifest.json`];
- (ii) dair-ai/emotion test (tweets, hashtag-derived labels), ISEAR (first-person event descriptions) and DailyDialog (utterances).

Class sets differ across sources, so predictions are restricted to each source's classes. Calm exists only in LLM-written text (confounded with writer) and is excluded from all headline numbers.

**Models.** Llama-3.2-3B-Instruct, Gemma-2-2B-it and Mistral-7B-Instruct, all 4-bit via MLX. The Mistral model is Mistral-7B-Instruct-v0.3 (`mlx-community/Mistral-7B-Instruct-v0.3-4bit`). Mistral was chosen as a third family at the largest size that runs comfortably on the hardware used; it is an older model, and results may differ on newer 7–9B models.

## 2. Results

**Figure 1** (`docs/figures/fig1_forest.svg`; `make_figures.py`; data `t11.json`, `external_eval.json`). Forest plot: probe − baseline (TF-IDF, sentence encoder, bag of embeddings), macro-F1 with 95% CI. Rows: frozen in-domain set (shaded) plus three external sources; panels: three models; a vertical line at 0. All systems are trained on the HUM set (LLM-written rows plus the human pool); the in-domain bag-of-embeddings difference is post hoc (`t11_posthoc.json`). This figure carries the paper.

### 2.1 In-domain, lexical features win [`t11.md`, `t11_posthoc.md`]
On the GoEmotions frozen set (8-way, chance 0.125), every system trained on the same HUM data (LLM-written rows plus the human GoEmotions-train pool, n=4,192). HUM is the training set for all comparisons in 2.1–2.7 and for Figure 1; the first pre-registered run, on a different training set, is Appendix A and reaches the same conclusion.

| system | Llama-3.2-3B | Gemma-2-2B | Mistral-7B |
|---|---|---|---|
| probe | 0.587 [0.554, 0.619] | 0.568 [0.534, 0.600] | 0.549 [0.513, 0.580] |
| trained classifier, same states | 0.641 [0.606, 0.670] | 0.634 [0.598, 0.666] | 0.653 [0.618, 0.683] |
| TF-IDF | 0.674 [0.640, 0.704] | 0.674 [0.640, 0.704] | 0.674 [0.640, 0.704] |
| sentence encoder | 0.563 [0.527, 0.594] | 0.563 [0.527, 0.594] | 0.563 [0.527, 0.594] |
| bag of embeddings | 0.489 | 0.488 | 0.458 |
| EmoLex (no training) | 0.248 [0.213, 0.278] | 0.248 [0.213, 0.278] | 0.248 [0.213, 0.278] |
| probe − TF-IDF | −0.086 [−0.124, −0.049] | −0.106 [−0.144, −0.065] | −0.126 [−0.165, −0.088] |
| probe − sentence encoder (H3) | +0.024 [−0.012, +0.061] | +0.005 [−0.031, +0.042] | −0.015 [−0.055, +0.023] |
| probe − EmoLex (H4) | +0.339 [+0.294, +0.384] | +0.320 [+0.276, +0.364] | +0.300 [+0.254, +0.344] |
| probe − bag of embeddings (H2; post hoc on HUM) | +0.098 [+0.058, +0.139] | +0.080 [+0.038, +0.122] | +0.090 [+0.048, +0.132] |
| probe − trained classifier (post hoc) | −0.053 [−0.081, −0.026] | −0.066 [−0.095, −0.037] | −0.106 [−0.133, −0.080] |

[`t11.md` and `t11_posthoc.md` in each model's directory.] TF-IDF, EmoLex and the sentence encoder do not use the LLM, so their rows are identical across models. The probe sits between the bag of embeddings and the lexical and encoder baselines. In-domain the probe and the sentence encoder are indistinguishable (H3 concerns the external sources, 2.2), and the trained classifier on the same states is above the probe on all three models, so one direction per emotion leaves in-domain performance on the table.

### 2.2 Across domains, the probe beats lexical models [`external_eval.md`]
Trained on the same GoEmotions-derived data and tested on other human sources, probe − TF-IDF (macro-F1 [95% CI], H1):
- Llama: dair +0.151 [+0.122, +0.178], ISEAR +0.132 [+0.101, +0.163], DailyDialog +0.103 [+0.077, +0.130];
- Gemma: +0.137, +0.190, +0.095;
- Mistral: +0.152, +0.127, +0.054.

All intervals exclude 0. Trained only on LLM-written text, the gaps are larger: Llama +0.167, +0.262, +0.125; Gemma +0.127, +0.264, +0.152; Mistral +0.135, +0.196, +0.092.

Against the zero-training lexicon (H4, descriptive; all intervals exclude 0 in the probe's favour): Llama dair +0.133 [+0.106, +0.159], ISEAR +0.251 [+0.219, +0.282], DailyDialog +0.118 [+0.083, +0.151]; Gemma +0.119, +0.309, +0.110; Mistral +0.134, +0.246, +0.069 [+0.036, +0.100]. EmoLex scores 0.226–0.316 on the external sources and 0.248 on the frozen set: close to TF-IDF trained on HUM on dair and DailyDialog (0.254, 0.240), below it on ISEAR (0.434), and below the probe on every source.

Against the frozen sentence encoder (H3; probe − encoder, macro-F1 [95% CI], all trained on HUM; `t11.md`):

| model | dair | ISEAR | DailyDialog |
|---|---|---|---|
| Llama | +0.003 [−0.024, +0.030] | −0.090 [−0.118, −0.063] | −0.038 [−0.069, −0.007] |
| Gemma | −0.011 [−0.038, +0.018] | −0.032 [−0.061, −0.004] | −0.046 [−0.077, −0.017] |
| Mistral | +0.004 [−0.024, +0.034] | −0.095 [−0.123, −0.067] | −0.087 [−0.118, −0.056] |

The probe's interval is above 0 on 0 of 3 sources for 0 of 3 models, so **H3 is not supported** (`t11_summary.md`). The encoder is better on ISEAR and DailyDialog for every model and tied on dair. We therefore make no claim that the probes carry emotion information beyond a generic sentence encoder.

**Hidden states versus the encoder (post hoc, not pre-registered; `t11_posthoc.md`).** H3 tested the single-direction probe. The trained classifier on the same hidden states, native LR minus the encoder (HUM training; paired bootstrap):

| model | frozen (in-domain) | dair | ISEAR | DailyDialog |
|---|---|---|---|---|
| Llama | +0.078 [+0.043, +0.112] | +0.003 [−0.025, +0.030] | −0.082 [−0.109, −0.054] | +0.029 [−0.003, +0.061] |
| Gemma | +0.071 [+0.035, +0.106] | +0.004 [−0.024, +0.032] | −0.061 [−0.087, −0.035] | +0.053 [+0.018, +0.089] |
| Mistral | +0.091 [+0.053, +0.127] | +0.006 [−0.024, +0.037] | −0.061 [−0.088, −0.034] | +0.018 [−0.012, +0.048] |

In-domain the states carry more than the encoder (all three intervals above 0). Across domains the classifier ties the encoder on dair, is below it on ISEAR, and is at or above it on DailyDialog (above only for Gemma). So the hidden states hold more in-domain information than the encoder, but a single direction per emotion does not extract it, and the cross-domain picture is mixed rather than a clear win for either. The H3 verdict for the probe is unchanged.

**Robustness to ISEAR (post hoc; `t11_leave_one_source_out.md`).** ISEAR is the weakest-provenance source (Hub copy, 5 classes) and the one where the encoder's lead is largest. Dropping it, H1 still holds for all three models, and the probe still ties the encoder on dair and loses to it on DailyDialog, so H3 stays unsupported; H1 holds and H3 fails under every single-source drop.

*Reading.* Lexical models fit the vocabulary of their training domain; contextual representations transfer better. The LLM probe is no better at this than a frozen sentence encoder (and worse on two sources), so the cross-domain advantage over lexical models is not specific to these LLMs.

Caveats: one encoder only (a larger or emotion-fine-tuned encoder could widen the gap); the encoder's training data include Reddit comments, so in-domain it may have seen text like GoEmotions (the external sources are less affected); the encoder and the probe see the same training rows but the encoder reads whole sentences with a model built for sentence similarity; dair labels are hashtag-derived and lexical; ISEAR provenance is unverified; sources differ in class sets.

### 2.3 LLM-written training data are not enough; human data are the lever [`frozen_eval.md`]
- With *no* human training data, the probes score 0.305–0.313 on the frozen human set (shuffle-null p95 ≈ 0.13); TF-IDF scores 0.146.
- Adding human rows (0 → 378 per class) raises frozen-set F1 steadily: native LR at k=378 vs k=80 gives +0.069 [+0.039, +0.099] on Llama, +0.103 on Gemma and +0.096 on Mistral (D2: data-limited).
- Once human data are present, LLM-written rows add nothing measurable: human-only training equals LLM + human at every k.
- With full HUM training the trained classifier on the same states is above the probe in-domain on all three models (−0.053 to −0.106 for probe − classifier) and on DailyDialog (−0.067 to −0.105); it ties the probe on dair and is mixed on ISEAR (Gemma probe +0.029, Mistral −0.035; post hoc, `t11_posthoc.md`). Restricting to a single direction per class therefore costs something where there is enough human data; the pre-registered nonlinear checks (2.4) still find nothing beyond a linear read-out.

**Figure 2** (`docs/figures/fig2_learning_curve.svg`; data `frozen_eval.json`). Learning curve: frozen-set macro-F1 against human rows per class (0–378), probe / native LR / TF-IDF, LLM + human vs human-only, mean ± sd over three subsampling seeds, three models; the human-only curve starts at k=20. The k=378 point is a single fit without seed variance, and the human-only curves dip there for Gemma and Mistral; this is not an effect we interpret.

This bears on the common practice of building emotion vectors from model-written stories (Sofroniew et al., 2026, and the replications in 1.2). Those studies target representational geometry and behaviour, not classifying human text, so our result does not contradict them. It does show that a probe built that way should not be assumed to read human-written emotion well.

### 2.4 Linear is enough; geometry is human-like on two models [`t8_experiments.md`]
- **Read-out.** A valence hierarchy beat flat logistic regression on at most 1 of 4 sets, and an MLP on 0 of 4 (rule: ≥ 3 of 4), on all three models.
- **Geometry.** The 21 pairwise dissimilarities of the probe directions correlate with pairwise valence–arousal–dominance distances between the seven emotion words (NRC VAD v2.1; Mohammad, 2025). Spearman ρ = +0.53 / +0.60 (Llama, lda / mean-diff) and +0.61 / +0.63 (Gemma), all p ≤ 0.016 under an exact permutation over 5040 labelings. On Mistral the mean-difference directions agree (+0.54, p=0.005) but the validation-selected logreg directions do not (+0.31, p=0.112), so the pre-registered rule is not met there. Seven points; nominal.

### 2.5 Pooling and prompting [`variants.md`]
Mean-pooling is a sound default. Last-token and last-4 pooling are worse on most sets (by up to −0.15; last-token is equal on two Llama sets). Max-pooling and concatenating plateau layers are equivalent. A prompted read-out helped on two of four sets on Llama (ISEAR +0.136, DailyDialog +0.072) but on none on Gemma; it does not count as a finding.

### 2.6 Calibration and neutral detection fail on human text [`length_calibration.md`, `t11.md`, `t11_summary.md`]
**Length.** On 5,014 held-out neutral human comments, the share of passages with any |z|>3 under the original calibration ranges with length:

| model | 1–2 words | 20+ words |
|---|---|---|
| Llama | 76.0% | 15.3% |
| Gemma | 83.4% | 20.1% |
| Mistral | 69.0% | 15.3% |

Per-bucket cross-validated recalibration reduces this to about 1–1.7% per emotion and 4.0–7.4% per passage. It still misses the pre-registered bounds (≤ 2% per emotion, ≤ 5% per passage, upper confidence limit) in every bucket for all three models, because the tails are heavier than Gaussian. Descriptively, after recalibration a single threshold of about 2.8–3.3 on max |z| keeps false alarms near 5% at every length; the unmodified scores need 3.7–10.9 depending on length.

**Neutral detection.** With τ chosen on LLM-written validation text, the probe's recall on human neutral text is low: on the frozen set 0.192 (Llama), 0.291 (Gemma) and 0.126 (Mistral), and on 2,507 held-out human neutral passages 0.181, 0.312 and 0.137, so the probe labels 69–86% of neutral human text as emotional. We pre-registered (`0d74f80`) a re-fit: τ_h = the 90th percentile of max z on a human neutral reserve (2,507 passages fit τ_h, an equal number of other passages check it). This restores held-out neutral recall to 0.896 / 0.898 / 0.899 (τ_h = 4.77 / 5.43 / 4.42 against the original 1.57 / 2.03 / 1.42), but frozen-set 8-way macro-F1 falls: 0.510 → 0.390 (Llama; difference −0.121 [−0.163, −0.084]), 0.513 → 0.340 (Gemma; −0.173 [−0.213, −0.135]), 0.515 → 0.382 (Mistral; −0.134 [−0.173, −0.096]). The pre-registered rule (recall within 0.05 of 0.90 and no macro-F1 loss) is **not met on any model**. The loss comes from emotional comments moving to neutral (post hoc breakdown, `t11_posthoc.md`): every one of the seven emotion classes loses recall, by 0.23–0.42 (Llama), 0.26–0.40 (Gemma) and 0.23–0.43 (Mistral), while neutral recall rises by 0.77, 0.66 and 0.79. Overall 60.7%, 58.8% and 59.9% of emotional test items whose prediction was an emotion under the original τ become neutral under τ_h. A single global threshold therefore either calls most neutral text emotional or sends most emotional text to neutral, because the z-scores of emotional and neutral human comments overlap heavily. For any live display or monitor this is the most practical finding here: without human-text calibration the probe reports emotion in most neutral human text, and the simple fix we pre-registered costs more emotion recall than it gains.

**Figure 3** (`docs/figures/fig3_neutral_false_alarm.svg`; data `length_calibration.json`). Neutral false-alarm rate (any |z|>3) by length bucket, raw vs recalibrated, three models, with Clopper-Pearson intervals.

### 2.7 Training-seed robustness [`t11.md`, `t11.json` in each model's directory]
The intervals in 2.1–2.2 resample test items only. Over 5 training subsamples (seeds 0–4; each draws 80% of the LLM-written rows and 80% of each human-pool class; validation set fixed), the sign of probe − TF-IDF was stable (same sign in at least 4 of 5 seeds) in all 12 model × test-set cells: negative in-domain (−0.122 to −0.074 across models and seeds) and positive on every external source (+0.039 to +0.211). The sign of probe − sentence encoder was stable in 9 of 12 cells: negative on ISEAR and DailyDialog for all three models (−0.104 to −0.011) and on dair for Gemma (−0.017 to −0.002); small and positive on the frozen set for Llama and Gemma. The unstable cells are Llama on dair and Mistral on the frozen set and on dair, where the difference is near zero, as in 2.2. The subsamples overlap heavily, so this measures sensitivity to the composition of the training set, not independent replication.

### 2.8 Steering: reading is not writing [`steering_results.md`]
Pre-registered (PLAN_v2 T10, committed before any generation). On Llama-3.2-3B we added a unit direction, scaled by 0.1, 0.2 or 0.4 times the mean token-state norm (16.2), to the residual stream after block 16 at every position, and generated three-sentence texts for 40 neutral prompts (greedy, 60 tokens; 4,120 generations). Directions: the audited LDA probe direction for each of 7 emotions, a v1-style mean-difference direction for each emotion, and 20 random unit directions at every strength. The strength grid came from a fluency-only pilot (S* = 0.4; perplexity of the steered text under the unsteered model stayed within 1.14x). The emotion expressed was labelled by a fixed zero-shot judge (Granite-4.2-8B, local) that passed its gate on human text (macro-F1 0.648, accuracy 0.646 on the frozen GoEmotions set; `judge_validation.md`).
- **Result: not established.** The rule needed an emotion's steered rate to exceed all 20 random directions' with a bootstrap interval excluding 0, for at least 4 of 7 emotions. **0 of 7** met it: changes in the judged rate of the target emotion were within ±0.075 (point estimates) for every probe direction at every strength, with the generation still fluent; the 95% interval bounds lie between −0.150 and +0.175. Judge failures (266 of 4,120 unparsed) were 6–7% in every condition.
- **Post hoc, not pre-registered:** the same rule applied to the mean-difference directions finds joy (+0.375 [+0.200, +0.525] at 0.4; perplexity ratio 1.40; random maximum 0.000) and anger (+0.125 [+0.025, +0.225]) specific, 2 of 7, still below the threshold; love is not fluent at its effective strength. Probe minus mean-difference for joy at 0.4 is −0.350 [−0.500, −0.175].
- **Power.** With 40 prompts the 95% intervals for the probe directions at S* = 0.4 are 0.075–0.225 wide (anger and love show no change at all, so their intervals are degenerate at 0). A shift of the size the mean-difference direction produced for joy (+0.375) would have been detected; a shift of up to about +0.15 (the upper bound of the widest interval, joy) cannot be excluded. This is a test against large effects; small effects are not ruled out.
- **Fluency rule.** "Fluent" means a median perplexity ratio ≤ 1.5, fixed in the T10 pre-registration (`PLAN_v2.md`) and applied identically to probe, mean-difference and random directions. The 1.14× figure is the largest ratio the probe directions reached at 0.4, not a threshold. Mean-difference joy at 0.4 (ratio 1.40) passes; love (1.57) and sadness (1.68) do not.
- **Norm-matched is not effect-matched (post hoc diagnostic, `steering_diagnostic.md`).** On cached layer-16 states, the unit probe direction shifts the projection onto the class mean-difference axis as much per unit norm as the mean-difference direction does (mean cosine 0.270 against 0.274; random 0.002), so the null is not explained by a smaller shift on the class axis. The two differ in where they point: the pooled states vary with sd 0.21 along the probe directions, 0.59 along the mean-difference directions and 0.10 along random ones, and at 0.4 the perplexity ratio is 1.10–1.14 for probe directions (random 1.08–1.15 in the pilot) but 1.37–1.68 for mean-difference directions. Adding the probe direction at matched norm is therefore about as disruptive as a random direction, whereas the mean-difference direction perturbs the model far more per unit norm. A probe strength between 0.4 and 0.8 was not tried (the pilot grid had no point there, and fluency fails at 0.8 even for random directions), so a stronger, fluency-matched probe intervention remains untested.
- **Relation to prior work.** The post hoc ordering (mean-difference directions steer, the discriminatively fitted ones do not) matches Marks & Tegmark (2023), who found difference-in-means directions more causally implicated than other probe directions for truth. We treat our result as consistent with theirs in a new domain, not as new.
- **Reading:** a direction that reads emotion well here (LDA directions whitened to separate classes) is not necessarily a direction that changes what a model writes when added; the plain difference of means did better. One model, one layer, one judge, 40 neutral prompts that already elicit mostly positive text (unsteered judge labels: joy 15, neutral 19 of 40), and a modest fluent strength ceiling. This does not support treating the probes as causal handles on expressed emotion.

## 3. What this does and does not show

**Supported:**
- emotion directions in mid-layer states carry information beyond a bag of the model's own embeddings (all models, all human sets);
- they transfer across human text domains better than a lexical classifier trained on the same data;
- the cross-domain advantage over lexical models is shared by a frozen sentence encoder: the probes are no better than one trained on the same data (H3 not supported; the encoder is better on ISEAR and DailyDialog, tied on dair). The hidden states themselves carry more than the encoder in-domain (trained classifier − encoder +0.071 to +0.091, post hoc), but one direction per emotion does not extract it;
- the information is linear at the chosen layer;
- probe geometry is related to human affect norms on two of three models.

**Not supported:**
- that probes beat lexical baselines in-domain on human text (they do not);
- that scores are comparable across lengths, or that uncalibrated scores separate neutral from emotional human text (they do not);
- that LLM-written emotion corpora are sufficient (about 0.31 vs 0.55–0.59 macro-F1 with human training data);
- that a larger model fixes the in-domain gap (7B does not);
- that probes carry more emotion information than a generic frozen sentence encoder (H3);
- that re-fitting the neutral threshold on human neutral text fixes neutral detection (H5: neutral recall is restored, macro-F1 falls 0.12–0.17);
- that prompting or another pooling helps;
- that adding a probe direction to the model's activations changes the emotion it expresses (0 of 7 against random controls; section 2.8).

These experiments classify *input text*. They test whether probes read the emotion expressed in a passage, which is a setting where the text itself carries most of the signal. They do not test whether probes reveal anything that is *absent* from the text, such as an emotion the model represents but does not express. That question needs different designs (section 5). Nothing here supports statements about a model's feelings or internal emotional state: the probes are linear read-outs correlated with emotion-concept text.

## 4. Limitations
- **Models.** Three small 4-bit models from three families (2B–7B); no larger or full-precision model.
- **Data.** Human evidence comes from Reddit comments plus three sources of mixed provenance (hashtag labels, an unverified ISEAR copy, utterances without context). GoEmotions labels are noisy. The single-label rule keeps items with weaker rater agreement; strict unanimity left 8–29 items per class.
- **Comparability.** Class sets differ across sources, so macro-F1 values are not comparable across sources.
- **Blinding.** A hand-written test set we used early was accidentally committed to the public repository. We treat all results on it as non-blind and base conclusions on the frozen and external sets.
- **Multiplicity.** We report 16–20 paired comparisons per model per experiment with unadjusted intervals. The ≥ 3-of-4 consistency rules are the guard, and some results (prompted read-out, geometry on Mistral) did not replicate.
- **Encoder baseline.** One encoder (all-mpnet-base-v2), whose own training data include Reddit comments, which may favour it in-domain; a stronger or emotion-fine-tuned encoder could widen its lead.
- **Steering.** One model (Llama-3.2-3B), one layer (16), addition at all positions, 40 neutral prompts that already lean positive (unsteered judge labels: joy 15, neutral 19 of 40), one zero-shot judge (macro-F1 0.648 on human text; not validated on the generated text), a fluent-strength ceiling of 0.4× the token-state norm, and limited power (interval widths up to 0.225; section 2.8). Small effects, and effects at strengths between 0.4 and 0.8, are not ruled out.
- **Post hoc analyses** (fusion with TF-IDF, lexicon strata, alarm thresholds) are labelled as such where reported.

## 5. Future work
- **Information not in the text.** Use minimal pairs in which the target sentence is identical and only the preceding context differs; any lexical model is constant on the target by construction. Probe the assistant turn in role-play where an emotion is described but hidden (following the "hidden emotion" condition of Sofroniew et al., 2026).
- **Causal tests.** Done for one model and layer (section 2.8: not established for the probe directions; mean-difference directions steered joy and anger post hoc). Repeat on other models and layers, add stronger or multi-layer interventions, and validate the judge against human ratings of the generated text.
- **Sparse autoencoders.** Gemma Scope SAE features (Lieberum et al., 2024) as an alternative to supervised directions, noting that those SAEs were trained on the base model, not the instruction-tuned one.
- **Scale.** Larger and full-precision models.

## 6. Reproducibility and compliance
Code, tests (22 model-free synthetic test files, all passing on 2026-11-20 (19 of them also from a clean checkout on 2026-11-18)), pre-registrations and reports are in the repository. A bundled synthetic example runs from install to a report in one command (`probekit-audit --demo`; `docs/AUDIT_QUICKSTART.md`). `probekit_audit.py` reproduces the headline table from cached activations exactly (Llama probe 0.510, native LR 0.521, bag of embeddings 0.403, TF-IDF 0.599). The T11 runs are `probekit_t11.py` (encoder, EmoLex, human-neutral threshold, seeds), the post hoc analyses are `probekit_t11_posthoc.py` and `probekit_steer_diag.py`, and the figures are `make_figures.py`; both read only committed reports and cached states.

Third-party data stay outside the repository and are rebuilt locally from hash-checked manifests:
- GoEmotions: not stored in the repository since 2026-11-19 (only ids, manifests and hashes); `rebuild_human_data.sh` rebuilds every GoEmotions-derived file from the original release and checks each against `data/DATA_HASHES.json` (verified byte-identical). Earlier commits still contain the subsets. Checked against the original release (2026-11-18): the `google-research` repository that hosts it is Apache-2.0 (top-level `LICENSE`), but the GoEmotions README states no separate data license, and the text is Reddit user content, so Reddit's terms and the authors' rights may also apply; we use it for non-commercial research;
- dair-ai/emotion: license "other", no terms stated;
- ISEAR: Hub copy, original host unreachable, uploader's license tag unverified;
- DailyDialog: CC-BY-NC-SA-4.0;
- NRC EmoLex and VAD lexicons: non-commercial research use, no redistribution; used locally and never committed.

License: The Unlicense (public domain) for our code and documents (`LICENSE` is committed); `NOTICE.md` lists the third-party terms that remain.

## Appendix A. First pre-registered run (T0c, merged2 training set)
The first pre-registered run trained on the merged2 training rows (LLM-written and human rows from the earlier human corpus) instead of HUM. Same conclusion: TF-IDF ahead in-domain, probe ahead of the bag of embeddings. [`frozen_eval.md` in each model's directory]

| system (GoEmotions frozen set) | Llama-3.2-3B | Gemma-2-2B | Mistral-7B |
|---|---|---|---|
| probe | 0.510 [0.476, 0.544] | 0.513 [0.478, 0.546] | 0.515 [0.481, 0.547] |
| trained classifier, same states | 0.521 | 0.537 | 0.559 |
| TF-IDF | 0.599 [0.564, 0.630] | 0.599 [0.564, 0.630] | 0.599 [0.564, 0.630] |
| bag of embeddings | 0.403 | 0.401 | 0.394 |

Probe − TF-IDF: −0.088, −0.085, −0.083 (all intervals exclude 0). Probe − bag of embeddings: +0.107, +0.112, +0.121 (all exclude 0). TF-IDF does not use the model, so it is identical across models. The TF-IDF advantage is larger on items with high rater agreement (Llama: 0.594 vs 0.749 at agreement ≥ 0.67, n=244), so label noise does not explain it.

## 7. References
*Section A was checked against the papers' own pages on 2026-11-18. In section B, title, authors and year of the items with an arXiv id were checked against the arXiv API on 2026-11-19; venues, and the four items without an arXiv id (listed under "Still to check" below), were not checked and are from memory.*

**A. Checked**
- Fang, T., Guibon, G., & Buscaldi, D. (2026). Some Emotions Run Deeper: Layer-wise Probing and Causal Intervention in Large Language Models. *Findings of EMNLP 2026*.
- Hollowell, A. (2026). Replicating the Geometry of Emotion Representations in a Base Open-Weights Model. arXiv:2609.22208.
- Jeong, J. (2026a). Extracting and Steering Emotion Representations in Small Language Models: A Methodological Comparison. arXiv:2604.04064.
- Jeong, J. (2026b). Shared Emotion Geometry Across Small Language Models: A Cross-Architecture Study of Representation, Behavior, and Methodological Confounds. arXiv:2604.11050.
- Mohammad, S. M. (2025). NRC VAD Lexicon v2: Norms for Valence, Arousal, and Dominance for over 55k English Terms. arXiv:2503.23547.
- Reichman, B., Avsian, A., & Heck, L. (2026). Emotions Where Art Thou: Understanding and Characterizing the Emotional Latent Space of Large Language Models. *ICLR 2026*. arXiv:2510.22042.
- Sofroniew, N., Kauvar, I., Saunders, W., Chen, R., et al. (2026). Emotion Concepts and their Function in a Large Language Model. Anthropic; arXiv:2604.07729.
- van der Ben, S., Baur, R., Metz, Y., & El-Assady, M. (2026). Where Do Models Find Happiness? Emotion Vectors in Open-Source LLMs. *ICML 2026 Mechanistic Interpretability Workshop*. arXiv:2606.26987.
- Zhang, J., & Zhong, L. (2025). Decoding Emotion in the Deep: A Systematic Study of How LLMs Represent, Retain, and Express Emotion. arXiv:2510.04064.
- Keeman, M. (2026). Whether, Not Which: Mechanistic Interpretability Reveals Dissociable Affect Reception and Emotion Categorization in LLMs. arXiv:2603.22295 (checked 2026-11-19; abstract read).

**B. To check**
Checked on arXiv (title, authors, year; venue from memory): Alain & Bengio (2016), Understanding intermediate layers using linear classifier probes, arXiv:1610.01644; Hewitt & Liang (2019), Designing and Interpreting Probes with Control Tasks, EMNLP, arXiv:1909.03368; Belinkov (2022), Probing Classifiers: Promises, Shortcomings, and Advances, *Computational Linguistics*, arXiv:2102.12452; Zou et al. (2023), Representation Engineering: A Top-Down Approach to AI Transparency, arXiv:2310.01405; Demszky et al. (2020), GoEmotions: A Dataset of Fine-Grained Emotions, ACL, arXiv:2005.00547; Li et al. (2017), DailyDialog: A Manually Labelled Multi-turn Dialogue Dataset, IJCNLP, arXiv:1710.03957; Reimers & Gurevych (2019), Sentence-BERT, EMNLP, arXiv:1908.10084; Gemma Team (2024), Gemma 2: Improving Open Language Models at a Practical Size, arXiv:2408.00118; Jiang et al. (2023), Mistral 7B, arXiv:2310.06825; Lieberum et al. (2024), Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2, arXiv:2408.05147.

Added for the steering section (title, authors and year checked against the arXiv API on 2026-11-19; venues from memory): Marks & Tegmark (2023), The Geometry of Truth: Emergent Linear Structure in Large Language Model Representations of True/False Datasets, arXiv:2310.06824 (COLM 2024); Turner, Thiergart, Leech, Udell, Vazquez, Mini & MacDiarmid (2023), Steering Language Models With Activation Engineering, arXiv:2308.10248; Panickssery, Gabrieli, Schulz, Tong, Hubinger & Turner (2023), Steering Llama 2 via Contrastive Activation Addition, arXiv:2312.06681 (ACL 2024).

Still to check (no arXiv id): Saravia et al. (2018), CARER, EMNLP; Scherer & Wallbott (1994), ISEAR; Mohammad & Turney (2013), NRC EmoLex, *Computational Intelligence*; Meta (2024), Llama 3.2 model card.

## 8. Open items before release
- [x] Pre-registrations for H3–H6 committed (`0d74f80`), run, and every placeholder filled from the reports.
- [x] Figures 1–3 (`docs/figures/`, `make_figures.py`).
- [x] Abstract variant B (H3 not supported); variant A deleted.
- [ ] Check the four remaining section 7B references (no arXiv id) and the venues of the others.
- [x] LICENSE file (The Unlicense) and `NOTICE.md` committed.
- [ ] Check the remaining dataset licenses against primary sources (GoEmotions done, see section 6; dair-ai/emotion and the ISEAR copy unresolved).
- [x] README, FINDINGS.md and CLAIMS.md aligned with this draft (hand-written results only in a "non-blind / superseded" section); README in-domain numbers updated to the HUM table on 2026-11-20.
- [ ] Optional before submission: T5 audit of one published model-written emotion corpus.
- [ ] Human read-through of the whole draft.
