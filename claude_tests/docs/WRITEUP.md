# What linear emotion probes on LLM hidden states do and do not buy you: an audit

*Draft v2, 2026-11-18. Every number is traceable to a committed report (paths in brackets) and to a pre-registration in `PLAN_v2.md` written before the run. This is a hobby-scale, non-commercial study of three small 4-bit open models; it makes no claim about model feelings or internal states.*

> **For the agent (delete this box before release).** Placeholders `[[AGENT: ...]]` mark numbers that do not exist yet or must be checked against the repo. Do not fill any placeholder from memory: run the code, quote the report, and add the report path. Before running the new experiments (encoder baseline, EmoLex baseline, human-neutral threshold, training seeds), write their hypotheses and decision rules into `PLAN_v2.md` exactly as stated in section 1.3 and commit that change first. If the encoder result lands in variant B, use abstract variant B and the alternative sentences marked "(B)". Remove the variant you don't use.

## Abstract

*Variant A (use if H3 holds: probe beats the sentence encoder cross-domain).*
Linear "emotion probes" read emotion-concept directions out of an open-weight language model's residual stream and are a common basis for visualisations and proposed monitors. We ask a narrow question: what do such probes carry beyond cheap and standard baselines, and where? We audit probes on three small instruction-tuned models (Llama-3.2-3B, Gemma-2-2B, Mistral-7B; all 4-bit). Evaluation uses a frozen, hash-verified human test set (GoEmotions, n=840), three further human-labelled sources (tweets, event descriptions, dialogue), pre-registered hypotheses and decision rules, and paired bootstrap intervals. Baselines are trained on the *same* data: TF-IDF, a zero-training emotion lexicon, a bag of the model's own input embeddings, a frozen sentence encoder, and a trained classifier on the same hidden states. The results hold on all three models. (1) In-domain, TF-IDF beats the probe by about 0.09 macro-F1, and a 7B model does not close the gap. (2) Across domains the order reverses: the probe beats TF-IDF on all three external sources (+0.054 to +0.190) and beats a frozen sentence encoder by [[AGENT: range]]. (3) Probes trained only on LLM-written emotional text, the recipe used by most emotion-vector studies, reach about 0.31 macro-F1 on human text, against 0.51–0.60 with human training data. Once human data are present, LLM-written data add nothing measurable. (4) The probes' z-scores are not comparable across text length (15–83% of neutral human passages raise some |z|>3, depending on length), and the probe labels [[AGENT: share]] of neutral human text as emotional. A linear read-out is sufficient, mean-pooling is a sound default, and probe geometry correlates with human valence–arousal–dominance norms on two of three models. We release the audit tool, the frozen test-set manifests, and all pre-registrations.

*Variant B (use if H3 fails: the probe ties with or loses to the encoder).* Same as A, but replace the end of result (2) with: "...but does no better than a frozen 110M-parameter sentence encoder trained on the same data ([[AGENT: range]]). Across domains, the emotion information a linear probe reads from these LLMs is comparable to what a generic sentence encoder provides."

## 1. Question, position and design

### 1.1 Why audit
Emotion probes are usually built from LLM-generated emotional passages and then shown to separate held-out passages from the same generator. That is weak evidence: such corpora can be lexically solvable. Our first corpus was (TF-IDF 0.953 vs probe 0.857 [`docs/FINDINGS.md`]). The interpretability literature has known since control tasks were introduced (Hewitt & Liang, 2019) that probe accuracy means little without baselines. Emotion-probe studies rarely include them.

### 1.2 Position relative to recent work
Sofroniew et al. (2026) extracted 171 emotion-concept vectors from Claude Sonnet 4.5 using model-written stories and showed that they influence behaviour. Several studies replicated the representational part on open models with model-generated corpora (van der Ben et al., 2026; Hollowell, 2026; Jeong, 2026a,b). Reichman et al. (2026) report an emotional subspace, found from a synthetic rewritten-sentence dataset, whose probes generalise across eight real datasets and five languages. Fang et al. (2026) probe eight 1B–9B open models, including Llama-3.2-3B, on GoEmotions and two other corpora, and test transfer across corpora. Zhang & Zhong (2025) release a probing toolkit and a Reddit emotion dataset.

None of these compares the probe with a lexical or bag-of-embeddings classifier trained on the same data, separates in-domain from cross-domain evaluation against such baselines, or checks calibration on neutral human text. Those comparisons are our contribution. We do not claim novelty for linear emotion probing, for cross-domain transfer as such, or for the valence geometry.

### 1.3 Contributions
1. **An in-domain/cross-domain reversal, on three model families.** A lexical classifier beats the probe in-domain; the probe beats it across domains. [[AGENT: (A) and beats a frozen sentence encoder across domains / (B) and matches a frozen sentence encoder]].
2. **A cost for the common recipe.** Probes built only from LLM-written emotional text transfer weakly to human text, and LLM-written data add nothing once human data are present.
3. **Calibration and neutral detection fail.** Raw z-scores depend strongly on text length, and recalibration does not meet pre-registered bounds. This matters for any live display or monitor.
4. **A reusable audit tool** (`probekit audit`) with frozen, hash-verified human test sets and fetch scripts.

### 1.4 Design
**Baselines.** Each comparison trains the baseline on the same data as the probe:
- (a) TF-IDF + logistic regression;
- (b) NRC EmoLex word counts (Mohammad & Turney, 2013), no training. This separates "lexical features fail across domains" from "features fitted to the training domain fail";
- (c) logistic regression on the mean of the model's input token embeddings ("bag of embeddings": vocabulary knowledge, no context);
- (d) a frozen pretrained sentence encoder + logistic regression. [[AGENT: name the encoder, e.g. sentence-transformers/all-mpnet-base-v2, with revision hash and parameter count]];
- (e) logistic regression on the same hidden states as the probe (the linear ceiling);
- (f) a shuffled-label null.

Comparisons are paired over the same test items and reported with bootstrap 95% intervals. Headline comparisons are also repeated over [[AGENT: 5]] training seeds or subsamples (section 2.7).

**Pre-registered hypotheses.** H1: probe − TF-IDF > 0 cross-domain (CI above 0 on ≥ 2 of 3 sources). H2: probe − bag of embeddings > 0. H3 (added before the run, [[AGENT: commit hash]]): probe − sentence encoder > 0, CI above 0 on ≥ 2 of 3 external sources per model, on ≥ 2 of 3 models. If H3 fails, we report the encoder as equivalent or better and do not claim LLM-specific information. D2: native LR at k=378 − k=80 > 0.03 means data-limited.

**Probes.** One direction per emotion (joy, sadness, fear, anger, surprise, disgust, love), fitted by shrinkage LDA, logistic regression or mean difference on mean-pooled mid-layer states (first 4 tokens skipped). Scores are z-scored against held-out neutral text, with a neutral threshold τ. Layer, estimator and τ are chosen on validation only. The extractor uses a causal mask and is checked against each model's own forward pass (relative difference 3e-4 to 6e-4); it handles Gemma's embedding scaling.

**Data.** Training: LLM-written passages from two writers (Llama-3.2-3B and Qwen3.8-27B) plus human GoEmotions comments (up to 378 per class, official train split only). Test:
- (i) a frozen GoEmotions test set: official test split, single-label items, near-duplicates of train/dev removed, n=840, 63–151 per class, SHA-256 manifest; the evaluator refuses a modified file [`data/testsets/*.manifest.json`];
- (ii) dair-ai/emotion test (tweets, hashtag-derived labels), ISEAR (first-person event descriptions) and DailyDialog (utterances).

Class sets differ across sources, so predictions are restricted to each source's classes. Calm exists only in LLM-written text (confounded with writer) and is excluded from all headline numbers.

**Models.** Llama-3.2-3B-Instruct, Gemma-2-2B-it and Mistral-7B-Instruct, all 4-bit via MLX. The Mistral model is Mistral-7B-Instruct-v0.3 (`mlx-community/Mistral-7B-Instruct-v0.3-4bit`). Mistral was chosen as a third family at the largest size that runs comfortably on the hardware used; it is an older model, and results may differ on newer 7–9B models.

## 2. Results

**Figure 1** [[AGENT: make]]. Forest plot: probe − baseline (TF-IDF, sentence encoder, bag of embeddings), macro-F1 with 95% CI. Rows: frozen in-domain set plus three external sources; panels: three models; a vertical line at 0. This figure carries the paper.

### 2.1 In-domain, lexical features win [`frozen_eval.md`]
On the GoEmotions frozen set (8-way, chance 0.125):

| system | Llama-3.2-3B | Gemma-2-2B | Mistral-7B |
|---|---|---|---|
| probe | 0.510 [0.476, 0.544] | 0.513 [0.478, 0.546] | 0.515 [0.481, 0.547] |
| trained classifier, same states | 0.521 | 0.537 | 0.559 |
| TF-IDF | 0.599 [0.564, 0.630] | [[AGENT]] | [[AGENT]] |
| sentence encoder | [[AGENT]] | [[AGENT]] | [[AGENT]] |
| EmoLex (no training) | [[AGENT]] | [[AGENT]] | [[AGENT]] |
| bag of embeddings | 0.403 | [[AGENT]] | [[AGENT]] |

[[AGENT: the v1 draft gave bag of embeddings as 0.394–0.403 and TF-IDF as 0.599 without saying whether TF-IDF is identical across models. Check: TF-IDF should be model-independent apart from item order.]]

Probe − TF-IDF: −0.088, −0.085, −0.083 (all intervals exclude 0). Probe − bag of embeddings: +0.107, +0.112, +0.121 (all exclude 0). Probe − sentence encoder: [[AGENT]].

The TF-IDF advantage is larger on items with high rater agreement (Llama: 0.594 vs 0.749 at agreement ≥ 0.67, n=244). Label noise therefore does not explain it.

### 2.2 Across domains, the probe beats lexical models [`external_eval.md`]
Trained on the same GoEmotions-derived data and tested on other human sources, probe − TF-IDF (macro-F1 [95% CI], H1):
- Llama: dair +0.151 [+0.122, +0.178], ISEAR +0.132 [+0.101, +0.163], DailyDialog +0.103 [+0.077, +0.130];
- Gemma: +0.137, +0.190, +0.095;
- Mistral: +0.152, +0.127, +0.054.

All intervals exclude 0. Trained only on LLM-written text, the gaps are larger: Llama +0.167, +0.262, +0.125; Gemma +0.127, +0.264, +0.152; Mistral +0.135, +0.196, +0.092.

Against the zero-training lexicon: [[AGENT: probe − EmoLex per source and model]]. Against the sentence encoder (H3): [[AGENT: probe − encoder per source and model, with CIs]].

*Reading (A).* Lexical models fit the vocabulary of their training domain. The probe transfers better than lexical models and than a frozen sentence encoder, which suggests these LLMs' mid-layer states carry emotion information that a general-purpose encoder does not provide in linearly readable form.
*Reading (B).* Lexical models fit the vocabulary of their training domain; contextual representations transfer better. The LLM probe is no better at this than a frozen sentence encoder, so the cross-domain advantage is a property of contextual encoders in general, not of these LLMs specifically.

Caveats: dair labels are hashtag-derived and lexical; ISEAR provenance is unverified; sources differ in class sets.

### 2.3 LLM-written training data are not enough; human data are the lever [`frozen_eval.md`]
- With *no* human training data, the probes score 0.305–0.313 on the frozen human set (shuffle-null p95 ≈ 0.13); TF-IDF scores 0.146.
- Adding human rows (0 → 378 per class) raises frozen-set F1 steadily: native LR at k=378 vs k=80 gives +0.069 [+0.039, +0.099] on Llama, +0.103 on Gemma and +0.096 on Mistral (D2: data-limited).
- Once human data are present, LLM-written rows add nothing measurable: human-only training equals LLM + human at every k.
- The trained classifier on the same states is statistically equal to the probe on most sets, so restricting to a single direction per class costs little. Exceptions: DailyDialog utterances, and Mistral on the frozen set (−0.044 [−0.070, −0.019]).

**Figure 2** [[AGENT: make]]. Learning curve: frozen-set macro-F1 against human rows per class (0–378), probe / native LR / TF-IDF, LLM + human vs human-only, mean ± sd over seeds, three models.

This bears on the common practice of building emotion vectors from model-written stories (Sofroniew et al., 2026, and the replications in 1.2). Those studies target representational geometry and behaviour, not classifying human text, so our result does not contradict them. It does show that a probe built that way should not be assumed to read human-written emotion well.

### 2.4 Linear is enough; geometry is human-like on two models [`t8_experiments.md`]
- **Read-out.** A valence hierarchy beat flat logistic regression on at most 1 of 4 sets, and an MLP on 0 of 4 (rule: ≥ 3 of 4), on all three models.
- **Geometry.** The 21 pairwise dissimilarities of the probe directions correlate with pairwise valence–arousal–dominance distances between the seven emotion words (NRC VAD v2.1; Mohammad, 2025). Spearman ρ = +0.53 / +0.60 (Llama, lda / mean-diff) and +0.61 / +0.63 (Gemma), all p ≤ 0.016 under an exact permutation over 5040 labelings. On Mistral the mean-difference directions agree (+0.54, p=0.005) but the validation-selected logreg directions do not (+0.31, p=0.112), so the pre-registered rule is not met there. Seven points; nominal.

### 2.5 Pooling and prompting [`variants.md`]
Mean-pooling is a sound default. Last-token and last-4 pooling are worse on most sets (by up to −0.15; last-token is equal on two Llama sets). Max-pooling and concatenating plateau layers are equivalent. A prompted read-out helped on two of four sets on Llama (ISEAR +0.136, DailyDialog +0.072) but on none on Gemma; it does not count as a finding.

### 2.6 Calibration and neutral detection fail on human text [`length_calibration.md`, [[AGENT: neutral threshold report path]]]
**Length.** On 5,014 held-out neutral human comments, the share of passages with any |z|>3 under the original calibration ranges with length:

| model | 1–2 words | 20+ words |
|---|---|---|
| Llama | 76.0% | 15.3% |
| Gemma | 83.4% | 20.1% |
| Mistral | 69.0% | 15.3% |

Per-bucket cross-validated recalibration reduces this to about 1–1.7% per emotion and 4.0–7.4% per passage. It still misses the pre-registered bounds (≤ 2% per emotion, ≤ 5% per passage, upper confidence limit) in every bucket for all three models, because the tails are heavier than Gaussian. Descriptively, after recalibration a single threshold of about 2.8–3.3 on max |z| keeps false alarms near 5% at every length; the unmodified scores need 3.7–10.9 depending on length.

**Neutral detection.** With τ chosen on LLM-written validation text, the probe's recall on human neutral text is low (Llama 0.19; [[AGENT: Gemma, Mistral]]). Re-fitting τ on a held-out human neutral reserve gives neutral recall [[AGENT]] and frozen-set macro-F1 [[AGENT]] (pre-registered before the run, [[AGENT: commit hash]]). For any live display or monitor this is the most practical finding here: without human-text calibration the probe reports emotion in most neutral human text.

**Figure 3** [[AGENT: make]]. Neutral false-alarm rate (any |z|>3) by length bucket, raw vs recalibrated, three models, with Clopper-Pearson intervals.

### 2.7 Training-seed robustness [[AGENT: report path]]
The intervals in 2.1–2.2 resample test items only. Over [[AGENT: 5]] training seeds or subsamples, the sign of probe − TF-IDF (in-domain and on each external source) and of probe − sentence encoder was [[AGENT: stable / not stable]]: range [[AGENT]].

### 2.8 Steering: reading is not writing [`steering_results.md`]
Pre-registered (PLAN_v2 T10, committed before any generation). On Llama-3.2-3B we added a unit direction, scaled by 0.1, 0.2 or 0.4 times the mean token-state norm (16.2), to the residual stream after block 16 at every position, and generated three-sentence texts for 40 neutral prompts (greedy, 60 tokens; 4,120 generations). Directions: the audited LDA probe direction for each of 7 emotions, a v1-style mean-difference direction for each emotion, and 20 random unit directions at every strength. The strength grid came from a fluency-only pilot (S* = 0.4; perplexity of the steered text under the unsteered model stayed within 1.14x). The emotion expressed was labelled by a fixed zero-shot judge (Granite-4.2-8B, local) that passed its gate on human text (macro-F1 0.648, accuracy 0.646 on the frozen GoEmotions set; `judge_validation.md`).
- **Result: not established.** The rule needed an emotion's steered rate to exceed all 20 random directions' with a bootstrap interval excluding 0, for at least 4 of 7 emotions. **0 of 7** met it: changes in the judged rate of the target emotion were within about ±0.075 for every probe direction at every strength, with the generation still fluent. Judge failures (266 of 4,120 unparsed) were 6–7% in every condition.
- **Post hoc, not pre-registered:** the same rule applied to the mean-difference directions finds joy (+0.375 [+0.200, +0.525] at 0.4; perplexity ratio 1.40; random maximum 0.000) and anger (+0.125 [+0.025, +0.225]) specific, 2 of 7, still below the threshold; love is not fluent at its effective strength. Probe minus mean-difference for joy at 0.4 is −0.350 [−0.500, −0.175].
- **Reading:** a direction that reads emotion well here (LDA directions whitened to separate classes) is not necessarily a direction that changes what a model writes when added; the plain difference of means did better. One model, one layer, one judge, 40 neutral prompts that already elicit mostly positive text (unsteered judge labels: joy 15, neutral 19 of 40), and a modest fluent strength ceiling. This does not support treating the probes as causal handles on expressed emotion.

## 3. What this does and does not show

**Supported:**
- emotion directions in mid-layer states carry information beyond a bag of the model's own embeddings (all models, all human sets);
- they transfer across human text domains better than a lexical classifier trained on the same data;
- [[AGENT: (A) better than a frozen sentence encoder / (B) no better than a frozen sentence encoder]];
- the information is linear at the chosen layer;
- probe geometry is related to human affect norms on two of three models.

**Not supported:**
- that probes beat lexical baselines in-domain on human text (they do not);
- that scores are comparable across lengths, or that uncalibrated scores separate neutral from emotional human text (they do not);
- that LLM-written emotion corpora are sufficient (about 0.31 vs 0.51–0.60 macro-F1);
- that a larger model fixes the in-domain gap (7B does not);
- that prompting or another pooling helps;
- that adding a probe direction to the model's activations changes the emotion it expresses (0 of 7 against random controls; section 2.8).

These experiments classify *input text*. They test whether probes read the emotion expressed in a passage, which is a setting where the text itself carries most of the signal. They do not test whether probes reveal anything that is *absent* from the text, such as an emotion the model represents but does not express. That question needs different designs (section 5). Nothing here supports statements about a model's feelings or internal emotional state: the probes are linear read-outs correlated with emotion-concept text.

## 4. Limitations
- **Models.** Three small 4-bit models from three families (2B–7B); no larger or full-precision model.
- **Data.** Human evidence comes from Reddit comments plus three sources of mixed provenance (hashtag labels, an unverified ISEAR copy, utterances without context). GoEmotions labels are noisy. The single-label rule keeps items with weaker rater agreement; strict unanimity left 8–29 items per class.
- **Comparability.** Class sets differ across sources, so macro-F1 values are not comparable across sources.
- **Blinding.** A hand-written test set we used early was accidentally committed to the public repository. We treat all results on it as non-blind and base conclusions on the frozen and external sets.
- **Multiplicity.** We report 16–20 paired comparisons per model per experiment with unadjusted intervals. The ≥ 3-of-4 consistency rules are the guard, and some results (prompted read-out, geometry on Mistral) did not replicate.
- **Post hoc analyses** (fusion with TF-IDF, lexicon strata, alarm thresholds) are labelled as such where reported.

## 5. Future work
- **Information not in the text.** Use minimal pairs in which the target sentence is identical and only the preceding context differs; any lexical model is constant on the target by construction. Probe the assistant turn in role-play where an emotion is described but hidden (following the "hidden emotion" condition of Sofroniew et al., 2026).
- **Causal tests.** Done for one model and layer (section 2.8: not established for the probe directions; mean-difference directions steered joy and anger post hoc). Repeat on other models and layers, add stronger or multi-layer interventions, and validate the judge against human ratings of the generated text.
- **Sparse autoencoders.** Gemma Scope SAE features (Lieberum et al., 2024) as an alternative to supervised directions, noting that those SAEs were trained on the base model, not the instruction-tuned one.
- **Scale.** Larger and full-precision models.

## 6. Reproducibility and compliance
Code, tests (19 model-free synthetic test files, all passing from a clean checkout of the repository on 2026-11-18), pre-registrations and reports are in the repository. `probekit_audit.py` reproduces the headline table from cached activations exactly (Llama probe 0.510, native LR 0.521, bag of embeddings 0.403, TF-IDF 0.599).

Third-party data stay outside the repository and are rebuilt locally from hash-checked manifests:
- GoEmotions: rebuilt from the original release by hash-checked scripts. Checked against the original release (2026-11-18): the `google-research` repository that hosts it is Apache-2.0 (top-level `LICENSE`), but the GoEmotions README states no separate data license, and the text is Reddit user content, so Reddit's terms and the authors' rights may also apply; we use it for non-commercial research;
- dair-ai/emotion: license "other", no terms stated;
- ISEAR: Hub copy, original host unreachable, uploader's license tag unverified;
- DailyDialog: CC-BY-NC-SA-4.0;
- NRC EmoLex and VAD lexicons: non-commercial research use, no redistribution; used locally and never committed.

License: The Unlicense (public domain) for our code and documents (`LICENSE` is committed); `NOTICE.md` lists the third-party terms that remain.

## 7. References
*Section A was checked against the papers' own pages on 2026-11-18. Section B is from memory: the agent must check each title, year and venue, and must not add a reference that has not been checked.*

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
- Whether, Not Which: Mechanistic Interpretability Reveals Dissociable Affect Reception and Emotion Categorization in LLMs. arXiv:2603.22295. [[AGENT: add authors and decide whether to cite in 1.2: it uses emotion vignettes with no emotion words, crossed with topic, which is close to our lexical-control motivation]]

**B. To check**
Alain & Bengio (2016/2017), linear classifier probes; Hewitt & Liang (2019), control tasks, EMNLP; Belinkov (2022), probing classifiers, *Computational Linguistics*; Zou et al. (2023), representation engineering; Demszky et al. (2020), GoEmotions, ACL; Saravia et al. (2018), CARER, EMNLP; Li et al. (2017), DailyDialog, IJCNLP; Scherer & Wallbott (1994), ISEAR; Mohammad & Turney (2013), NRC EmoLex, *Computational Intelligence*; Reimers & Gurevych (2019), Sentence-BERT, EMNLP [needed for the encoder baseline]; Gemma Team (2024), Gemma 2; Jiang et al. (2023), Mistral 7B; Meta (2024), Llama 3.2 model card; Lieberum et al. (2024), Gemma Scope.

## 8. Open items before release
- [ ] Commit the pre-registrations for H3, the human-neutral τ and training seeds, then run them; fill every `[[AGENT]]`.
- [ ] Figures 1–3.
- [ ] Choose abstract variant A or B from the H3 result; delete the other.
- [ ] Check section 7B references.
- [x] LICENSE file (The Unlicense) and `NOTICE.md` committed.
- [ ] Check the remaining dataset licenses against primary sources (GoEmotions done, see section 6; dair-ai/emotion and the ISEAR copy unresolved).
- [ ] Align README, FINDINGS.md and CLAIMS.md with this draft (hand-written results only in a "non-blind" section).
- [ ] Optional before submission: T5 audit of one published model-written emotion corpus.
- [ ] Human read-through of the whole draft.
