# T0c frozen-set evaluation (n=840; manifest sha256 f47b22ae79d3...)

Layer 10; selected estimator logreg (chosen on VAL-hard of the pooled eval, not on this set). Trained on pooled merged2 TRAIN without calm (n=1654); tau and C on VAL-hard (n=225). 8-way macro-F1 over joy, sadness, fear, anger, surprise, disgust, love, neutral; bootstrap 95% CI over test items.

| system | macro-F1 [95% CI] | accuracy | shuffle-null p95 | coarse valence F1 |
|---|---|---|---|---|
| probe_lda | 0.547 [0.512, 0.581] | 0.551 | 0.142 | 0.609 |
| probe_meandiff | 0.504 [0.470, 0.537] | 0.517 | 0.139 | 0.580 |
| probe_logreg (selected) | 0.515 [0.481, 0.547] | 0.531 | 0.136 | 0.584 |
| native_lr | 0.559 [0.524, 0.594] | 0.567 | 0.142 | 0.641 |
| embed | 0.394 [0.360, 0.427] | 0.394 | 0.142 | 0.462 |
| block0 | 0.432 [0.398, 0.467] | 0.439 | 0.143 | 0.531 |
| tfidf | 0.599 [0.564, 0.630] | 0.599 | 0.144 | 0.628 |
| fusion_tfidf_native (post hoc) | 0.656 [0.620, 0.691] | 0.662 | 0.143 | 0.701 |

## Paired differences, macro-F1 (mean [95% interval]); * = excludes 0

| comparison | difference |
|---|---|
| probe_logreg - tfidf | -0.083 [-0.125, -0.042]* |
| probe_logreg - embed | +0.121 [+0.082, +0.160]* |
| probe_logreg - block0 | +0.083 [+0.043, +0.122]* |
| probe_logreg - native_lr | -0.044 [-0.070, -0.019]* |
| fusion_tfidf_native (post hoc) - tfidf (post hoc) | +0.058 [+0.023, +0.092]* |
| fusion_tfidf_native (post hoc) - native_lr (post hoc) | +0.097 [+0.073, +0.121]* |

## Per class (selected probe)

| class | n | recall | F1 | native LR recall |
|---|---|---|---|---|
| joy | 91 | 0.64 | 0.62 | 0.68 |
| sadness | 98 | 0.36 | 0.49 | 0.52 |
| fear | 63 | 0.71 | 0.51 | 0.60 |
| anger | 128 | 0.51 | 0.55 | 0.60 |
| surprise | 84 | 0.64 | 0.61 | 0.56 |
| disgust | 74 | 0.59 | 0.45 | 0.55 |
| love | 151 | 0.83 | 0.72 | 0.72 |
| neutral | 151 | 0.13 | 0.18 | 0.34 |

## Strata (macro-F1, systems: probe, native_lr, embed, tfidf)

- rater agreement >= 0.67 (n=244): probe 0.568, native_lr 0.651, embed 0.488, tfidf 0.749
- rater agreement < 0.67 (n=596): probe 0.495, native_lr 0.523, embed 0.352, tfidf 0.544
- <= 12 words (n=431): probe 0.537, native_lr 0.590, embed 0.369, tfidf 0.572
- > 12 words (n=409): probe 0.485, native_lr 0.524, embed 0.419, tfidf 0.623
- subtypes (too small to score; n only): {'no_emotion_lexicon_match': 403, 'negation': 10, 'emotion_word_neutral': 6}

## Human-data learning curve (frozen-set macro-F1, mean ± sd over subsampling seeds; k=max is 378 per class, fear-limited)

| training | k per class | n train | probe F1 | native LR F1 | TF-IDF F1 (post hoc) | fusion F1 (post hoc) |
|---|---|---|---|---|---|---|
| LLM + human | 0 | 1014 | 0.305 ± 0.000 | 0.246 ± 0.000 | 0.146 ± 0.000 | 0.250 ± 0.000 |
| LLM + human | 20 | 1174 | 0.419 ± 0.010 | 0.437 ± 0.008 | 0.451 ± 0.028 | 0.493 ± 0.020 |
| LLM + human | 40 | 1334 | 0.462 ± 0.007 | 0.499 ± 0.005 | 0.526 ± 0.007 | 0.581 ± 0.015 |
| LLM + human | 80 | 1654 | 0.516 ± 0.011 | 0.559 ± 0.002 | 0.583 ± 0.016 | 0.658 ± 0.004 |
| LLM + human | 160 | 2294 | 0.527 ± 0.011 | 0.604 ± 0.002 | 0.620 ± 0.016 | 0.703 ± 0.003 |
| LLM + human | 320 | 3574 | 0.556 ± 0.006 | 0.634 ± 0.001 | 0.659 ± 0.004 | 0.734 ± 0.004 |
| LLM + human | 378 | 4038 | 0.572 ± 0.000 | 0.657 ± 0.000 | 0.666 ± 0.000 | 0.744 ± 0.000 |
| human only | 20 | 160 | 0.438 ± 0.008 | 0.488 ± 0.011 | 0.450 ± 0.047 | 0.563 ± 0.016 |
| human only | 40 | 320 | 0.449 ± 0.012 | 0.536 ± 0.014 | 0.521 ± 0.027 | 0.625 ± 0.014 |
| human only | 80 | 640 | 0.511 ± 0.003 | 0.581 ± 0.015 | 0.566 ± 0.007 | 0.665 ± 0.020 |
| human only | 160 | 1280 | 0.538 ± 0.006 | 0.624 ± 0.005 | 0.621 ± 0.006 | 0.715 ± 0.013 |
| human only | 320 | 2560 | 0.546 ± 0.000 | 0.643 ± 0.008 | 0.644 ± 0.010 | 0.742 ± 0.006 |
| human only | 378 | 3024 | 0.531 ± 0.000 | 0.650 ± 0.000 | 0.665 ± 0.000 | 0.746 ± 0.000 |

D2 check: native LR at k=378 minus at k=80 = +0.098 (threshold 0.03; k=378 has one fit, so no seed variance, k=80 mean over 3 seeds).

Paired on the frozen items (post hoc CIs for the pre-registered D2 and for equal-data comparisons):

- D2 native LR k=378 - k=80: +0.096 [+0.067, +0.126]*
- probe - TF-IDF at k=378: -0.094 [-0.134, -0.055]*
- native LR - TF-IDF at k=378: -0.008 [-0.047, +0.032]
- fusion - TF-IDF at k=378: +0.078 [+0.050, +0.108]*

Zero human training data (LLM-written rows only), paired on the frozen items (post hoc):

- probe - TF-IDF at k=0: +0.160 [+0.123, +0.198]*
- native LR - TF-IDF at k=0: +0.101 [+0.066, +0.137]*
- probe at k=0: macro-F1 0.305 [0.273, 0.337] vs shuffle-null p95 0.129; TF-IDF 0.146
