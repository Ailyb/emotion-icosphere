# T0c frozen-set evaluation (n=840; manifest sha256 f47b22ae79d3...)

Layer 14; selected estimator lda (chosen on VAL-hard of the pooled eval, not on this set). Trained on pooled merged2 TRAIN without calm (n=1654); tau and C on VAL-hard (n=225). 8-way macro-F1 over joy, sadness, fear, anger, surprise, disgust, love, neutral; bootstrap 95% CI over test items.

| system | macro-F1 [95% CI] | accuracy | shuffle-null p95 | coarse valence F1 |
|---|---|---|---|---|
| probe_lda (selected) | 0.513 [0.478, 0.546] | 0.517 | 0.141 | 0.580 |
| probe_meandiff | 0.441 [0.407, 0.473] | 0.449 | 0.139 | 0.517 |
| probe_logreg | 0.509 [0.475, 0.542] | 0.508 | 0.141 | 0.588 |
| native_lr | 0.537 [0.503, 0.571] | 0.542 | 0.143 | 0.628 |
| embed | 0.401 [0.370, 0.434] | 0.400 | 0.141 | 0.470 |
| block0 | 0.452 [0.418, 0.483] | 0.451 | 0.143 | 0.516 |
| tfidf | 0.599 [0.564, 0.630] | 0.599 | 0.144 | 0.628 |
| fusion_tfidf_native (post hoc) | 0.661 [0.627, 0.693] | 0.662 | 0.142 | 0.696 |

## Paired differences, macro-F1 (mean [95% interval]); * = excludes 0

| comparison | difference |
|---|---|
| probe_lda - tfidf | -0.085 [-0.128, -0.046]* |
| probe_lda - embed | +0.112 [+0.073, +0.150]* |
| probe_lda - block0 | +0.061 [+0.022, +0.100]* |
| probe_lda - native_lr | -0.024 [-0.053, +0.005] |
| fusion_tfidf_native (post hoc) - tfidf (post hoc) | +0.062 [+0.027, +0.095]* |
| fusion_tfidf_native (post hoc) - native_lr (post hoc) | +0.124 [+0.097, +0.150]* |

## Per class (selected probe)

| class | n | recall | F1 | native LR recall |
|---|---|---|---|---|
| joy | 91 | 0.48 | 0.52 | 0.66 |
| sadness | 98 | 0.44 | 0.55 | 0.61 |
| fear | 63 | 0.71 | 0.56 | 0.60 |
| anger | 128 | 0.42 | 0.50 | 0.55 |
| surprise | 84 | 0.58 | 0.52 | 0.54 |
| disgust | 74 | 0.50 | 0.44 | 0.42 |
| love | 151 | 0.78 | 0.70 | 0.66 |
| neutral | 151 | 0.29 | 0.31 | 0.34 |

## Strata (macro-F1, systems: probe, native_lr, embed, tfidf)

- rater agreement >= 0.67 (n=244): probe 0.570, native_lr 0.666, embed 0.492, tfidf 0.749
- rater agreement < 0.67 (n=596): probe 0.486, native_lr 0.482, embed 0.361, tfidf 0.544
- <= 12 words (n=431): probe 0.536, native_lr 0.576, embed 0.370, tfidf 0.572
- > 12 words (n=409): probe 0.485, native_lr 0.491, embed 0.430, tfidf 0.623
- subtypes (too small to score; n only): {'no_emotion_lexicon_match': 403, 'negation': 10, 'emotion_word_neutral': 6}

## Human-data learning curve (frozen-set macro-F1, mean ± sd over subsampling seeds; k=max is 378 per class, fear-limited)

| training | k per class | n train | probe F1 | native LR F1 | TF-IDF F1 (post hoc) | fusion F1 (post hoc) |
|---|---|---|---|---|---|---|
| LLM + human | 0 | 1014 | 0.313 ± 0.000 | 0.290 ± 0.000 | 0.146 ± 0.000 | 0.289 ± 0.000 |
| LLM + human | 20 | 1174 | 0.428 ± 0.010 | 0.440 ± 0.005 | 0.451 ± 0.028 | 0.511 ± 0.013 |
| LLM + human | 40 | 1334 | 0.463 ± 0.019 | 0.481 ± 0.012 | 0.526 ± 0.007 | 0.576 ± 0.018 |
| LLM + human | 80 | 1654 | 0.507 ± 0.004 | 0.536 ± 0.011 | 0.583 ± 0.016 | 0.657 ± 0.009 |
| LLM + human | 160 | 2294 | 0.539 ± 0.012 | 0.566 ± 0.009 | 0.620 ± 0.016 | 0.673 ± 0.022 |
| LLM + human | 320 | 3574 | 0.566 ± 0.005 | 0.615 ± 0.008 | 0.659 ± 0.004 | 0.715 ± 0.019 |
| LLM + human | 378 | 4038 | 0.568 ± 0.000 | 0.628 ± 0.000 | 0.666 ± 0.000 | 0.739 ± 0.000 |
| human only | 20 | 160 | 0.398 ± 0.003 | 0.465 ± 0.008 | 0.450 ± 0.047 | 0.555 ± 0.019 |
| human only | 40 | 320 | 0.444 ± 0.010 | 0.521 ± 0.005 | 0.521 ± 0.027 | 0.629 ± 0.010 |
| human only | 80 | 640 | 0.499 ± 0.018 | 0.564 ± 0.001 | 0.566 ± 0.007 | 0.670 ± 0.013 |
| human only | 160 | 1280 | 0.524 ± 0.023 | 0.597 ± 0.013 | 0.621 ± 0.006 | 0.709 ± 0.031 |
| human only | 320 | 2560 | 0.575 ± 0.013 | 0.639 ± 0.008 | 0.644 ± 0.010 | 0.744 ± 0.004 |
| human only | 378 | 3024 | 0.543 ± 0.000 | 0.642 ± 0.000 | 0.665 ± 0.000 | 0.747 ± 0.000 |

D2 check: native LR at k=378 minus at k=80 = +0.092 (threshold 0.03; k=378 has one fit, so no seed variance, k=80 mean over 3 seeds).

Paired on the frozen items (post hoc CIs for the pre-registered D2 and for equal-data comparisons):

- D2 native LR k=378 - k=80: +0.103 [+0.074, +0.132]*
- probe - TF-IDF at k=378: -0.098 [-0.136, -0.059]*
- native LR - TF-IDF at k=378: -0.037 [-0.078, +0.004]
- fusion - TF-IDF at k=378: +0.073 [+0.044, +0.101]*

Zero human training data (LLM-written rows only), paired on the frozen items (post hoc):

- probe - TF-IDF at k=0: +0.167 [+0.128, +0.206]*
- native LR - TF-IDF at k=0: +0.144 [+0.107, +0.181]*
- probe at k=0: macro-F1 0.313 [0.283, 0.344] vs shuffle-null p95 0.130; TF-IDF 0.146
