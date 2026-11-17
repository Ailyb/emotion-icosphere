# T0c frozen-set evaluation (n=840; manifest sha256 f47b22ae79d3...)

Layer 16; selected estimator lda (chosen on VAL-hard of the pooled eval, not on this set). Trained on pooled merged2 TRAIN without calm (n=1654); tau and C on VAL-hard (n=225). 8-way macro-F1 over joy, sadness, fear, anger, surprise, disgust, love, neutral; bootstrap 95% CI over test items.

| system | macro-F1 [95% CI] | accuracy | shuffle-null p95 | coarse valence F1 |
|---|---|---|---|---|
| probe_lda (selected) | 0.510 [0.476, 0.544] | 0.521 | 0.140 | 0.579 |
| probe_meandiff | 0.489 [0.454, 0.521] | 0.485 | 0.144 | 0.559 |
| probe_logreg | 0.517 [0.483, 0.550] | 0.521 | 0.141 | 0.588 |
| native_lr | 0.521 [0.486, 0.554] | 0.531 | 0.144 | 0.617 |
| embed | 0.403 [0.373, 0.435] | 0.401 | 0.145 | 0.456 |
| block0 | 0.432 [0.401, 0.465] | 0.433 | 0.143 | 0.503 |
| tfidf | 0.599 [0.564, 0.630] | 0.599 | 0.144 | 0.628 |

## Paired differences, macro-F1 (mean [95% interval]); * = excludes 0

| comparison | difference |
|---|---|
| probe_lda - tfidf | -0.088 [-0.128, -0.046]* |
| probe_lda - embed | +0.107 [+0.065, +0.148]* |
| probe_lda - block0 | +0.078 [+0.039, +0.116]* |
| probe_lda - native_lr | -0.010 [-0.036, +0.016] |

## Per class (selected probe)

| class | n | recall | F1 | native LR recall |
|---|---|---|---|---|
| joy | 91 | 0.58 | 0.56 | 0.62 |
| sadness | 98 | 0.53 | 0.54 | 0.56 |
| fear | 63 | 0.63 | 0.54 | 0.62 |
| anger | 128 | 0.52 | 0.55 | 0.52 |
| surprise | 84 | 0.58 | 0.53 | 0.56 |
| disgust | 74 | 0.47 | 0.40 | 0.38 |
| love | 151 | 0.75 | 0.70 | 0.70 |
| neutral | 151 | 0.19 | 0.26 | 0.32 |

## Strata (macro-F1, systems: probe, native_lr, embed, tfidf)

- rater agreement >= 0.67 (n=244): probe 0.594, native_lr 0.639, embed 0.465, tfidf 0.749
- rater agreement < 0.67 (n=596): probe 0.475, native_lr 0.473, embed 0.378, tfidf 0.544
- <= 12 words (n=431): probe 0.541, native_lr 0.553, embed 0.378, tfidf 0.572
- > 12 words (n=409): probe 0.475, native_lr 0.485, embed 0.428, tfidf 0.623
- subtypes (too small to score; n only): {'no_emotion_lexicon_match': 403, 'negation': 10, 'emotion_word_neutral': 6}

## Human-data learning curve (frozen-set macro-F1, mean ± sd over subsampling seeds; k=max is 378 per class, fear-limited)

| training | k per class | n train | probe F1 | native LR F1 |
|---|---|---|---|---|
| LLM + human | 0 | 1014 | 0.305 ± 0.000 | 0.271 ± 0.000 |
| LLM + human | 20 | 1174 | 0.439 ± 0.011 | 0.448 ± 0.004 |
| LLM + human | 40 | 1334 | 0.471 ± 0.009 | 0.478 ± 0.009 |
| LLM + human | 80 | 1654 | 0.512 ± 0.012 | 0.548 ± 0.005 |
| LLM + human | 160 | 2294 | 0.542 ± 0.007 | 0.588 ± 0.005 |
| LLM + human | 320 | 3574 | 0.566 ± 0.002 | 0.632 ± 0.012 |
| LLM + human | 378 | 4038 | 0.576 ± 0.000 | 0.617 ± 0.000 |
| human only | 20 | 160 | 0.435 ± 0.033 | 0.485 ± 0.012 |
| human only | 40 | 320 | 0.440 ± 0.009 | 0.521 ± 0.004 |
| human only | 80 | 640 | 0.509 ± 0.009 | 0.570 ± 0.017 |
| human only | 160 | 1280 | 0.535 ± 0.025 | 0.594 ± 0.010 |
| human only | 320 | 2560 | 0.559 ± 0.024 | 0.635 ± 0.018 |
| human only | 378 | 3024 | 0.596 ± 0.000 | 0.628 ± 0.000 |

D2 check: native LR at k=378 minus at k=80 = +0.069 (threshold 0.03; k=378 has one fit, so no seed variance, k=80 mean over 3 seeds).
