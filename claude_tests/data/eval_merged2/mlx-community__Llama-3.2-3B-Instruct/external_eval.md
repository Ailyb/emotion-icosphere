# T0d external cross-domain evaluation (layer 16, lda; pre-registered in PLAN_v2)

Training sizes: HUM n=4192 (tau 2.01), LLM n=1014 (tau 1.55). Nothing tuned on the external sources. Frozen GoEmotions results are in frozen_eval.md.

## Pre-registered tally (H1: probe - TF-IDF, CI excluding 0)

- trained on HUM: probe > TF-IDF on 3/3 sources ['dair', 'isear', 'dailydialog']; probe < TF-IDF on 0 []
- trained on LLM: probe > TF-IDF on 3/3 sources ['dair', 'isear', 'dailydialog']; probe < TF-IDF on 0 []

## dair: n=1999, classes ['anger', 'fear', 'joy', 'love', 'sadness', 'surprise']; chance 0.167
License/provenance note: other (no terms stated on the card); labels: machine-generated from hashtags (distant supervision).

| training | system | macro-F1 [95% CI] | accuracy | shuffle-null p95 |
|---|---|---|---|---|
| HUM | probe | 0.405 [0.382, 0.427] | 0.485 | 0.176 |
| HUM | native_lr | 0.405 [0.377, 0.431] | 0.510 | 0.176 |
| HUM | embed | 0.329 [0.306, 0.352] | 0.424 | 0.177 |
| HUM | tfidf | 0.254 [0.232, 0.279] | 0.313 | 0.164 |
| HUM | fusion (post hoc) | 0.388 [0.362, 0.414] | 0.481 | 0.172 |
| LLM | probe | 0.334 [0.312, 0.355] | 0.447 | 0.172 |
| LLM | native_lr | 0.327 [0.306, 0.348] | 0.457 | 0.176 |
| LLM | embed | 0.179 [0.163, 0.195] | 0.249 | 0.136 |
| LLM | tfidf | 0.167 [0.151, 0.182] | 0.251 | 0.144 |
| LLM | fusion (post hoc) | 0.303 [0.282, 0.324] | 0.434 | 0.172 |

Paired differences, macro-F1 (mean [95% interval]); * = excludes 0

| training | probe - tfidf (H1) | probe - embed (H2) | probe - native_lr | fusion - tfidf |
|---|---|---|---|---|
| HUM | +0.151 [+0.122, +0.178]* | +0.076 [+0.049, +0.103]* | +0.000 [-0.020, +0.021] | +0.134 [+0.107, +0.159]* |
| LLM | +0.167 [+0.141, +0.192]* | +0.155 [+0.131, +0.178]* | +0.007 [-0.014, +0.028] | +0.136 [+0.113, +0.159]* |

Per-class recall (probe / TF-IDF), n in brackets (post hoc, descriptive):

- anger [275]: HUM: 0.36 / 0.07; LLM: 0.25 / 0.07
- fear [224]: HUM: 0.51 / 0.41; LLM: 0.38 / 0.38
- joy [695]: HUM: 0.54 / 0.23; LLM: 0.75 / 0.51
- love [159]: HUM: 0.25 / 0.21; LLM: 0.26 / 0.12
- sadness [580]: HUM: 0.54 / 0.53; LLM: 0.29 / 0.02
- surprise [66]: HUM: 0.39 / 0.23; LLM: 0.14 / 0.17
## isear: n=1500, classes ['anger', 'disgust', 'fear', 'joy', 'sadness']; chance 0.200
License/provenance note: Hub tag apache-2.0 (uploader's claim; original terms unverified); labels: self-reported emotion of the writer for their own event.

| training | system | macro-F1 [95% CI] | accuracy | shuffle-null p95 |
|---|---|---|---|---|
| HUM | probe | 0.567 [0.539, 0.589] | 0.563 | 0.215 |
| HUM | native_lr | 0.575 [0.548, 0.597] | 0.576 | 0.206 |
| HUM | embed | 0.380 [0.353, 0.405] | 0.391 | 0.198 |
| HUM | tfidf | 0.434 [0.409, 0.458] | 0.437 | 0.211 |
| HUM | fusion (post hoc) | 0.560 [0.535, 0.583] | 0.561 | 0.206 |
| LLM | probe | 0.486 [0.458, 0.511] | 0.485 | 0.203 |
| LLM | native_lr | 0.511 [0.486, 0.537] | 0.532 | 0.205 |
| LLM | embed | 0.274 [0.251, 0.298] | 0.297 | 0.192 |
| LLM | tfidf | 0.224 [0.204, 0.245] | 0.243 | 0.202 |
| LLM | fusion (post hoc) | 0.472 [0.448, 0.496] | 0.496 | 0.204 |

Paired differences, macro-F1 (mean [95% interval]); * = excludes 0

| training | probe - tfidf (H1) | probe - embed (H2) | probe - native_lr | fusion - tfidf |
|---|---|---|---|---|
| HUM | +0.132 [+0.101, +0.163]* | +0.186 [+0.155, +0.216]* | -0.008 [-0.028, +0.011] | +0.126 [+0.101, +0.150]* |
| LLM | +0.262 [+0.230, +0.293]* | +0.212 [+0.180, +0.245]* | -0.025 [-0.049, -0.002]* | +0.248 [+0.221, +0.274]* |

Per-class recall (probe / TF-IDF), n in brackets (post hoc, descriptive):

- anger [300]: HUM: 0.49 / 0.38; LLM: 0.63 / 0.30
- disgust [300]: HUM: 0.47 / 0.32; LLM: 0.52 / 0.16
- fear [300]: HUM: 0.57 / 0.41; LLM: 0.28 / 0.10
- joy [300]: HUM: 0.61 / 0.47; LLM: 0.64 / 0.50
- sadness [300]: HUM: 0.67 / 0.61; LLM: 0.36 / 0.15
## dailydialog: n=2193, classes ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']; chance 0.143
License/provenance note: CC-BY-NC-SA-4.0; labels: human annotated per utterance.

| training | system | macro-F1 [95% CI] | accuracy | shuffle-null p95 |
|---|---|---|---|---|
| HUM | probe | 0.343 [0.315, 0.367] | 0.465 | 0.140 |
| HUM | native_lr | 0.410 [0.381, 0.439] | 0.573 | 0.148 |
| HUM | embed | 0.234 [0.213, 0.254] | 0.392 | 0.140 |
| HUM | tfidf | 0.240 [0.217, 0.263] | 0.390 | 0.140 |
| HUM | fusion (post hoc) | 0.390 [0.360, 0.418] | 0.565 | 0.147 |
| LLM | probe | 0.280 [0.254, 0.305] | 0.416 | 0.122 |
| LLM | native_lr | 0.326 [0.294, 0.355] | 0.439 | 0.137 |
| LLM | embed | 0.178 [0.160, 0.194] | 0.282 | 0.129 |
| LLM | tfidf | 0.154 [0.139, 0.170] | 0.295 | 0.132 |
| LLM | fusion (post hoc) | 0.316 [0.283, 0.348] | 0.442 | 0.138 |

Paired differences, macro-F1 (mean [95% interval]); * = excludes 0

| training | probe - tfidf (H1) | probe - embed (H2) | probe - native_lr | fusion - tfidf |
|---|---|---|---|---|
| HUM | +0.103 [+0.077, +0.130]* | +0.109 [+0.083, +0.137]* | -0.067 [-0.089, -0.045]* | +0.150 [+0.125, +0.177]* |
| LLM | +0.125 [+0.097, +0.154]* | +0.102 [+0.074, +0.130]* | -0.046 [-0.071, -0.018]* | +0.161 [+0.130, +0.195]* |

Per-class recall (probe / TF-IDF), n in brackets (post hoc, descriptive):

- anger [109]: HUM: 0.42 / 0.18; LLM: 0.27 / 0.09
- disgust [45]: HUM: 0.33 / 0.22; LLM: 0.69 / 0.27
- fear [16]: HUM: 0.56 / 0.19; LLM: 0.44 / 0.06
- joy [916]: HUM: 0.36 / 0.30; LLM: 0.69 / 0.16
- neutral [915]: HUM: 0.57 / 0.53; LLM: 0.15 / 0.49
- sadness [94]: HUM: 0.45 / 0.47; LLM: 0.23 / 0.06
- surprise [98]: HUM: 0.57 / 0.24; LLM: 0.58 / 0.27
