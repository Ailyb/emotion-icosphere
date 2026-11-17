# T0d external cross-domain evaluation (layer 10, logreg; pre-registered in PLAN_v2)

Training sizes: HUM n=4192 (tau 1.48), LLM n=1014 (tau 1.81). Nothing tuned on the external sources. Frozen GoEmotions results are in frozen_eval.md.

## Pre-registered tally (H1: probe - TF-IDF, CI excluding 0)

- trained on HUM: probe > TF-IDF on 3/3 sources ['dair', 'isear', 'dailydialog']; probe < TF-IDF on 0 []
- trained on LLM: probe > TF-IDF on 3/3 sources ['dair', 'isear', 'dailydialog']; probe < TF-IDF on 0 []

## dair: n=1999, classes ['anger', 'fear', 'joy', 'love', 'sadness', 'surprise']; chance 0.167
License/provenance note: other (no terms stated on the card); labels: machine-generated from hashtags (distant supervision).

| training | system | macro-F1 [95% CI] | accuracy | shuffle-null p95 |
|---|---|---|---|---|
| HUM | probe | 0.407 [0.381, 0.430] | 0.498 | 0.171 |
| HUM | native_lr | 0.408 [0.379, 0.434] | 0.516 | 0.175 |
| HUM | embed | 0.305 [0.284, 0.326] | 0.376 | 0.178 |
| HUM | tfidf | 0.254 [0.232, 0.279] | 0.313 | 0.164 |
| HUM | fusion (post hoc) | 0.392 [0.364, 0.420] | 0.494 | 0.170 |
| LLM | probe | 0.303 [0.281, 0.323] | 0.414 | 0.164 |
| LLM | native_lr | 0.325 [0.302, 0.346] | 0.432 | 0.176 |
| LLM | embed | 0.160 [0.145, 0.176] | 0.235 | 0.139 |
| LLM | tfidf | 0.167 [0.151, 0.182] | 0.251 | 0.144 |
| LLM | fusion (post hoc) | 0.313 [0.290, 0.334] | 0.421 | 0.173 |

Paired differences, macro-F1 (mean [95% interval]); * = excludes 0

| training | probe - tfidf (H1) | probe - embed (H2) | probe - native_lr | fusion - tfidf |
|---|---|---|---|---|
| HUM | +0.152 [+0.120, +0.183]* | +0.101 [+0.072, +0.131]* | -0.001 [-0.022, +0.020] | +0.137 [+0.111, +0.164]* |
| LLM | +0.135 [+0.109, +0.160]* | +0.142 [+0.118, +0.166]* | -0.023 [-0.039, -0.005]* | +0.145 [+0.120, +0.170]* |

Per-class recall (probe / TF-IDF), n in brackets (post hoc, descriptive):

- anger [275]: HUM: 0.21 / 0.07; LLM: 0.27 / 0.07
- fear [224]: HUM: 0.53 / 0.41; LLM: 0.44 / 0.38
- joy [695]: HUM: 0.50 / 0.23; LLM: 0.76 / 0.51
- love [159]: HUM: 0.23 / 0.21; LLM: 0.14 / 0.12
- sadness [580]: HUM: 0.71 / 0.53; LLM: 0.16 / 0.02
- surprise [66]: HUM: 0.36 / 0.23; LLM: 0.21 / 0.17
## isear: n=1500, classes ['anger', 'disgust', 'fear', 'joy', 'sadness']; chance 0.200
License/provenance note: Hub tag apache-2.0 (uploader's claim; original terms unverified); labels: self-reported emotion of the writer for their own event.

| training | system | macro-F1 [95% CI] | accuracy | shuffle-null p95 |
|---|---|---|---|---|
| HUM | probe | 0.561 [0.537, 0.583] | 0.561 | 0.211 |
| HUM | native_lr | 0.596 [0.573, 0.619] | 0.603 | 0.207 |
| HUM | embed | 0.398 [0.373, 0.423] | 0.403 | 0.208 |
| HUM | tfidf | 0.434 [0.409, 0.458] | 0.437 | 0.211 |
| HUM | fusion (post hoc) | 0.570 [0.545, 0.594] | 0.575 | 0.205 |
| LLM | probe | 0.420 [0.395, 0.442] | 0.457 | 0.191 |
| LLM | native_lr | 0.436 [0.409, 0.461] | 0.477 | 0.193 |
| LLM | embed | 0.252 [0.231, 0.274] | 0.281 | 0.189 |
| LLM | tfidf | 0.224 [0.204, 0.245] | 0.243 | 0.202 |
| LLM | fusion (post hoc) | 0.391 [0.364, 0.415] | 0.436 | 0.190 |

Paired differences, macro-F1 (mean [95% interval]); * = excludes 0

| training | probe - tfidf (H1) | probe - embed (H2) | probe - native_lr | fusion - tfidf |
|---|---|---|---|---|
| HUM | +0.127 [+0.097, +0.157]* | +0.163 [+0.133, +0.194]* | -0.035 [-0.052, -0.017]* | +0.136 [+0.111, +0.161]* |
| LLM | +0.196 [+0.166, +0.226]* | +0.168 [+0.136, +0.199]* | -0.016 [-0.040, +0.008] | +0.167 [+0.142, +0.192]* |

Per-class recall (probe / TF-IDF), n in brackets (post hoc, descriptive):

- anger [300]: HUM: 0.36 / 0.38; LLM: 0.57 / 0.30
- disgust [300]: HUM: 0.53 / 0.32; LLM: 0.58 / 0.16
- fear [300]: HUM: 0.51 / 0.41; LLM: 0.23 / 0.10
- joy [300]: HUM: 0.62 / 0.47; LLM: 0.81 / 0.50
- sadness [300]: HUM: 0.78 / 0.61; LLM: 0.10 / 0.15
## dailydialog: n=2193, classes ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']; chance 0.143
License/provenance note: CC-BY-NC-SA-4.0; labels: human annotated per utterance.

| training | system | macro-F1 [95% CI] | accuracy | shuffle-null p95 |
|---|---|---|---|---|
| HUM | probe | 0.294 [0.272, 0.315] | 0.409 | 0.133 |
| HUM | native_lr | 0.399 [0.369, 0.427] | 0.569 | 0.149 |
| HUM | embed | 0.219 [0.201, 0.237] | 0.344 | 0.133 |
| HUM | tfidf | 0.240 [0.217, 0.263] | 0.390 | 0.140 |
| HUM | fusion (post hoc) | 0.385 [0.353, 0.413] | 0.561 | 0.149 |
| LLM | probe | 0.246 [0.220, 0.274] | 0.390 | 0.121 |
| LLM | native_lr | 0.328 [0.293, 0.360] | 0.453 | 0.145 |
| LLM | embed | 0.182 [0.165, 0.198] | 0.293 | 0.132 |
| LLM | tfidf | 0.154 [0.139, 0.170] | 0.295 | 0.132 |
| LLM | fusion (post hoc) | 0.325 [0.287, 0.358] | 0.452 | 0.144 |

Paired differences, macro-F1 (mean [95% interval]); * = excludes 0

| training | probe - tfidf (H1) | probe - embed (H2) | probe - native_lr | fusion - tfidf |
|---|---|---|---|---|
| HUM | +0.054 [+0.029, +0.079]* | +0.074 [+0.052, +0.097]* | -0.105 [-0.127, -0.083]* | +0.145 [+0.120, +0.171]* |
| LLM | +0.092 [+0.063, +0.124]* | +0.064 [+0.036, +0.096]* | -0.080 [-0.108, -0.053]* | +0.169 [+0.134, +0.204]* |

Per-class recall (probe / TF-IDF), n in brackets (post hoc, descriptive):

- anger [109]: HUM: 0.33 / 0.18; LLM: 0.21 / 0.09
- disgust [45]: HUM: 0.38 / 0.22; LLM: 0.76 / 0.27
- fear [16]: HUM: 0.50 / 0.19; LLM: 0.31 / 0.06
- joy [916]: HUM: 0.48 / 0.30; LLM: 0.64 / 0.16
- neutral [915]: HUM: 0.31 / 0.53; LLM: 0.17 / 0.49
- sadness [94]: HUM: 0.50 / 0.47; LLM: 0.02 / 0.06
- surprise [98]: HUM: 0.64 / 0.24; LLM: 0.48 / 0.27
