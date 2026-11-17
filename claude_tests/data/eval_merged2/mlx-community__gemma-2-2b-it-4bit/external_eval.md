# T0d external cross-domain evaluation (layer 14, lda; pre-registered in PLAN_v2)

Training sizes: HUM n=4192 (tau 1.74), LLM n=1014 (tau 2.39). Nothing tuned on the external sources. Frozen GoEmotions results are in frozen_eval.md.

## Pre-registered tally (H1: probe - TF-IDF, CI excluding 0)

- trained on HUM: probe > TF-IDF on 3/3 sources ['dair', 'isear', 'dailydialog']; probe < TF-IDF on 0 []
- trained on LLM: probe > TF-IDF on 3/3 sources ['dair', 'isear', 'dailydialog']; probe < TF-IDF on 0 []

## dair: n=1999, classes ['anger', 'fear', 'joy', 'love', 'sadness', 'surprise']; chance 0.167
License/provenance note: other (no terms stated on the card); labels: machine-generated from hashtags (distant supervision).

| training | system | macro-F1 [95% CI] | accuracy | shuffle-null p95 |
|---|---|---|---|---|
| HUM | probe | 0.391 [0.368, 0.413] | 0.464 | 0.175 |
| HUM | native_lr | 0.406 [0.379, 0.432] | 0.500 | 0.174 |
| HUM | embed | 0.329 [0.306, 0.352] | 0.409 | 0.176 |
| HUM | tfidf | 0.254 [0.232, 0.279] | 0.313 | 0.164 |
| HUM | fusion (post hoc) | 0.386 [0.361, 0.410] | 0.483 | 0.172 |
| LLM | probe | 0.294 [0.273, 0.313] | 0.404 | 0.157 |
| LLM | native_lr | 0.340 [0.317, 0.363] | 0.458 | 0.178 |
| LLM | embed | 0.190 [0.172, 0.207] | 0.257 | 0.140 |
| LLM | tfidf | 0.167 [0.151, 0.182] | 0.251 | 0.144 |
| LLM | fusion (post hoc) | 0.310 [0.290, 0.333] | 0.432 | 0.174 |

Paired differences, macro-F1 (mean [95% interval]); * = excludes 0

| training | probe - tfidf (H1) | probe - embed (H2) | probe - native_lr | fusion - tfidf |
|---|---|---|---|---|
| HUM | +0.137 [+0.107, +0.166]* | +0.062 [+0.036, +0.089]* | -0.014 [-0.035, +0.006] | +0.131 [+0.106, +0.156]* |
| LLM | +0.127 [+0.102, +0.150]* | +0.104 [+0.080, +0.129]* | -0.046 [-0.066, -0.027]* | +0.143 [+0.119, +0.165]* |

Per-class recall (probe / TF-IDF), n in brackets (post hoc, descriptive):

- anger [275]: HUM: 0.34 / 0.07; LLM: 0.26 / 0.07
- fear [224]: HUM: 0.54 / 0.41; LLM: 0.59 / 0.38
- joy [695]: HUM: 0.54 / 0.23; LLM: 0.73 / 0.51
- love [159]: HUM: 0.30 / 0.21; LLM: 0.13 / 0.12
- sadness [580]: HUM: 0.47 / 0.53; LLM: 0.11 / 0.02
- surprise [66]: HUM: 0.32 / 0.23; LLM: 0.23 / 0.17
## isear: n=1500, classes ['anger', 'disgust', 'fear', 'joy', 'sadness']; chance 0.200
License/provenance note: Hub tag apache-2.0 (uploader's claim; original terms unverified); labels: self-reported emotion of the writer for their own event.

| training | system | macro-F1 [95% CI] | accuracy | shuffle-null p95 |
|---|---|---|---|---|
| HUM | probe | 0.624 [0.600, 0.649] | 0.624 | 0.217 |
| HUM | native_lr | 0.595 [0.569, 0.618] | 0.601 | 0.209 |
| HUM | embed | 0.375 [0.351, 0.400] | 0.385 | 0.201 |
| HUM | tfidf | 0.434 [0.409, 0.458] | 0.437 | 0.211 |
| HUM | fusion (post hoc) | 0.588 [0.564, 0.612] | 0.593 | 0.207 |
| LLM | probe | 0.487 [0.462, 0.512] | 0.513 | 0.202 |
| LLM | native_lr | 0.506 [0.481, 0.531] | 0.529 | 0.200 |
| LLM | embed | 0.266 [0.244, 0.291] | 0.290 | 0.191 |
| LLM | tfidf | 0.224 [0.204, 0.245] | 0.243 | 0.202 |
| LLM | fusion (post hoc) | 0.457 [0.432, 0.480] | 0.487 | 0.197 |

Paired differences, macro-F1 (mean [95% interval]); * = excludes 0

| training | probe - tfidf (H1) | probe - embed (H2) | probe - native_lr | fusion - tfidf |
|---|---|---|---|---|
| HUM | +0.190 [+0.159, +0.222]* | +0.249 [+0.219, +0.278]* | +0.029 [+0.008, +0.050]* | +0.155 [+0.130, +0.180]* |
| LLM | +0.264 [+0.232, +0.293]* | +0.221 [+0.190, +0.251]* | -0.019 [-0.042, +0.007] | +0.233 [+0.207, +0.260]* |

Per-class recall (probe / TF-IDF), n in brackets (post hoc, descriptive):

- anger [300]: HUM: 0.54 / 0.38; LLM: 0.72 / 0.30
- disgust [300]: HUM: 0.53 / 0.32; LLM: 0.40 / 0.16
- fear [300]: HUM: 0.68 / 0.41; LLM: 0.47 / 0.10
- joy [300]: HUM: 0.72 / 0.47; LLM: 0.82 / 0.50
- sadness [300]: HUM: 0.65 / 0.61; LLM: 0.16 / 0.15
## dailydialog: n=2193, classes ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']; chance 0.143
License/provenance note: CC-BY-NC-SA-4.0; labels: human annotated per utterance.

| training | system | macro-F1 [95% CI] | accuracy | shuffle-null p95 |
|---|---|---|---|---|
| HUM | probe | 0.335 [0.311, 0.358] | 0.454 | 0.142 |
| HUM | native_lr | 0.434 [0.400, 0.465] | 0.587 | 0.149 |
| HUM | embed | 0.230 [0.210, 0.250] | 0.364 | 0.135 |
| HUM | tfidf | 0.240 [0.217, 0.263] | 0.390 | 0.140 |
| HUM | fusion (post hoc) | 0.396 [0.364, 0.426] | 0.567 | 0.148 |
| LLM | probe | 0.306 [0.281, 0.332] | 0.497 | 0.141 |
| LLM | native_lr | 0.365 [0.329, 0.396] | 0.473 | 0.142 |
| LLM | embed | 0.174 [0.157, 0.193] | 0.277 | 0.128 |
| LLM | tfidf | 0.154 [0.139, 0.170] | 0.295 | 0.132 |
| LLM | fusion (post hoc) | 0.355 [0.316, 0.388] | 0.472 | 0.143 |

Paired differences, macro-F1 (mean [95% interval]); * = excludes 0

| training | probe - tfidf (H1) | probe - embed (H2) | probe - native_lr | fusion - tfidf |
|---|---|---|---|---|
| HUM | +0.095 [+0.069, +0.121]* | +0.105 [+0.079, +0.131]* | -0.098 [-0.122, -0.076]* | +0.156 [+0.128, +0.184]* |
| LLM | +0.152 [+0.124, +0.179]* | +0.132 [+0.106, +0.161]* | -0.058 [-0.089, -0.027]* | +0.200 [+0.165, +0.235]* |

Per-class recall (probe / TF-IDF), n in brackets (post hoc, descriptive):

- anger [109]: HUM: 0.35 / 0.18; LLM: 0.23 / 0.09
- disgust [45]: HUM: 0.33 / 0.22; LLM: 0.47 / 0.27
- fear [16]: HUM: 0.56 / 0.19; LLM: 0.62 / 0.06
- joy [916]: HUM: 0.47 / 0.30; LLM: 0.74 / 0.16
- neutral [915]: HUM: 0.44 / 0.53; LLM: 0.32 / 0.49
- sadness [94]: HUM: 0.51 / 0.47; LLM: 0.09 / 0.06
- surprise [98]: HUM: 0.57 / 0.24; LLM: 0.53 / 0.27
