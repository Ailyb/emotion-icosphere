# Probe audit (layer 16, lda; trained on n=960, validated on n=320; classes ['anger', 'disgust', 'fear', 'joy', 'love', 'neutral', 'sadness', 'surprise'])

Train classes (n): {'anger': 120, 'disgust': 120, 'fear': 120, 'joy': 120, 'love': 120, 'neutral': 120, 'sadness': 120, 'surprise': 120}; excluded by request: none. tau (neutral threshold) = 6.22, chosen on VAL.

Cache provenance: model mlx-community/Llama-3.2-3B-Instruct, pooling mean over tokens[skip:], skip 4, code 318cf6ad

## t5_test: n=320, evaluated classes ['anger', 'disgust', 'fear', 'joy', 'love', 'neutral', 'sadness', 'surprise']; chance 0.125
Per-class n: anger 40, disgust 40, fear 40, joy 40, love 40, neutral 40, sadness 40, surprise 40

| system | macro-F1 [95% CI] | accuracy | shuffle-null p95 |
|---|---|---|---|
| probe | 0.861 [0.823, 0.896] | 0.869 | 0.152 |
| native_lr | 0.953 [0.929, 0.974] | 0.953 | 0.156 |
| embed | 0.891 [0.850, 0.923] | 0.891 | 0.152 |
| tfidf | 0.930 [0.901, 0.957] | 0.931 | 0.156 |
| fusion (post hoc) | 0.975 [0.956, 0.991] | 0.975 | 0.156 |

Paired differences (macro-F1, mean [95% CI]); * = excludes 0:

- probe - tfidf: -0.071 [-0.115, -0.028]*
- probe - embed: -0.031 [-0.077, +0.016]
- probe - native_lr: -0.093 [-0.134, -0.051]*
- stratum <= 12 words (n=0): n < 20, not reported
- stratum > 12 words (n=320): probe 0.861, native_lr 0.953, embed 0.891, tfidf 0.930, fusion (post hoc) 0.975

## goemotions: n=840, evaluated classes ['anger', 'disgust', 'fear', 'joy', 'love', 'neutral', 'sadness', 'surprise']; chance 0.125
Per-class n: anger 128, disgust 74, fear 63, joy 91, love 151, neutral 151, sadness 98, surprise 84

| system | macro-F1 [95% CI] | accuracy | shuffle-null p95 |
|---|---|---|---|
| probe | 0.236 [0.205, 0.265] | 0.256 | 0.109 |
| native_lr | 0.302 [0.268, 0.337] | 0.314 | 0.114 |
| embed | 0.188 [0.163, 0.216] | 0.200 | 0.128 |
| tfidf | 0.120 [0.096, 0.142] | 0.129 | 0.131 |
| fusion (post hoc) | 0.289 [0.256, 0.322] | 0.304 | 0.115 |

Paired differences (macro-F1, mean [95% CI]); * = excludes 0:

- probe - tfidf: +0.116 [+0.080, +0.152]*
- probe - embed: +0.048 [+0.013, +0.084]*
- probe - native_lr: -0.066 [-0.098, -0.033]*
- stratum <= 12 words (n=431): probe 0.252, native_lr 0.330, embed 0.203, tfidf 0.113, fusion (post hoc) 0.313
- stratum > 12 words (n=409): probe 0.215, native_lr 0.268, embed 0.167, tfidf 0.125, fusion (post hoc) 0.257

## dair: n=1999, evaluated classes ['anger', 'fear', 'joy', 'love', 'sadness', 'surprise']; chance 0.167
Per-class n: anger 275, fear 224, joy 695, love 159, sadness 580, surprise 66

| system | macro-F1 [95% CI] | accuracy | shuffle-null p95 |
|---|---|---|---|
| probe | 0.286 [0.264, 0.305] | 0.346 | 0.153 |
| native_lr | 0.304 [0.282, 0.325] | 0.368 | 0.155 |
| embed | 0.212 [0.194, 0.230] | 0.307 | 0.164 |
| tfidf | 0.154 [0.138, 0.170] | 0.192 | 0.149 |
| fusion (post hoc) | 0.275 [0.254, 0.295] | 0.333 | 0.156 |

Paired differences (macro-F1, mean [95% CI]); * = excludes 0:

- probe - tfidf: +0.131 [+0.107, +0.155]*
- probe - embed: +0.073 [+0.047, +0.099]*
- probe - native_lr: -0.019 [-0.039, +0.001]
- stratum <= 12 words (n=680): probe 0.369, native_lr 0.383, embed 0.240, tfidf 0.163, fusion (post hoc) 0.362
- stratum > 12 words (n=1319): probe 0.243, native_lr 0.261, embed 0.189, tfidf 0.150, fusion (post hoc) 0.227

## isear: n=1500, evaluated classes ['anger', 'disgust', 'fear', 'joy', 'sadness']; chance 0.200
Per-class n: anger 300, disgust 300, fear 300, joy 300, sadness 300

| system | macro-F1 [95% CI] | accuracy | shuffle-null p95 |
|---|---|---|---|
| probe | 0.446 [0.420, 0.468] | 0.471 | 0.190 |
| native_lr | 0.586 [0.559, 0.609] | 0.593 | 0.213 |
| embed | 0.264 [0.242, 0.285] | 0.270 | 0.207 |
| tfidf | 0.233 [0.211, 0.254] | 0.243 | 0.205 |
| fusion (post hoc) | 0.464 [0.438, 0.490] | 0.475 | 0.209 |

Paired differences (macro-F1, mean [95% CI]); * = excludes 0:

- probe - tfidf: +0.212 [+0.181, +0.245]*
- probe - embed: +0.182 [+0.152, +0.212]*
- probe - native_lr: -0.141 [-0.167, -0.116]*
- stratum <= 12 words (n=424): probe 0.418, native_lr 0.566, embed 0.234, tfidf 0.233, fusion (post hoc) 0.497
- stratum > 12 words (n=1076): probe 0.456, native_lr 0.588, embed 0.275, tfidf 0.232, fusion (post hoc) 0.448

## dailydialog: n=2193, evaluated classes ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']; chance 0.143
Per-class n: anger 109, disgust 45, fear 16, joy 916, neutral 915, sadness 94, surprise 98

| system | macro-F1 [95% CI] | accuracy | shuffle-null p95 |
|---|---|---|---|
| probe | 0.209 [0.183, 0.234] | 0.361 | 0.114 |
| native_lr | 0.226 [0.191, 0.258] | 0.440 | 0.111 |
| embed | 0.130 [0.114, 0.148] | 0.224 | 0.116 |
| tfidf | 0.153 [0.136, 0.171] | 0.267 | 0.129 |
| fusion (post hoc) | 0.237 [0.202, 0.268] | 0.442 | 0.114 |

Paired differences (macro-F1, mean [95% CI]); * = excludes 0:

- probe - tfidf: +0.056 [+0.026, +0.085]*
- probe - embed: +0.078 [+0.050, +0.107]*
- probe - native_lr: -0.017 [-0.049, +0.015]
- stratum <= 12 words (n=1439): probe 0.229, native_lr 0.261, embed 0.114, tfidf 0.155, fusion (post hoc) 0.266
- stratum > 12 words (n=754): probe 0.151, native_lr 0.145, embed 0.149, tfidf 0.145, fusion (post hoc) 0.171

