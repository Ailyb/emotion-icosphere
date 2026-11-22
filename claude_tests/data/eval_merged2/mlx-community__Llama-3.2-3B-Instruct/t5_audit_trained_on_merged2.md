# Probe audit (layer 16, lda; trained on n=1654, validated on n=225; classes ['anger', 'disgust', 'fear', 'joy', 'love', 'neutral', 'sadness', 'surprise'])

Train classes (n): {'anger': 211, 'disgust': 210, 'fear': 206, 'joy': 193, 'love': 208, 'neutral': 212, 'sadness': 209, 'surprise': 205}; excluded by request: ['calm']. tau (neutral threshold) = 1.57, chosen on VAL.

Cache provenance: NO META SIDECAR (provenance unknown)

## t5_test: n=320, evaluated classes ['anger', 'disgust', 'fear', 'joy', 'love', 'neutral', 'sadness', 'surprise']; chance 0.125
Per-class n: anger 40, disgust 40, fear 40, joy 40, love 40, neutral 40, sadness 40, surprise 40

| system | macro-F1 [95% CI] | accuracy | shuffle-null p95 |
|---|---|---|---|
| probe | 0.544 [0.487, 0.594] | 0.528 | 0.126 |
| native_lr | 0.617 [0.562, 0.662] | 0.613 | 0.134 |
| embed | 0.392 [0.340, 0.439] | 0.416 | 0.122 |
| tfidf | 0.395 [0.340, 0.442] | 0.403 | 0.145 |
| fusion (post hoc) | 0.622 [0.570, 0.667] | 0.616 | 0.133 |

Paired differences (macro-F1, mean [95% CI]); * = excludes 0:

- probe - tfidf: +0.149 [+0.088, +0.210]*
- probe - embed: +0.151 [+0.088, +0.215]*
- probe - native_lr: -0.073 [-0.112, -0.036]*
- stratum <= 12 words (n=0): n < 20, not reported
- stratum > 12 words (n=320): probe 0.544, native_lr 0.617, embed 0.392, tfidf 0.395, fusion (post hoc) 0.622


## P3 paired: T5-trained minus merged2-trained, on T5-TEST (n = 320)

- probe: +0.319 [+0.255, +0.382]*
- native_lr: +0.339 [+0.287, +0.390]*
- tfidf: +0.539 [+0.481, +0.595]*
- embed: +0.501 [+0.446, +0.555]*
