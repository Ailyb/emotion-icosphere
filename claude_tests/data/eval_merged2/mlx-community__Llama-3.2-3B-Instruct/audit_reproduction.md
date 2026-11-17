# Probe audit (layer 16, lda; trained on n=1654, validated on n=225; classes ['anger', 'disgust', 'fear', 'joy', 'love', 'neutral', 'sadness', 'surprise'])

Train classes (n): {'anger': 211, 'disgust': 210, 'fear': 206, 'joy': 193, 'love': 208, 'neutral': 212, 'sadness': 209, 'surprise': 205}; excluded by request: ['calm']. tau (neutral threshold) = 1.57, chosen on VAL.

Cache provenance: NO META SIDECAR (provenance unknown)

## goemotions: n=840, evaluated classes ['anger', 'disgust', 'fear', 'joy', 'love', 'neutral', 'sadness', 'surprise']; chance 0.125
Per-class n: anger 128, disgust 74, fear 63, joy 91, love 151, neutral 151, sadness 98, surprise 84

| system | macro-F1 [95% CI] | accuracy | shuffle-null p95 |
|---|---|---|---|
| probe | 0.510 [0.476, 0.544] | 0.521 | 0.140 |
| native_lr | 0.521 [0.486, 0.554] | 0.531 | 0.144 |
| embed | 0.403 [0.373, 0.435] | 0.401 | 0.145 |
| tfidf | 0.599 [0.564, 0.630] | 0.599 | 0.144 |
| fusion (post hoc) | 0.617 [0.582, 0.650] | 0.625 | 0.143 |

Paired differences (macro-F1, mean [95% CI]); * = excludes 0:

- probe - tfidf: -0.088 [-0.128, -0.046]*
- probe - embed: +0.107 [+0.065, +0.148]*
- probe - native_lr: -0.010 [-0.036, +0.016]
- stratum <= 12 words (n=431): probe 0.541, native_lr 0.553, embed 0.378, tfidf 0.572, fusion (post hoc) 0.644
- stratum > 12 words (n=409): probe 0.475, native_lr 0.485, embed 0.428, tfidf 0.623, fusion (post hoc) 0.586

