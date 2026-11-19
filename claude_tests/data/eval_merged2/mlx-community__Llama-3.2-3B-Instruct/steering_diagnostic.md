# Steering diagnostic: effect per unit norm (POST HOC; not pre-registered; REVIEW_v3 A3.3)

Llama-3.2-3B, layer 16, mean-pooled states of merged2 TRAIN rows (n=1654, calm excluded). m_e = mean(e) - mean(neutral), unit length. cos = cosine of the unit steering direction with m_e (shift of the projection onto m_e per unit norm); sd = standard deviation of the pooled states along the direction; shift/sd = cos / sd. Random = mean over 20 random unit directions.

| emotion | cos probe | cos v1 | cos random | sd probe | sd v1 | sd random | shift/sd probe | shift/sd v1 | shift/sd random |
|---|---|---|---|---|---|---|---|---|---|
| joy | +0.296 | +0.285 | +0.001 | 0.23 | 0.57 | 0.10 | +1.288 | +0.502 | +0.007 |
| sadness | +0.230 | +0.422 | +0.001 | 0.20 | 0.68 | 0.10 | +1.177 | +0.622 | +0.011 |
| fear | +0.287 | +0.356 | +0.004 | 0.22 | 0.59 | 0.10 | +1.314 | +0.604 | +0.039 |
| anger | +0.259 | +0.256 | +0.003 | 0.20 | 0.76 | 0.10 | +1.287 | +0.338 | +0.029 |
| surprise | +0.258 | +0.043 | +0.004 | 0.19 | 0.58 | 0.10 | +1.338 | +0.075 | +0.039 |
| disgust | +0.244 | +0.239 | +0.001 | 0.21 | 0.46 | 0.10 | +1.143 | +0.517 | +0.008 |
| love | +0.318 | +0.320 | +0.002 | 0.23 | 0.52 | 0.10 | +1.373 | +0.613 | +0.018 |

Means over emotions: cos probe +0.270, v1 +0.274, random +0.002; sd probe 0.21, v1 0.59, random 0.10.
