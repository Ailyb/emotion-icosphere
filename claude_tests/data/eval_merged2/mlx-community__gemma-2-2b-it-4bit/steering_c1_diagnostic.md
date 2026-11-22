# Steering diagnostic: effect per unit norm (POST HOC; not pre-registered; REVIEW_v3 A3.3)

mlx-community/gemma-2-2b-it-4bit, layer 14, mean-pooled states of merged2 TRAIN rows (n=1654, calm excluded). m_e = mean(e) - mean(neutral), unit length. cos = cosine of the unit steering direction with m_e (shift of the projection onto m_e per unit norm); sd = standard deviation of the pooled states along the direction; shift/sd = cos / sd. Random = mean over 20 random unit directions.

| emotion | cos probe | cos v1 | cos random | sd probe | sd v1 | sd random | shift/sd probe | shift/sd v1 | shift/sd random |
|---|---|---|---|---|---|---|---|---|---|
| joy | +0.280 | +0.244 | -0.003 | 2.59 | 7.49 | 1.27 | +0.108 | +0.033 | -0.001 |
| sadness | +0.233 | +0.389 | -0.003 | 2.36 | 7.17 | 1.27 | +0.099 | +0.054 | -0.002 |
| fear | +0.282 | +0.420 | -0.006 | 2.51 | 9.50 | 1.27 | +0.113 | +0.044 | -0.005 |
| anger | +0.235 | +0.282 | -0.001 | 2.22 | 10.01 | 1.27 | +0.106 | +0.028 | -0.001 |
| surprise | +0.254 | +0.123 | -0.003 | 2.29 | 7.08 | 1.27 | +0.111 | +0.017 | -0.002 |
| disgust | +0.258 | +0.229 | -0.005 | 2.55 | 6.08 | 1.27 | +0.101 | +0.038 | -0.004 |
| love | +0.319 | +0.342 | +0.001 | 2.69 | 6.44 | 1.27 | +0.119 | +0.053 | +0.001 |

Means over emotions: cos probe +0.266, v1 +0.290, random -0.003; sd probe 2.46, v1 7.68, random 1.27.
