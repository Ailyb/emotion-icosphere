# H5 repeated with the HUM-trained probe (POST HOC, not pre-registered; mlx-community/Llama-3.2-3B-Instruct, layer 16, lda; training n = 4192)

tau (LLM-written VAL-hard) = 2.01; tau_h (90th percentile of max z, human neutral reserve) = 4.33.

- Held-out human neutral recall (odd half, n check): original tau 0.439 -> tau_h 0.900.
- Frozen set (n = 840) neutral recall: 0.490 -> 0.993.
- Frozen 8-way macro-F1: 0.587 -> 0.416; difference -0.171 [-0.209, -0.136].
- Share of emotional frozen items predicted as an emotion under tau that become neutral under tau_h: 0.624.

| class | recall at tau | recall at tau_h | change |
|---|---|---|---|
| anger | 0.523 | 0.242 | -0.281 |
| disgust | 0.514 | 0.324 | -0.189 |
| fear | 0.698 | 0.397 | -0.302 |
| joy | 0.670 | 0.297 | -0.374 |
| love | 0.662 | 0.265 | -0.397 |
| neutral | 0.490 | 0.993 | +0.503 |
| sadness | 0.500 | 0.204 | -0.296 |
| surprise | 0.631 | 0.238 | -0.393 |
