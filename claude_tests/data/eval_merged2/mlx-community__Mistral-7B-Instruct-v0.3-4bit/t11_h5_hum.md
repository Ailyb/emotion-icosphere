# H5 repeated with the HUM-trained probe (POST HOC, not pre-registered; mlx-community/Mistral-7B-Instruct-v0.3-4bit, layer 10, logreg; training n = 4192)

tau (LLM-written VAL-hard) = 1.48; tau_h (90th percentile of max z, human neutral reserve) = 4.19.

- Held-out human neutral recall (odd half, n check): original tau 0.193 -> tau_h 0.902.
- Frozen set (n = 840) neutral recall: 0.159 -> 0.954.
- Frozen 8-way macro-F1: 0.549 -> 0.422; difference -0.127 [-0.167, -0.088].
- Share of emotional frozen items predicted as an emotion under tau that become neutral under tau_h: 0.637.

| class | recall at tau | recall at tau_h | change |
|---|---|---|---|
| anger | 0.461 | 0.133 | -0.328 |
| disgust | 0.662 | 0.270 | -0.392 |
| fear | 0.667 | 0.333 | -0.333 |
| joy | 0.681 | 0.253 | -0.429 |
| love | 0.781 | 0.510 | -0.272 |
| neutral | 0.159 | 0.954 | +0.795 |
| sadness | 0.582 | 0.214 | -0.367 |
| surprise | 0.702 | 0.357 | -0.345 |
