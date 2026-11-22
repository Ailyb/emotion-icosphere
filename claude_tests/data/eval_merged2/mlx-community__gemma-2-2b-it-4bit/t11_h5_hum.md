# H5 repeated with the HUM-trained probe (POST HOC, not pre-registered; mlx-community/gemma-2-2b-it-4bit, layer 14, lda; training n = 4192)

tau (LLM-written VAL-hard) = 1.74; tau_h (90th percentile of max z, human neutral reserve) = 4.42.

- Held-out human neutral recall (odd half, n check): original tau 0.291 -> tau_h 0.901.
- Frozen set (n = 840) neutral recall: 0.272 -> 0.934.
- Frozen 8-way macro-F1: 0.568 -> 0.406; difference -0.163 [-0.202, -0.123].
- Share of emotional frozen items predicted as an emotion under tau that become neutral under tau_h: 0.642.

| class | recall at tau | recall at tau_h | change |
|---|---|---|---|
| anger | 0.539 | 0.219 | -0.320 |
| disgust | 0.541 | 0.243 | -0.297 |
| fear | 0.746 | 0.317 | -0.429 |
| joy | 0.670 | 0.352 | -0.319 |
| love | 0.755 | 0.470 | -0.285 |
| neutral | 0.272 | 0.934 | +0.662 |
| sadness | 0.541 | 0.163 | -0.378 |
| surprise | 0.583 | 0.179 | -0.405 |
