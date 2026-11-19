# T11 post hoc analyses (mlx-community/Llama-3.2-3B-Instruct; layer 16, lda; NOT pre-registered; requested in docs/REVIEW_v3.md)

## A5: trained classifier on the same hidden states (native LR) vs the frozen sentence encoder (HUM training; paired bootstrap)

| set | n | native LR | encoder | native LR - encoder | probe - native LR | probe - bag of embeddings |
|---|---|---|---|---|---|---|
| frozen | 840 | 0.641 [0.606, 0.670] | 0.563 [0.527, 0.594] | +0.078 [+0.043, +0.112]* | -0.053 [-0.081, -0.026]* | +0.098 [+0.058, +0.139]* |
| dair | 1999 | 0.405 [0.377, 0.431] | 0.402 [0.376, 0.429] | +0.003 [-0.025, +0.030] | +0.000 [-0.020, +0.021] | +0.076 [+0.049, +0.103]* |
| isear | 1500 | 0.575 [0.548, 0.597] | 0.656 [0.633, 0.679] | -0.082 [-0.109, -0.054]* | -0.008 [-0.028, +0.011] | +0.186 [+0.155, +0.216]* |
| dailydialog | 2193 | 0.410 [0.381, 0.439] | 0.381 [0.349, 0.412] | +0.029 [-0.003, +0.061] | -0.067 [-0.089, -0.045]* | +0.109 [+0.083, +0.137]* |

## B2: per-class recall on the frozen set, original tau vs human-neutral tau_h (H5 model; post hoc breakdown)

tau 1.57 -> tau_h 4.77

| class | n | recall (tau) | recall (tau_h) | change |
|---|---|---|---|---|
| anger | 128 | 0.52 | 0.20 | -0.31 |
| disgust | 74 | 0.47 | 0.24 | -0.23 |
| fear | 63 | 0.63 | 0.32 | -0.32 |
| joy | 91 | 0.58 | 0.24 | -0.34 |
| love | 151 | 0.75 | 0.42 | -0.33 |
| neutral | 151 | 0.19 | 0.96 | +0.77 |
| sadness | 98 | 0.53 | 0.24 | -0.29 |
| surprise | 84 | 0.58 | 0.17 | -0.42 |

Share of emotional (non-neutral) test items whose prediction changed from an emotion to neutral when tau became tau_h: 0.607.
