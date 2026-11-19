# T11 post hoc analyses (mlx-community/Mistral-7B-Instruct-v0.3-4bit; layer 10, logreg; NOT pre-registered; requested in docs/REVIEW_v3.md)

## A5: trained classifier on the same hidden states (native LR) vs the frozen sentence encoder (HUM training; paired bootstrap)

| set | n | native LR | encoder | native LR - encoder | probe - native LR | probe - bag of embeddings |
|---|---|---|---|---|---|---|
| frozen | 840 | 0.653 [0.618, 0.683] | 0.563 [0.527, 0.594] | +0.091 [+0.053, +0.127]* | -0.106 [-0.133, -0.080]* | +0.090 [+0.048, +0.132]* |
| dair | 1999 | 0.408 [0.379, 0.434] | 0.402 [0.376, 0.429] | +0.006 [-0.024, +0.037] | -0.001 [-0.022, +0.020] | +0.101 [+0.072, +0.131]* |
| isear | 1500 | 0.596 [0.573, 0.619] | 0.656 [0.633, 0.679] | -0.061 [-0.088, -0.034]* | -0.035 [-0.052, -0.017]* | +0.163 [+0.133, +0.194]* |
| dailydialog | 2193 | 0.399 [0.369, 0.427] | 0.381 [0.349, 0.412] | +0.018 [-0.012, +0.048] | -0.105 [-0.127, -0.083]* | +0.074 [+0.052, +0.097]* |

## B2: per-class recall on the frozen set, original tau vs human-neutral tau_h (H5 model; post hoc breakdown)

tau 1.42 -> tau_h 4.42

| class | n | recall (tau) | recall (tau_h) | change |
|---|---|---|---|---|
| anger | 128 | 0.51 | 0.21 | -0.30 |
| disgust | 74 | 0.59 | 0.27 | -0.32 |
| fear | 63 | 0.71 | 0.29 | -0.43 |
| joy | 91 | 0.64 | 0.22 | -0.42 |
| love | 151 | 0.83 | 0.54 | -0.30 |
| neutral | 151 | 0.13 | 0.91 | +0.79 |
| sadness | 98 | 0.36 | 0.12 | -0.23 |
| surprise | 84 | 0.64 | 0.23 | -0.42 |

Share of emotional (non-neutral) test items whose prediction changed from an emotion to neutral when tau became tau_h: 0.599.
