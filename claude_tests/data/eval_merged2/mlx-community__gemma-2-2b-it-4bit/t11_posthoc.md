# T11 post hoc analyses (mlx-community/gemma-2-2b-it-4bit; layer 14, lda; NOT pre-registered; requested in docs/REVIEW_v3.md)

## A5: trained classifier on the same hidden states (native LR) vs the frozen sentence encoder (HUM training; paired bootstrap)

| set | n | native LR | encoder | native LR - encoder | probe - native LR | probe - bag of embeddings |
|---|---|---|---|---|---|---|
| frozen | 840 | 0.634 [0.598, 0.666] | 0.563 [0.527, 0.594] | +0.071 [+0.035, +0.106]* | -0.066 [-0.095, -0.037]* | +0.080 [+0.038, +0.122]* |
| dair | 1999 | 0.406 [0.379, 0.432] | 0.402 [0.376, 0.429] | +0.004 [-0.024, +0.032] | -0.014 [-0.035, +0.006] | +0.062 [+0.036, +0.089]* |
| isear | 1500 | 0.595 [0.569, 0.618] | 0.656 [0.633, 0.679] | -0.061 [-0.087, -0.035]* | +0.029 [+0.008, +0.050]* | +0.249 [+0.219, +0.278]* |
| dailydialog | 2193 | 0.434 [0.400, 0.465] | 0.381 [0.349, 0.412] | +0.053 [+0.018, +0.089]* | -0.098 [-0.122, -0.076]* | +0.105 [+0.079, +0.131]* |

## B2: per-class recall on the frozen set, original tau vs human-neutral tau_h (H5 model; post hoc breakdown)

tau 2.03 -> tau_h 5.43

| class | n | recall (tau) | recall (tau_h) | change |
|---|---|---|---|---|
| anger | 128 | 0.42 | 0.12 | -0.30 |
| disgust | 74 | 0.50 | 0.18 | -0.32 |
| fear | 63 | 0.71 | 0.32 | -0.40 |
| joy | 91 | 0.48 | 0.22 | -0.26 |
| love | 151 | 0.78 | 0.42 | -0.36 |
| neutral | 151 | 0.29 | 0.95 | +0.66 |
| sadness | 98 | 0.44 | 0.10 | -0.34 |
| surprise | 84 | 0.58 | 0.21 | -0.37 |

Share of emotional (non-neutral) test items whose prediction changed from an emotion to neutral when tau became tau_h: 0.588.
