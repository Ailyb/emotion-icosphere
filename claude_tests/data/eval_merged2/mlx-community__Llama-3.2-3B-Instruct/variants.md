# T9 pooling / prompting variants (layer 16; trained on HUM; pre-registered in PLAN_v2)

Macro-F1 per set (predictions restricted to the source's classes); paired differences vs V0 are [95% CI]; * = excludes 0.

| variant | goemotions (n=840) | dair (n=1999) | isear (n=1500) | dailydialog (n=2193) |
|---|---|---|---|---|
| V0 mean | 0.641 | 0.405 | 0.575 | 0.410 |
| V1 last | 0.489 | 0.319 | 0.579 | 0.428 |
| V2 max | 0.634 | 0.407 | 0.545 | 0.415 |
| V3 last4 | 0.526 | 0.335 | 0.531 | 0.386 |
| V4 prompted | 0.669 | 0.392 | 0.710 | 0.482 |
| V5 concat | 0.642 | 0.394 | 0.564 | 0.389 |

| variant - V0 | goemotions | dair | isear | dailydialog | sets with CI > 0 (claim needs >= 3) |
|---|---|---|---|---|---|
| V1 last | -0.151 [-0.187, -0.116]* | -0.085 [-0.114, -0.057]* | +0.005 [-0.022, +0.033] | +0.018 [-0.013, +0.048] | 0/4 |
| V2 max | -0.007 [-0.032, +0.020] | +0.003 [-0.018, +0.023] | -0.029 [-0.052, -0.007]* | +0.005 [-0.018, +0.028] | 0/4 |
| V3 last4 | -0.114 [-0.149, -0.078]* | -0.069 [-0.095, -0.045]* | -0.043 [-0.070, -0.017]* | -0.024 [-0.051, +0.003] | 0/4 |
| V4 prompted | +0.028 [-0.003, +0.058] | -0.013 [-0.042, +0.015] | +0.136 [+0.110, +0.164]* | +0.072 [+0.039, +0.106]* | 2/4 |
| V5 concat | +0.002 [-0.016, +0.020] | -0.011 [-0.023, +0.001] | -0.011 [-0.027, +0.004] | -0.021 [-0.035, -0.008]* | 0/4 |

Pre-registered reading: no variant improves on mean-pooling on >= 3 of 4 sets; the representation is not limited by pooling or prompt framing at this layer.
