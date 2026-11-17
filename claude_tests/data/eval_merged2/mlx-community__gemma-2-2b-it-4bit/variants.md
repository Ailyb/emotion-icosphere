# T9 pooling / prompting variants (layer 14; trained on HUM; pre-registered in PLAN_v2)

Macro-F1 per set (predictions restricted to the source's classes); paired differences vs V0 are [95% CI]; * = excludes 0.

| variant | goemotions (n=840) | dair (n=1999) | isear (n=1500) | dailydialog (n=2193) |
|---|---|---|---|---|
| V0 mean | 0.634 | 0.406 | 0.595 | 0.434 |
| V1 last | 0.497 | 0.315 | 0.545 | 0.385 |
| V2 max | 0.618 | 0.399 | 0.602 | 0.418 |
| V3 last4 | 0.525 | 0.333 | 0.559 | 0.367 |
| V4 prompted | 0.611 | 0.368 | 0.566 | 0.415 |
| V5 concat | 0.633 | 0.407 | 0.601 | 0.419 |

| variant - V0 | goemotions | dair | isear | dailydialog | sets with CI > 0 (claim needs >= 3) |
|---|---|---|---|---|---|
| V1 last | -0.136 [-0.173, -0.101]* | -0.091 [-0.118, -0.064]* | -0.050 [-0.077, -0.023]* | -0.049 [-0.083, -0.014]* | 0/4 |
| V2 max | -0.016 [-0.042, +0.010] | -0.007 [-0.029, +0.015] | +0.007 [-0.015, +0.029] | -0.016 [-0.045, +0.014] | 0/4 |
| V3 last4 | -0.109 [-0.142, -0.075]* | -0.072 [-0.096, -0.048]* | -0.036 [-0.063, -0.009]* | -0.066 [-0.098, -0.035]* | 0/4 |
| V4 prompted | -0.023 [-0.056, +0.011] | -0.037 [-0.065, -0.009]* | -0.029 [-0.057, +0.000] | -0.019 [-0.055, +0.016] | 0/4 |
| V5 concat | -0.001 [-0.020, +0.016] | +0.002 [-0.012, +0.015] | +0.006 [-0.009, +0.020] | -0.015 [-0.034, +0.003] | 0/4 |

Pre-registered reading: no variant improves on mean-pooling on >= 3 of 4 sets; the representation is not limited by pooling or prompt framing at this layer.
