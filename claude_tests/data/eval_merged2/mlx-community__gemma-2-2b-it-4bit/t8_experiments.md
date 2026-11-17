# T8 cached-state experiments (mlx-community/gemma-2-2b-it-4bit, layer 14; pre-registered in PLAN_v2)

Trained on HUM (n=4192); hierarchy C1=0.003, C2=0.003; MLP alpha=0.1. Paired macro-F1 differences [95% CI]; * = excludes 0.

| set (n, classes) | flat LR | hierarchical | MLP | hierarchical - flat (a) | MLP - flat (e) |
|---|---|---|---|---|---|
| goemotions (frozen) (840, 8) | 0.634 | 0.628 | 0.626 | -0.006 [-0.022, +0.011] | -0.008 [-0.030, +0.013] |
| dair (1999, 6) | 0.406 | 0.402 | 0.404 | -0.003 [-0.021, +0.013] | -0.002 [-0.021, +0.018] |
| isear (1500, 5) | 0.595 | 0.612 | 0.595 | +0.017 [+0.004, +0.030]* | +0.000 [-0.018, +0.017] |
| dailydialog (2193, 7) | 0.434 | 0.440 | 0.430 | +0.006 [-0.009, +0.022] | -0.003 [-0.024, +0.017] |

Pre-registered tally: hierarchical > flat on 1/4 sets (claim needs >= 3); MLP > flat on 0/4 sets (claim needs >= 3).
Reading: no evidence that hierarchy helps; the linear read-out is sufficient at this layer.

## (b) Probe-direction dissimilarity vs NRC VAD distance (7 emotions, 21 pairs; exact permutation over 5040 labelings)

| estimator | Spearman rho | one-sided permutation p |
|---|---|---|
| lda | +0.61 | 0.003 |
| meandiff | +0.63 | 0.001 |

Claim rule: only if p < 0.05 for both estimators. Seven points have low power; a non-significant result does not show the geometries differ.
