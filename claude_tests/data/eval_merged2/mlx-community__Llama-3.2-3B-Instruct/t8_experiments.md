# T8 cached-state experiments (mlx-community/Llama-3.2-3B-Instruct, layer 16; pre-registered in PLAN_v2)

Trained on HUM (n=4192); hierarchy C1=0.003, C2=0.003; MLP alpha=0.01. Paired macro-F1 differences [95% CI]; * = excludes 0.

| set (n, classes) | flat LR | hierarchical | MLP | hierarchical - flat (a) | MLP - flat (e) |
|---|---|---|---|---|---|
| goemotions (frozen) (840, 8) | 0.641 | 0.631 | 0.609 | -0.009 [-0.026, +0.006] | -0.031 [-0.058, -0.005]* |
| dair (1999, 6) | 0.405 | 0.404 | 0.411 | -0.000 [-0.012, +0.012] | +0.007 [-0.012, +0.027] |
| isear (1500, 5) | 0.575 | 0.593 | 0.567 | +0.019 [+0.004, +0.033]* | -0.008 [-0.030, +0.014] |
| dailydialog (2193, 7) | 0.410 | 0.412 | 0.396 | +0.002 [-0.011, +0.015] | -0.014 [-0.033, +0.006] |

Pre-registered tally: hierarchical > flat on 1/4 sets (claim needs >= 3); MLP > flat on 0/4 sets (claim needs >= 3).
Reading: no evidence that hierarchy helps; the linear read-out is sufficient at this layer.

## (b) Probe-direction dissimilarity vs NRC VAD distance (7 emotions, 21 pairs; exact permutation over 5040 labelings)

| estimator | Spearman rho | one-sided permutation p |
|---|---|---|
| lda | +0.53 | 0.016 |
| meandiff | +0.60 | 0.001 |

Claim rule: only if p < 0.05 for both estimators. Seven points have low power; a non-significant result does not show the geometries differ.
