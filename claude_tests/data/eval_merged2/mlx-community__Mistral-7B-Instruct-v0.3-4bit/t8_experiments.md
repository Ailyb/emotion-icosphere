# T8 cached-state experiments (mlx-community/Mistral-7B-Instruct-v0.3-4bit, layer 10; pre-registered in PLAN_v2)

Trained on HUM (n=4192); hierarchy C1=0.003, C2=0.003; MLP alpha=0.1. Paired macro-F1 differences [95% CI]; * = excludes 0.

| set (n, classes) | flat LR | hierarchical | MLP | hierarchical - flat (a) | MLP - flat (e) |
|---|---|---|---|---|---|
| goemotions (frozen) (840, 8) | 0.653 | 0.648 | 0.633 | -0.006 [-0.024, +0.013] | -0.021 [-0.043, +0.001] |
| dair (1999, 6) | 0.408 | 0.398 | 0.399 | -0.009 [-0.027, +0.008] | -0.008 [-0.027, +0.012] |
| isear (1500, 5) | 0.596 | 0.596 | 0.593 | +0.001 [-0.015, +0.015] | -0.003 [-0.020, +0.015] |
| dailydialog (2193, 7) | 0.399 | 0.406 | 0.396 | +0.007 [-0.007, +0.023] | -0.004 [-0.024, +0.016] |

Pre-registered tally: hierarchical > flat on 0/4 sets (claim needs >= 3); MLP > flat on 0/4 sets (claim needs >= 3).
Reading: no evidence that hierarchy helps; the linear read-out is sufficient at this layer.

## (b) Probe-direction dissimilarity vs NRC VAD distance (7 emotions, 21 pairs; exact permutation over 5040 labelings)

| estimator | Spearman rho | one-sided permutation p |
|---|---|---|
| logreg | +0.31 | 0.112 |
| meandiff | +0.54 | 0.005 |

Claim rule: only if p < 0.05 for both estimators. Seven points have low power; a non-significant result does not show the geometries differ.
