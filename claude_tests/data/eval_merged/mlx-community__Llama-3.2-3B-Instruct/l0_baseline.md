# T1 (i): layer-0 baselines vs the selected probe (lda @ layer 14)

Macro-F1 (9-way unless a class is absent from the set). `embed` = bag of input embeddings; `block0` = after decoder block 0. Differences are paired bootstrap on the same items (B=2000), 95% interval.

| system | VAL-hard | TEST-hard | TEST-all | hand-written |
|---|---|---|---|---|
| tfidf | 0.228 | 0.279 | 0.596 | 0.221 |
| embed | 0.356 | 0.404 | 0.534 | 0.270 |
| block0 | 0.422 | 0.459 | 0.586 | 0.265 |
| native_lr_L | 0.644 | 0.669 | 0.734 | 0.680 |
| probe | 0.642 | 0.616 | 0.670 | 0.723 |

n per set: VAL-hard=153, TEST-hard=153, TEST-all=307, hand-written=92
hand-written accuracy: tfidf 25/92, embed 32/92, block0 33/92, native_lr_L 64/92, probe 67/92

## Paired differences (macro-F1, mean [95% interval]); * = interval excludes 0

| comparison | VAL-hard | TEST-hard | TEST-all | hand-written |
|---|---|---|---|---|
| TF-IDF - embed | -0.125 [-0.205, -0.042]* | -0.123 [-0.211, -0.038]* | +0.061 [-0.009, +0.128] | -0.047 [-0.143, +0.048] |
| embed - block0 | -0.066 [-0.149, +0.016] | -0.055 [-0.130, +0.019] | -0.053 [-0.106, -0.002]* | +0.005 [-0.079, +0.087] |
| probe - embed (the T1 test) | +0.285 [+0.185, +0.383]* | +0.210 [+0.114, +0.305]* | +0.137 [+0.067, +0.206]* | +0.451 [+0.331, +0.572]* |
| probe - block0 | +0.220 [+0.118, +0.313]* | +0.156 [+0.062, +0.248]* | +0.085 [+0.022, +0.149]* | +0.456 [+0.333, +0.578]* |
| probe - native LR at probe layer | -0.004 [-0.075, +0.064] | -0.055 [-0.117, +0.005] | -0.065 [-0.105, -0.026]* | +0.044 [-0.045, +0.129] |
