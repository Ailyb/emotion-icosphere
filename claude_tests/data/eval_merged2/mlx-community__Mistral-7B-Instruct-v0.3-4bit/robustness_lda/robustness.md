# Robustness (lda @ layer 10, cfg {'var_frac': 0.5, 'shrinkage': 'auto'})

## Leave-one-writing-style-out

| held-out style | n test | test F1 | hand-written F1 (this fold's model) |
|---|---|---|---|
| dry, understated first-person account | 113 | 0.714 | 0.698 |
| informal social-media comment (reddit) | 160 | 0.277 | 0.478 |
| plain report-style account where the feeling shows only in what the person decides or does | 114 | 0.740 | 0.674 |
| reflective paragraph in which the writer reasons about what to do next | 102 | 0.804 | 0.667 |
| short dialogue between two people (use quotation marks) | 109 | 0.813 | 0.667 |
| text-message exchange between two people | 110 | 0.783 | 0.649 |
| third-person narrative focused on one person's actions and choices | 108 | 0.844 | 0.682 |

Mean test F1 0.711, worst 0.277.

## Leave-one-writer-out

| held-out writer | n test | test F1 | hand-written F1 |
|---|---|---|---|
| llama-3.2-3b | 297 | 0.719 | 0.636 |
| qwen3.8-27b | 359 | 0.634 | 0.720 |
| reddit-goemotions | 160 | 0.277 | 0.478 |

## Learning curve (stratified subsamples of TRAIN)

| fraction | n train | test-hard F1 | hand-written F1 |
|---|---|---|---|
| 0.1 | 177 | 0.543 ± 0.045 | 0.513 ± 0.041 |
| 0.2 | 354 | 0.607 ± 0.018 | 0.580 ± 0.055 |
| 0.4 | 708 | 0.689 ± 0.021 | 0.654 ± 0.029 |
| 0.7 | 1240 | 0.717 ± 0.024 | 0.655 ± 0.012 |
| 1.0 | 1771 | 0.707 ± 0.000 | 0.665 ± 0.000 |

Gain from 40% -> 100% of the data: +0.018 test F1. Flattening: more of the same data won't help much; vary the data instead.
