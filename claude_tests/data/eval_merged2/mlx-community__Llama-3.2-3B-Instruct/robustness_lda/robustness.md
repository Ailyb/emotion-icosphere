# Robustness (lda @ layer 16, cfg {'var_frac': 0.5, 'shrinkage': 'auto'})

## Leave-one-writing-style-out

| held-out style | n test | test F1 | hand-written F1 (this fold's model) |
|---|---|---|---|
| dry, understated first-person account | 113 | 0.780 | 0.693 |
| informal social-media comment (reddit) | 160 | 0.237 | 0.577 |
| plain report-style account where the feeling shows only in what the person decides or does | 114 | 0.772 | 0.662 |
| reflective paragraph in which the writer reasons about what to do next | 102 | 0.838 | 0.670 |
| short dialogue between two people (use quotation marks) | 109 | 0.854 | 0.673 |
| text-message exchange between two people | 110 | 0.759 | 0.655 |
| third-person narrative focused on one person's actions and choices | 108 | 0.893 | 0.679 |

Mean test F1 0.734, worst 0.237.

## Leave-one-writer-out

| held-out writer | n test | test F1 | hand-written F1 |
|---|---|---|---|
| llama-3.2-3b | 297 | 0.747 | 0.644 |
| qwen3.8-27b | 359 | 0.658 | 0.682 |
| reddit-goemotions | 160 | 0.237 | 0.577 |

## Learning curve (stratified subsamples of TRAIN)

| fraction | n train | test-hard F1 | hand-written F1 |
|---|---|---|---|
| 0.1 | 177 | 0.531 ± 0.047 | 0.523 ± 0.071 |
| 0.2 | 354 | 0.623 ± 0.029 | 0.599 ± 0.043 |
| 0.4 | 708 | 0.729 ± 0.057 | 0.677 ± 0.024 |
| 0.7 | 1240 | 0.770 ± 0.023 | 0.667 ± 0.013 |
| 1.0 | 1771 | 0.807 ± 0.000 | 0.665 ± 0.000 |

Gain from 40% -> 100% of the data: +0.078 test F1. Still climbing: more/varied passages should help.
