# Robustness (meandiff @ layer 16, cfg {'var_frac': 0.5})

## Leave-one-writing-style-out

| held-out style | n test | test F1 | hand-written F1 (this fold's model) |
|---|---|---|---|
| dry, understated first-person account | 113 | 0.696 | 0.644 |
| informal social-media comment (reddit) | 160 | 0.238 | 0.531 |
| plain report-style account where the feeling shows only in what the person decides or does | 114 | 0.684 | 0.592 |
| reflective paragraph in which the writer reasons about what to do next | 102 | 0.766 | 0.639 |
| short dialogue between two people (use quotation marks) | 109 | 0.777 | 0.593 |
| text-message exchange between two people | 110 | 0.689 | 0.611 |
| third-person narrative focused on one person's actions and choices | 108 | 0.679 | 0.628 |

Mean test F1 0.647, worst 0.238.

## Leave-one-writer-out

| held-out writer | n test | test F1 | hand-written F1 |
|---|---|---|---|
| llama-3.2-3b | 297 | 0.661 | 0.620 |
| qwen3.8-27b | 359 | 0.590 | 0.600 |
| reddit-goemotions | 160 | 0.238 | 0.531 |

## Learning curve (stratified subsamples of TRAIN)

| fraction | n train | test-hard F1 | hand-written F1 |
|---|---|---|---|
| 0.1 | 177 | 0.468 ± 0.025 | 0.469 ± 0.039 |
| 0.2 | 354 | 0.560 ± 0.050 | 0.576 ± 0.019 |
| 0.4 | 708 | 0.616 ± 0.023 | 0.606 ± 0.016 |
| 0.7 | 1240 | 0.645 ± 0.013 | 0.616 ± 0.024 |
| 1.0 | 1771 | 0.676 ± 0.000 | 0.596 ± 0.000 |

Gain from 40% -> 100% of the data: +0.060 test F1. Still climbing: more/varied passages should help.
