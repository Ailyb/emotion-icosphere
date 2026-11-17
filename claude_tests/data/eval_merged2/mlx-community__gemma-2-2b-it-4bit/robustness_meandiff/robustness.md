# Robustness (meandiff @ layer 14, cfg {'var_frac': 0.5})

## Leave-one-writing-style-out

| held-out style | n test | test F1 | hand-written F1 (this fold's model) |
|---|---|---|---|
| dry, understated first-person account | 113 | 0.573 | 0.541 |
| informal social-media comment (reddit) | 160 | 0.211 | 0.480 |
| plain report-style account where the feeling shows only in what the person decides or does | 114 | 0.697 | 0.540 |
| reflective paragraph in which the writer reasons about what to do next | 102 | 0.704 | 0.539 |
| short dialogue between two people (use quotation marks) | 109 | 0.750 | 0.588 |
| text-message exchange between two people | 110 | 0.678 | 0.572 |
| third-person narrative focused on one person's actions and choices | 108 | 0.733 | 0.555 |

Mean test F1 0.621, worst 0.211.

## Leave-one-writer-out

| held-out writer | n test | test F1 | hand-written F1 |
|---|---|---|---|
| llama-3.2-3b | 297 | 0.679 | 0.569 |
| qwen3.8-27b | 359 | 0.605 | 0.571 |
| reddit-goemotions | 160 | 0.211 | 0.480 |

## Learning curve (stratified subsamples of TRAIN)

| fraction | n train | test-hard F1 | hand-written F1 |
|---|---|---|---|
| 0.1 | 177 | 0.461 ± 0.036 | 0.457 ± 0.044 |
| 0.2 | 354 | 0.576 ± 0.027 | 0.486 ± 0.042 |
| 0.4 | 708 | 0.633 ± 0.023 | 0.570 ± 0.051 |
| 0.7 | 1240 | 0.631 ± 0.034 | 0.572 ± 0.016 |
| 1.0 | 1771 | 0.613 ± 0.000 | 0.546 ± 0.000 |

Gain from 40% -> 100% of the data: -0.020 test F1. Flattening: more of the same data won't help much; vary the data instead.
