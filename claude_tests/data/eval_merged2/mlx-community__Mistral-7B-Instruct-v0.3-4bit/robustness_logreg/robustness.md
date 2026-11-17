# Robustness (logreg @ layer 10, cfg {'var_frac': 0.5, 'C': 0.003})

## Leave-one-writing-style-out

| held-out style | n test | test F1 | hand-written F1 (this fold's model) |
|---|---|---|---|
| dry, understated first-person account | 113 | 0.708 | 0.622 |
| informal social-media comment (reddit) | 160 | 0.211 | 0.517 |
| plain report-style account where the feeling shows only in what the person decides or does | 114 | 0.782 | 0.633 |
| reflective paragraph in which the writer reasons about what to do next | 102 | 0.783 | 0.627 |
| short dialogue between two people (use quotation marks) | 109 | 0.814 | 0.619 |
| text-message exchange between two people | 110 | 0.817 | 0.601 |
| third-person narrative focused on one person's actions and choices | 108 | 0.833 | 0.587 |

Mean test F1 0.707, worst 0.211.

## Leave-one-writer-out

| held-out writer | n test | test F1 | hand-written F1 |
|---|---|---|---|
| llama-3.2-3b | 297 | 0.680 | 0.557 |
| qwen3.8-27b | 359 | 0.608 | 0.692 |
| reddit-goemotions | 160 | 0.211 | 0.517 |

## Learning curve (stratified subsamples of TRAIN)

| fraction | n train | test-hard F1 | hand-written F1 |
|---|---|---|---|
| 0.1 | 177 | 0.533 ± 0.051 | 0.501 ± 0.027 |
| 0.2 | 354 | 0.602 ± 0.013 | 0.564 ± 0.040 |
| 0.4 | 708 | 0.676 ± 0.031 | 0.634 ± 0.018 |
| 0.7 | 1240 | 0.693 ± 0.040 | 0.595 ± 0.015 |
| 1.0 | 1771 | 0.712 ± 0.000 | 0.621 ± 0.000 |

Gain from 40% -> 100% of the data: +0.036 test F1. Still climbing: more/varied passages should help.
