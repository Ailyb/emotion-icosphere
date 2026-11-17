# Robustness (logreg @ layer 14, cfg {'var_frac': 0.5, 'C': 0.003})

## Leave-one-writing-style-out

| held-out style | n test | test F1 | hand-written F1 (this fold's model) |
|---|---|---|---|
| dry, understated first-person account | 113 | 0.649 | 0.649 |
| informal social-media comment (reddit) | 160 | 0.250 | 0.559 |
| plain report-style account where the feeling shows only in what the person decides or does | 114 | 0.776 | 0.688 |
| reflective paragraph in which the writer reasons about what to do next | 102 | 0.801 | 0.604 |
| short dialogue between two people (use quotation marks) | 109 | 0.784 | 0.609 |
| text-message exchange between two people | 110 | 0.757 | 0.618 |
| third-person narrative focused on one person's actions and choices | 108 | 0.873 | 0.658 |

Mean test F1 0.698, worst 0.250.

## Leave-one-writer-out

| held-out writer | n test | test F1 | hand-written F1 |
|---|---|---|---|
| llama-3.2-3b | 297 | 0.743 | 0.709 |
| qwen3.8-27b | 359 | 0.688 | 0.612 |
| reddit-goemotions | 160 | 0.250 | 0.559 |

## Learning curve (stratified subsamples of TRAIN)

| fraction | n train | test-hard F1 | hand-written F1 |
|---|---|---|---|
| 0.1 | 177 | 0.572 ± 0.041 | 0.506 ± 0.041 |
| 0.2 | 354 | 0.643 ± 0.005 | 0.591 ± 0.051 |
| 0.4 | 708 | 0.718 ± 0.015 | 0.638 ± 0.035 |
| 0.7 | 1240 | 0.729 ± 0.016 | 0.612 ± 0.029 |
| 1.0 | 1771 | 0.743 ± 0.000 | 0.633 ± 0.000 |

Gain from 40% -> 100% of the data: +0.025 test F1. Flattening: more of the same data won't help much; vary the data instead.
