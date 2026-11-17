# Robustness (logreg @ layer 16, cfg {'var_frac': 0.5, 'C': 0.003})

## Leave-one-writing-style-out

| held-out style | n test | test F1 | hand-written F1 (this fold's model) |
|---|---|---|---|
| dry, understated first-person account | 113 | 0.687 | 0.640 |
| informal social-media comment (reddit) | 160 | 0.261 | 0.564 |
| plain report-style account where the feeling shows only in what the person decides or does | 114 | 0.753 | 0.636 |
| reflective paragraph in which the writer reasons about what to do next | 102 | 0.849 | 0.651 |
| short dialogue between two people (use quotation marks) | 109 | 0.836 | 0.625 |
| text-message exchange between two people | 110 | 0.744 | 0.611 |
| third-person narrative focused on one person's actions and choices | 108 | 0.902 | 0.625 |

Mean test F1 0.719, worst 0.261.

## Leave-one-writer-out

| held-out writer | n test | test F1 | hand-written F1 |
|---|---|---|---|
| llama-3.2-3b | 297 | 0.764 | 0.628 |
| qwen3.8-27b | 359 | 0.644 | 0.640 |
| reddit-goemotions | 160 | 0.261 | 0.564 |

## Learning curve (stratified subsamples of TRAIN)

| fraction | n train | test-hard F1 | hand-written F1 |
|---|---|---|---|
| 0.1 | 177 | 0.513 ± 0.051 | 0.509 ± 0.066 |
| 0.2 | 354 | 0.616 ± 0.045 | 0.571 ± 0.022 |
| 0.4 | 708 | 0.673 ± 0.023 | 0.646 ± 0.030 |
| 0.7 | 1240 | 0.749 ± 0.029 | 0.614 ± 0.013 |
| 1.0 | 1771 | 0.781 ± 0.000 | 0.628 ± 0.000 |

Gain from 40% -> 100% of the data: +0.108 test F1. Still climbing: more/varied passages should help.
