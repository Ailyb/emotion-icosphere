# Robustness (logreg @ layer 14, cfg {'var_frac': 0.5, 'C': 0.03})

## Leave-one-writing-style-out

| held-out style | n test | test F1 | hand-written F1 (this fold's model) |
|---|---|---|---|
| dry, understated first-person account | 54 | 0.635 | 0.691 |
| informal social-media comment (reddit) | 320 | 0.247 | 0.595 |
| plain report-style account where the feeling shows only in what the person decides or does | 52 | 0.770 | 0.589 |
| reflective paragraph in which the writer reasons about what to do next | 42 | 0.698 | 0.590 |
| short dialogue between two people (use quotation marks) | 50 | 0.751 | 0.609 |
| text-message exchange between two people | 49 | 0.650 | 0.607 |
| third-person narrative focused on one person's actions and choices | 50 | 0.785 | 0.611 |

Mean test F1 0.648, worst 0.247.

## Leave-one-writer-out

| held-out writer | n test | test F1 | hand-written F1 |
|---|---|---|---|
| llama-3.2-3b * | 265 | 0.347 | 0.418 |
| reddit-goemotions | 320 | 0.247 | 0.595 |

(* = partial fold)
- `llama-3.2-3b`: partial: no training data for ['calm']; those rows excluded

## Learning curve (stratified subsamples of TRAIN)

| fraction | n train | test-hard F1 | hand-written F1 |
|---|---|---|---|
| 0.1 | 114 | 0.422 ± 0.061 | 0.402 ± 0.038 |
| 0.2 | 227 | 0.493 ± 0.030 | 0.526 ± 0.054 |
| 0.4 | 451 | 0.592 ± 0.037 | 0.608 ± 0.038 |
| 0.7 | 790 | 0.585 ± 0.029 | 0.597 ± 0.034 |
| 1.0 | 1128 | 0.602 ± 0.000 | 0.616 ± 0.000 |

Gain from 40% -> 100% of the data: +0.010 test F1. Flattening: more of the same data won't help much; vary the data instead.
