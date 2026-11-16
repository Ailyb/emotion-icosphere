# Robustness (meandiff @ layer 14, cfg {'var_frac': 0.5})

## Leave-one-writing-style-out

| held-out style | n test | test F1 | hand-written F1 (this fold's model) |
|---|---|---|---|
| dry, understated first-person account | 54 | 0.604 | 0.664 |
| informal social-media comment (reddit) | 320 | 0.284 | 0.493 |
| plain report-style account where the feeling shows only in what the person decides or does | 52 | 0.655 | 0.613 |
| reflective paragraph in which the writer reasons about what to do next | 42 | 0.643 | 0.618 |
| short dialogue between two people (use quotation marks) | 50 | 0.728 | 0.636 |
| text-message exchange between two people | 49 | 0.721 | 0.575 |
| third-person narrative focused on one person's actions and choices | 50 | 0.689 | 0.679 |

Mean test F1 0.618, worst 0.284.

## Leave-one-writer-out

| held-out writer | n test | test F1 | hand-written F1 |
|---|---|---|---|
| llama-3.2-3b * | 265 | 0.329 | 0.462 |
| reddit-goemotions | 320 | 0.284 | 0.493 |

(* = partial fold)
- `llama-3.2-3b`: partial: no training data for ['calm']; those rows excluded

## Learning curve (stratified subsamples of TRAIN)

| fraction | n train | test-hard F1 | hand-written F1 |
|---|---|---|---|
| 0.1 | 114 | 0.372 ± 0.060 | 0.342 ± 0.025 |
| 0.2 | 227 | 0.454 ± 0.054 | 0.477 ± 0.072 |
| 0.4 | 451 | 0.570 ± 0.020 | 0.548 ± 0.047 |
| 0.7 | 790 | 0.542 ± 0.016 | 0.617 ± 0.024 |
| 1.0 | 1128 | 0.540 ± 0.000 | 0.669 ± 0.000 |

Gain from 40% -> 100% of the data: -0.030 test F1. Flattening: more of the same data won't help much; vary the data instead.
