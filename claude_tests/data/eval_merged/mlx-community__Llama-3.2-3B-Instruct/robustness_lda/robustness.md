# Robustness (lda @ layer 14, cfg {'var_frac': 0.5, 'shrinkage': 'auto'})

## Leave-one-writing-style-out

| held-out style | n test | test F1 | hand-written F1 (this fold's model) |
|---|---|---|---|
| dry, understated first-person account | 54 | 0.651 | 0.749 |
| informal social-media comment (reddit) | 320 | 0.235 | 0.547 |
| plain report-style account where the feeling shows only in what the person decides or does | 52 | 0.796 | 0.677 |
| reflective paragraph in which the writer reasons about what to do next | 42 | 0.739 | 0.721 |
| short dialogue between two people (use quotation marks) | 50 | 0.776 | 0.699 |
| text-message exchange between two people | 49 | 0.683 | 0.658 |
| third-person narrative focused on one person's actions and choices | 50 | 0.767 | 0.723 |

Mean test F1 0.664, worst 0.235.

## Leave-one-writer-out

| held-out writer | n test | test F1 | hand-written F1 |
|---|---|---|---|
| llama-3.2-3b * | 265 | 0.377 | 0.444 |
| reddit-goemotions | 320 | 0.235 | 0.547 |

(* = partial fold)
- `llama-3.2-3b`: partial: no training data for ['calm']; those rows excluded

## Learning curve (stratified subsamples of TRAIN)

| fraction | n train | test-hard F1 | hand-written F1 |
|---|---|---|---|
| 0.1 | 114 | 0.436 ± 0.044 | 0.405 ± 0.042 |
| 0.2 | 227 | 0.504 ± 0.026 | 0.534 ± 0.060 |
| 0.4 | 451 | 0.595 ± 0.040 | 0.616 ± 0.038 |
| 0.7 | 790 | 0.607 ± 0.017 | 0.678 ± 0.042 |
| 1.0 | 1128 | 0.616 ± 0.000 | 0.723 ± 0.000 |

Gain from 40% -> 100% of the data: +0.020 test F1. Flattening: more of the same data won't help much; vary the data instead.
