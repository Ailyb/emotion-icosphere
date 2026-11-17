# Robustness (lda @ layer 14, cfg {'var_frac': 0.5, 'shrinkage': 'auto'})

## Leave-one-writing-style-out

| held-out style | n test | test F1 | hand-written F1 (this fold's model) |
|---|---|---|---|
| dry, understated first-person account | 113 | 0.709 | 0.717 |
| informal social-media comment (reddit) | 160 | 0.307 | 0.566 |
| plain report-style account where the feeling shows only in what the person decides or does | 114 | 0.755 | 0.687 |
| reflective paragraph in which the writer reasons about what to do next | 102 | 0.836 | 0.701 |
| short dialogue between two people (use quotation marks) | 109 | 0.843 | 0.674 |
| text-message exchange between two people | 110 | 0.772 | 0.667 |
| third-person narrative focused on one person's actions and choices | 108 | 0.861 | 0.669 |

Mean test F1 0.726, worst 0.307.

## Leave-one-writer-out

| held-out writer | n test | test F1 | hand-written F1 |
|---|---|---|---|
| llama-3.2-3b | 297 | 0.782 | 0.684 |
| qwen3.8-27b | 359 | 0.723 | 0.712 |
| reddit-goemotions | 160 | 0.307 | 0.566 |

## Learning curve (stratified subsamples of TRAIN)

| fraction | n train | test-hard F1 | hand-written F1 |
|---|---|---|---|
| 0.1 | 177 | 0.562 ± 0.031 | 0.505 ± 0.069 |
| 0.2 | 354 | 0.640 ± 0.025 | 0.610 ± 0.059 |
| 0.4 | 708 | 0.726 ± 0.019 | 0.655 ± 0.039 |
| 0.7 | 1240 | 0.760 ± 0.020 | 0.670 ± 0.024 |
| 1.0 | 1771 | 0.763 ± 0.000 | 0.683 ± 0.000 |

Gain from 40% -> 100% of the data: +0.037 test F1. Still climbing: more/varied passages should help.
