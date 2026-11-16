# Robustness (lda @ layer 14, cfg {'var_frac': 0.8, 'shrinkage': 'auto'})

## Leave-one-writing-style-out

| held-out style | n test | test F1 | hand-written F1 (this fold's model) |
|---|---|---|---|
| dry, understated first-person account | 54 | 0.767 | 0.569 |
| plain report-style account where the feeling shows only in what the person decides or does | 52 | 0.725 | 0.482 |
| reflective paragraph in which the writer reasons about what to do next | 42 | 0.668 | 0.519 |
| short dialogue between two people (use quotation marks) | 50 | 0.714 | 0.494 |
| text-message exchange between two people | 49 | 0.755 | 0.520 |
| third-person narrative focused on one person's actions and choices | 50 | 0.615 | 0.465 |

Mean test F1 0.707, worst 0.615.

## Leave-one-writer-out

Only one writer in this corpus (or too few rows per writer). Add text from another model: see probekit_corpus_api.py.

## Learning curve (stratified subsamples of TRAIN)

| fraction | n train | test-hard F1 | hand-written F1 |
|---|---|---|---|
| 0.1 | 54 | 0.514 ± 0.032 | 0.402 ± 0.037 |
| 0.2 | 99 | 0.513 ± 0.080 | 0.415 ± 0.057 |
| 0.4 | 195 | 0.616 ± 0.068 | 0.465 ± 0.033 |
| 0.7 | 342 | 0.695 ± 0.025 | 0.541 ± 0.034 |
| 1.0 | 488 | 0.712 ± 0.000 | 0.565 ± 0.000 |

Gain from 40% -> 100% of the data: +0.096 test F1. Still climbing: more/varied passages should help.
