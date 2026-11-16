# Robustness (meandiff @ layer 14, cfg {'var_frac': 0.5})

## Leave-one-writing-style-out

| held-out style | n test | test F1 | hand-written F1 (this fold's model) |
|---|---|---|---|
| dry, understated first-person account | 54 | 0.670 | 0.495 |
| plain report-style account where the feeling shows only in what the person decides or does | 52 | 0.768 | 0.407 |
| reflective paragraph in which the writer reasons about what to do next | 42 | 0.600 | 0.449 |
| short dialogue between two people (use quotation marks) | 50 | 0.703 | 0.403 |
| text-message exchange between two people | 49 | 0.758 | 0.469 |
| third-person narrative focused on one person's actions and choices | 50 | 0.681 | 0.548 |

Mean test F1 0.697, worst 0.600.

## Leave-one-writer-out

Only one writer in this corpus (or too few rows per writer). Add text from another model: see probekit_corpus_api.py.

## Learning curve (stratified subsamples of TRAIN)

| fraction | n train | test-hard F1 | hand-written F1 |
|---|---|---|---|
| 0.1 | 54 | 0.444 ± 0.048 | 0.302 ± 0.061 |
| 0.2 | 99 | 0.447 ± 0.057 | 0.344 ± 0.050 |
| 0.4 | 195 | 0.630 ± 0.034 | 0.431 ± 0.027 |
| 0.7 | 342 | 0.692 ± 0.035 | 0.434 ± 0.059 |
| 1.0 | 488 | 0.683 ± 0.000 | 0.499 ± 0.000 |

Gain from 40% -> 100% of the data: +0.054 test F1. Still climbing: more/varied passages should help.
