# Robustness (logreg @ layer 14, cfg {'var_frac': 0.8, 'C': 0.003})

## Leave-one-writing-style-out

| held-out style | n test | test F1 | hand-written F1 (this fold's model) |
|---|---|---|---|
| dry, understated first-person account | 54 | 0.674 | 0.600 |
| plain report-style account where the feeling shows only in what the person decides or does | 52 | 0.746 | 0.533 |
| reflective paragraph in which the writer reasons about what to do next | 42 | 0.593 | 0.543 |
| short dialogue between two people (use quotation marks) | 50 | 0.722 | 0.508 |
| text-message exchange between two people | 49 | 0.726 | 0.535 |
| third-person narrative focused on one person's actions and choices | 50 | 0.607 | 0.542 |

Mean test F1 0.678, worst 0.593.

## Leave-one-writer-out

Only one writer in this corpus (or too few rows per writer). Add text from another model: see probekit_corpus_api.py.

## Learning curve (stratified subsamples of TRAIN)

| fraction | n train | test-hard F1 | hand-written F1 |
|---|---|---|---|
| 0.1 | 54 | 0.531 ± 0.040 | 0.400 ± 0.054 |
| 0.2 | 99 | 0.522 ± 0.083 | 0.431 ± 0.042 |
| 0.4 | 195 | 0.689 ± 0.041 | 0.490 ± 0.019 |
| 0.7 | 342 | 0.720 ± 0.013 | 0.532 ± 0.060 |
| 1.0 | 488 | 0.721 ± 0.000 | 0.542 ± 0.000 |

Gain from 40% -> 100% of the data: +0.032 test F1. Still climbing: more/varied passages should help.
