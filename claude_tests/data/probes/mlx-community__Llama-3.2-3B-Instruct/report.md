# Probe report: mlx-community/Llama-3.2-3B-Instruct

- layer **6** (after block), nuisance cutoff 0.8 (24 PCs removed), contrast `mean_emotion`, pooled tokens skip=4
- corpus: 333 train / 101 val / 98 test passages (split by TOPIC)

## Held-out TEST topics (never used for layer or hyperparameter selection)

- accuracy **0.860** (chance 0.125); macro-F1 **0.857**
- label-shuffle null macro-F1: mean 0.072, p95 0.110, max 0.111
- emotional-vs-neutral AUC: 0.955

| emotion | n test | AUC (1-vs-rest) | split-half cosine |
|---|---|---|---|
| joy | 11 | 0.997 | 0.77 |
| sadness | 12 | 0.978 | 0.61 |
| fear | 11 | 0.997 | 0.68 |
| anger | 10 | 0.993 | 0.68 |
| surprise | 12 | 0.995 | 0.64 |
| disgust | 9 | 1.000 | 0.74 |
| calm | 12 | 0.998 | 0.67 |
| love | 9 | 0.999 | 0.69 |

## Layer sweep (best val macro-F1 per layer)

| layer | val macro-F1 |
|---|---|
| 6 | 0.967 |
| 8 | 0.930 |
| 10 | 0.896 |
| 12 | 0.932 |
| 14 | 0.876 |
| 16 | 0.897 |
| 18 | 0.886 |
| 20 | 0.879 |
| 22 | 0.873 |
| 24 | 0.884 |

## Verdict

All concepts clear the AUC/stability bar on held-out topics.
Probes are only meaningful if test macro-F1 is far above the null p95.
