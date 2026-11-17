# T3 length-aware calibration (mlx-community/Mistral-7B-Instruct-v0.3-4bit, layer 10, logreg; n=5014; pre-registered in PLAN_v2)

Held-out neutral GoEmotions-TRAIN comments; |z| > 3; Clopper-Pearson 95% upper bounds; acceptance: <= 2% per emotion (worst of 7), <= 5% per passage, in every bucket with n >= 500.

| words | n | mean z (RAW) | sd z (RAW) | worst-emotion rate (upper) RAW | per-passage rate (upper) RAW | RAW | worst-emotion rate (upper) RECAL | per-passage rate (upper) RECAL | RECAL |
|---|---|---|---|---|---|---|---|---|---|
| 1-2 | 416 | +0.41 | 2.20 | 25.7% (30.2%) | 69.0% (73.4%) | FAIL | 1.2% (2.8%) | 5.0% (7.6%) | FAIL |
| 3-4 | 900 | +0.25 | 2.06 | 22.2% (25.1%) | 61.7% (64.9%) | FAIL | 1.1% (2.0%) | 5.3% (7.0%) | FAIL |
| 5-7 | 900 | +0.22 | 1.69 | 15.2% (17.7%) | 44.9% (48.2%) | FAIL | 1.0% (1.9%) | 4.0% (5.5%) | FAIL |
| 8-11 | 900 | +0.23 | 1.47 | 10.1% (12.3%) | 28.8% (31.9%) | FAIL | 0.9% (1.7%) | 4.0% (5.5%) | FAIL |
| 12-19 | 998 | +0.24 | 1.31 | 6.2% (7.9%) | 22.0% (24.7%) | FAIL | 1.0% (1.8%) | 4.8% (6.3%) | FAIL |
| 20+ | 900 | +0.26 | 1.17 | 4.1% (5.6%) | 15.3% (17.9%) | FAIL | 1.2% (2.2%) | 4.6% (6.1%) | FAIL |

RAW: no bucket qualifies; 1-2: n=416 < 500: cannot certify; 3-4: per-emotion per-passage bound exceeded; 5-7: per-emotion per-passage bound exceeded; 8-11: per-emotion per-passage bound exceeded; 12-19: per-emotion per-passage bound exceeded; 20+: per-emotion per-passage bound exceeded

RECAL: no bucket qualifies; 1-2: n=416 < 500: cannot certify; 3-4: per-emotion per-passage bound exceeded; 5-7: per-passage bound exceeded; 8-11: per-passage bound exceeded; 12-19: per-passage bound exceeded; 20+: per-emotion per-passage bound exceeded

## Post hoc, descriptive: alarm thresholds that would hold per bucket

Quantile of max |z| over the 7 emotions on neutral text. To keep false alarms at 5% (or 1%) of neutral passages, only highlight a passage when its max |z| exceeds the value shown. RAW = existing calibration; RECAL = per-bucket recalibrated z (out of fold).

| words | n | RAW q95 | RAW q99 | RECAL q95 | RECAL q99 |
|---|---|---|---|---|---|
| 1-2 | 416 | 7.7 | 9.8 | 3.0 | 4.4 |
| 3-4 | 900 | 6.8 | 9.0 | 3.1 | 4.4 |
| 5-7 | 900 | 5.2 | 7.0 | 2.9 | 3.7 |
| 8-11 | 900 | 4.5 | 6.0 | 2.8 | 3.9 |
| 12-19 | 998 | 4.2 | 5.3 | 3.0 | 3.6 |
| 20+ | 900 | 3.7 | 4.6 | 3.0 | 3.6 |
