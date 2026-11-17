# T3 length-aware calibration (mlx-community/Llama-3.2-3B-Instruct, layer 16, lda; n=5014; pre-registered in PLAN_v2)

Held-out neutral GoEmotions-TRAIN comments; |z| > 3; Clopper-Pearson 95% upper bounds; acceptance: <= 2% per emotion (worst of 7), <= 5% per passage, in every bucket with n >= 500.

| words | n | mean z (RAW) | sd z (RAW) | worst-emotion rate (upper) RAW | per-passage rate (upper) RAW | RAW | worst-emotion rate (upper) RECAL | per-passage rate (upper) RECAL | RECAL |
|---|---|---|---|---|---|---|---|---|---|
| 1-2 | 416 | +0.36 | 2.41 | 34.1% (38.9%) | 76.0% (80.0%) | FAIL | 1.4% (3.1%) | 6.0% (8.7%) | FAIL |
| 3-4 | 900 | +0.28 | 2.33 | 27.4% (30.5%) | 63.4% (66.6%) | FAIL | 1.4% (2.5%) | 7.4% (9.4%) | FAIL |
| 5-7 | 900 | +0.25 | 1.78 | 14.9% (17.4%) | 47.3% (50.7%) | FAIL | 1.1% (2.0%) | 5.4% (7.1%) | FAIL |
| 8-11 | 900 | +0.23 | 1.53 | 8.6% (10.6%) | 33.4% (36.6%) | FAIL | 1.1% (2.0%) | 4.9% (6.5%) | FAIL |
| 12-19 | 998 | +0.25 | 1.32 | 7.1% (8.9%) | 23.0% (25.8%) | FAIL | 1.1% (2.0%) | 5.3% (6.9%) | FAIL |
| 20+ | 900 | +0.25 | 1.19 | 4.0% (5.5%) | 15.3% (17.9%) | FAIL | 1.2% (2.2%) | 4.4% (6.0%) | FAIL |

RAW: no bucket qualifies; 1-2: n=416 < 500: cannot certify; 3-4: per-emotion per-passage bound exceeded; 5-7: per-emotion per-passage bound exceeded; 8-11: per-emotion per-passage bound exceeded; 12-19: per-emotion per-passage bound exceeded; 20+: per-emotion per-passage bound exceeded

RECAL: no bucket qualifies; 1-2: n=416 < 500: cannot certify; 3-4: per-emotion per-passage bound exceeded; 5-7: per-emotion per-passage bound exceeded; 8-11: per-emotion per-passage bound exceeded; 12-19: per-passage bound exceeded; 20+: per-emotion per-passage bound exceeded

## Post hoc, descriptive: alarm thresholds that would hold per bucket

Quantile of max |z| over the 7 emotions on neutral text. To keep false alarms at 5% (or 1%) of neutral passages, only highlight a passage when its max |z| exceeds the value shown. RAW = existing calibration; RECAL = per-bucket recalibrated z (out of fold).

| words | n | RAW q95 | RAW q99 | RECAL q95 | RECAL q99 |
|---|---|---|---|---|---|
| 1-2 | 416 | 8.4 | 11.1 | 3.0 | 4.1 |
| 3-4 | 900 | 7.8 | 12.1 | 3.3 | 4.8 |
| 5-7 | 900 | 5.7 | 8.0 | 3.1 | 4.2 |
| 8-11 | 900 | 4.7 | 6.3 | 3.0 | 4.1 |
| 12-19 | 998 | 4.2 | 5.6 | 3.0 | 4.0 |
| 20+ | 900 | 3.8 | 4.8 | 2.9 | 3.9 |
