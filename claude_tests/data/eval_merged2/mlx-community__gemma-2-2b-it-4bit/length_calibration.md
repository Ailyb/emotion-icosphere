# T3 length-aware calibration (mlx-community/gemma-2-2b-it-4bit, layer 14, lda; n=5014; pre-registered in PLAN_v2)

Held-out neutral GoEmotions-TRAIN comments; |z| > 3; Clopper-Pearson 95% upper bounds; acceptance: <= 2% per emotion (worst of 7), <= 5% per passage, in every bucket with n >= 500.

| words | n | mean z (RAW) | sd z (RAW) | worst-emotion rate (upper) RAW | per-passage rate (upper) RAW | RAW | worst-emotion rate (upper) RECAL | per-passage rate (upper) RECAL | RECAL |
|---|---|---|---|---|---|---|---|---|---|
| 1-2 | 416 | +0.66 | 3.04 | 42.1% (47.0%) | 83.4% (86.9%) | FAIL | 1.7% (3.4%) | 6.2% (9.0%) | FAIL |
| 3-4 | 900 | +0.35 | 2.53 | 32.3% (35.5%) | 72.7% (75.6%) | FAIL | 1.3% (2.3%) | 6.4% (8.3%) | FAIL |
| 5-7 | 900 | +0.28 | 1.93 | 18.8% (21.5%) | 55.2% (58.5%) | FAIL | 1.2% (2.2%) | 5.6% (7.3%) | FAIL |
| 8-11 | 900 | +0.26 | 1.61 | 14.6% (17.0%) | 41.4% (44.7%) | FAIL | 1.0% (1.9%) | 4.9% (6.5%) | FAIL |
| 12-19 | 998 | +0.27 | 1.40 | 11.0% (13.1%) | 26.9% (29.7%) | FAIL | 1.2% (2.1%) | 4.7% (6.2%) | FAIL |
| 20+ | 900 | +0.28 | 1.27 | 7.4% (9.4%) | 20.1% (22.9%) | FAIL | 1.0% (1.9%) | 4.8% (6.4%) | FAIL |

RAW: no bucket qualifies; 1-2: n=416 < 500: cannot certify; 3-4: per-emotion per-passage bound exceeded; 5-7: per-emotion per-passage bound exceeded; 8-11: per-emotion per-passage bound exceeded; 12-19: per-emotion per-passage bound exceeded; 20+: per-emotion per-passage bound exceeded

RECAL: no bucket qualifies; 1-2: n=416 < 500: cannot certify; 3-4: per-emotion per-passage bound exceeded; 5-7: per-emotion per-passage bound exceeded; 8-11: per-passage bound exceeded; 12-19: per-emotion per-passage bound exceeded; 20+: per-passage bound exceeded

## Post hoc, descriptive: alarm thresholds that would hold per bucket

Quantile of max |z| over the 7 emotions on neutral text. To keep false alarms at 5% (or 1%) of neutral passages, only highlight a passage when its max |z| exceeds the value shown. RAW = existing calibration; RECAL = per-bucket recalibrated z (out of fold).

| words | n | RAW q95 | RAW q99 | RECAL q95 | RECAL q99 |
|---|---|---|---|---|---|
| 1-2 | 416 | 10.9 | 15.3 | 3.1 | 4.5 |
| 3-4 | 900 | 8.4 | 12.4 | 3.1 | 4.4 |
| 5-7 | 900 | 6.4 | 8.5 | 3.1 | 4.0 |
| 8-11 | 900 | 5.2 | 6.9 | 2.9 | 4.1 |
| 12-19 | 998 | 4.7 | 6.1 | 3.0 | 4.1 |
| 20+ | 900 | 4.1 | 5.4 | 3.0 | 3.9 |
