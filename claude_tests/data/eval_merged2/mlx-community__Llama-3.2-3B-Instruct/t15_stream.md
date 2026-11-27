# T15 stream splice test (pre-registered 2026-11-27, commit 2cbb564; mlx-community/Llama-3.2-3B-Instruct, layer 16, lda, HUM-trained; synthetic streams, intended labels unvalidated)

## VAL (window selection)

| window | neutral FA share | hit | onset median | offset median | offset (contiguous run, post hoc) | qualifies |
|---|---|---|---|---|---|---|
| 8 | 0.010 | 0.693 | 9.0 | 1.0 | 1.0 | yes |
| 16 | 0.010 | 0.471 | 12.0 | 0.0 | 0.0 | yes |
| 32 | 0.010 | 0.107 | 17.0 | 0.0 | 0.0 | yes |
| prefix | 0.010 | 0.014 | 14.0 | 0.0 | 0.0 | yes |

## TEST (window 8 chosen on VAL; threshold 3.48 on recalibrated max z)

- n splice streams = 140, neutral streams = 140.
- Neutral per-position false-alarm share: 0.018 (rule <= 0.05).
- Hit rate: 0.700 [0.621, 0.786] (rule >= 0.6); shuffled-label chance 0.115 (95th pct 0.157).
- Onset latency, tokens, median: 8.0 [7.0, 8.0] (rule <= 12).
- Offset latency, tokens, median: 2.0 [1.0, 2.0] (rule <= 12); contiguous-run variant (post hoc): 1.0 [0.0, 2.0].
- Hit rate per emotion (descriptive): joy 0.75, sadness 0.45, fear 0.85, anger 0.90, surprise 0.30, disgust 0.95, love 0.70

**Rule MET: token-level colour usable.**
