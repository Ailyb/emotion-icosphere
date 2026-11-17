# T2 step 1: GoEmotions official TEST split, label counts (n items in split: 5427; with rater data: 5427)

Counts only; nothing was built. `_unmapped` = agreed label is a GoEmotions class we do not use (e.g. admiration); `_no_agreement` = fails the rule. Criterion (PLAN_v2 T2): at least 6 of the 7 emotions, and neutral, reach n >= 50.

## rule=strict, length filter 10-80 words: off

| joy | sadness | fear | anger | surprise | disgust | love | neutral | unmapped | no agreement |
|---|---|---|---|---|---|---|---|---|---|
| 10 | 12 | 8 | 14 | 9 | 8 | 29 | 372 | 328 | 4637 |

classes with n >= 50: 1/8 (neutral); emotions among them: 0/7

## rule=strict, length filter 10-80 words: on

| joy | sadness | fear | anger | surprise | disgust | love | neutral | unmapped | no agreement |
|---|---|---|---|---|---|---|---|---|---|
| 3 | 4 | 5 | 6 | 6 | 3 | 11 | 200 | 149 | 3049 |

classes with n >= 50: 1/8 (neutral); emotions among them: 0/7

## rule=shared, length filter 10-80 words: off

| joy | sadness | fear | anger | surprise | disgust | love | neutral | unmapped | no agreement |
|---|---|---|---|---|---|---|---|---|---|
| 32 | 30 | 22 | 30 | 20 | 16 | 78 | 372 | 671 | 4156 |

classes with n >= 50: 2/8 (love, neutral); emotions among them: 1/7

## rule=shared, length filter 10-80 words: on

| joy | sadness | fear | anger | surprise | disgust | love | neutral | unmapped | no agreement |
|---|---|---|---|---|---|---|---|---|---|
| 20 | 16 | 15 | 16 | 16 | 8 | 48 | 200 | 385 | 2712 |

classes with n >= 50: 1/8 (neutral); emotions among them: 0/7

## rule=simple, length filter 10-80 words: off

| joy | sadness | fear | anger | surprise | disgust | love | neutral | unmapped | no agreement |
|---|---|---|---|---|---|---|---|---|---|
| 93 | 102 | 65 | 131 | 87 | 76 | 160 | 1606 | 2270 | 837 |

classes with n >= 50: 8/8 (joy, sadness, fear, anger, surprise, disgust, love, neutral); emotions among them: 7/7

## rule=simple, length filter 10-80 words: on

| joy | sadness | fear | anger | surprise | disgust | love | neutral | unmapped | no agreement |
|---|---|---|---|---|---|---|---|---|---|
| 59 | 65 | 41 | 71 | 55 | 50 | 94 | 976 | 1428 | 597 |

classes with n >= 50: 7/8 (joy, sadness, anger, surprise, disgust, love, neutral); emotions among them: 6/7

