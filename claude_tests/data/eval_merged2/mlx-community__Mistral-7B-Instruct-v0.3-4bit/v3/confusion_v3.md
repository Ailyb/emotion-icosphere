# Confusion diagnostics @ layer 10

hand-written passages: 92 (own-voice 45). Chance for 9-way = 0.11; own-voice intervals are wide (n=45): differences under ~0.15 between estimators are not resolvable.

## Accuracy by estimator

| estimator | own-voice | other hand-written | own-voice F1 | coarse (4-way) own-voice | coarse (4-way) other |
|---|---|---|---|---|---|
| meandiff | 25/45 = 0.56 (95% CI 0.41-0.69) | 28/47 = 0.60 (95% CI 0.45-0.72) | 0.578 | 0.62 | 0.62 |
| lda | 28/45 = 0.62 (95% CI 0.48-0.75) | 34/47 = 0.72 (95% CI 0.58-0.83) | 0.626 | 0.71 | 0.77 |
| logreg | 25/45 = 0.56 (95% CI 0.41-0.69) | 32/47 = 0.68 (95% CI 0.54-0.80) | 0.576 | 0.64 | 0.72 |
| native_lr | 27/45 = 0.60 (95% CI 0.45-0.73) | 35/47 = 0.74 (95% CI 0.60-0.85) | 0.590 | 0.69 | 0.79 |

## Own-voice accuracy by length (short = 12 words or fewer)

| estimator | short | longer |
|---|---|---|
| meandiff | 1/1 = 1.00 (95% CI 0.21-1.00) | 24/44 = 0.55 (95% CI 0.40-0.68) |
| lda | 1/1 = 1.00 (95% CI 0.21-1.00) | 27/44 = 0.61 (95% CI 0.47-0.74) |
| logreg | 1/1 = 1.00 (95% CI 0.21-1.00) | 24/44 = 0.55 (95% CI 0.40-0.68) |
| native_lr | 0/1 = 0.00 (95% CI 0.00-0.79) | 27/44 = 0.61 (95% CI 0.47-0.74) |

## Own-voice confusion: meandiff

| true \ pred | joy | sadn | fear | ange | surp | disg | calm | love | neut |
|---|---|---|---|---|---|---|---|---|---|
| joy | 3 | . | . | . | 2 | . | . | . | . |
| sadness | . | 3 | . | 1 | . | . | . | 1 | . |
| fear | . | . | 1 | . | 4 | . | . | . | . |
| anger | . | . | . | 3 | 2 | . | . | . | . |
| surprise | 1 | . | . | . | 4 | . | . | . | . |
| disgust | . | . | . | . | 2 | 2 | . | . | 1 |
| calm | 1 | . | . | 1 | . | . | 3 | . | . |
| love | 1 | . | . | . | . | . | . | 3 | 1 |
| neutral | . | . | . | . | 2 | . | . | . | 3 |

## Own-voice confusion: lda

| true \ pred | joy | sadn | fear | ange | surp | disg | calm | love | neut |
|---|---|---|---|---|---|---|---|---|---|
| joy | 4 | . | . | . | 1 | . | . | . | . |
| sadness | . | 2 | . | 1 | . | . | . | 1 | 1 |
| fear | . | . | 2 | . | 3 | . | . | . | . |
| anger | . | . | . | 2 | . | 1 | . | . | 2 |
| surprise | 1 | . | . | . | 4 | . | . | . | . |
| disgust | . | . | . | . | 2 | 3 | . | . | . |
| calm | 1 | . | . | 1 | . | . | 3 | . | . |
| love | 1 | . | . | . | . | . | . | 4 | . |
| neutral | . | . | . | . | 1 | . | . | . | 4 |

## Own-voice confusion: logreg

| true \ pred | joy | sadn | fear | ange | surp | disg | calm | love | neut |
|---|---|---|---|---|---|---|---|---|---|
| joy | 2 | . | . | . | 3 | . | . | . | . |
| sadness | . | 3 | . | 1 | . | . | . | 1 | . |
| fear | . | . | 2 | . | 3 | . | . | . | . |
| anger | . | . | . | 2 | 1 | 1 | . | . | 1 |
| surprise | 1 | . | . | . | 4 | . | . | . | . |
| disgust | . | . | . | . | 2 | 2 | . | . | 1 |
| calm | 1 | . | . | 1 | . | . | 3 | . | . |
| love | 1 | . | . | . | . | . | . | 4 | . |
| neutral | . | . | . | . | 1 | 1 | . | . | 3 |

## Own-voice confusion: native_lr

| true \ pred | joy | sadn | fear | ange | surp | disg | calm | love | neut |
|---|---|---|---|---|---|---|---|---|---|
| joy | 4 | . | . | . | 1 | . | . | . | . |
| sadness | . | 2 | . | 1 | . | . | . | . | 2 |
| fear | . | . | 2 | . | 2 | . | . | . | 1 |
| anger | . | . | . | 4 | . | . | . | . | 1 |
| surprise | 1 | . | . | . | 4 | . | . | . | . |
| disgust | . | . | . | . | 1 | 1 | . | . | 3 |
| calm | 2 | . | . | 1 | . | . | 2 | . | . |
| love | 1 | . | . | . | . | . | . | 3 | 1 |
| neutral | . | . | . | . | . | . | . | . | 5 |

## Reading this

- If coarse accuracy is much higher than 9-way accuracy, the probes mostly separate positive/negative/neutral and blur fine categories.
- Off-diagonal mass concentrated in one column means an 'attractor' class that absorbs unfamiliar text.
- A big gap between short and longer passages points at length-unreliable calibration, not at emotion.
