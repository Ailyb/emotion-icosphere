# Confusion diagnostics @ layer 14

hand-written passages: 92 (own-voice 45). Chance for 9-way = 0.11; own-voice intervals are wide (n=45): differences under ~0.15 between estimators are not resolvable.

## Accuracy by estimator

| estimator | own-voice | other hand-written | own-voice F1 | coarse (4-way) own-voice | coarse (4-way) other |
|---|---|---|---|---|---|
| meandiff | 24/45 = 0.53 (95% CI 0.39-0.67) | 25/47 = 0.53 (95% CI 0.39-0.67) | 0.524 | 0.67 | 0.55 |
| lda | 28/45 = 0.62 (95% CI 0.48-0.75) | 35/47 = 0.74 (95% CI 0.60-0.85) | 0.632 | 0.71 | 0.77 |
| logreg | 25/45 = 0.56 (95% CI 0.41-0.69) | 34/47 = 0.72 (95% CI 0.58-0.83) | 0.552 | 0.69 | 0.74 |
| native_lr | 30/45 = 0.67 (95% CI 0.52-0.79) | 37/47 = 0.79 (95% CI 0.65-0.88) | 0.663 | 0.84 | 0.85 |

## Own-voice accuracy by length (short = 12 words or fewer)

| estimator | short | longer |
|---|---|---|
| meandiff | 1/1 = 1.00 (95% CI 0.21-1.00) | 23/44 = 0.52 (95% CI 0.38-0.66) |
| lda | 1/1 = 1.00 (95% CI 0.21-1.00) | 27/44 = 0.61 (95% CI 0.47-0.74) |
| logreg | 1/1 = 1.00 (95% CI 0.21-1.00) | 24/44 = 0.55 (95% CI 0.40-0.68) |
| native_lr | 1/1 = 1.00 (95% CI 0.21-1.00) | 29/44 = 0.66 (95% CI 0.51-0.78) |

## Own-voice confusion: meandiff

| true \ pred | joy | sadn | fear | ange | surp | disg | calm | love | neut |
|---|---|---|---|---|---|---|---|---|---|
| joy | 3 | . | . | . | 2 | . | . | . | . |
| sadness | . | 2 | . | . | 1 | . | . | 1 | 1 |
| fear | . | . | 4 | . | 1 | . | . | . | . |
| anger | . | . | . | 4 | . | 1 | . | . | . |
| surprise | 1 | . | . | . | 4 | . | . | . | . |
| disgust | . | . | . | . | 2 | 3 | . | . | . |
| calm | 2 | . | . | 1 | . | . | 2 | . | . |
| love | 3 | . | . | . | . | . | . | 2 | . |
| neutral | . | 2 | . | . | 2 | . | . | 1 | . |

## Own-voice confusion: lda

| true \ pred | joy | sadn | fear | ange | surp | disg | calm | love | neut |
|---|---|---|---|---|---|---|---|---|---|
| joy | 3 | . | . | . | 2 | . | . | . | . |
| sadness | . | 2 | . | . | 1 | . | . | 1 | 1 |
| fear | . | . | 3 | . | 2 | . | . | . | . |
| anger | . | . | . | 3 | . | 1 | . | . | 1 |
| surprise | 1 | . | . | . | 4 | . | . | . | . |
| disgust | . | . | . | . | . | 5 | . | . | . |
| calm | 2 | . | . | 1 | . | . | 2 | . | . |
| love | 1 | . | . | . | . | . | . | 4 | . |
| neutral | . | . | . | . | 3 | . | . | . | 2 |

## Own-voice confusion: logreg

| true \ pred | joy | sadn | fear | ange | surp | disg | calm | love | neut |
|---|---|---|---|---|---|---|---|---|---|
| joy | 2 | . | . | . | 3 | . | . | . | . |
| sadness | . | 1 | . | . | . | 1 | . | 1 | 2 |
| fear | . | . | 3 | . | 2 | . | . | . | . |
| anger | . | . | . | 3 | . | 1 | . | . | 1 |
| surprise | 1 | . | . | . | 4 | . | . | . | . |
| disgust | . | . | . | . | . | 5 | . | . | . |
| calm | 2 | . | . | 1 | . | . | 2 | . | . |
| love | 2 | . | . | . | . | . | . | 3 | . |
| neutral | . | . | 1 | . | 2 | . | . | . | 2 |

## Own-voice confusion: native_lr

| true \ pred | joy | sadn | fear | ange | surp | disg | calm | love | neut |
|---|---|---|---|---|---|---|---|---|---|
| joy | 3 | . | . | . | 2 | . | . | . | . |
| sadness | . | 4 | . | 1 | . | . | . | . | . |
| fear | . | 1 | 3 | . | 1 | . | . | . | . |
| anger | . | . | . | 4 | . | . | . | . | 1 |
| surprise | 1 | . | . | . | 4 | . | . | . | . |
| disgust | . | . | . | . | . | 4 | . | . | 1 |
| calm | 3 | . | . | 1 | . | . | 1 | . | . |
| love | 3 | . | . | . | . | . | . | 2 | . |
| neutral | . | . | . | . | . | . | . | . | 5 |

## Reading this

- If coarse accuracy is much higher than 9-way accuracy, the probes mostly separate positive/negative/neutral and blur fine categories.
- Off-diagonal mass concentrated in one column means an 'attractor' class that absorbs unfamiliar text.
- A big gap between short and longer passages points at length-unreliable calibration, not at emotion.
