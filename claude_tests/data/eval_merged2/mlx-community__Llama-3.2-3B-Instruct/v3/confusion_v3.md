# Confusion diagnostics @ layer 16

hand-written passages: 92 (own-voice 45). Chance for 9-way = 0.11; own-voice intervals are wide (n=45): differences under ~0.15 between estimators are not resolvable.

## Accuracy by estimator

| estimator | own-voice | other hand-written | own-voice F1 | coarse (4-way) own-voice | coarse (4-way) other |
|---|---|---|---|---|---|
| meandiff | 23/45 = 0.51 (95% CI 0.37-0.65) | 32/47 = 0.68 (95% CI 0.54-0.80) | 0.510 | 0.60 | 0.77 |
| lda | 25/45 = 0.56 (95% CI 0.41-0.69) | 36/47 = 0.77 (95% CI 0.63-0.86) | 0.564 | 0.64 | 0.81 |
| logreg | 25/45 = 0.56 (95% CI 0.41-0.69) | 33/47 = 0.70 (95% CI 0.56-0.81) | 0.548 | 0.64 | 0.77 |
| native_lr | 24/45 = 0.53 (95% CI 0.39-0.67) | 35/47 = 0.74 (95% CI 0.60-0.85) | 0.528 | 0.64 | 0.83 |

## Own-voice accuracy by length (short = 12 words or fewer)

| estimator | short | longer |
|---|---|---|
| meandiff | 0/1 = 0.00 (95% CI 0.00-0.79) | 23/44 = 0.52 (95% CI 0.38-0.66) |
| lda | 0/1 = 0.00 (95% CI 0.00-0.79) | 25/44 = 0.57 (95% CI 0.42-0.70) |
| logreg | 0/1 = 0.00 (95% CI 0.00-0.79) | 25/44 = 0.57 (95% CI 0.42-0.70) |
| native_lr | 0/1 = 0.00 (95% CI 0.00-0.79) | 24/44 = 0.55 (95% CI 0.40-0.68) |

## Own-voice confusion: meandiff

| true \ pred | joy | sadn | fear | ange | surp | disg | calm | love | neut |
|---|---|---|---|---|---|---|---|---|---|
| joy | 3 | . | . | . | 2 | . | . | . | . |
| sadness | . | 2 | . | . | . | . | 1 | 1 | 1 |
| fear | . | 1 | 1 | . | 3 | . | . | . | . |
| anger | . | . | . | 4 | . | . | . | . | 1 |
| surprise | . | . | . | . | 5 | . | . | . | . |
| disgust | . | . | . | . | . | 2 | . | . | 3 |
| calm | 1 | . | . | 1 | . | . | 3 | . | . |
| love | 2 | . | . | . | . | . | . | 2 | 1 |
| neutral | . | . | 2 | . | 2 | . | . | . | 1 |

## Own-voice confusion: lda

| true \ pred | joy | sadn | fear | ange | surp | disg | calm | love | neut |
|---|---|---|---|---|---|---|---|---|---|
| joy | 3 | . | . | . | 1 | . | . | . | 1 |
| sadness | . | 3 | . | 1 | . | . | . | 1 | . |
| fear | . | . | 2 | . | 2 | 1 | . | . | . |
| anger | . | . | . | 2 | 1 | . | . | 1 | 1 |
| surprise | 1 | . | . | . | 4 | . | . | . | . |
| disgust | . | . | . | . | . | 4 | . | . | 1 |
| calm | . | . | . | 1 | . | . | 4 | . | . |
| love | 2 | . | . | . | . | . | . | 2 | 1 |
| neutral | . | . | . | . | 3 | 1 | . | . | 1 |

## Own-voice confusion: logreg

| true \ pred | joy | sadn | fear | ange | surp | disg | calm | love | neut |
|---|---|---|---|---|---|---|---|---|---|
| joy | 2 | . | . | . | 2 | . | . | . | 1 |
| sadness | . | 3 | . | 1 | . | . | . | 1 | . |
| fear | . | . | 1 | . | 2 | 1 | . | . | 1 |
| anger | . | . | . | 3 | 1 | . | . | 1 | . |
| surprise | 1 | . | . | . | 4 | . | . | . | . |
| disgust | . | . | . | . | . | 4 | 1 | . | . |
| calm | . | . | . | 1 | . | . | 4 | . | . |
| love | 2 | . | . | . | . | . | . | 3 | . |
| neutral | . | . | . | . | 3 | 1 | . | . | 1 |

## Own-voice confusion: native_lr

| true \ pred | joy | sadn | fear | ange | surp | disg | calm | love | neut |
|---|---|---|---|---|---|---|---|---|---|
| joy | 3 | . | . | . | 1 | . | . | . | 1 |
| sadness | . | 5 | . | . | . | . | . | . | . |
| fear | . | . | 2 | . | 2 | . | . | . | 1 |
| anger | . | . | . | 2 | 1 | . | . | . | 2 |
| surprise | 1 | . | . | . | 4 | . | . | . | . |
| disgust | . | . | . | . | . | 1 | . | . | 4 |
| calm | 3 | . | . | 1 | . | . | 1 | . | . |
| love | 2 | . | . | . | . | . | . | 2 | 1 |
| neutral | . | . | . | . | . | 1 | . | . | 4 |

## Reading this

- If coarse accuracy is much higher than 9-way accuracy, the probes mostly separate positive/negative/neutral and blur fine categories.
- Off-diagonal mass concentrated in one column means an 'attractor' class that absorbs unfamiliar text.
- A big gap between short and longer passages points at length-unreliable calibration, not at emotion.
