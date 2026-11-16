# Confusion diagnostics @ layer 14

hand-written passages: 92 (own-voice 45). Chance for 9-way = 0.11; own-voice intervals are wide (n=45): differences under ~0.15 between estimators are not resolvable.

## Accuracy by estimator

| estimator | own-voice | other hand-written | own-voice F1 | coarse (4-way) own-voice | coarse (4-way) other |
|---|---|---|---|---|---|
| meandiff | 21/45 = 0.47 (95% CI 0.33-0.61) | 28/47 = 0.60 (95% CI 0.45-0.72) | 0.430 | 0.64 | 0.66 |
| lda | 24/45 = 0.53 (95% CI 0.39-0.67) | 31/47 = 0.66 (95% CI 0.52-0.78) | 0.511 | 0.71 | 0.77 |
| logreg | 22/45 = 0.49 (95% CI 0.35-0.63) | 30/47 = 0.64 (95% CI 0.50-0.76) | 0.453 | 0.78 | 0.74 |
| native_lr | 22/45 = 0.49 (95% CI 0.35-0.63) | 38/47 = 0.81 (95% CI 0.67-0.90) | 0.430 | 0.60 | 0.89 |

## Own-voice accuracy by length (short = 12 words or fewer)

| estimator | short | longer |
|---|---|---|
| meandiff | 2/5 = 0.40 (95% CI 0.12-0.77) | 19/40 = 0.47 (95% CI 0.33-0.63) |
| lda | 2/5 = 0.40 (95% CI 0.12-0.77) | 22/40 = 0.55 (95% CI 0.40-0.69) |
| logreg | 1/5 = 0.20 (95% CI 0.04-0.62) | 21/40 = 0.53 (95% CI 0.37-0.67) |
| native_lr | 4/5 = 0.80 (95% CI 0.38-0.96) | 18/40 = 0.45 (95% CI 0.31-0.60) |

## Own-voice confusion: meandiff

| true \ pred | joy | sadn | fear | ange | surp | disg | calm | love | neut |
|---|---|---|---|---|---|---|---|---|---|
| joy | 4 | . | . | . | 1 | . | . | . | . |
| sadness | 2 | . | . | 1 | 1 | . | . | . | 1 |
| fear | . | . | 1 | . | 4 | . | . | . | . |
| anger | . | . | . | 5 | . | . | . | . | . |
| surprise | 1 | . | . | . | 4 | . | . | . | . |
| disgust | . | . | . | . | 2 | 2 | . | . | 1 |
| calm | 3 | . | . | 1 | . | . | 1 | . | . |
| love | 4 | . | . | . | . | . | . | 1 | . |
| neutral | . | 1 | . | . | 1 | . | . | . | 3 |

## Own-voice confusion: lda

| true \ pred | joy | sadn | fear | ange | surp | disg | calm | love | neut |
|---|---|---|---|---|---|---|---|---|---|
| joy | 4 | . | . | . | . | . | . | . | 1 |
| sadness | 1 | . | . | 1 | . | . | . | . | 3 |
| fear | . | . | 2 | . | . | 1 | . | . | 2 |
| anger | . | . | 1 | 4 | . | . | . | . | . |
| surprise | 1 | . | . | . | 3 | . | . | 1 | . |
| disgust | . | . | . | . | 1 | 4 | . | . | . |
| calm | 1 | . | . | 1 | . | . | 3 | . | . |
| love | 4 | . | . | . | . | . | . | 1 | . |
| neutral | 2 | . | . | . | . | . | . | . | 3 |

## Own-voice confusion: logreg

| true \ pred | joy | sadn | fear | ange | surp | disg | calm | love | neut |
|---|---|---|---|---|---|---|---|---|---|
| joy | 3 | . | . | . | . | . | . | 1 | 1 |
| sadness | . | . | 2 | 1 | 1 | . | . | 1 | . |
| fear | . | . | 3 | . | . | 1 | . | . | 1 |
| anger | . | . | 1 | 4 | . | . | . | . | . |
| surprise | 1 | . | . | . | 4 | . | . | . | . |
| disgust | . | . | . | . | 1 | 4 | . | . | . |
| calm | 2 | . | . | 1 | . | . | 2 | . | . |
| love | 5 | . | . | . | . | . | . | . | . |
| neutral | 2 | . | . | . | . | 1 | . | . | 2 |

## Own-voice confusion: native_lr

| true \ pred | joy | sadn | fear | ange | surp | disg | calm | love | neut |
|---|---|---|---|---|---|---|---|---|---|
| joy | 3 | . | . | . | 2 | . | . | . | . |
| sadness | 1 | . | . | 1 | 2 | . | . | . | 1 |
| fear | . | . | 2 | . | 2 | . | . | . | 1 |
| anger | . | . | . | 4 | . | . | . | . | 1 |
| surprise | 1 | . | . | . | 4 | . | . | . | . |
| disgust | . | . | . | . | 1 | 1 | . | . | 3 |
| calm | 1 | . | . | 1 | . | . | 3 | . | . |
| love | 3 | . | . | . | 1 | . | . | . | 1 |
| neutral | . | . | . | . | . | . | . | . | 5 |

## Reading this

- If coarse accuracy is much higher than 9-way accuracy, the probes mostly separate positive/negative/neutral and blur fine categories.
- Off-diagonal mass concentrated in one column means an 'attractor' class that absorbs unfamiliar text.
- A big gap between short and longer passages points at length-unreliable calibration, not at emotion.
