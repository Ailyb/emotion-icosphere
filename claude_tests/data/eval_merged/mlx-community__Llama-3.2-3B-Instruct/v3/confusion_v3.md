# Confusion diagnostics @ layer 14

hand-written passages: 92 (own-voice 45). Chance for 9-way = 0.11; own-voice intervals are wide (n=45): differences under ~0.15 between estimators are not resolvable.

## Accuracy by estimator

| estimator | own-voice | other hand-written | own-voice F1 | coarse (4-way) own-voice | coarse (4-way) other |
|---|---|---|---|---|---|
| meandiff | 29/45 = 0.64 (95% CI 0.50-0.77) | 31/47 = 0.66 (95% CI 0.52-0.78) | 0.639 | 0.76 | 0.68 |
| lda | 32/45 = 0.71 (95% CI 0.57-0.82) | 35/47 = 0.74 (95% CI 0.60-0.85) | 0.709 | 0.78 | 0.79 |
| logreg | 29/45 = 0.64 (95% CI 0.50-0.77) | 29/47 = 0.62 (95% CI 0.47-0.74) | 0.623 | 0.73 | 0.64 |
| native_lr | 29/45 = 0.64 (95% CI 0.50-0.77) | 35/47 = 0.74 (95% CI 0.60-0.85) | 0.646 | 0.78 | 0.79 |

## Own-voice accuracy by length (short = 12 words or fewer)

| estimator | short | longer |
|---|---|---|
| meandiff | 1/1 = 1.00 (95% CI 0.21-1.00) | 28/44 = 0.64 (95% CI 0.49-0.76) |
| lda | 0/1 = 0.00 (95% CI 0.00-0.79) | 32/44 = 0.73 (95% CI 0.58-0.84) |
| logreg | 0/1 = 0.00 (95% CI 0.00-0.79) | 29/44 = 0.66 (95% CI 0.51-0.78) |
| native_lr | 0/1 = 0.00 (95% CI 0.00-0.79) | 29/44 = 0.66 (95% CI 0.51-0.78) |

## Own-voice confusion: meandiff

| true \ pred | joy | sadn | fear | ange | surp | disg | calm | love | neut |
|---|---|---|---|---|---|---|---|---|---|
| joy | 4 | . | . | . | 1 | . | . | . | . |
| sadness | . | 3 | . | 1 | . | . | . | . | 1 |
| fear | . | 1 | 3 | . | 1 | . | . | . | . |
| anger | . | . | . | 5 | . | . | . | . | . |
| surprise | 1 | . | . | . | 4 | . | . | . | . |
| disgust | . | . | . | . | . | 4 | . | . | 1 |
| calm | 1 | . | . | 1 | . | . | 3 | . | . |
| love | 2 | . | . | . | 1 | . | . | 1 | 1 |
| neutral | 1 | . | . | 1 | . | . | . | 1 | 2 |

## Own-voice confusion: lda

| true \ pred | joy | sadn | fear | ange | surp | disg | calm | love | neut |
|---|---|---|---|---|---|---|---|---|---|
| joy | 3 | . | . | . | . | . | . | 1 | 1 |
| sadness | . | 3 | . | 1 | . | . | . | . | 1 |
| fear | . | . | 3 | . | 1 | . | . | . | 1 |
| anger | . | . | . | 5 | . | . | . | . | . |
| surprise | 1 | . | . | . | 4 | . | . | . | . |
| disgust | . | . | . | . | . | 4 | . | . | 1 |
| calm | . | . | . | 1 | . | . | 4 | . | . |
| love | 1 | . | . | . | . | . | . | 1 | 3 |
| neutral | . | . | . | . | . | . | . | . | 5 |

## Own-voice confusion: logreg

| true \ pred | joy | sadn | fear | ange | surp | disg | calm | love | neut |
|---|---|---|---|---|---|---|---|---|---|
| joy | 3 | . | . | . | . | . | . | . | 2 |
| sadness | . | 3 | . | 1 | . | . | . | . | 1 |
| fear | . | . | 3 | . | 1 | . | . | . | 1 |
| anger | . | . | 1 | 4 | . | . | . | . | . |
| surprise | 1 | . | . | . | 4 | . | . | . | . |
| disgust | . | . | . | 1 | 1 | 3 | . | . | . |
| calm | . | . | . | 1 | . | . | 4 | . | . |
| love | 1 | . | . | . | . | . | . | . | 4 |
| neutral | . | . | . | . | . | . | . | . | 5 |

## Own-voice confusion: native_lr

| true \ pred | joy | sadn | fear | ange | surp | disg | calm | love | neut |
|---|---|---|---|---|---|---|---|---|---|
| joy | 2 | . | . | . | . | . | . | . | 3 |
| sadness | . | 4 | . | 1 | . | . | . | . | . |
| fear | . | 1 | 3 | . | 1 | . | . | . | . |
| anger | . | . | 1 | 3 | . | . | . | . | 1 |
| surprise | . | . | . | . | 5 | . | . | . | . |
| disgust | . | . | . | . | 1 | 3 | . | . | 1 |
| calm | 2 | . | . | . | . | . | 2 | . | 1 |
| love | 1 | . | . | . | . | . | . | 2 | 2 |
| neutral | . | . | . | . | . | . | . | . | 5 |

## Reading this

- If coarse accuracy is much higher than 9-way accuracy, the probes mostly separate positive/negative/neutral and blur fine categories.
- Off-diagonal mass concentrated in one column means an 'attractor' class that absorbs unfamiliar text.
- A big gap between short and longer passages points at length-unreliable calibration, not at emotion.
