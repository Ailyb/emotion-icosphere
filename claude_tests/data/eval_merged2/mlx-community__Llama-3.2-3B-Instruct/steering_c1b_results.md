# T13/C1b fluency-matched steering on Llama-3.2-3B (layer 16; pre-registered in PLAN_v2, 85e7e53); 40 prompts, 20 random directions

Pilot (perplexity only): s_F = {'probe': 0.4, 'v1': 0.3} (grid [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], limit 1.3, every tested direction); mean token norm 16.2. T10 strengths (0.1, 0.2, 0.4) are not pooled. Pilot table: `pilot.json`.

## Pre-registered decisions (T13/C1b)

- **H8a:** LDA probe directions specifically steerable at >= 1 of the two new strengths: none (0/7; claim needs >= 4) -> **not established**.
- **H8b:** paired Delta(mean-difference at s_F) - Delta(LDA at s_F), per emotion [95% CI over prompts]:

| emotion | difference [CI] | CI above 0 | mean-difference specific at s_F |
|---|---|---|---|
| joy | +0.275 [+0.125, +0.425] | yes | yes |
| sadness | +0.150 [+0.050, +0.275] | yes | yes |
| fear | +0.025 [-0.050, +0.100] | no | no |
| anger | +0.075 [+0.000, +0.175] | no | no |
| surprise | -0.050 [-0.125, +0.000] | no | no |
| disgust | +0.000 [-0.075, +0.075] | no | no |
| love | +0.075 [+0.000, +0.175] | no | no |

Emotions with the interval above 0: ['joy', 'sadness'] (2/7; need >= 3); of those, mean-difference specific: ['joy', 'sadness'] (need >= 2) -> **not established**.
- Mean-difference directions specifically steerable at any new fluent strength (descriptive): ['joy', 'sadness'].
- Descriptive (joy / anger mean-difference, any new strength): {"joy": true, "anger": false}.

# T13/C1b LDA probe directions (T10 rule); 40 prompts, 20 random directions)

Delta = change in the judge's rate of the target emotion vs the unsteered text of the same prompts; * = bootstrap 95% CI over prompts excludes 0; fluent = median perplexity ratio <= 1.5; specific = fluent AND Delta above ALL random directions AND CI excludes 0.

| emotion | strength | Delta [95% CI] | random max | ppl ratio | fluent | specific | probe - v1 [CI] |
|---|---|---|---|---|---|---|---|
| joy | 0.2 | +0.050 [-0.075, +0.175] | +0.075 | 1.01 | yes | no | - |
| joy | 0.4 | +0.075 [-0.025, +0.200] | +0.050 | 1.12 | yes | no | - |
| sadness | 0.2 | -0.025 [-0.075, +0.000] | +0.050 | 1.02 | yes | no | - |
| sadness | 0.4 | +0.025 [-0.050, +0.100] | +0.025 | 1.14 | yes | no | - |
| fear | 0.2 | +0.025 [+0.000, +0.075] | +0.050 | 1.02 | yes | no | - |
| fear | 0.4 | +0.050 [+0.000, +0.125] | +0.050 | 1.09 | yes | no | - |
| anger | 0.2 | +0.050 [+0.000, +0.125] | +0.025 | 1.02 | yes | no | - |
| anger | 0.4 | +0.000 [+0.000, +0.000] | +0.025 | 1.11 | yes | no | - |
| surprise | 0.2 | +0.025 [+0.000, +0.075] | +0.025 | 1.00 | yes | no | - |
| surprise | 0.4 | +0.050 [+0.000, +0.125] | +0.025 | 1.12 | yes | no | - |
| disgust | 0.2 | +0.025 [+0.000, +0.075] | +0.050 | 1.01 | yes | no | - |
| disgust | 0.4 | +0.025 [+0.000, +0.075] | +0.025 | 1.13 | yes | no | - |
| love | 0.2 | -0.025 [-0.075, +0.000] | +0.025 | 1.03 | yes | no | - |
| love | 0.4 | -0.025 [-0.075, +0.000] | +0.050 | 1.14 | yes | no | - |

Emotions that are specifically steerable at >= 1 fluent strength: none (0/7; claim needs >= 4).
**Pre-registered conclusion: NOT established that probe directions shift expressed emotion beyond random controls.**


# T13/C1b mean-difference directions (T10 rule); 40 prompts, 20 random directions)

Delta = change in the judge's rate of the target emotion vs the unsteered text of the same prompts; * = bootstrap 95% CI over prompts excludes 0; fluent = median perplexity ratio <= 1.5; specific = fluent AND Delta above ALL random directions AND CI excludes 0.

| emotion | strength | Delta [95% CI] | random max | ppl ratio | fluent | specific | probe - v1 [CI] |
|---|---|---|---|---|---|---|---|
| joy | 0.15 | +0.125 [-0.025, +0.275] | +0.050 | 1.07 | yes | no | - |
| joy | 0.3 | +0.350 [+0.175, +0.525]* | +0.100 | 1.24 | yes | **yes** | - |
| sadness | 0.15 | -0.025 [-0.075, +0.000] | +0.025 | 1.06 | yes | no | - |
| sadness | 0.3 | +0.175 [+0.075, +0.300]* | +0.025 | 1.28 | yes | **yes** | - |
| fear | 0.15 | +0.050 [+0.000, +0.125] | +0.050 | 1.02 | yes | no | - |
| fear | 0.3 | +0.075 [+0.000, +0.175] | +0.075 | 1.22 | yes | no | - |
| anger | 0.15 | +0.000 [+0.000, +0.000] | +0.025 | 1.05 | yes | no | - |
| anger | 0.3 | +0.075 [+0.000, +0.175] | +0.025 | 1.19 | yes | no | - |
| surprise | 0.15 | -0.025 [-0.075, +0.000] | +0.025 | 1.03 | yes | no | - |
| surprise | 0.3 | +0.000 [-0.075, +0.075] | +0.025 | 1.17 | yes | no | - |
| disgust | 0.15 | +0.000 [+0.000, +0.000] | +0.025 | 1.03 | yes | no | - |
| disgust | 0.3 | +0.025 [+0.000, +0.075] | +0.025 | 1.18 | yes | no | - |
| love | 0.15 | -0.025 [-0.075, +0.000] | +0.025 | 1.06 | yes | no | - |
| love | 0.3 | +0.050 [+0.000, +0.125] | +0.025 | 1.24 | yes | no | - |

Emotions that are specifically steerable at >= 1 fluent strength: ['joy', 'sadness'] (2/7; claim needs >= 4).
**Pre-registered conclusion: NOT established that probe directions shift expressed emotion beyond random controls.**


Unparsed judge labels by condition (post hoc): base@0: 1/40; probe@0.2: 17/280; probe@0.4: 18/280; random@0.15: 54/800; random@0.2: 52/800; random@0.3: 61/800; random@0.4: 62/800; v1@0.15: 16/280; v1@0.3: 13/280
