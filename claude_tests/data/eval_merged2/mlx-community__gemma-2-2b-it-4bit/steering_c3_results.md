> **T14/C3 (pre-registered f4cabfe).** In this report the type labelled 'mean-difference' / 'v1' / 'H7b' is **md-same**: mean-difference directions fitted on the merged2 TRAIN states that the LDA directions were fitted on (not the v1 corpus). H7a/H7b lines below are the H9 contrast. cos(md-same, LDA): joy 0.43, sadness 0.56, fear 0.59, anger 0.46, surprise 0.49, disgust 0.57, love 0.47.

# T12/C1 steering replication on Gemma-2-2B (layer 14; pre-registered in PLAN_v2, a2da166); 40 prompts, 20 random directions

Pilot (perplexity only): S* = 0.2 (limit 1.5); s_F = {'probe': 0.3, 'v1': 0.1} (limit 1.3, every tested direction); mean token norm 214.4. Pilot table: `pilot.json`.

## Pre-registered decisions (T12/C1)

Strengths: regime N S* = 0.2 (x mean token norm; strengths 0.05, 0.1, 0.2); regime F s_F = 0.3 (LDA) and 0.1 (mean-difference), strengths s_F/2 and s_F.

- **H7a:** LDA probe directions specifically steerable (>= 1 fluent strength, regimes N and F): ['joy', 'sadness', 'fear', 'surprise'] (4/7; claim needs >= 4) -> **claim holds**.
- **H7b:** paired Delta(mean-difference at s_F) - Delta(LDA at s_F), per emotion [95% CI over prompts]:

| emotion | difference [CI] | CI above 0 | mean-difference specific at s_F |
|---|---|---|---|
| joy | -0.125 [-0.300, +0.050] | no | yes |
| sadness | -0.100 [-0.225, +0.000] | no | yes |
| fear | -0.075 [-0.151, +0.000] | no | no |
| anger | -0.075 [-0.175, +0.000] | no | no |
| surprise | -0.125 [-0.225, -0.025] | no | no |
| disgust | -0.025 [-0.075, +0.000] | no | no |
| love | +0.000 [+0.000, +0.000] | no | no |

Emotions with the interval above 0: none (0/7; need >= 3); of those, mean-difference specifically steerable: none (need >= 2) -> **not established**.
- Descriptive replication (mean-difference specific for joy / anger): {"joy": {"N": true, "F": true}, "anger": {"N": true, "F": false}}.
- Mean-difference directions specifically steerable at any fluent strength (descriptive): ['joy', 'sadness', 'fear', 'anger'].
- Sensitivity by regime (post hoc, descriptive; specifically steerable at some strength of the regime): probe_N_only: ['joy'] (1/7); probe_F_only: ['joy', 'sadness', 'fear', 'surprise'] (4/7); v1_N_only: ['joy', 'sadness', 'fear', 'anger'] (4/7); v1_F_only: ['joy', 'sadness'] (2/7).

# T12/C1 LDA probe directions (T10 rule); 40 prompts, 20 random directions)

Delta = change in the judge's rate of the target emotion vs the unsteered text of the same prompts; * = bootstrap 95% CI over prompts excludes 0; fluent = median perplexity ratio <= 1.5; specific = fluent AND Delta above ALL random directions AND CI excludes 0.

| emotion | strength | Delta [95% CI] | random max | ppl ratio | fluent | specific | probe - v1 [CI] |
|---|---|---|---|---|---|---|---|
| joy | 0.05 | +0.050 [+0.000, +0.125] | +0.050 | 1.00 | yes | no | -0.025 [-0.075, +0.000] |
| joy | 0.1 | +0.100 [+0.025, +0.200]* | +0.075 | 1.02 | yes | **yes** | -0.150 [-0.275, -0.025] |
| joy | 0.15 | +0.200 [+0.050, +0.350]* | +0.075 | 1.05 | yes | **yes** | - |
| joy | 0.2 | +0.225 [+0.100, +0.350]* | +0.075 | 1.13 | yes | **yes** | -0.100 [-0.250, +0.050] |
| joy | 0.3 | +0.375 [+0.225, +0.525]* | +0.075 | 1.26 | yes | **yes** | - |
| sadness | 0.05 | +0.000 [-0.075, +0.075] | +0.050 | 1.01 | yes | no | -0.025 [-0.125, +0.050] |
| sadness | 0.1 | +0.050 [-0.050, +0.150] | +0.075 | 1.02 | yes | no | -0.100 [-0.225, +0.025] |
| sadness | 0.15 | +0.125 [+0.000, +0.250] | +0.075 | 1.05 | yes | no | - |
| sadness | 0.2 | +0.100 [+0.000, +0.225] | +0.075 | 1.09 | yes | no | -0.150 [-0.275, -0.025] |
| sadness | 0.3 | +0.250 [+0.100, +0.400]* | +0.100 | 1.32 | yes | **yes** | - |
| fear | 0.05 | +0.050 [+0.000, +0.125] | +0.050 | 1.00 | yes | no | +0.025 [-0.050, +0.100] |
| fear | 0.1 | +0.025 [-0.050, +0.100] | +0.050 | 1.00 | yes | no | -0.025 [-0.075, +0.000] |
| fear | 0.15 | +0.050 [-0.050, +0.150] | +0.050 | 1.04 | yes | no | - |
| fear | 0.2 | +0.100 [+0.000, +0.225] | +0.050 | 1.11 | yes | no | -0.075 [-0.200, +0.025] |
| fear | 0.3 | +0.125 [+0.025, +0.250]* | +0.050 | 1.29 | yes | **yes** | - |
| anger | 0.05 | +0.025 [+0.000, +0.075] | +0.025 | 1.00 | yes | no | +0.000 [+0.000, +0.000] |
| anger | 0.1 | +0.000 [+0.000, +0.000] | +0.025 | 0.99 | yes | no | +0.000 [+0.000, +0.000] |
| anger | 0.15 | +0.025 [+0.000, +0.075] | +0.025 | 1.03 | yes | no | - |
| anger | 0.2 | +0.025 [+0.000, +0.075] | +0.050 | 1.09 | yes | no | -0.125 [-0.225, -0.025] |
| anger | 0.3 | +0.075 [+0.000, +0.175] | +0.050 | 1.26 | yes | no | - |
| surprise | 0.05 | -0.025 [-0.075, +0.000] | +0.000 | 1.00 | yes | no | +0.025 [+0.000, +0.075] |
| surprise | 0.1 | -0.025 [-0.075, +0.000] | -0.025 | 1.01 | yes | no | -0.025 [-0.125, +0.050] |
| surprise | 0.15 | -0.025 [-0.075, +0.000] | -0.050 | 1.03 | yes | no | - |
| surprise | 0.2 | -0.025 [-0.075, +0.000] | -0.025 | 1.04 | yes | no | +0.000 [-0.075, +0.075] |
| surprise | 0.3 | +0.125 [+0.025, +0.225]* | -0.025 | 1.16 | yes | **yes** | - |
| disgust | 0.05 | +0.000 [+0.000, +0.000] | +0.000 | 1.00 | yes | no | -0.025 [-0.075, +0.000] |
| disgust | 0.1 | +0.025 [+0.000, +0.075] | +0.000 | 1.01 | yes | no | +0.000 [+0.000, +0.000] |
| disgust | 0.15 | +0.025 [+0.000, +0.075] | +0.025 | 1.04 | yes | no | - |
| disgust | 0.2 | +0.025 [+0.000, +0.075] | +0.000 | 1.09 | yes | no | -0.025 [-0.075, +0.000] |
| disgust | 0.3 | +0.050 [+0.000, +0.125] | +0.025 | 1.38 | yes | no | - |
| love | 0.05 | +0.000 [+0.000, +0.000] | +0.025 | 1.00 | yes | no | -0.025 [-0.075, +0.000] |
| love | 0.1 | +0.000 [+0.000, +0.000] | +0.025 | 1.03 | yes | no | +0.000 [+0.000, +0.000] |
| love | 0.15 | +0.025 [+0.000, +0.075] | +0.050 | 1.06 | yes | no | - |
| love | 0.2 | +0.025 [+0.000, +0.075] | +0.050 | 1.09 | yes | no | +0.000 [-0.075, +0.075] |
| love | 0.3 | +0.000 [+0.000, +0.000] | +0.025 | 1.29 | yes | no | - |

Emotions that are specifically steerable at >= 1 fluent strength: ['joy', 'sadness', 'fear', 'surprise'] (4/7; claim needs >= 4).
**Pre-registered conclusion: probe directions shift expressed emotion beyond random-direction controls.**


# T12/C1 mean-difference directions (T10 rule); 40 prompts, 20 random directions)

Delta = change in the judge's rate of the target emotion vs the unsteered text of the same prompts; * = bootstrap 95% CI over prompts excludes 0; fluent = median perplexity ratio <= 1.5; specific = fluent AND Delta above ALL random directions AND CI excludes 0.

| emotion | strength | Delta [95% CI] | random max | ppl ratio | fluent | specific | probe - v1 [CI] |
|---|---|---|---|---|---|---|---|
| joy | 0.05 | +0.075 [+0.000, +0.150] | +0.050 | 1.00 | yes | no | - |
| joy | 0.1 | +0.250 [+0.125, +0.375]* | +0.075 | 1.04 | yes | **yes** | - |
| joy | 0.2 | +0.325 [+0.175, +0.475]* | +0.075 | 1.21 | yes | **yes** | - |
| sadness | 0.05 | +0.025 [+0.000, +0.075] | +0.050 | 1.01 | yes | no | - |
| sadness | 0.1 | +0.150 [+0.050, +0.275]* | +0.075 | 1.06 | yes | **yes** | - |
| sadness | 0.2 | +0.250 [+0.125, +0.375]* | +0.075 | 1.33 | yes | **yes** | - |
| fear | 0.05 | +0.025 [-0.050, +0.100] | +0.050 | 1.00 | yes | no | - |
| fear | 0.1 | +0.050 [-0.050, +0.150] | +0.050 | 1.05 | yes | no | - |
| fear | 0.2 | +0.175 [+0.050, +0.325]* | +0.050 | 1.23 | yes | **yes** | - |
| anger | 0.05 | +0.025 [+0.000, +0.075] | +0.025 | 1.00 | yes | no | - |
| anger | 0.1 | +0.000 [+0.000, +0.000] | +0.025 | 1.03 | yes | no | - |
| anger | 0.2 | +0.150 [+0.050, +0.275]* | +0.050 | 1.36 | yes | **yes** | - |
| surprise | 0.05 | -0.050 [-0.125, +0.000] | +0.000 | 1.00 | yes | no | - |
| surprise | 0.1 | +0.000 [-0.075, +0.075] | -0.025 | 1.01 | yes | no | - |
| surprise | 0.2 | -0.025 [-0.125, +0.050] | -0.025 | 1.17 | yes | no | - |
| disgust | 0.05 | +0.025 [+0.000, +0.075] | +0.000 | 1.01 | yes | no | - |
| disgust | 0.1 | +0.025 [+0.000, +0.075] | +0.000 | 1.04 | yes | no | - |
| disgust | 0.2 | +0.050 [+0.000, +0.125] | +0.000 | 1.43 | yes | no | - |
| love | 0.05 | +0.025 [+0.000, +0.075] | +0.025 | 1.00 | yes | no | - |
| love | 0.1 | +0.000 [+0.000, +0.000] | +0.025 | 1.04 | yes | no | - |
| love | 0.2 | +0.025 [+0.000, +0.075] | +0.050 | 1.22 | yes | no | - |

Emotions that are specifically steerable at >= 1 fluent strength: ['joy', 'sadness', 'fear', 'anger'] (4/7; claim needs >= 4).
**Pre-registered conclusion: probe directions shift expressed emotion beyond random-direction controls.**


Unparsed judge labels by condition (post hoc): base@0: 8/40; probe@0.05: 44/280; probe@0.1: 45/280; probe@0.15: 38/280; probe@0.2: 32/280; probe@0.3: 26/280; random@0.05: 145/800; random@0.1: 127/800; random@0.15: 118/800; random@0.2: 112/800; random@0.3: 89/800; v1@0.05: 35/280; v1@0.1: 32/280; v1@0.2: 18/280

Judge: granite4.2:8b; unparsed labels 869/6280. Unsteered label distribution: fear 1, joy 17, neutral 8, sadness 3, surprise 3, unparsed 8
