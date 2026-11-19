# T10 steering results (pre-registered in PLAN_v2; 40 prompts, 20 random directions)

Delta = change in the judge's rate of the target emotion vs the unsteered text of the same prompts; * = bootstrap 95% CI over prompts excludes 0; fluent = median perplexity ratio <= 1.5; specific = fluent AND Delta above ALL random directions AND CI excludes 0.

| emotion | strength | Delta [95% CI] | random max | ppl ratio | fluent | specific | probe - v1 [CI] |
|---|---|---|---|---|---|---|---|
| joy | 0.1 | -0.050 [-0.150, +0.050] | +0.025 | 1.00 | yes | no | -0.050 [-0.150, +0.050] |
| joy | 0.2 | +0.000 [-0.125, +0.125] | +0.000 | 1.02 | yes | no | -0.125 [-0.250, +0.000] |
| joy | 0.4 | +0.025 [-0.075, +0.150] | +0.000 | 1.11 | yes | no | -0.350 [-0.500, -0.175] |
| sadness | 0.1 | -0.025 [-0.125, +0.050] | +0.025 | 1.01 | yes | no | +0.000 [-0.075, +0.075] |
| sadness | 0.2 | -0.050 [-0.125, +0.000] | +0.025 | 1.02 | yes | no | -0.125 [-0.250, -0.025] |
| sadness | 0.4 | +0.025 [-0.050, +0.100] | +0.025 | 1.14 | yes | no | -0.100 [-0.225, +0.000] |
| fear | 0.1 | +0.000 [+0.000, +0.000] | +0.025 | 1.00 | yes | no | +0.025 [+0.000, +0.075] |
| fear | 0.2 | +0.000 [+0.000, +0.000] | +0.025 | 1.02 | yes | no | -0.025 [-0.125, +0.050] |
| fear | 0.4 | +0.000 [-0.075, +0.075] | +0.025 | 1.11 | yes | no | -0.075 [-0.176, +0.025] |
| anger | 0.1 | +0.000 [+0.000, +0.000] | +0.025 | 1.00 | yes | no | +0.000 [+0.000, +0.000] |
| anger | 0.2 | +0.050 [+0.000, +0.125] | +0.025 | 1.04 | yes | no | +0.025 [+0.000, +0.075] |
| anger | 0.4 | +0.000 [+0.000, +0.000] | +0.025 | 1.10 | yes | no | -0.125 [-0.225, -0.025] |
| surprise | 0.1 | +0.025 [+0.000, +0.075] | +0.050 | 1.00 | yes | no | +0.025 [+0.000, +0.075] |
| surprise | 0.2 | +0.050 [+0.000, +0.125] | +0.075 | 1.01 | yes | no | +0.050 [+0.000, +0.125] |
| surprise | 0.4 | +0.075 [+0.000, +0.175] | +0.050 | 1.11 | yes | no | +0.000 [-0.100, +0.100] |
| disgust | 0.1 | +0.000 [+0.000, +0.000] | +0.025 | 1.01 | yes | no | -0.025 [-0.075, +0.000] |
| disgust | 0.2 | +0.000 [+0.000, +0.000] | +0.050 | 1.03 | yes | no | +0.000 [+0.000, +0.000] |
| disgust | 0.4 | +0.025 [+0.000, +0.075] | +0.025 | 1.12 | yes | no | -0.050 [-0.125, +0.000] |
| love | 0.1 | +0.025 [+0.000, +0.075] | +0.050 | 0.99 | yes | no | +0.000 [-0.075, +0.075] |
| love | 0.2 | +0.000 [+0.000, +0.000] | +0.050 | 1.03 | yes | no | -0.050 [-0.125, +0.000] |
| love | 0.4 | +0.000 [+0.000, +0.000] | +0.075 | 1.11 | yes | no | -0.100 [-0.200, -0.025] |

Emotions that are specifically steerable at >= 1 fluent strength: none (0/7; claim needs >= 4).
**Pre-registered conclusion: NOT established that probe directions shift expressed emotion beyond random controls.**

Judge: granite4.2:8b; unparsed labels 266/4120; see judge_validation.md for the judge's accuracy on human text.

# POST HOC (not pre-registered): the same rule applied to the v1-style mean-difference directions (PLAN_v2 T10 (ii); 40 prompts, 20 random directions)

Delta = change in the judge's rate of the target emotion vs the unsteered text of the same prompts; * = bootstrap 95% CI over prompts excludes 0; fluent = median perplexity ratio <= 1.5; specific = fluent AND Delta above ALL random directions AND CI excludes 0.

| emotion | strength | Delta [95% CI] | random max | ppl ratio | fluent | specific | probe - v1 [CI] |
|---|---|---|---|---|---|---|---|
| joy | 0.1 | +0.000 [-0.100, +0.100] | +0.025 | 1.00 | yes | no | - |
| joy | 0.2 | +0.125 [-0.025, +0.275] | +0.000 | 1.09 | yes | no | - |
| joy | 0.4 | +0.375 [+0.200, +0.525]* | +0.000 | 1.40 | yes | **yes** | - |
| sadness | 0.1 | -0.025 [-0.075, +0.000] | +0.025 | 1.02 | yes | no | - |
| sadness | 0.2 | +0.075 [+0.000, +0.175] | +0.025 | 1.09 | yes | no | - |
| sadness | 0.4 | +0.125 [+0.000, +0.250] | +0.025 | 1.68 | no | no | - |
| fear | 0.1 | -0.025 [-0.075, +0.000] | +0.025 | 1.00 | yes | no | - |
| fear | 0.2 | +0.025 [-0.050, +0.125] | +0.025 | 1.06 | yes | no | - |
| fear | 0.4 | +0.075 [-0.025, +0.175] | +0.025 | 1.42 | yes | no | - |
| anger | 0.1 | +0.000 [+0.000, +0.000] | +0.025 | 1.03 | yes | no | - |
| anger | 0.2 | +0.025 [+0.000, +0.075] | +0.025 | 1.08 | yes | no | - |
| anger | 0.4 | +0.125 [+0.025, +0.225]* | +0.025 | 1.43 | yes | **yes** | - |
| surprise | 0.1 | +0.000 [-0.075, +0.075] | +0.050 | 1.00 | yes | no | - |
| surprise | 0.2 | +0.000 [-0.075, +0.075] | +0.075 | 1.07 | yes | no | - |
| surprise | 0.4 | +0.075 [+0.000, +0.175] | +0.050 | 1.37 | yes | no | - |
| disgust | 0.1 | +0.025 [+0.000, +0.075] | +0.025 | 1.03 | yes | no | - |
| disgust | 0.2 | +0.000 [+0.000, +0.000] | +0.050 | 1.10 | yes | no | - |
| disgust | 0.4 | +0.075 [+0.000, +0.175] | +0.025 | 1.43 | yes | no | - |
| love | 0.1 | +0.025 [+0.000, +0.075] | +0.050 | 1.02 | yes | no | - |
| love | 0.2 | +0.050 [+0.000, +0.125] | +0.050 | 1.13 | yes | no | - |
| love | 0.4 | +0.100 [+0.025, +0.200]* | +0.075 | 1.57 | no | no | - |

Emotions that are specifically steerable at >= 1 fluent strength: ['joy', 'anger'] (2/7; claim needs >= 4).
**Pre-registered conclusion: NOT established that probe directions shift expressed emotion beyond random controls.**

Unparsed judge labels by condition (post hoc): base@0: 2/40; probe@0.1: 17/280; probe@0.2: 17/280; probe@0.4: 19/280; random@0.1: 56/800; random@0.2: 48/800; random@0.4: 56/800; v1@0.1: 20/280; v1@0.2: 15/280; v1@0.4: 16/280
Unsteered label distribution: fear 1, joy 15, neutral 19, sadness 2, surprise 1, unparsed 2
