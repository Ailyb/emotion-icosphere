# T16 judge consistency on generated text (pre-registered 2026-11-27, commit 2cbb564)

Judges: granite4.2:8b (existing labels) vs qwen3.8:27b (fresh). Cohen's kappa over rows where both parsed; bootstrap 95% CI over rows. **Not a validation against humans.**

| set | stratum | n | agreement | kappa [95% CI] | unparsed granite / qwen |
|---|---|---|---|---|---|
| llama | all | 400 | 0.772 | 0.642 [0.582, 0.696] | 0 / 1 |
| llama | base | 40 | 0.800 | 0.685 [0.499, 0.865] | 0 / 0 |
| llama | probe_max | 120 | 0.773 | 0.646 [0.537, 0.742] | 0 / 1 |
| llama | random_max | 120 | 0.700 | 0.532 [0.423, 0.638] | 0 / 0 |
| llama | probe_other | 120 | 0.833 | 0.734 [0.630, 0.822] | 0 / 0 |
| gemma | all | 400 | 0.682 | 0.556 [0.499, 0.610] | 0 / 1 |
| gemma | base | 40 | 0.615 | 0.471 [0.308, 0.643] | 0 / 1 |
| gemma | probe_max | 120 | 0.658 | 0.530 [0.432, 0.625] | 0 / 0 |
| gemma | random_max | 120 | 0.658 | 0.540 [0.447, 0.635] | 0 / 0 |
| gemma | probe_other | 120 | 0.750 | 0.627 [0.522, 0.732] | 0 / 0 |

Synthetic set (n = 1559; intended label = the writer's request, writer was granite4.2:8b so granite may be favoured): macro-F1 vs intended: granite 0.858, qwen 0.924; granite-qwen kappa 0.908; unparsed granite 0, qwen 0.

**Decision rule (overall kappa >= 0.60 on both models AND kappa >= 0.60 in the probe-steered strata of both models): NOT MET: judge labels on steered text are not validated.** This is agreement between two models, not accuracy.
