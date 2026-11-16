# Eval v2: mlx-community/Llama-3.2-3B-Instruct

n: {'train': 488, 'val': 73, 'test_hard': 73, 'test_all': 147, 'hardset': 47} (train on `all`); selected layer **10** by VAL-hard probe macro-F1 only.

## Headline (9-way macro-F1; chance ~0.11)

| | VAL-hard | TEST-hard | TEST-all | HAND-WRITTEN hard set | word-shuffled TEST-hard |
|---|---|---|---|---|---|
| probe @ L10 | 0.689 | 0.617 | 0.685 | **0.449** | 0.185 |
| hidden-state LR @ L10 | 0.785 | 0.875 | 0.919 | **0.720** | 0.282 |
| lexical TF-IDF | 0.351 | 0.506 | 0.710 | **0.235** | (same by construction) |

Hard-set probe accuracy 25/47 (95% Wilson CI 0.39-0.67). Label-shuffle null (8-way, test-hard): mean 0.099, p95 0.178.

## By hard-set subtype (accuracy at the selected layer)

| subtype | n | probe | hidden-LR | lexical |
|---|---|---|---|---|
| emotion_word_neutral | 8 | 0.62 | 0.88 | 0.75 |
| implicit | 24 | 0.46 | 0.62 | 0.21 |
| negation | 9 | 0.56 | 0.89 | 0.33 |
| plain_neutral | 6 | 0.67 | 1.00 | 0.67 |

## Layer curve (macro-F1)

| layer | probe val | probe test-hard | probe HARD SET | hidden-LR HARD SET | probe shuffled |
|---|---|---|---|---|---|
| 0 | 0.410 | 0.426 | 0.115 | 0.195 | 0.302 |
| 1 | 0.437 | 0.506 | 0.169 | 0.200 | 0.209 |
| 2 | 0.497 | 0.422 | 0.213 | 0.316 | 0.188 |
| 3 | 0.606 | 0.503 | 0.231 | 0.257 | 0.207 |
| 4 | 0.586 | 0.575 | 0.227 | 0.335 | 0.147 |
| 6 | 0.662 | 0.606 | 0.311 | 0.588 | 0.151 |
| 8 | 0.676 | 0.592 | 0.348 | 0.683 | 0.152 |
| 10 | 0.689 | 0.617 | 0.449 | 0.720 | 0.185 |
| 12 | 0.678 | 0.599 | 0.507 | 0.775 | 0.192 |
| 14 | 0.657 | 0.601 | 0.462 | 0.738 | 0.191 |
| 16 | 0.658 | 0.619 | 0.448 | 0.683 | 0.331 |
| 18 | 0.663 | 0.573 | 0.454 | 0.664 | 0.301 |
| 20 | 0.669 | 0.561 | 0.428 | 0.627 | 0.279 |
| 22 | 0.638 | 0.587 | 0.460 | 0.570 | 0.362 |
| 24 | 0.643 | 0.608 | 0.440 | 0.508 | 0.279 |

## Reading this (mechanical, not a proof)

- Probe minus lexical on the hand-written set: **+0.214** macro-F1 at the layer chosen WITHOUT looking at it. With n=47, differences under ~0.10 are within noise.
- Layer with the best hand-written-set probe score: 12 (for inspection only; picking it would be test-set selection).
- Shuffle drop for the probe: +0.432. A large drop means the probe uses word order/context; ~0 means it behaves like a bag of words (not proof either way: a concept could also survive shuffling).
- Weak spots to check first: `negation` and `emotion_word_neutral` rows. Lexical-style shortcuts fail there.
- **Consistent with the probes carrying information beyond vocabulary.** Still needs: other writers, other models, steering tests.
