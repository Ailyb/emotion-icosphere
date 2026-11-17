# Eval v3: mlx-community/Mistral-7B-Instruct-v0.3-4bit

n: {'train': 1771, 'val_hard': 243, 'test_hard': 163, 'test_all': 327, 'hardset': 92, 'calib_neutral': 58}. Selected **logreg @ layer 10** by VAL-hard macro-F1 only. Config: `{'var_frac': 0.5, 'C': 0.003, 'method': 'logreg'}`, tau 1.69.
Layers within 0.03 val-F1 of the best (the plateau): **[10, 12, 18]**

## Headline (9-way macro-F1; chance ~0.11)

| | VAL-hard | TEST-hard | TEST-all | HAND-WRITTEN set |
|---|---|---|---|---|
| meandiff @ L10 | 0.628 | 0.603 | 0.724 | 0.591 |
| lda @ L10 | 0.688 | 0.707 | 0.804 | 0.665 |
| logreg @ L10 | 0.705 | 0.712 | 0.811 | 0.621 |
| native_lr (ceiling) @ L10 | 0.741 | 0.848 | 0.906 | 0.640 |
| lexical TF-IDF | 0.352 | 0.354 | 0.648 | 0.337 |

Selected probe hand-written accuracy 57/92 (95% Wilson 0.52-0.71).

## Paired bootstrap on the hand-written set (macro-F1 difference, 95% interval)

| comparison | mean diff | 95% interval | excludes 0? |
|---|---|---|---|
| selected vs lexical | +0.281 | [+0.135, +0.416] | yes |
| selected vs native LR (ceiling) | -0.017 | [-0.106, +0.072] | no |
| selected vs meandiff (same layer) | +0.031 | [-0.036, +0.096] | no |

## By hand-written subtype (accuracy at the selected layer)

| subtype | n | selected | native LR | lexical |
|---|---|---|---|---|
| emotion_word_neutral | 8 | 0.62 | 0.88 | 0.62 |
| implicit | 24 | 0.67 | 0.67 | 0.25 |
| negation | 9 | 0.89 | 0.67 | 0.33 |
| own_voice | 45 | 0.56 | 0.60 | 0.36 |
| plain_neutral | 6 | 0.50 | 1.00 | 0.83 |

## Layer x method (VAL-hard / HAND-WRITTEN macro-F1)

| layer | meandiff | lda | logreg | native_lr |
|---|---|---|---|---|
| 8 | 0.59 / 0.58 | 0.66 / 0.66 | 0.65 / 0.65 | 0.72 / 0.67 |
| 10 | 0.63 / 0.59 | 0.69 / 0.66 | 0.70 / 0.62 | 0.74 / 0.64 |
| 12 | 0.65 / 0.55 | 0.70 / 0.71 | 0.69 / 0.60 | 0.73 / 0.71 |
| 14 | 0.62 / 0.60 | 0.67 / 0.66 | 0.66 / 0.67 | 0.75 / 0.67 |
| 16 | 0.63 / 0.61 | 0.64 / 0.70 | 0.66 / 0.66 | 0.73 / 0.67 |
| 18 | 0.60 / 0.62 | 0.67 / 0.70 | 0.69 / 0.64 | 0.71 / 0.65 |
| 20 | 0.58 / 0.62 | 0.66 / 0.64 | 0.66 / 0.65 | 0.72 / 0.64 |
| 22 | 0.57 / 0.58 | 0.67 / 0.70 | 0.66 / 0.66 | 0.71 / 0.63 |
| 24 | 0.56 / 0.62 | 0.67 / 0.70 | 0.67 / 0.67 | 0.69 / 0.65 |
| 26 | 0.58 / 0.70 | 0.66 / 0.68 | 0.67 / 0.59 | 0.70 / 0.66 |

## Chosen hyperparameters at the selected layer

- meandiff: `{'var_frac': 0.5, 'method': 'meandiff'}`
- lda: `{'var_frac': 0.5, 'shrinkage': 'auto', 'method': 'lda'}`
- logreg: `{'var_frac': 0.5, 'C': 0.003, 'method': 'logreg'}`
