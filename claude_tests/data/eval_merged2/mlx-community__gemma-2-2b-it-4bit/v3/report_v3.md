# Eval v3: mlx-community/gemma-2-2b-it-4bit

n: {'train': 1771, 'val_hard': 243, 'test_hard': 163, 'test_all': 327, 'hardset': 92, 'calib_neutral': 58}. Selected **lda @ layer 14** by VAL-hard macro-F1 only. Config: `{'var_frac': 0.5, 'shrinkage': 'auto', 'method': 'lda'}`, tau 1.74.
Layers within 0.03 val-F1 of the best (the plateau): **[14, 16]**

## Headline (9-way macro-F1; chance ~0.11)

| | VAL-hard | TEST-hard | TEST-all | HAND-WRITTEN set |
|---|---|---|---|---|
| meandiff @ L14 | 0.609 | 0.613 | 0.706 | 0.546 |
| lda @ L14 | 0.720 | 0.763 | 0.844 | 0.683 |
| logreg @ L14 | 0.682 | 0.743 | 0.819 | 0.633 |
| native_lr (ceiling) @ L14 | 0.730 | 0.873 | 0.913 | 0.701 |
| lexical TF-IDF | 0.352 | 0.354 | 0.648 | 0.337 |

Selected probe hand-written accuracy 63/92 (95% Wilson 0.58-0.77).

## Paired bootstrap on the hand-written set (macro-F1 difference, 95% interval)

| comparison | mean diff | 95% interval | excludes 0? |
|---|---|---|---|
| selected vs lexical | +0.345 | [+0.208, +0.476] | yes |
| selected vs native LR (ceiling) | -0.017 | [-0.102, +0.067] | no |
| selected vs meandiff (same layer) | +0.137 | [+0.056, +0.216] | yes |

## By hand-written subtype (accuracy at the selected layer)

| subtype | n | selected | native LR | lexical |
|---|---|---|---|---|
| emotion_word_neutral | 8 | 0.88 | 1.00 | 0.62 |
| implicit | 24 | 0.71 | 0.71 | 0.25 |
| negation | 9 | 0.78 | 0.67 | 0.33 |
| own_voice | 45 | 0.62 | 0.67 | 0.36 |
| plain_neutral | 6 | 0.67 | 1.00 | 0.83 |

## Layer x method (VAL-hard / HAND-WRITTEN macro-F1)

| layer | meandiff | lda | logreg | native_lr |
|---|---|---|---|---|
| 6 | 0.46 / 0.43 | 0.60 / 0.56 | 0.63 / 0.53 | 0.69 / 0.53 |
| 8 | 0.54 / 0.54 | 0.63 / 0.61 | 0.65 / 0.58 | 0.72 / 0.59 |
| 10 | 0.56 / 0.56 | 0.67 / 0.60 | 0.67 / 0.59 | 0.74 / 0.62 |
| 12 | 0.55 / 0.56 | 0.66 / 0.61 | 0.65 / 0.57 | 0.72 / 0.66 |
| 14 | 0.61 / 0.55 | 0.72 / 0.68 | 0.68 / 0.63 | 0.73 / 0.70 |
| 16 | 0.65 / 0.61 | 0.70 / 0.66 | 0.70 / 0.61 | 0.75 / 0.65 |
| 18 | 0.63 / 0.60 | 0.68 / 0.66 | 0.66 / 0.65 | 0.71 / 0.71 |

## Chosen hyperparameters at the selected layer

- meandiff: `{'var_frac': 0.5, 'method': 'meandiff'}`
- lda: `{'var_frac': 0.5, 'shrinkage': 'auto', 'method': 'lda'}`
- logreg: `{'var_frac': 0.5, 'C': 0.003, 'method': 'logreg'}`
