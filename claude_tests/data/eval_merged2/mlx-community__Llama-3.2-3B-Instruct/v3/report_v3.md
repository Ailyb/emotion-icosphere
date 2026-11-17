# Eval v3: mlx-community/Llama-3.2-3B-Instruct

n: {'train': 1771, 'val_hard': 243, 'test_hard': 163, 'test_all': 327, 'hardset': 92, 'calib_neutral': 58}. Selected **lda @ layer 16** by VAL-hard macro-F1 only. Config: `{'var_frac': 0.5, 'shrinkage': 'auto', 'method': 'lda'}`, tau 1.89.
Layers within 0.03 val-F1 of the best (the plateau): **[14, 16, 18]**

## Headline (9-way macro-F1; chance ~0.11)

| | VAL-hard | TEST-hard | TEST-all | HAND-WRITTEN set |
|---|---|---|---|---|
| meandiff @ L16 | 0.640 | 0.676 | 0.746 | 0.596 |
| lda @ L16 | 0.719 | 0.807 | 0.862 | 0.665 |
| logreg @ L16 | 0.708 | 0.781 | 0.852 | 0.628 |
| native_lr (ceiling) @ L16 | 0.726 | 0.828 | 0.890 | 0.622 |
| lexical TF-IDF | 0.352 | 0.354 | 0.648 | 0.337 |

Selected probe hand-written accuracy 61/92 (95% Wilson 0.56-0.75).

## Paired bootstrap on the hand-written set (macro-F1 difference, 95% interval)

| comparison | mean diff | 95% interval | excludes 0? |
|---|---|---|---|
| selected vs lexical | +0.326 | [+0.213, +0.438] | yes |
| selected vs native LR (ceiling) | +0.046 | [-0.049, +0.137] | no |
| selected vs meandiff (same layer) | +0.070 | [-0.015, +0.159] | no |

## By hand-written subtype (accuracy at the selected layer)

| subtype | n | selected | native LR | lexical |
|---|---|---|---|---|
| emotion_word_neutral | 8 | 0.88 | 1.00 | 0.62 |
| implicit | 24 | 0.71 | 0.67 | 0.25 |
| negation | 9 | 0.89 | 0.67 | 0.33 |
| own_voice | 45 | 0.56 | 0.53 | 0.36 |
| plain_neutral | 6 | 0.67 | 0.83 | 0.83 |

## Layer x method (VAL-hard / HAND-WRITTEN macro-F1)

| layer | meandiff | lda | logreg | native_lr |
|---|---|---|---|---|
| 6 | 0.57 / 0.52 | 0.66 / 0.60 | 0.63 / 0.59 | 0.71 / 0.55 |
| 8 | 0.58 / 0.58 | 0.67 / 0.65 | 0.66 / 0.60 | 0.73 / 0.67 |
| 10 | 0.60 / 0.63 | 0.68 / 0.72 | 0.67 / 0.72 | 0.73 / 0.67 |
| 12 | 0.63 / 0.56 | 0.65 / 0.74 | 0.64 / 0.68 | 0.74 / 0.76 |
| 14 | 0.61 / 0.63 | 0.69 / 0.69 | 0.67 / 0.62 | 0.74 / 0.66 |
| 16 | 0.64 / 0.60 | 0.72 / 0.66 | 0.71 / 0.63 | 0.73 / 0.62 |
| 18 | 0.66 / 0.59 | 0.69 / 0.65 | 0.67 / 0.58 | 0.70 / 0.64 |

## Chosen hyperparameters at the selected layer

- meandiff: `{'var_frac': 0.5, 'method': 'meandiff'}`
- lda: `{'var_frac': 0.5, 'shrinkage': 'auto', 'method': 'lda'}`
- logreg: `{'var_frac': 0.5, 'C': 0.003, 'method': 'logreg'}`
