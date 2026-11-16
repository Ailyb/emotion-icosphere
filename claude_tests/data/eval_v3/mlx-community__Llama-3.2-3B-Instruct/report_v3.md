# Eval v3: mlx-community/Llama-3.2-3B-Instruct

n: {'train': 488, 'val_hard': 73, 'test_hard': 73, 'test_all': 147, 'hardset': 92, 'calib_neutral': 18}. Selected **meandiff @ layer 14** by VAL-hard macro-F1 only. Config: `{'var_frac': 0.5, 'method': 'meandiff'}`, tau 1.80.
Layers within 0.03 val-F1 of the best (the plateau): **[10, 12, 14]**

## Headline (9-way macro-F1; chance ~0.11)

| | VAL-hard | TEST-hard | TEST-all | HAND-WRITTEN set |
|---|---|---|---|---|
| meandiff @ L14 | 0.798 | 0.683 | 0.758 | 0.499 |
| lda @ L14 | 0.723 | 0.712 | 0.781 | 0.565 |
| logreg @ L14 | 0.764 | 0.721 | 0.779 | 0.542 |
| native_lr (ceiling) @ L14 | 0.804 | 0.885 | 0.924 | 0.599 |
| lexical TF-IDF | 0.351 | 0.506 | 0.710 | 0.237 |

Selected probe hand-written accuracy 49/92 (95% Wilson 0.43-0.63).

## Paired bootstrap on the hand-written set (macro-F1 difference, 95% interval)

| comparison | mean diff | 95% interval | excludes 0? |
|---|---|---|---|
| selected vs lexical | +0.257 | [+0.132, +0.383] | yes |
| selected vs native LR (ceiling) | -0.100 | [-0.186, -0.010] | yes |

## By hand-written subtype (accuracy at the selected layer)

| subtype | n | selected | native LR | lexical |
|---|---|---|---|---|
| emotion_word_neutral | 8 | 0.62 | 1.00 | 0.75 |
| implicit | 24 | 0.54 | 0.67 | 0.21 |
| negation | 9 | 0.67 | 0.89 | 0.33 |
| own_voice | 45 | 0.47 | 0.49 | 0.24 |
| plain_neutral | 6 | 0.67 | 1.00 | 0.67 |

## Layer x method (VAL-hard / HAND-WRITTEN macro-F1)

| layer | meandiff | lda | logreg | native_lr |
|---|---|---|---|---|
| 6 | 0.68 / 0.38 | 0.70 / 0.47 | 0.66 / 0.47 | 0.76 / 0.48 |
| 8 | 0.68 / 0.40 | 0.71 / 0.49 | 0.74 / 0.45 | 0.79 / 0.54 |
| 10 | 0.76 / 0.44 | 0.77 / 0.49 | 0.79 / 0.54 | 0.78 / 0.58 |
| 12 | 0.75 / 0.46 | 0.74 / 0.50 | 0.79 / 0.52 | 0.79 / 0.63 |
| 14 | 0.80 / 0.50 | 0.72 / 0.57 | 0.76 / 0.54 | 0.80 / 0.60 |
| 16 | 0.73 / 0.42 | 0.72 / 0.53 | 0.74 / 0.59 | 0.81 / 0.59 |
| 18 | 0.69 / 0.41 | 0.70 / 0.49 | 0.74 / 0.60 | 0.80 / 0.57 |

## Chosen hyperparameters at the selected layer

- meandiff: `{'var_frac': 0.5, 'method': 'meandiff'}`
- lda: `{'var_frac': 0.8, 'shrinkage': 'auto', 'method': 'lda'}`
- logreg: `{'var_frac': 0.8, 'C': 0.003, 'method': 'logreg'}`
