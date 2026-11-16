# Eval v3: mlx-community/Llama-3.2-3B-Instruct

n: {'train': 1128, 'val_hard': 153, 'test_hard': 153, 'test_all': 307, 'hardset': 92, 'calib_neutral': 38}. Selected **lda @ layer 14** by VAL-hard macro-F1 only. Config: `{'var_frac': 0.5, 'shrinkage': 'auto', 'method': 'lda'}`, tau 1.78.
Layers within 0.03 val-F1 of the best (the plateau): **[12, 14, 16, 18]**

## Headline (9-way macro-F1; chance ~0.11)

| | VAL-hard | TEST-hard | TEST-all | HAND-WRITTEN set |
|---|---|---|---|---|
| meandiff @ L14 | 0.616 | 0.540 | 0.627 | 0.669 |
| lda @ L14 | 0.642 | 0.616 | 0.670 | 0.723 |
| logreg @ L14 | 0.625 | 0.602 | 0.627 | 0.616 |
| native_lr (ceiling) @ L14 | 0.644 | 0.669 | 0.734 | 0.680 |
| lexical TF-IDF | 0.228 | 0.279 | 0.596 | 0.227 |

Selected probe hand-written accuracy 67/92 (95% Wilson 0.63-0.81).

## Paired bootstrap on the hand-written set (macro-F1 difference, 95% interval)

| comparison | mean diff | 95% interval | excludes 0? |
|---|---|---|---|
| selected vs lexical | +0.492 | [+0.370, +0.607] | yes |
| selected vs native LR (ceiling) | +0.044 | [-0.045, +0.129] | no |
| selected vs meandiff (same layer) | +0.056 | [-0.019, +0.128] | no |

## By hand-written subtype (accuracy at the selected layer)

| subtype | n | selected | native LR | lexical |
|---|---|---|---|---|
| emotion_word_neutral | 8 | 0.88 | 1.00 | 0.25 |
| implicit | 24 | 0.62 | 0.62 | 0.12 |
| negation | 9 | 0.89 | 0.67 | 0.33 |
| own_voice | 45 | 0.71 | 0.64 | 0.31 |
| plain_neutral | 6 | 0.83 | 1.00 | 0.67 |

## Layer x method (VAL-hard / HAND-WRITTEN macro-F1)

| layer | meandiff | lda | logreg | native_lr |
|---|---|---|---|---|
| 6 | 0.51 / 0.55 | 0.58 / 0.60 | 0.56 / 0.59 | 0.58 / 0.61 |
| 8 | 0.56 / 0.59 | 0.59 / 0.67 | 0.57 / 0.63 | 0.63 / 0.65 |
| 10 | 0.55 / 0.65 | 0.58 / 0.70 | 0.58 / 0.69 | 0.60 / 0.69 |
| 12 | 0.62 / 0.62 | 0.62 / 0.65 | 0.62 / 0.66 | 0.66 / 0.69 |
| 14 | 0.62 / 0.67 | 0.64 / 0.72 | 0.62 / 0.62 | 0.64 / 0.68 |
| 16 | 0.57 / 0.71 | 0.62 / 0.71 | 0.60 / 0.67 | 0.62 / 0.64 |
| 18 | 0.59 / 0.64 | 0.62 / 0.63 | 0.58 / 0.56 | 0.61 / 0.63 |

## Chosen hyperparameters at the selected layer

- meandiff: `{'var_frac': 0.5, 'method': 'meandiff'}`
- lda: `{'var_frac': 0.5, 'shrinkage': 'auto', 'method': 'lda'}`
- logreg: `{'var_frac': 0.5, 'C': 0.03, 'method': 'logreg'}`
