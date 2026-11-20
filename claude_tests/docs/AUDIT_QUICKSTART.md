# probekit-audit quickstart

`probekit-audit` checks whether a linear emotion probe on an LLM's hidden states carries information beyond cheap baselines trained on the **same** data: TF-IDF, a bag of the model's input embeddings, and a classifier on the same hidden states, with paired bootstrap intervals and a shuffled-label null. It reads texts, labels and **cached activations**; it never loads a model.

## From install to a report: one command

```bash
pip install "git+https://github.com/Ailyb/emotion-icosphere.git#subdirectory=claude_tests" && probekit-audit --demo
```

(From a checkout: `cd claude_tests && pip install . && probekit-audit --demo`.) Needs Python 3.10+ and installs only numpy and scikit-learn. It takes a few seconds, writes `audit_demo.md` in the current directory and prints the same report.

The demo is **synthetic**: random vectors and made-up cue words, no model, no human data. It has an in-domain test set where the cue words match training and a `shifted` set where they do not. Expected pattern (this is a toy, not a result): TF-IDF is best in-domain and collapses on the shifted set, while the probe and the hidden-state classifier hold up, and the bag of embeddings is near chance. Add `--demo-dir DIR` to keep the generated files and see the input format.

## On your own data

```bash
probekit-audit --train train.jsonl --train-cache train.npz \
    --test mydata=test.jsonl=test.npz --layer 16 --exclude calm --out audit.md
```

- JSONL rows: `id`, `text`, `category`; training rows also `split` (`train` or `val`; optional `hard` restricts validation to hard passages). The class set must include `neutral`.
- Cache `.npz` (same row order as the JSONL): `ids`, `skip`, `L<layer>` (N x d mean-pooled hidden states), optional `E` (N x d mean input embeddings, for the bag-of-embeddings baseline). A `<cache>.meta.json` sidecar records provenance (`probekit_audit.write_meta`).
- The probe, the neutral threshold and the logistic-regression strength are fitted or chosen on `val` only; nothing is tuned on the test sets. Classes missing from train or test are skipped and reported, never dropped silently.
- Extracting caches from a model needs `pip install ".[mlx]"` on Apple Silicon (`probekit_capture.py`); the audit itself does not.

## How to read the report

Per test set: macro-F1 with 95% bootstrap CI and the shuffle-null 95th percentile for the probe, hidden-state LR, bag of embeddings and TF-IDF; paired differences (`probe - tfidf`, `probe - embed`, `probe - native_lr`); length strata. Say "better than baseline X on test set Y, n, CI", not "the probe reads emotion". Compare in-domain and out-of-domain separately: in our study (`docs/WRITEUP.md`) the order of probe and TF-IDF reverses between them.
