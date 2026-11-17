# Claims checklist (PLAN_v2 standing rule; finish in T7)

Each line links to the file that holds the number. Wording rule: say "information beyond bag-of-words baselines", never "beyond topic or semantics" and never "internal state".

## Supported (Llama-3.2-3B-Instruct, one model, merged corpus)
- Probe beats a lexical TF-IDF baseline on the 92-passage hand-written set: +0.492 [+0.370, +0.607] macro-F1 (paired bootstrap). Source: `data/eval_merged/mlx-community__Llama-3.2-3B-Instruct/v3/report_v3.md`. Caveat: that set is public in git history, so this is not a blind test.
- Estimators (meandiff, lda, logreg) are not distinguishable on that set (lda - meandiff +0.056 [-0.019, +0.128]). Same file.

## Not established
- Probe vs trained classifier on the same states: +0.044 [-0.045, +0.129] on the hand-written set (CI includes 0); weaker on TEST-hard / TEST-all (no CI computed). Pre-merge it was worse (-0.100 [-0.186, -0.010]). Why it changed is not isolated (no ablation). `report_v3.md`, `docs/FINDINGS.md`.
- Beyond a layer-0 (embedding) baseline: not yet measured (T1).
- Short-text reliability: only 1 short own-voice passage; untested. `v3/confusion_v3.md`.
- Calm: appears only in LLM-written text (the human corpus has no calm), so calm results are writer-confounded.
- Generalisation across models: one model so far (T6).
- Qwen-written corpus is lexically harder than v1 but not clearly hard: TF-IDF 8-way macro-F1 0.708 [0.678, 0.741] (gate was < 0.7). Command: `python probekit_tfidf_gate.py --corpus data/corpus/corpus_v3_qwen3.8-27b.jsonl`.

## Rejected
- Any claim that the probes read the model's "internal emotional state" or feelings.
- The v1 result (acc 0.860) as evidence of anything: lexical TF-IDF scored 0.953 on that corpus. `docs/FINDINGS.md`.
- "Short-text z-scores are valid": the neutral short sentence "The sky is blue." scored sadness z +4.9 (anecdote that motivates T3).
- Writer-shift-free: leave-one-writer-out with reddit held out gives F1 0.235-0.284. `robustness_*/robustness.md` under `data/eval_merged/...`.
