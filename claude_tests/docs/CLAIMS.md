# Claims checklist (PLAN_v2 standing rule; finish in T7)

Each line links to the file that holds the number. Wording rule: say "information beyond bag-of-words baselines", never "beyond topic or semantics" and never "internal state".

## Supported (Llama-3.2-3B-Instruct, one model)
- Frozen human test set (GoEmotions official test, n=840, 8-way, hash-verified; `data/eval_merged2/mlx-community__Llama-3.2-3B-Instruct/frozen_eval.md`): probe beats a bag-of-embeddings baseline, +0.107 [+0.065, +0.148]; and beats block-0 states, +0.078 [+0.039, +0.116]. Wording: "information beyond a bag-of-embeddings baseline".
- Cross-writer transfer: probes trained only on LLM-written text score 0.305 [0.272, 0.338] on the frozen human set (shuffle-null p95 0.137, TF-IDF 0.146; probe − TF-IDF +0.159 [+0.122, +0.197]). Low in absolute terms. Same file.
- More human training data helps: native LR k=378 vs k=80 per class, +0.069 [+0.039, +0.099] (data-limited, not saturated). Same file.
- Once human training text is present, LLM-written training text adds no measurable gain (human-only vs LLM+human curves overlap). Same file.
- Probe beats a lexical TF-IDF baseline on the 92-passage hand-written set: +0.492 [+0.370, +0.607] macro-F1 (paired bootstrap). Source: `data/eval_merged/mlx-community__Llama-3.2-3B-Instruct/v3/report_v3.md`. Caveat: that set is public in git history, so this is not a blind test.
- Estimators (meandiff, lda, logreg) are not distinguishable on that set (lda - meandiff +0.056 [-0.019, +0.128]). Same file.
- Probe beats a bag-of-embeddings (mean input-token-embedding) baseline: TEST-all +0.137 [+0.067, +0.206] (n=307), TEST-hard +0.210 [+0.114, +0.305] (n=153), hand-written +0.451 [+0.331, +0.572] (n=92, non-blind); also beats block-0 states. Source: `data/eval_merged/mlx-community__Llama-3.2-3B-Instruct/l0_baseline.md`. Interim: the plan's decision test is on the T2 frozen sets, not built yet. Wording: "beyond bag-of-words baselines" only.

## Not established
- That the probe carries information beyond TF-IDF on human text: it does NOT on the frozen set; TF-IDF is better, −0.088 [−0.128, −0.046] (k=378: −0.090 [−0.126, −0.053]), and adding hidden states to TF-IDF gives +0.018 [−0.018, +0.053] (fusion, post hoc). The earlier "probe beats TF-IDF" result is on LLM-written and hand-written (non-blind) text only.
- Probe vs trained classifier on the same states: +0.044 [-0.045, +0.129] on the hand-written set (CI includes 0); weaker on TEST-all (-0.065 [-0.105, -0.026]) and TEST-hard (-0.055 [-0.117, +0.005]). Pre-merge it was worse (-0.100 [-0.186, -0.010]). Why it changed is not isolated (no ablation). `report_v3.md`, `docs/FINDINGS.md`.
- Short-text reliability: only 1 short own-voice passage; untested. `v3/confusion_v3.md`.
- Calm: appears only in LLM-written text (the human corpus has no calm), so calm results are writer-confounded.
- Generalisation across models: one model so far (T6).
- Qwen-written corpus is lexically harder than v1 but not clearly hard: TF-IDF 8-way macro-F1 0.708 [0.678, 0.741] (gate was < 0.7). Command: `python probekit_tfidf_gate.py --corpus data/corpus/corpus_v3_qwen3.8-27b.jsonl`.

## Rejected
- "The probe beats a lexical baseline on human text" (frozen set, equal training data: it does not).
- "LLM-written corpora are sufficient for probes that work on human text" (0.305 with no human rows vs 0.51+ with them).
- Any claim that the probes read the model's "internal emotional state" or feelings.
- The v1 result (acc 0.860) as evidence of anything: lexical TF-IDF scored 0.953 on that corpus. `docs/FINDINGS.md`.
- "Short-text z-scores are valid": the neutral short sentence "The sky is blue." scored sadness z +4.9 (anecdote that motivates T3).
- Writer-shift-free: leave-one-writer-out with reddit held out gives F1 0.235-0.284. `robustness_*/robustness.md` under `data/eval_merged/...`.
