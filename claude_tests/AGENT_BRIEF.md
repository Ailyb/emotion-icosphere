# Agent brief: Emotion-probe toolkit (probekit)

You are helping Justin build and honestly validate linear "emotion probes" for small open-weight LLMs.
Your job is **running, generating, porting and reporting**. Interpreting what the results mean, deciding labels, and
changing how things are measured are **not** your job; those stay with Justin and his assistant (Claude). When unsure, stop and ask.

## 1. The project in six lines
- Probes = directions in a model's residual stream (one per emotion) that score how emotional a text is, token by token.
- Corpus: LLM-written passages per emotion + neutral, split TRAIN/VAL/TEST **by topic** (never mix topics across splits).
- Hand-written HARD SET (`data/hardset/hardset.jsonl`, `own_voice.jsonl`): negation, implicit emotion, emotion words in neutral text. It is the honest yardstick.
- Baselines that must be beaten: TF-IDF word model (lexical) and logistic regression on the same hidden states (ceiling).
- Earlier result: word statistics solved the first corpus (0.95), so we built harder data. Current state: probes beat words on the hard set, but are weaker than a trained classifier. Layers ~10-14 look best (validation-chosen).
- Claims are limited to what the numbers show. We do **not** claim the model "feels" anything or that probes are "validated".

## 2. Golden rules (violating any of these invalidates results)
1. **Never touch** `data/hardset/hardset.jsonl` or `data/hardset/own_voice.jsonl`. Never show their text to any generator model, never use them as few-shot examples, never train on them, never upload them anywhere (own_voice is personal writing: keep it local).
2. **Never select** layers, hyperparameters, thresholds or methods using hard-set numbers. Selection uses VAL only (the scripts already do this; do not change that).
3. **Do not edit** split definitions, metrics, selection logic, or the banned-word filter without asking first. Propose the diff in your report instead.
4. Don't hand-edit generated corpus files to "fix" labels. If passages look wrong, report counts and examples; regenerate with the scripts.
5. If any score on the hand-written set exceeds **0.90** macro-F1, assume a leak or bug first and report it before anything else.
6. Change one thing at a time. After **any** code change run all three test files (see section 4). They must still print their PASSED lines.
7. Never invent numbers. Copy figures verbatim from generated reports, with file paths.
8. Stay inside the project directory. No network access except downloading the model weights Justin named.

## 3. Directory map (all scripts run from the project root)
```
probekit_capture.py      residual-stream extraction (MLX), causal mask, `--check`
probekit_probes.py       ProbeSet, fit/score/evaluate (numpy)
probekit_estimators.py   meandiff / lda / logreg directions, held-out calibration
probekit_corpus.py       v1 corpus (obsolete, keep)       probekit_corpus_v2.py   v2 corpus via MLX
probekit_corpus_api.py   corpus via ANY OpenAI-compatible server, adds 24 topics, records `writer`
adversarial_filter.py    marks passages that word statistics find hard
probekit_hardset.py      writes the hand-written set + own-voice template
probekit_eval_v2.py      layer sweep and caches hidden states        probekit_eval_v3.py   estimator comparison, best_config.json
probekit_build_v3.py     builds deployable probes.npz                  probekit_stream_check.py  per-token check + HTML viewer
probekit_robustness.py   leave-one-style/writer-out, learning curve   backend.py   live server (not your concern yet)
test_probes_synthetic.py  test_bundle_synthetic.py  test_v3_synthetic.py   model-free tests
data/corpus/ data/hardset/ data/eval_v2/<model>/ data/eval_v3/<model>/ data/probes_v3/<model>/
```
`<model>` = the Hugging Face name with `/` replaced by `__`.

## 4. Tests (run after every code change, from the project root)
```
python test_probes_synthetic.py && python test_bundle_synthetic.py && python test_v3_synthetic.py
```
Expected last lines: `ALL SYNTHETIC TESTS PASSED`, `ALL BUNDLE TESTS PASSED`, `ALL V3 TESTS PASSED`.

## 5. Tasks (do them in order; stop after each and write RESULT.md; see section 6)

### Task A: smoke-run the pipeline (no code changes)
```
python probekit_hardset.py
python probekit_eval_v3.py --fast
```
Acceptance: both finish; `data/eval_v3/<model>/report_v3.md` exists. Report the selected method/layer and the table rows verbatim.
(This needs `data/corpus/corpus_v2_scored.jsonl` and the cached states in `data/eval_v2/<model>/` from Justin's earlier runs. If missing, stop and say so.)

### Task B: generate a second-writer corpus with the 27B model
1. Serve yourself (or the 27B) behind an OpenAI-compatible endpoint, e.g. `llama-server`, Ollama, LM Studio or `mlx_lm.server`.
2. Smoke test: `python probekit_corpus_api.py --api-base <url> --api-model <name> --writer <short-name> --topics ext --samples 1 --max-new 10 --out /tmp/smoke.jsonl`
3. Real run: `--topics all --samples 2 --out data/corpus/corpus_v3_<short-name>.jsonl` (resumable; re-run if interrupted).
4. Quality audit (read, don't edit): for each of the 9 categories print 3 random passages. Report: rejection-reason counts, passages per (category, split), and any category where more than 20% of the samples you read are clearly off-target (e.g. "calm" text that is actually anxious). Do not change prompts yourself; quote the problem.
5. Merge and filter:
```
cat data/corpus/corpus_v2.jsonl data/corpus/corpus_v3_<short-name>.jsonl > data/corpus/corpus_merged.jsonl
python adversarial_filter.py --corpus data/corpus/corpus_merged.jsonl --out data/corpus/corpus_merged_scored.jsonl
```
Acceptance: merged file has no duplicate ids; filter prints per-split hard counts; report any test cell with fewer than 6 hard passages.

### Task C: run the evaluation on the merged corpus (Justin's machine/model only)
```
python probekit_eval_v2.py --corpus data/corpus/corpus_merged_scored.jsonl
python probekit_eval_v3.py --corpus data/corpus/corpus_merged_scored.jsonl
python probekit_robustness.py --corpus data/corpus/corpus_merged_scored.jsonl
```
(These re-extract hidden states for the new corpus; that is expected.) Report headline tables and the leave-one-writer-out table verbatim.

### Task D: port to a second model (e.g. Qwen2.5-3B-Instruct, a Gemma-class model)
1. `python probekit_capture.py --model <hf-id> --check` must print PASS (masked mismatch tiny, unmasked clearly larger). If it errors, the usual cause is the attribute layout (`model.model.layers`, `embed_tokens`) or the layer call signature; make the **smallest** fix in `probekit_capture.py` only, then re-run the check and the tests.
2. Generate the corpus with `probekit_corpus_api.py` (the writer does not need to be the model being probed) or reuse the merged corpus; run `probekit_eval_v2.py --model <hf-id> ...` then `probekit_eval_v3.py --model <hf-id> ...`.
3. Report per-model: selected method/layer, plateau layers, hard-set numbers vs lexical, and whether the mid-layer pattern repeats. Do **not** compare models in prose beyond quoting the tables.

## 6. What to deliver after each task: `RESULT.md`
- Exact commands run (copy/paste), date, git commit hash (or "no git"), Python/mlx-lm versions.
- Paths of every file produced.
- Key numbers copied verbatim from reports (with the file they came from).
- Anomalies: errors, warnings, skipped rows, anything surprising. Say "none" only if true.
- Open questions for Justin. Never resolve them silently.
- A list of any code you changed, as a diff, with the test output after the change.

## 7. Stop and ask when
- a test fails and you cannot fix it with a change under ~20 lines;
- a number looks too good (see rule 5) or the opposite (an unexpected collapse);
- generated passages systematically contain emotion words, clichés, or off-target emotions;
- you think the evaluation itself should change;
- anything would require seeing, uploading, or using the hard-set or own-voice text outside the provided scripts.

## 8. Style
Short, factual, no hype. Say "the scripts report X" rather than "this proves X". If you are guessing, say so.
