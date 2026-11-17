# Emotion Icosphere / probekit — project guide for Claude Code

Read this first, then PLAN_v2.md (what to do; supersedes docs/ROADMAP.md), docs/FINDINGS.md (what we know) and docs/CLAIMS.md (what may be claimed). Work WITH Justin: he is the owner, runs the Mac (M4 MacBook Pro, Apple Silicon, MLX), and decides direction. Propose, then do; keep him informed in short summaries.

## What this is
A public hobby/research project. It extracts linear "emotion" probe directions from a local open-weight LLM's residual stream (default `mlx-community/Llama-3.2-3B-Instruct`), scores text/tokens as z-scores against neutral text, and visualizes them live on an icosphere (FastAPI + WebSocket backend, Three.js frontend). Inspired by Anthropic's April 2026 paper "Emotion Concepts and their Function in a Large Language Model".

## Honesty rules (non-negotiable)
- Never describe probe scores as the model's "internal emotional state" or "feelings". They are linear read-outs correlated with emotion-concept text. Say "emotion-concept probe activations".
- Every accuracy claim needs: test set, n, CI, and a baseline (lexical TF-IDF, native hidden-state LR, shuffle null).
- Report negative results (sadness 0/5, short-text unreliability, writer shift). README must match docs/FINDINGS.md.
- Hand-written hard set and own_voice are TEST ONLY. Never train, tune, select, or fit thresholds on them. Never paste their contents into prompts, logs, commits, or reports beyond aggregate numbers.
- Don't invent numbers. Run the code, quote the output.

## Layout (flat module repo; run from repo root)
- `probekit_probes.py` ProbeSet, fit_probes, evaluate, null_distribution (no MLX import; shared by builder+backend)
- `probekit_estimators.py` meandiff / lda / logreg directions; `calibrate_pooled` (calibrate on HELD-OUT neutral)
- `probekit_capture.py` Extractor (causal-mask-correct manual layer loop; `--check` verifies causality)
- `probekit_corpus*.py` corpus generation (MLX v1/v2, OpenAI-compatible API writer, HF GoEmotions/dair loader)
- `adversarial_filter.py` marks `hard` passages; `probekit_hardset.py` hand-written set + own-voice template
- `probekit_eval_v2.py`, `probekit_eval_v3.py` sweeps/selection/reports; `probekit_robustness.py`; `probekit_confusion.py`; `probekit_stream_check.py`; `make_config.py`; `probekit_build_v3.py`
- `backend.py` (KV-cached live server; MLX streams are thread-bound: generate on the main thread), `frontend/index.html`
- `run_merged.sh` merged-corpus pipeline (`SKIP_EVAL=1` to resume; `DRY=1` to print)
- Tests (model-free, synthetic): test_probes_synthetic.py, test_bundle_synthetic.py, test_v3_synthetic.py, test_confusion.py, test_hf_loader.py. Run all before and after any change: `for t in test_*.py; do python $t || break; done`
- Obsolete top-level files (emotion_vectors.py, capture_hidden_state.py, score_emotions.py, config.py, geometry_prototype.py) were deleted on 2026-11-16; recover from commit e6ad62f if needed. The top-level demo `backend.py` still uses the old contrastive vectors; see the root README.

## Conventions
- Layer L = output of decoder block L (old "18" == new "17"). Mean-pool passage states, skip first 4 tokens.
- Splits are topic-disjoint (train/val/test). Neutral threshold tau is chosen on VAL only.
- Any new fold/eval that can lack a class must skip-and-report, never crash silently or silently drop (see leave_one_group_out).
- Prefer small, tested changes. Add a synthetic test for every bug fixed. Don't add heavy deps; numpy/sklearn only in core modules.
- Long jobs (model extraction, corpus generation) are run by Justin or in background; tell him the exact command and what file to send back.

## Privacy / safety
`.claude/settings.json` denies Read on hardset/own_voice data. Deny rules don't stop subprocesses (python scripts read them legitimately for aggregate scoring), so the rule above is the real control: only aggregate outputs leave the machine.

Git policy for `data/` (the repo is PUBLIC):
- DO commit and push small (<~5 MB each) text outputs: corpora (`data/corpus/*.jsonl`, synthetic/GoEmotions only), aggregate reports and configs (`report_*.md/json`, `robustness*.md/json`, `confusion_*.md`, `best_config*.json`), probe metadata.
- NEVER commit or push: `*.npz` (hidden-state caches, hundreds of MB), `*.png`, anything over ~5 MB, `data/hardset/` (hardset.jsonl, own_voice*.jsonl), or `errors_*.md` (they quote test passages). These are in `.gitignore`; keep it that way.
- Before every `git add`, run `git status --short` and check for the above. Ask Justin before any push that adds a new top-level data directory.
- The hardset/own_voice/errors files were committed to the public remote by mistake before this rule existed (commit 242e2dd and later); they are now untracked but remain in git history.

## Current state / next
See PLAN_v2.md. Done: merged pipeline resumed, FINDINGS/README updated, obsolete files deleted, Qwen second-writer corpus generated (`data/corpus/corpus_v3_qwen3.8-27b.jsonl`, 1002 passages). T0a gate on it FAILED (TF-IDF 8-way macro-F1 0.708, 95% CI 0.678-0.741 vs gate <0.7): treat it as an "easy" stratum until a harder one is generated. Next per plan v2.2: T2 (frozen human test set; counts first), T1 (layer-0 baseline), then T0; T0b (regenerate a harder stratum) only if triggered. `clean()` was fixed on 2026-11-17 to keep dialogue quotes and line breaks; older corpora were generated with the flattening version. Hand-written/own_voice data is NOT purged from history (Justin's decision); report it as non-blind.
