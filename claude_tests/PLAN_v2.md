# PLAN v2.1 — validation toolkit + audit

Supersedes docs/ROADMAP.md "Now/Next". Housekeeping to do first: point CLAUDE.md and ROADMAP.md at this file, commit it.

Scope: v1 = `probekit audit` + a short negative-results writeup. Out of scope for v1: monitoring, visualizer v2, steering (v1.1), more than two probe models.

Rules from CLAUDE.md still apply: hard set and own_voice are TEST ONLY; aggregates only; run all `test_*.py` before and after each change; every number needs test set, n, CI, baseline.

Owner key: **[J]** = Justin's hands (model runs, decisions), **[A]** = agent only.

**Execution order: T0a -> T2 -> T0 -> T1 -> T3 -> T4 -> T5 -> T6 -> T7.** (Task ids kept from v2 so earlier reviews still map.)

## Standing rules added in v2.1
- **Class coverage.** Any class missing from a writer is writer-confounded. The human corpus has no `calm`, so calm is LLM-only. Primary results use the 7 classes present in every source. Calm is reported separately, labelled "writer-confounded", and never counted in transfer or leave-one-writer-out means.
- **Claim wording.** Beating TF-IDF and layer 0 supports "information beyond bag-of-words baselines". It does not support "beyond topic or semantics". Use the first wording only.
- **Paired statistics.** Every comparison resamples the same test items for all systems.
- **CLAIMS.md** (start now, finish in T7): three lists — supported, not established, rejected — each line linked to a committed report.
- **Writers.** Record writer id per passage (v2 writer, qwen3.8:27b, human). Qwen is a training writer, so the second probe model (T6) must not be Qwen-family.

## T0a [A, model-free, do now] Post-hoc gate on the Qwen corpus
- Context: the pilot gate was skipped; full generation with qwen3.8:27b is running.
- Run: TF-IDF + LR on the saved passages, topic-disjoint CV, 3 split seeds; report mean macro-F1 with bootstrap CI and the top 20 features per class. Verify topics are disjoint by topic id, not random groups. Then run `probekit_corpus_audit.py` and re-run `probekit_overlap.py` on the complete file.
- Accept: TF-IDF macro-F1 CI lies below 0.7; top features are not emotion words, stock openers, or single-topic vocabulary.
- Change course if: CI overlaps or exceeds 0.7. Then keep the corpus as an "easy" stratum only (as with v1), fix the prompt, and regenerate a hard stratum before T0.

## T2 [A build, J approve] Frozen human test sets — before any T1 decision
- Step 1, counts only: GoEmotions official TEST split, rater-unanimous single-label items, mapped to the 7 common classes. Report n per class before building anything.
- Step 2: rebuild the human TRAINING corpus from the GoEmotions TRAIN split only (the current loader samples all splits at random). Drop its existing 160 "test" rows from every use.
- Step 3: `probekit_testset.py` writes the set and a manifest: dataset version, original ids, label mapping, unanimity and filter rules, dedup rule, class counts, SHA-256. Eval refuses to run if the hash changes.
- Second source, planned not fallback: one human-labeled dataset from a different domain (candidate: ISEAR; agent checks source and license first). Reported separately, never pooled. Same-source GoEmotions train -> test is in-domain; only the second source counts as cross-domain.
- Subtypes mined by rule, then frozen: `negation`; `emotion_word_neutral`; `no_emotion_lexicon_match` (not called "implicit").
- Accept: manifest committed; leakage check passes (ids and near-duplicates vs all training corpora); n per class and per subtype reported.
- Change course if: fewer than 6 of the 7 classes reach n >= 50 in GoEmotions. Then the second source becomes primary for the short classes, stated as such.
- The hand-written set is demoted to a footnote. Separate decision for Justin: it is exposed in the public repo; decide whether to remove it from history.

## T0 [J] Transfer matrix
- Goal: 3 training sources (v2 writer, Qwen, human) x test sources (each LLM writer's test split, GoEmotions frozen test, second human source), 7 classes.
- Run: `SKIP_EVAL=1 ./run_merged.sh ...` with the rebuilt human corpus and the complete Qwen corpus (agent supplies the exact command).
- Diagnostic: cosine between each class direction and the LLM-vs-human mean-difference direction; report calm separately.
- Accept: matrix with n and CI per cell in FINDINGS.md; real leave-one-writer-out over three writers.
- Change course if: LLM-trained probes score at or below TF-IDF on human text. Then writer shift is the headline and T5 is the main experiment.

## T1 [A, then J for one rerun] Baselines beyond TF-IDF
- (i) Layer-0 embedding mean-pool + LR. Needs one extraction pass.
- (ii) Passage-level control task (secondary). Each word type gets a seeded random class, matched to class priors; a passage's control label is the plurality class of its content-word types. Same splits and classifier settings as the real task. Report task and control accuracy for probe, L0 and TF-IDF, not only the difference. Sanity check: TF-IDF must beat chance on the control task, or the control is unlearnable and is dropped.
- Report four paired comparisons: TF-IDF vs L0; L0 vs probe; probe vs native LR; task vs control.
- Accept: synthetic tests where a purely lexical signal gives ~0 gain over L0; report printed with paired-bootstrap CIs.
- Change course if: probe - L0 CI includes 0 on the T2 frozen test sets. Then the writeup is a pure negative result.

## T3 [A] Length-aware calibration
- Blocker: only 38 held-out neutral passages exist; zero exceedances in 38 cannot bound the rate below ~8%.
- Inputs: >= 500 neutral human texts per length bucket (GoEmotions TRAIN neutral-unanimous for short buckets; one window per longer neutral passage, so samples stay independent).
- Report: per-emotion rate of |z| > 3, and per-passage rate of max |z| over all emotions > 3.
- Accept: upper 95% bound <= 2% per emotion and <= 5% per passage in every bucket with n >= 500.
- Change course if: a bucket lacks data or fails. Then suppress scores for that length and state the minimum length as a limit.

## T4 [A] `probekit audit` — the artifact
- Design: extraction (MLX or other backend) writes cached activations (numpy + metadata); the audit is model-agnostic and reads only the cache.
- Metadata per extraction: model id and revision, quantization, tokenizer revision, layer, pooling, prompt template, tensor shape, dataset hash, code version.
- Accept: reproduces the FINDINGS v3 table from cached states within bootstrap noise; runs on synthetic data in tests; core stays numpy/sklearn only.

## T5 [A prepare, J run] Audit published emotion-vector corpora
- Inputs: 1-2 openly licensed story corpora from other replications. Agent reports availability, license and label format before downloading.
- Accept: per corpus, TF-IDF / L0 / probe / native LR with CIs and n; classes missing from a corpus are skipped and reported.
- Change course if: TF-IDF is well below the probe there. Then drop the comparative claim; ship toolkit + own results.

## T6 [J run] Second probe model
- Choose and write down the model before looking at final Llama results: different family from Llama and from Qwen, similar size, instruction-tuned, same quantization level.
- Accept: same audit report; layer plateau as relative depth; differences with CIs.
- Change course if: conclusions flip. Then report as model-dependent.

## T7 [A draft, J edit] Writeup + release
- README and FINDINGS match the reports (check the README's hand-written numbers against FINDINGS.md now; an outside reader reported a mismatch). CLAIMS.md complete. 4-6 page note, pyproject.
- Accept: every number traces to a committed report.

## v1.1: does validation predict steering?
- Directions: (i) audited probe direction; (ii) direction fit on the lexically solvable v1 corpus; (iii) >= 20 norm-matched random directions; 3-4 strengths. Same prompts, sampling parameters and seeds across conditions.
- Evaluator, fixed before any run: a public emotion classifier that is a different model from the probe model and not trained on our corpora (agent proposes one, documents its training data and label set; classes it lacks are not evaluated). Plus perplexity under the unsteered model for breakage.
- Accept: effect of (i) lies outside the random-direction distribution at matched fluency; (i) vs (ii) with CIs.
- Wording: this shows the direction changes expressed emotion in output more than random directions do. It does not show an isolated emotion mechanism or a change in decisions.

## Deferred beyond v1.1
Behavioural steering (decisions, not expressed emotion); stream check redesign; backend/visualizer integration; monitoring.
