# PLAN v2.2 — validation toolkit + audit

Supersedes docs/ROADMAP.md "Now/Next" (CLAUDE.md and ROADMAP.md now point here). v2.2 (2026-11-17) records the T0a result, the clean() fix, the class definition, and decisions below.

Decisions (Justin, 2026-11-17): the hand-written set and own_voice are NOT purged from git history; they are demoted to a non-blind footnote (T2). `clean()` bug fixed in `probekit_corpus.py` (dialogue quotes and line breaks are kept); existing corpora keep the old formatting and are not regenerated unless T0b runs.

Scope: v1 = `probekit audit` + a short negative-results writeup. Out of scope for v1: monitoring, visualizer v2, steering (v1.1), more than two probe models.

Rules from CLAUDE.md still apply: hard set and own_voice are TEST ONLY; aggregates only; run all `test_*.py` before and after each change; every number needs test set, n, CI, baseline.

Owner key: **[J]** = Justin's hands (model runs, decisions), **[A]** = agent only.

**Execution order: T0a (done, failed) -> T2 -> T1 -> T0 -> T0b (only if triggered) -> T3 -> T4 -> T5 -> T6 -> T7.** (Task ids kept from v2 so earlier reviews still map. T1 moves before T0 so the layer-0 baseline decides whether regenerating a hard stratum is needed.)

## Standing rules added in v2.1
- **Class coverage.** Any class missing from a writer is writer-confounded. The human corpus has no `calm`, so calm is LLM-only. **The common classes are the 7 emotions joy, sadness, fear, anger, surprise, disgust, love, plus neutral.** Primary results are 8-way (7 emotions + neutral); a 7-way neutral-excluded number is reported alongside so it stays comparable with the v1 gate. "n >= 50 per class" in T2 applies to all 8. Calm is reported separately, labelled "writer-confounded", and never counted in transfer or leave-one-writer-out means.
- **Claim wording.** Beating TF-IDF and layer 0 supports "information beyond bag-of-words baselines". It does not support "beyond topic or semantics". Use the first wording only.
- **Paired statistics.** Every comparison resamples the same test items for all systems.
- **CLAIMS.md** (start now, finish in T7): three lists — supported, not established, rejected — each line linked to a committed report.
- **Writers.** Record writer id per passage (v2 writer, qwen3.8:27b, human). Qwen is a training writer, so the second probe model (T6) must not be Qwen-family.

## T0a [A, model-free] Post-hoc gate on the Qwen corpus — DONE 2026-11-17: FAILED (marginally)
- Result: 1002 passages, 56 topics, no topic in two splits. TF-IDF 8-way (neutral excluded) macro-F1 0.708, 95% CI [0.678, 0.741] (v2: 0.616 [0.580, 0.651]; v1: 0.953); 9-way 0.732 [0.706, 0.761]. No top feature matches the emotion-word filter; features are behaviour vocabulary (`checking`, `clock`, `smell`, `sip`) and stock openers ("did you" 4-8% in several classes, "i had" 11% of surprise). Report: `data/corpus/corpus_v3_qwen3.8-27b_T0a_report.md`; command `python probekit_tfidf_gate.py --corpus ...`.
- Resolution: do not regenerate yet. The hard-passage filter (`adversarial_filter.py`) keeps the rows TF-IDF finds hardest, so gating TF-IDF on hard rows would pass by construction and proves nothing. The non-circular test is T1 (layer-0 and native-LR baselines on the hard rows). Treat the corpus as "easy + hard" strata, as the eval already does, and trigger T0b only if T1 or T2 says so.
- Original plan text follows.
- Context: the pilot gate was skipped; full generation with qwen3.8:27b was run before the gate.
- Run: TF-IDF + LR on the saved passages, topic-disjoint CV, 3 split seeds; report mean macro-F1 with bootstrap CI and the top 20 features per class. Verify topics are disjoint by topic id, not random groups. Then run `probekit_corpus_audit.py` and re-run `probekit_overlap.py` on the complete file.
- Accept: TF-IDF macro-F1 CI lies below 0.7; top features are not emotion words, stock openers, or single-topic vocabulary.
- Change course if: CI overlaps or exceeds 0.7. Then keep the corpus as an "easy" stratum only (as with v1), fix the prompt, and regenerate a hard stratum before T0.

## T2 [A build, J approve] Frozen human test sets — before any T1 decision
- Step 1, counts only: GoEmotions official TEST split, rater-unanimous single-label items, mapped to the 7 common emotions plus neutral. Report n per class before building anything.
  - **Step 1 DONE 2026-11-17** (`docs/T2_step1_counts.md`, `probekit_testset_counts.py`; 5427 official test items, per-rater CSVs from the Google release). Strict unanimity (all raters, one label, none flagged unclear): joy 10, sadness 12, fear 8, anger 14, surprise 9, disgust 8, love 29, neutral 372. Shared-label rule: 16-32 per emotion except love 78. Dataset-simplified single label (what the loader uses): joy 93, sadness 102, fear 65, anger 131, surprise 87, disgust 76, love 160, neutral 1606 (all 8 >= 50); with the 10-80 word filter fear drops to 41.
  - **The "change course" trigger fired for unanimity** (0 of 7 emotions reach n >= 50). Proposal for Justin to approve: freeze GoEmotions test using the dataset's single-label rule WITHOUT the length filter (n >= 65 per emotion), report length strata and per-item rater agreement as a covariate, state that single-label means rater-majority agreement and is noisier than unanimity; add the second-domain source (ISEAR) as planned for a cleaner check. Neutral is capped to a class-balanced sample for evaluation so it does not dominate macro-F1.
- Step 2: rebuild the human TRAINING corpus from the GoEmotions TRAIN split only (the current loader samples all splits at random). Drop its existing 160 "test" rows from every use.
- Step 3: `probekit_testset.py` writes the set and a manifest: dataset version, original ids, label mapping, unanimity and filter rules, dedup rule, class counts, SHA-256. Eval refuses to run if the hash changes.
- Second source, planned not fallback: one human-labeled dataset from a different domain (candidate: ISEAR; agent checks source and license first). Reported separately, never pooled. Same-source GoEmotions train -> test is in-domain; only the second source counts as cross-domain.
- Subtypes mined by rule, then frozen: `negation`; `emotion_word_neutral`; `no_emotion_lexicon_match` (not called "implicit").
- Accept: manifest committed; leakage check passes (ids and near-duplicates vs all training corpora); n per class and per subtype reported.
- Change course if: fewer than 6 of the 7 classes reach n >= 50 in GoEmotions. Then the second source becomes primary for the short classes, stated as such.
- The hand-written set is demoted to a footnote and reported as non-blind (it is in the public repo history; decision: no purge).
- Step 0: the GoEmotions official TEST split already contributed 160 rows to the current merged corpus's test split; the leakage check must confirm none of the frozen-set ids appear in any training corpus.

## T0 [J] Transfer matrix
- Goal: 3 training sources (v2 writer, Qwen, human) x test sources (each LLM writer's test split, GoEmotions frozen test, second human source), 8-way (7 emotions + neutral; calm separate, see class-coverage rule). Existing Qwen and v2 passages have the old flattened formatting; both writers share it, so it is not a writer confound, but note it in the matrix.
- Run: `SKIP_EVAL=1 ./run_merged.sh ...` with the rebuilt human corpus and the complete Qwen corpus (agent supplies the exact command).
- Diagnostic: cosine between each class direction and the LLM-vs-human mean-difference direction; report calm separately.
- Accept: matrix with n and CI per cell in FINDINGS.md; real leave-one-writer-out over three writers.
- Change course if: LLM-trained probes score at or below TF-IDF on human text. Then writer shift is the headline and T5 is the main experiment.

## T0b [A prepare, J run] Regenerate a harder, correctly formatted stratum — CONDITIONAL
- Trigger (any one): T1 shows probe - L0 CI includes 0 on the hard rows; the hard stratum has fewer than 6 passages in some test cell; or T0 shows LLM-trained probes are weak on human text and a more varied LLM corpus is the obvious lever.
- Change: use the fixed `clean()` (dialogue quotes and line breaks kept); revise the prompt to ban the stock openers measured in T0a ("did you ...", "i had ...", "i kept ..."), vary narrator and register, and drop the "feeling shows only in what they do" frame for half the passages so behaviour vocabulary is not the only cue. Same writer (qwen3.8:27b, `--no-think`) so the change is attributable. Regenerate Qwen only; v2 stays as is and is reported as old-format.
- Gate before the full run: pilot of ~25 passages per class through `probekit_tfidf_gate.py`; accept only if the upper CI is below 0.7 on the pilot; the full run starts only after Justin approves the pilot.
- Accept: audit report with format adherence (dialogue passages start and end with a quote; text-message passages have speaker lines); T0a-style report.

## T1 [A, then J for one rerun] Baselines beyond TF-IDF
- (i) Layer-0 embedding mean-pool + LR. Needs one extraction pass.
  - **(i) DONE 2026-11-17** (`probekit_l0_baseline.py`, `data/eval_merged/<model>/l0_baseline.md`): probe (lda@14) - bag-of-embeddings: TEST-all +0.137 [+0.067, +0.206], TEST-hard +0.210 [+0.114, +0.305], hand-written +0.451 [+0.331, +0.572]; all exclude 0, so the change-course trigger did NOT fire on the current sets. Re-run on the T2 frozen sets once built (that is the plan's decision test). Probe vs native LR on TEST-all: -0.065 [-0.105, -0.026].
- (ii) Passage-level control task (secondary). Each word type gets a seeded random class, matched to class priors; a passage's control label is the plurality class of its content-word types. Same splits and classifier settings as the real task. Report task and control accuracy for probe, L0 and TF-IDF, not only the difference. Sanity check: TF-IDF must beat chance on the control task, or the control is unlearnable and is dropped.
- Report four paired comparisons: TF-IDF vs L0; L0 vs probe; probe vs native LR; task vs control.
- Accept: synthetic tests where a purely lexical signal gives ~0 gain over L0; report printed with paired-bootstrap CIs.
- Change course if: probe - L0 CI includes 0 on the T2 frozen test sets. Then the writeup is a pure negative result.

## T3 [A] Length-aware calibration
- Blocker: only 38 held-out neutral passages exist; zero exceedances in 38 cannot bound the rate below ~8%.
- Inputs: >= 500 neutral human texts per length bucket (GoEmotions TRAIN neutral-unanimous for short buckets; one window per longer neutral passage, so samples stay independent).
- Disjointness: calibration texts must have ids disjoint from everything used to fit probes, from the neutral reference set that defines the z-scores, and from the T2 frozen test sets. Draw them from a GoEmotions TRAIN subset reserved up front (record the ids in a manifest) and exclude that subset when rebuilding the human training corpus in T2 step 2.
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
