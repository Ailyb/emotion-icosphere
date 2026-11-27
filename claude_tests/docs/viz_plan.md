# Visualization plan (after the paper is submitted)

*Written 2026-11-22. Start only after the paper is submitted. The aim is honest visuals that show what the probes do and where they fail, not a flashy "inside the model's feelings" display.*

## Why this matters, and the risk
Strong visuals bring people to the science. Here they can also undo it. A glowing display of "emotions firing inside the model" is the overclaim the paper argues against. The visuals must look exactly as confident as the evidence, and no more. Every number on screen comes from the same code and calibration as the paper.

Current mismatch to fix first: the demo `backend.py` still uses the old contrastive vectors at layer 18, with no length recalibration (`CAPTURE_LAYER_INDEX = 18`). Do not promote the current demo next to the paper.

## Design rules (apply to all three options)
1. **Language.** Say "emotion-concept probe activation", never "the model feels" or "internal state" (CLAUDE.md honesty rules). Every view has a one-line "what this is not" note.
2. **Baselines in view.** Wherever a probe score appears, show what TF-IDF and the sentence encoder say about the same text. The paper's main finding is that the probe often does no better.
3. **Calibration in view.** Use per-length recalibrated z-scores, never raw ones (WRITEUP 2.6). Show the false-alarm threshold for that text's length. Below the minimum length, mark the score as low-confidence instead of hiding it.
4. **Neutral is the default look.** Scores below threshold render dim or grey. Under the old LLM-validation threshold the probe called 69–86% of neutral human text emotional, so the display must not read as "emotional" by default.
5. **Uncertainty is visible.** Show intervals or confidence bands where they exist. Never show a single bar with no reference.
6. **Reproducible.** Numbers come from committed code paths (`probekit_*`), each with a synthetic test. No hand-tuned display constants that change what a score means.
7. **Data.** Do not display GoEmotions/Reddit text, external test text (dair, ISEAR, DailyDialog), the hand-written set or own-voice text. Example passages are new synthetic text written for the demo, marked as such, and never used to tune anything.

## Option 1: "Read the probe honestly" explainer (first; a few days; agent-only)
**Goal:** a page that lets a reader see the paper's findings on their own text.

**What it shows**
- A text box and a set of example passages. For each passage, four rows side by side: probe (per emotion), TF-IDF, sentence encoder, trained classifier on the same states.
- Recalibrated max |z| with the 5% false-alarm threshold for that length bucket. Neutral if below.
- A length gauge: words in the passage, and the false-alarm rate of raw vs recalibrated scores at that length (from Figure 3's data).
- Three small panels linking to the paper: in-domain vs cross-domain (Figure 1), human data is the lever (Figure 2), calibration (Figure 3).

**Two modes**
- *Static (public):* precomputed scores for the bundled example passages, served as one self-contained page. No model needed.
- *Live (local):* the same page backed by a small local endpoint that runs the audited probes (`probes_v3` bundle) on Llama-3.2-3B via MLX.

**Acceptance**
- Static numbers equal a fresh run of the scoring code on the same passages (test).
- With the shipped threshold, at most about 5% of a held-out neutral synthetic set shows as "emotional" at every length bucket with n ≥ 100. Report the observed rate on the page.
- Copy reviewed against CLAIMS.md: nothing on the page claims more than the "Supported" list.

## Option 2: Icosphere v2 on the audited probes (second; 1–2 weeks; agent, with Justin on look and feel)
**Goal:** keep the project's visual identity, rebuilt on what the paper validated.

**Changes**
- **Backend:** load the `probes_v3` bundle (`probekit_build_v3.py`; layer 16 LDA for Llama, 14 for Gemma, as selected), apply per-length recalibration, and expose per emotion `{z_recal, threshold, low_confidence}` plus a coarse valence mode. Keep `/probe_report`.
- **Streaming:** first redesign the stream check (old ROADMAP item 5). Use per-token z against a neutral baseline, plus a splice test (neutral | emotional | neutral passages) to measure onset and offset latency. The old onset metric is invalid. Only stream token-level colour if the splice test shows a usable latency. Otherwise animate at the sentence level.
- **Frontend:** z-scaled colour against a fixed anchor, so neutral is dim; a confidence ring that fades for short or low-confidence text; an always-visible small "TF-IDF / encoder agree?" indicator; verify the icosphere face count (1280).
- **Model switch:** Llama-3.2-3B, Gemma-2-2B and Mistral-7B, each with its own bundle and calibration, never mixed.

**Acceptance**
- Splice-test report committed, with onset and offset latency and intervals.
- Neutral synthetic passages render dim at the same ≤ ~5% rate as Option 1.
- A short screen recording plus the reproducibility test pass before anything is posted publicly.

## Option 3: Steering lab (third; only after its prerequisites; separate effort)
**Goal:** let people see for themselves that reading is not writing.

**What it does**
- Pick a direction type (audited LDA probe, mean-difference, or random) and an emotion, set a strength in units of the token-state norm, and generate from a neutral prompt.
- Show the steered and unsteered text side by side, the perplexity ratio (fluency cost), the judge's label, and where this direction's effect falls among 20 random directions at the same strength.
- Default to the fluent range from the pilots (Llama ≤ 0.4, Gemma ≤ 0.3 of the token norm). Allow stronger settings, but mark them "not fluent".
- A standing note: "Results are model-dependent: probe directions 0/7 on Llama-3.2-3B, 4/7 at the largest fluent strength on Gemma-2-2B (a threshold pass). Random directions are shown so you can judge for yourself."

**Prerequisites (do these first)**
1. **Separate estimator from training corpus.** In T10/T12/T13 the LDA directions were fit on merged2 and the mean-difference directions on the v1 corpus, so "mean-difference vs LDA" mixes estimator and corpus. Pre-register and run a mean-difference arm fit on the same data as the LDA directions.
2. **Validate the judge on generated text.** Its gate used human comments, not steered model output.
3. Optionally, a logit-lens check per direction: does it promote the emotion's own words? If yes, the lab should say the effect may be word-level.

**Safety and scope:** local models only. Steering here changes expressed emotion in short neutral texts. It is not a tool for manipulating users, and the lab is not offered as a hosted service.

**Acceptance:** for every preset, the displayed random-direction distribution and judge labels reproduce from committed code; the prerequisites' reports are committed.

## Not in scope
Monitoring, "inner thoughts" or distress dashboards, and anything served to third-party users' conversations. The paper does not support them (WRITEUP sections 2.6 and 3).

## Order and owners
1. Option 1, static then live: agent, a few days; Justin reviews copy and look.
2. Option 2: agent builds, Justin decides look and feel; stream-check redesign first.
3. Option 3: run the prerequisites as a pre-registered experiment, then build.

## Status (2026-11-23)
- **Option 1 static: built** (`probekit_viz_static.py`, `docs/viz/static.html`, `test_viz_static.py`). 1,559 model-written synthetic passages (`data/viz/synth.jsonl`; writer granite4.2:8b; 120 neutral per length bucket, about 20 per emotion per bucket; generator pre-registered in `probekit_viz_gen.py`, with one documented neutral top-up deviation) are scored in `docs/viz/synth_summary.json`.
- **Acceptance (neutral false alarms <= ~5% per bucket, n >= 100): met on the synthetic set** for the probe (0-2% in every bucket). Caveat: model-written neutral text is easier than human neutral text; the human-text check is `length_calibration.md` (recalibrated passage false-alarm rate 6-7% in the shortest buckets, so those are marked low confidence).
- **Finding shown on the page:** the probe's threshold is conservative: it misses 66-71% of short (1-4 word) emotional passages and 19-22% of passages of 12+ words; native-state LR and the encoder have higher hit rates but no false-alarm control.
- Not done: live local mode, CLAIMS.md copy review, Options 2-3.
- **Option 1 live (local): built** (`probekit_viz_live.py`, `test_viz_live.py`). Serves on 127.0.0.1 only, single-threaded (MLX), scores typed text in memory with no storage or logging. Verified 2026-11-26: all 32 bundled passages give identical labels and recalibrated z within 0.05 of `docs/viz/static_scores.json` when sent to the live endpoint. Run: `python probekit_viz_live.py --model mlx-community/Llama-3.2-3B-Instruct --cache-dir data/eval_merged2/mlx-community__Llama-3.2-3B-Instruct --eval-dir data/eval_merged2/mlx-community__Llama-3.2-3B-Instruct/v3 --corpus data/corpus/corpus_merged2_scored.jsonl`.
