# Visualization artifacts (branch `viz-plan`)

Everything here follows `docs/viz_plan.md` and the wording rules in `docs/CLAIMS.md` ("Visualization pages"). One model per run; scores are emotion-concept probe read-outs, not feelings.
Model-written synthetic text only; no third-party dataset text is shown.

| Artifact | What it is | How to run / regenerate |
|---|---|---|
| `static.html` | Option 1, static explainer: summary tables (false alarms, hits, misses per length bucket, four systems) plus 32 hand-written synthetic examples | `python probekit_viz_static.py --model mlx-community/Llama-3.2-3B-Instruct --cache-dir data/eval_merged2/mlx-community__Llama-3.2-3B-Instruct --eval-dir <cache-dir>/v3 --corpus data/corpus/corpus_merged2_scored.jsonl` |
| live explainer | Option 1, live: type text, get the same scoring | `python probekit_viz_live.py` (same arguments), open http://127.0.0.1:8765 |
| `steering_lab.html`, `lab_data.json` | Option 3, static steering lab: pre-registered results per direction type with the random-direction strip, fluency flag, example pairs | `python probekit_viz_lab.py` (needs the local, git-ignored steering generations) |
| steering lab, live | Option 3, live: one topic, one sample, plus a few random directions | `python probekit_viz_lab_live.py --model mlx-community/Llama-3.2-3B-Instruct --layer 16 --dir data/steering` (Gemma: layer 14, `data/steering_gemma` or `data/steering_gemma_md`), http://127.0.0.1:8766 |
| icosphere v2 | Option 2: `backend_v2.py` + `../../frontend/v2.html` | `python backend_v2.py` (same arguments as the explainer), http://127.0.0.1:8000. `?text=...` in the URL scores a text on load. |
| `judge_annotation_sample.csv` | 120 generated texts with the judge labels withheld; fill the last column to validate the judge against a human (T16) | edit the CSV, then ask for the comparison |

Experiments behind them (pre-registered in `docs/viz_plan.md`, 2026-11-27): T15 stream splice test (`data/eval_merged2/<slug>/t15_stream.md`) and T16 judge consistency on generated text (`data/eval_merged2/judge_gen/judge_gen.md`).

Not done / for Justin: the look and feel of v2 (the page was only checked for syntax and driven over the WebSocket, not looked at in a browser); a human check of the judge (annotation sheet); Mistral-7B pages; hosting (none: everything is localhost, by design).
