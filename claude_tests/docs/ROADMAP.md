# Roadmap

> **Superseded by `claude_tests/PLAN_v2.md` (v2.1) for planning.** The items below are kept as history; follow PLAN_v2.md's task order (T0a -> T2 -> T0 -> T1 -> T3 -> T4 -> T5 -> T6 -> T7).

## Now (in order)
1. DONE (2026-11-16): merged pipeline resumed. Result: hand-written gap to native LR closed (+0.044, CI includes 0), sadness 3/5 not 0/5; writer shift NOT fixed (held-out reddit F1 0.24-0.28); learning curve flat. No ablation isolates the effect of human data.
2. DONE except deletion: FINDINGS.md and README updated to match. Still to do: delete obsolete files (ask Justin).
3. Second-writer corpus (AGENT_BRIEF.md Task B, via probekit_corpus_api.py with a larger model) -> real leave-one-writer-out.
4. Length-aware calibration + minimum-length rule (short text z-scores unreliable): calibrate sigma per length bucket on neutral text; suppress/flag display below N tokens.

## Next
5. Redesign stream check: neutral-baseline per-token z, splice test (neutral|emotional|neutral passages) for onset/offset latency; drop old onset metric.
6. Second model port (e.g. Qwen/Gemma/Llama-1B via mlx-community) to test whether layer plateau and findings generalize; `--model` already parameterized.
7. Backend integration: load probes_v3 npz (from probekit_build_v3.py) in backend.py, expose z-scores, coarse-valence mode, uncertainty/low-confidence flag; keep `/probe_report`.
8. Visualizer v2: z-scaled colors with fixed anchor (meta z_full), neutral = dim, show confidence; fix time-phrase→fear / "coffee"→love tint issues via the stream-check token report; verify icosphere face count (1280).

## Later (research-y)
9. Steering / causal tests: add scaled probe direction at layer L during generation; measure behavior change against random-direction controls. Only claim "causal" if controls fail to reproduce.
10. Monitoring prototypes: conversation-level z trajectories, threshold alerts (z>3 sustained), jailbreak/distress detection experiments with proper baselines; LLM-serving add-on only after (7).
11. Related work check before release: Anthropic emotion-concepts paper, representation engineering / repeng, EmotionScope-style projects — position as validation toolkit + negative results, not novelty of probing.
12. Packaging: pyproject, README with honest limits, FINDINGS as a short writeup/blog, release when 1–4 and 7 are done.

## Working agreement
Claude Code proposes the next task from this list, makes small tested changes, runs tests, and asks Justin to run anything needing the model for hours. Update this file as items complete.
