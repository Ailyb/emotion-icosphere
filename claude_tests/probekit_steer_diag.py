#!/usr/bin/env python3
"""
probekit_steer_diag.py -- POST HOC diagnostic requested in docs/REVIEW_v3.md A3.3 (not pre-registered).
Norm-matched is not effect-matched: how much does one unit of each steering direction move the representation?
From cached mean-pooled layer-16 states of the merged2 TRAIN rows (Llama-3.2-3B), for each emotion e and each direction type
(audited LDA probe, v1-style mean difference, 20 random unit directions):
  - cosine with the class mean-difference axis m_e = mean(states of e) - mean(states of neutral): the shift of the projection onto that axis per unit norm;
  - sd of the pooled states along the direction (a direction in a low-variance subspace is one the model's own states rarely use);
  - shift per unit norm in units of that sd (shift / sd).
    python probekit_steer_diag.py --out data/eval_merged2/mlx-community__Llama-3.2-3B-Instruct/steering_diagnostic.md
"""
import argparse
import json
from pathlib import Path

import numpy as np


def diagnose(X, y, dirs, emotions):
    """X (N,H) pooled states, y labels, dirs {type: (K,H) unit rows aligned with `emotions` (random: any K)}. Returns {type: {emotion: (cos, sd, shift_per_sd)}}."""
    neu = X[y == "neutral"].mean(0); out = {}
    for t, D in dirs.items():
        out[t] = {}
        for i, e in enumerate(emotions):
            m = X[y == e].mean(0) - neu; m = m / np.linalg.norm(m)
            rows = D if t == "random" else D[i:i + 1]
            cs = [float(d @ m / np.linalg.norm(d)) for d in rows]; sd = [float((X @ (d / np.linalg.norm(d))).std()) for d in rows]
            out[t][e] = (float(np.mean(cs)), float(np.mean(sd)), float(np.mean([c / s for c, s in zip(cs, sd)])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Llama-3.2-3B-Instruct")
    ap.add_argument("--cache-dir", default="data/eval_merged2/mlx-community__Llama-3.2-3B-Instruct")
    ap.add_argument("--corpus", default="data/corpus/corpus_merged2_scored.jsonl"); ap.add_argument("--directions", default="data/steering/directions.npz")
    ap.add_argument("--layer", type=int, default=16); ap.add_argument("--skip", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from probekit_eval_v2 import load_jsonl
    from probekit_eval_v3 import load_states
    rows = load_jsonl(a.corpus); z = np.load(a.directions, allow_pickle=True)
    X = load_states(rows, Path(a.cache_dir) / "states_corpus.npz", [a.layer], a.skip, a.model)[a.layer]
    cat = np.array([r["category"] for r in rows]); spl = np.array([r["split"] for r in rows]); tr = (spl == "train") & (cat != "calm")
    emotions = [str(e) for e in z["emotions"]]
    res = diagnose(X[tr], cat[tr], {"probe (LDA)": z["probe"], "v1 mean-difference": z["v1"], "random": z["random"]}, emotions)
    L = ["# Steering diagnostic: effect per unit norm (POST HOC; not pre-registered; REVIEW_v3 A3.3)", "",
         f"{a.model}, layer {a.layer}, mean-pooled states of merged2 TRAIN rows (n={int(tr.sum())}, calm excluded). m_e = mean(e) - mean(neutral), unit length. "
         "cos = cosine of the unit steering direction with m_e (shift of the projection onto m_e per unit norm); sd = standard deviation of the pooled states along the direction; "
         "shift/sd = cos / sd. Random = mean over 20 random unit directions.", "",
         "| emotion | cos probe | cos v1 | cos random | sd probe | sd v1 | sd random | shift/sd probe | shift/sd v1 | shift/sd random |", "|---|---|---|---|---|---|---|---|---|---|"]
    for e in emotions:
        p, v, r = res["probe (LDA)"][e], res["v1 mean-difference"][e], res["random"][e]
        L.append(f"| {e} | {p[0]:+.3f} | {v[0]:+.3f} | {r[0]:+.3f} | {p[1]:.2f} | {v[1]:.2f} | {r[1]:.2f} | {p[2]:+.3f} | {v[2]:+.3f} | {r[2]:+.3f} |")
    mean = lambda t, k: float(np.mean([res[t][e][k] for e in emotions]))
    L += ["", f"Means over emotions: cos probe {mean('probe (LDA)', 0):+.3f}, v1 {mean('v1 mean-difference', 0):+.3f}, random {mean('random', 0):+.3f}; "
              f"sd probe {mean('probe (LDA)', 1):.2f}, v1 {mean('v1 mean-difference', 1):.2f}, random {mean('random', 1):.2f}."]
    Path(a.out).write_text("\n".join(L) + "\n"); Path(a.out).with_suffix(".json").write_text(json.dumps(res, indent=1)); print("\n".join(L))


if __name__ == "__main__":
    main()
