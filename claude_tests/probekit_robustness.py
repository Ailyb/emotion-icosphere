#!/usr/bin/env python3
"""
probekit_robustness.py -- does the chosen probe survive changes of writing style, writer, and data size?

With the config from probekit_eval_v3.py (cached hidden states, no model load):
  1. leave-one-STYLE-out: train on passages in the other writing modes, test on the held-out mode
     (train rows come from TRAIN topics only; test rows from VAL+TEST topics, so topics are unseen too)
  2. leave-one-WRITER-out: only if the corpus has several writers (field `writer`, e.g. from probekit_corpus_api.py):
     train on one writer's text, test on another's. The v2 corpus counts as writer "llama-3.2-3b".
  3. learning curve: macro-F1 vs fraction of training data (stratified, repeated): tells us whether more
     passages would help or whether we are at the ceiling.
Each fold also reports the hand-written-set F1 of that fold's model, so you see how stable it is.

    python probekit_robustness.py [--reps 5]
"""
import argparse
import json
from pathlib import Path

import numpy as np

from probekit_estimators import calibrate_pooled, fit_from_config
from probekit_eval_v2 import f1_9, load_jsonl, pick_tau, probe_predict
from probekit_probes import DEFAULT_EMOTIONS, NEUTRAL


def _fit_eval(Xl, Xhl, cat, hcat, tr, cal, va, te, emotions, layer, cfg):
    ps = fit_from_config(Xl[tr], cat[tr], emotions, layer, cfg)
    calibrate_pooled(ps, Xl[cal])
    tau = pick_tau(ps, Xl[va], cat[va])
    return {"n_train": int(tr.sum()), "n_test": int(te.sum()),
            "test_f1": f1_9(cat[te], probe_predict(ps, Xl[te], tau)),
            "hardset_f1": f1_9(hcat, probe_predict(ps, Xhl, tau))}


def leave_one_group_out(Xl, Xhl, rows, hcat, emotions, layer, cfg, field, default="unknown"):
    cat = np.array([r["category"] for r in rows]); spl = np.array([r["split"] for r in rows])
    grp = np.array([r.get(field, default) for r in rows])
    res = {}
    for g in sorted(set(grp)):
        tr = (spl == "train") & (grp != g)
        te = np.isin(spl, ["val", "test"]) & (grp == g)
        cal = (spl == "val") & (cat == NEUTRAL) & (grp != g)
        va = (spl == "val") & (grp != g)                      # threshold chosen on groups the model has not been tested on
        if te.sum() < 10 or tr.sum() < 30 or cal.sum() < 5 or len(set(cat[te])) < 3:
            continue
        res[g] = _fit_eval(Xl, Xhl, cat, hcat, tr, cal, va, te, emotions, layer, cfg)
    return res


def learning_curve(Xl, Xhl, rows, hcat, emotions, layer, cfg, fracs=(0.1, 0.2, 0.4, 0.7, 1.0), reps=5, seed=0):
    cat = np.array([r["category"] for r in rows]); spl = np.array([r["split"] for r in rows])
    hard = np.array([bool(r.get("hard", True)) for r in rows])
    tr_all, va, te = spl == "train", (spl == "val") & hard, (spl == "test") & hard
    cal = (spl == "val") & (cat == NEUTRAL)
    rng = np.random.default_rng(seed); out = {}
    for f in fracs:
        vals = []
        for _ in range(reps if f < 1.0 else 1):
            keep = np.zeros(len(rows), bool)
            for c in np.unique(cat[tr_all]):
                idx = np.where(tr_all & (cat == c))[0]
                keep[rng.permutation(idx)[:max(6, int(round(f * len(idx))))]] = True
            vals.append(_fit_eval(Xl, Xhl, cat, hcat, keep, cal, va, te, emotions, layer, cfg))
        out[f] = {"n_train": int(np.mean([v["n_train"] for v in vals])),
                  "test_f1_mean": float(np.mean([v["test_f1"] for v in vals])), "test_f1_sd": float(np.std([v["test_f1"] for v in vals])),
                  "hardset_f1_mean": float(np.mean([v["hardset_f1"] for v in vals])), "hardset_f1_sd": float(np.std([v["hardset_f1"] for v in vals]))}
    return out


def render(best, style, writer, curve):
    L = [f"# Robustness ({best['method']} @ layer {best['layer']}, cfg {best['cfg']})", "",
         "## Leave-one-writing-style-out", "", "| held-out style | n test | test F1 | hand-written F1 (this fold's model) |", "|---|---|---|---|"]
    L += [f"| {g} | {v['n_test']} | {v['test_f1']:.3f} | {v['hardset_f1']:.3f} |" for g, v in style.items()]
    if style:
        L.append(f"\nMean test F1 {np.mean([v['test_f1'] for v in style.values()]):.3f}, worst {min(v['test_f1'] for v in style.values()):.3f}.")
    L += ["", "## Leave-one-writer-out", ""]
    if writer:
        L += ["| held-out writer | n test | test F1 | hand-written F1 |", "|---|---|---|---|"]
        L += [f"| {g} | {v['n_test']} | {v['test_f1']:.3f} | {v['hardset_f1']:.3f} |" for g, v in writer.items()]
    else:
        L.append("Only one writer in this corpus (or too few rows per writer). Add text from another model: see probekit_corpus_api.py.")
    L += ["", "## Learning curve (stratified subsamples of TRAIN)", "", "| fraction | n train | test-hard F1 | hand-written F1 |", "|---|---|---|---|"]
    L += [f"| {f:.1f} | {v['n_train']} | {v['test_f1_mean']:.3f} ± {v['test_f1_sd']:.3f} | {v['hardset_f1_mean']:.3f} ± {v['hardset_f1_sd']:.3f} |"
          for f, v in curve.items()]
    fr = sorted(curve)
    if len(fr) >= 3:
        gain = curve[fr[-1]]["test_f1_mean"] - curve[fr[-3]]["test_f1_mean"]
        L.append(f"\nGain from 40% -> 100% of the data: {gain:+.3f} test F1. "
                 + ("Still climbing: more/varied passages should help." if gain > 0.03 else "Flattening: more of the same data won't help much; vary the data instead."))
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Llama-3.2-3B-Instruct")
    ap.add_argument("--corpus", default="data/corpus/corpus_v2_scored.jsonl")
    ap.add_argument("--hardset", default="data/hardset/hardset.jsonl")
    ap.add_argument("--own-voice", default="data/hardset/own_voice.jsonl")
    ap.add_argument("--config", default=None)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--reps", type=int, default=5)
    a = ap.parse_args()
    from build_emotion_vectors import slug
    from probekit_eval_v3 import load_states
    best = json.loads(Path(a.config or f"data/eval_v3/{slug(a.model)}/best_config.json").read_text())
    cfg = dict(best["cfg"], method=best["method"]); layer = best["layer"]
    cache_dir = Path(a.cache_dir or f"data/eval_v2/{slug(a.model)}")
    out = Path(a.out or f"data/eval_v3/{slug(a.model)}"); out.mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(a.corpus); hrows = load_jsonl(a.hardset)
    if Path(a.own_voice).exists():
        hrows += [dict(r, id=f"own|{i}") for i, r in enumerate(load_jsonl(a.own_voice)) if r.get("text", "").strip()]
    emotions = [e for e in DEFAULT_EMOTIONS if any(r["category"] == e for r in rows)]
    Xl = load_states(rows, cache_dir / "states_corpus.npz", [layer], best["skip"], a.model)[layer]
    Xhl = load_states(hrows, cache_dir / "states_hardset.npz", [layer], best["skip"], a.model)[layer]
    hcat = np.array([r["category"] for r in hrows])
    for r in rows:
        r.setdefault("writer", "llama-3.2-3b")
    style = leave_one_group_out(Xl, Xhl, rows, hcat, emotions, layer, cfg, "style")
    writer = leave_one_group_out(Xl, Xhl, rows, hcat, emotions, layer, cfg, "writer") if len({r["writer"] for r in rows}) > 1 else {}
    curve = learning_curve(Xl, Xhl, rows, hcat, emotions, layer, cfg, reps=a.reps)
    md = render(best, style, writer, curve)
    (out / "robustness.md").write_text(md); print(md)
    (out / "robustness.json").write_text(json.dumps({"style": style, "writer": writer, "curve": {str(k): v for k, v in curve.items()}}, indent=2))


if __name__ == "__main__":
    main()
