#!/usr/bin/env python3
"""
probekit_confusion.py -- WHERE do the probes go wrong on human-written text?

For each estimator (meandiff / lda / logreg, with the hyperparameters eval_v3 chose on VAL) at the selected layer,
plus the native-classifier ceiling, report on the hand-written + own-voice passages:
  * own-voice and non-own accuracy with 95% Wilson intervals (own-voice is the writer-shift test)
  * the 9x9 confusion matrix for own-voice (rows = true, columns = predicted)
  * COARSE accuracy: positive {joy, love, calm} / negative {sadness, fear, anger, disgust} / surprise / neutral.
    Hypothesis under test: the probes capture coarse valence more reliably than fine categories.
  * accuracy by passage length (own-voice): short texts pool over very few tokens and have unreliable z-scores.
No selection happens here; it is a diagnostic. Uses cached hidden states (no model load).

    python probekit_confusion.py            # -> data/eval_v3/<model>/confusion_v3.md
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from probekit_estimators import METHODS, calibrate_pooled, fit_from_config
from probekit_eval_v2 import acc, f1_9, load_jsonl, probe_predict, wilson
from probekit_probes import DEFAULT_EMOTIONS, NEUTRAL

GROUP = {"joy": "positive", "love": "positive", "calm": "positive", "sadness": "negative", "fear": "negative",
         "anger": "negative", "disgust": "negative", "surprise": "surprise", NEUTRAL: "neutral"}
SHORT_WORDS = 12


def coarse(labels):
    return np.array([GROUP[x] for x in labels])


def confusion_md(y, p, labels):
    idx = {l: i for i, l in enumerate(labels)}
    M = np.zeros((len(labels), len(labels)), int)
    for a, b in zip(y, p):
        M[idx[a], idx[b]] += 1
    head = "| true \\ pred | " + " | ".join(l[:4] for l in labels) + " |"
    rows = [head, "|---|" + "---|" * len(labels)]
    rows += [f"| {l} | " + " | ".join(str(v) if v else "." for v in M[i]) + " |" for i, l in enumerate(labels)]
    return "\n".join(rows)


def ci_str(k, n):
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {k / max(1, n):.2f} (95% CI {lo:.2f}-{hi:.2f})"


def predictions(Xl, Xhl, rows, emotions, layer, report):
    """-> {name: predicted labels on the hand-written rows}"""
    cat = np.array([r["category"] for r in rows]); spl = np.array([r["split"] for r in rows])
    tr, cal = spl == "train", (spl == "val") & (cat == NEUTRAL)
    entry = report["per_layer"][str(layer)]
    out = {}
    for m in METHODS:
        ps = fit_from_config(Xl[tr], cat[tr], emotions, layer, entry[m]["cfg"])
        calibrate_pooled(ps, Xl[cal])
        out[m] = probe_predict(ps, Xhl, entry[m]["tau"])
    sc = StandardScaler().fit(Xl[tr])
    clf = LogisticRegression(C=entry["native_lr"]["cfg"]["C"], max_iter=3000).fit(sc.transform(Xl[tr]), cat[tr])
    out["native_lr"] = clf.predict(sc.transform(Xhl))
    return out


def report(preds, hrows, layer):
    y = np.array([r["category"] for r in hrows]); sub = np.array([r.get("subtype", "") for r in hrows])
    nwords = np.array([len(r["text"].split()) for r in hrows])
    own, rest = sub == "own_voice", sub != "own_voice"
    labels = DEFAULT_EMOTIONS + [NEUTRAL]
    L = [f"# Confusion diagnostics @ layer {layer}", "",
         f"hand-written passages: {len(y)} (own-voice {int(own.sum())}). Chance for 9-way = 0.11; "
         "own-voice intervals are wide (n=45): differences under ~0.15 between estimators are not resolvable.", "",
         "## Accuracy by estimator", "", "| estimator | own-voice | other hand-written | own-voice F1 | coarse (4-way) own-voice | coarse (4-way) other |",
         "|---|---|---|---|---|---|"]
    for name, p in preds.items():
        k1, k2 = int((y[own] == p[own]).sum()), int((y[rest] == p[rest]).sum())
        c = coarse(y) == coarse(p)
        L.append(f"| {name} | {ci_str(k1, int(own.sum()))} | {ci_str(k2, int(rest.sum()))} | {f1_9(y[own], p[own]):.3f} | "
                 f"{c[own].mean():.2f} | {c[rest].mean():.2f} |")
    L += ["", f"## Own-voice accuracy by length (short = {SHORT_WORDS} words or fewer)", "",
          "| estimator | short | longer |", "|---|---|---|"]
    sh = own & (nwords <= SHORT_WORDS); lg = own & (nwords > SHORT_WORDS)
    for name, p in preds.items():
        L.append(f"| {name} | {ci_str(int((y[sh] == p[sh]).sum()), int(sh.sum()))} | {ci_str(int((y[lg] == p[lg]).sum()), int(lg.sum()))} |")
    for name in ("meandiff", "lda", "logreg", "native_lr"):
        if name in preds:
            L += ["", f"## Own-voice confusion: {name}", "", confusion_md(y[own], preds[name][own], labels)]
    L += ["", "## Reading this", "",
          "- If coarse accuracy is much higher than 9-way accuracy, the probes mostly separate positive/negative/neutral and blur fine categories.",
          "- Off-diagonal mass concentrated in one column means an 'attractor' class that absorbs unfamiliar text.",
          "- A big gap between short and longer passages points at length-unreliable calibration, not at emotion."]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Llama-3.2-3B-Instruct")
    ap.add_argument("--corpus", default="data/corpus/corpus_v2_scored.jsonl")
    ap.add_argument("--hardset", default="data/hardset/hardset.jsonl")
    ap.add_argument("--own-voice", default="data/hardset/own_voice.jsonl")
    ap.add_argument("--eval-dir", default=None)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--layer", type=int, default=None)
    a = ap.parse_args()
    from build_emotion_vectors import slug
    from probekit_eval_v3 import load_states
    d = Path(a.eval_dir or f"data/eval_v3/{slug(a.model)}")
    best = json.loads((d / "best_config.json").read_text()); rep = json.loads((d / "report_v3.json").read_text())
    layer = a.layer or best["layer"]
    cache = Path(a.cache_dir or f"data/eval_v2/{slug(a.model)}")
    rows = load_jsonl(a.corpus); hrows = load_jsonl(a.hardset)
    if Path(a.own_voice).exists():
        hrows += [dict(r, id=f"own|{i}") for i, r in enumerate(load_jsonl(a.own_voice)) if r.get("text", "").strip()]
    emotions = [e for e in DEFAULT_EMOTIONS if any(r["category"] == e for r in rows)]
    Xl = load_states(rows, cache / "states_corpus.npz", [layer], best["skip"], a.model)[layer]
    Xhl = load_states(hrows, cache / "states_hardset.npz", [layer], best["skip"], a.model)[layer]
    md = report(predictions(Xl, Xhl, rows, emotions, layer, rep), hrows, layer)
    (d / "confusion_v3.md").write_text(md); print(md)


if __name__ == "__main__":
    main()
