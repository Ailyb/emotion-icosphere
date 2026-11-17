#!/usr/bin/env python3
"""
probekit_l0_baseline.py -- PLAN_v2 T1 (i): do the probes carry anything beyond a bag-of-embeddings baseline?

Representations (all mean-pooled over tokens [skip:], same as the probes):
  embed   mean of the INPUT token embeddings (before any decoder block): purely lexical, no context  [new extraction]
  block0  residual stream after decoder block 0 (cached as "L0"): one layer of context
  probe   the selected probe (best_config.json of the eval-v3 run), refit exactly as probekit_eval_v3 does
Classifiers on embed/block0 use the SAME recipe as the native-LR ceiling: StandardScaler + logistic regression, C chosen on
VAL-hard. Also reports native LR at the probe layer and TF-IDF, and paired-bootstrap macro-F1 differences on each test set.

    python probekit_l0_baseline.py --corpus data/corpus/corpus_merged_scored.jsonl \
        --cache-dir data/eval_merged/<slug> --eval-dir data/eval_merged/<slug>/v3 --out <slug>/l0_baseline.md

The first run loads the model once to extract embeddings (cached in <cache-dir>/states_embed_*.npz).
Hand-written and own-voice are TEST ONLY here; nothing is selected on them. Aggregates only are printed.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from probekit_eval_v2 import LR_CS, acc, f1_9, load_jsonl, probe_predict
from probekit_eval_v3 import load_states, paired_bootstrap, sweep
from probekit_probes import DEFAULT_EMOTIONS


def fit_native_lr(Xtr, ytr, Xva, yva):
    """Same recipe as the native-LR ceiling in probekit_eval_v3.sweep: C picked on VAL-hard. Returns predict(X)."""
    sc = StandardScaler().fit(Xtr); best = None
    for C in LR_CS:
        clf = LogisticRegression(C=C, max_iter=3000).fit(sc.transform(Xtr), ytr)
        f = f1_9(yva, clf.predict(sc.transform(Xva)))
        if best is None or f > best[0]:
            best = (f, clf, C)
    return (lambda X: best[1].predict(sc.transform(X))), best[2]


def fit_tfidf(texts_tr, ytr):
    vec = TfidfVectorizer(sublinear_tf=True, min_df=2, stop_words="english")
    clf = LogisticRegression(max_iter=3000, C=5.0).fit(vec.fit_transform(texts_tr), ytr)
    return lambda t: clf.predict(vec.transform(t))


def load_embed(items, cache: Path, skip, model):
    ids = [i["id"] for i in items]
    if cache.exists():
        d = np.load(cache, allow_pickle=False)
        if list(d["ids"]) == ids and int(d["skip"]) == skip:
            print(f"  cache hit: {cache}"); return d["E"]
    print(f"  extracting input embeddings for {cache.name} with the model ({len(items)} passages)")
    from probekit_capture import Extractor
    ex = Extractor(model)
    E = np.stack([ex.embedded_pooled(it["text"], skip) for it in items])
    np.savez(cache, ids=np.array(ids), skip=skip, E=E)
    return E


def run(Xs, Xhs, rows, hrows, ps_pred, texts_fn=None):
    """Xs/Xhs: {name: (N,d)} corpus / hand-written features; ps_pred: callable(set) -> probe predictions.
    Returns per-set predictions and truth for every system."""
    cat = np.array([r["category"] for r in rows]); spl = np.array([r["split"] for r in rows])
    hard = np.array([bool(r.get("hard", True)) for r in rows]); texts = np.array([r["text"] for r in rows])
    hcat = np.array([r["category"] for r in hrows]); htxt = np.array([r["text"] for r in hrows])
    tr, va = (spl == "train"), (spl == "val") & hard
    sets = {"VAL-hard": va, "TEST-hard": (spl == "test") & hard, "TEST-all": spl == "test"}
    preds = {k: {} for k in list(sets) + ["hand-written"]}
    truth = {k: cat[m] for k, m in sets.items()}; truth["hand-written"] = hcat
    for name in Xs:
        f, C = fit_native_lr(Xs[name][tr], cat[tr], Xs[name][va], cat[va])
        for k, m in sets.items():
            preds[k][name] = f(Xs[name][m])
        preds["hand-written"][name] = f(Xhs[name])
    tf = fit_tfidf(texts[tr], cat[tr])
    for k, m in sets.items():
        preds[k]["tfidf"] = tf(texts[m])
    preds["hand-written"]["tfidf"] = tf(htxt)
    for k in preds:
        preds[k]["probe"] = ps_pred(k, sets)
    return preds, truth


COMPARISONS = [("tfidf", "embed", "TF-IDF - embed"), ("embed", "block0", "embed - block0"),
               ("probe", "embed", "probe - embed (the T1 test)"), ("probe", "block0", "probe - block0"),
               ("probe", "native_lr_L", "probe - native LR at probe layer")]


def render(preds, truth, layer, method):
    L = [f"# T1 (i): layer-0 baselines vs the selected probe ({method} @ layer {layer})", "",
         "Macro-F1 (9-way unless a class is absent from the set). `embed` = bag of input embeddings; `block0` = after decoder "
         "block 0. Differences are paired bootstrap on the same items (B=2000), 95% interval.", "",
         "| system | " + " | ".join(preds) + " |", "|---|" + "---|" * len(preds)]
    for s in ("tfidf", "embed", "block0", "native_lr_L", "probe"):
        L.append(f"| {s} | " + " | ".join(f"{f1_9(truth[k], preds[k][s]):.3f}" for k in preds) + " |")
    L += ["", f"n per set: " + ", ".join(f"{k}={len(truth[k])}" for k in preds),
          f"hand-written accuracy: " + ", ".join(f"{s} {int(round(acc(truth['hand-written'], preds['hand-written'][s]) * len(truth['hand-written'])))}/{len(truth['hand-written'])}"
                                                  for s in ("tfidf", "embed", "block0", "native_lr_L", "probe")), "",
          "## Paired differences (macro-F1, mean [95% interval]); * = interval excludes 0", "",
          "| comparison | " + " | ".join(preds) + " |", "|---|" + "---|" * len(preds)]
    for a, b, name in COMPARISONS:
        cells = []
        for k in preds:
            m, lo, hi = paired_bootstrap(truth[k], preds[k][a], preds[k][b])
            cells.append(f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]{'*' if lo > 0 or hi < 0 else ''}")
        L.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Llama-3.2-3B-Instruct")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--hardset", default="data/hardset/hardset.jsonl")
    ap.add_argument("--own-voice", default="data/hardset/own_voice.jsonl")
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--eval-dir", required=True, help="dir with best_config.json from probekit_eval_v3.py")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    best = json.loads((Path(a.eval_dir) / "best_config.json").read_text())
    layer, skip, cfg = best["layer"], best["skip"], dict(best["cfg"]); method = cfg.pop("method")
    cache = Path(a.cache_dir)
    rows = load_jsonl(a.corpus); hrows = load_jsonl(a.hardset)
    if Path(a.own_voice).exists():
        hrows += [dict(r, id=f"own|{i}") for i, r in enumerate(load_jsonl(a.own_voice)) if r.get("text", "").strip()]
    emotions = [e for e in DEFAULT_EMOTIONS if any(r["category"] == e for r in rows)]
    Xc = load_states(rows, cache / "states_corpus.npz", [0, layer], skip, a.model)
    Xh = load_states(hrows, cache / "states_hardset.npz", [0, layer], skip, a.model)
    Ec = load_embed(rows, cache / "states_embed_corpus.npz", skip, a.model)
    Eh = load_embed(hrows, cache / "states_embed_hardset.npz", skip, a.model)
    # refit the selected probe exactly as probekit_eval_v3 does (single config, no selection on the hand-written set)
    res = sweep([layer], Xc, rows, Xh, hrows, emotions, {method: [cfg]})
    ent = res["per_layer"][layer][method]; ps, tau = ent["_ps"], ent["tau"]
    cat = np.array([r["category"] for r in rows])
    probe_pred = lambda k, sets: probe_predict(ps, Xh[layer], tau) if k == "hand-written" else probe_predict(ps, Xc[layer][sets[k]], tau)
    Xs = {"embed": Ec, "block0": Xc[0], "native_lr_L": Xc[layer]}
    Xhs = {"embed": Eh, "block0": Xh[0], "native_lr_L": Xh[layer]}
    preds, truth = run(Xs, Xhs, rows, hrows, probe_pred)
    md = render(preds, truth, layer, method)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
