#!/usr/bin/env python3
"""
probekit_external_eval.py -- PLAN_v2 T0d (pre-registered): cross-domain evaluation on external human-labelled test sets
(dair-ai/emotion, ISEAR, DailyDialog; see probekit_external.py). Nothing is tuned on these sources.

    python probekit_external_eval.py --cache-dir data/eval_merged2/<slug> --eval-dir data/eval_merged2/<slug>/v3 \
        --corpus data/corpus/corpus_merged2_scored.jsonl --out data/eval_merged2/<slug>/external_eval.md

Training sets: HUM = LLM-written rows + all human GoEmotions-train pool rows; LLM = LLM-written rows only (calm excluded in both).
Predictions are restricted to the classes present in each source (a source without 'neutral' never gets a neutral prediction);
macro-F1 is over those classes.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from probekit_estimators import calibrate_pooled, fit_directions
from probekit_eval_v2 import LR_CS, f1_9, load_jsonl, pick_tau
from probekit_eval_v3 import paired_bootstrap
from probekit_frozen_eval import EMOTIONS, EXTRACT_LAYERS, boot_ci, cfg_for, shuffle_null_p95

SOURCES = ("dair", "isear", "dailydialog")


# ---------------------------------------------------------------- restricted predictors
def fit_lr(Xtr, ytr, Xva, yva):
    """StandardScaler + LR with C chosen on VAL macro-F1 (same recipe as the native-LR ceiling)."""
    sc = StandardScaler().fit(Xtr); best = None
    for C in LR_CS:
        clf = LogisticRegression(C=C, max_iter=3000).fit(sc.transform(Xtr), ytr)
        f = f1_9(yva, clf.predict(sc.transform(Xva)))
        if best is None or f > best[0]:
            best = (f, clf)
    return sc, best[1]


def fit_tfidf_model(texts, y):
    vec = TfidfVectorizer(sublinear_tf=True, min_df=2, stop_words="english")
    return vec, LogisticRegression(max_iter=3000, C=5.0).fit(vec.fit_transform(texts), y)


def columns(clf, P, classes):
    """Probability matrix restricted to `classes` (zeros for classes the model never saw)."""
    out = np.zeros((P.shape[0], len(classes)))
    idx = {c: i for i, c in enumerate(clf.classes_)}
    for j, c in enumerate(classes):
        if c in idx:
            out[:, j] = P[:, idx[c]]
    return out


def argmax_classes(P, classes):
    return np.array(classes, dtype=object)[P.argmax(1)]


def probe_predict_restricted(ps, X, tau, classes):
    """z-score argmax among the source's emotions; 'neutral' (threshold tau) only if the source has a neutral class."""
    emo = [c for c in classes if c != "neutral"]
    Z = ps.score(X, "pooled")[:, [ps.emotions.index(c) for c in emo]]
    pred = np.array(emo, dtype=object)[Z.argmax(1)]
    if "neutral" in classes:
        pred[Z.max(1) < tau] = "neutral"
    return pred


def fit_systems(Xtr, Etr, Ttr, ytr, Xva, Eva, yva, Xcal, layer, cfg, method="lda", emotions=None):
    """Everything is fit on one training set; returns ({name: predict(X, E, texts, classes)}, tau).
    `emotions` = the non-neutral classes of the training set (default: the 7 project emotions); `Etr=None` omits the embedding baseline."""
    emotions = EMOTIONS if emotions is None else emotions
    ps = fit_directions(Xtr, ytr, emotions, layer, method=method, **cfg)
    calibrate_pooled(ps, Xcal)
    tau = pick_tau(ps, Xva, yva)
    sc_s, lr_s = fit_lr(Xtr, ytr, Xva, yva)
    vec, lex = fit_tfidf_model(Ttr, ytr)
    P_lex = lambda T, cl: columns(lex, lex.predict_proba(vec.transform(T)), cl)
    P_lr = lambda X, cl: columns(lr_s, lr_s.predict_proba(sc_s.transform(X)), cl)
    systems = {
        "probe": lambda X, E, T, cl: probe_predict_restricted(ps, X, tau, cl),
        "native_lr": lambda X, E, T, cl: argmax_classes(P_lr(X, cl), cl),
        "tfidf": lambda X, E, T, cl: argmax_classes(P_lex(T, cl), cl),
        "fusion (post hoc)": lambda X, E, T, cl: argmax_classes(P_lex(T, cl) + P_lr(X, cl), cl),
    }
    if Etr is not None:
        sc_e, lr_e = fit_lr(Etr, ytr, Eva, yva)
        systems["embed"] = lambda X, E, T, cl: argmax_classes(columns(lr_e, lr_e.predict_proba(sc_e.transform(E)), cl), cl)
    order = ["probe", "native_lr", "embed", "tfidf", "fusion (post hoc)"]
    return {k: systems[k] for k in order if k in systems}, tau


def evaluate_source(systems, X, E, T, y, classes):
    preds = {n: f(X, E, T, classes) for n, f in systems.items()}
    return preds


# ---------------------------------------------------------------- report
def render_source(name, manifest, y, preds_by_train):
    classes = sorted(set(y))
    L = [f"## {name}: n={len(y)}, classes {classes}; chance {1 / len(classes):.3f}",
         f"License/provenance note: {manifest['license_note']}; labels: {manifest['label_provenance']}.", "",
         "| training | system | macro-F1 [95% CI] | accuracy | shuffle-null p95 |", "|---|---|---|---|---|"]
    for tr, preds in preds_by_train.items():
        for s, p in preds.items():
            lo, hi = boot_ci(y, p)
            L.append(f"| {tr} | {s} | {f1_9(y, p):.3f} [{lo:.3f}, {hi:.3f}] | {(np.asarray(y) == p).mean():.3f} | {shuffle_null_p95(y, p):.3f} |")
    L += ["", "Paired differences, macro-F1 (mean [95% interval]); * = excludes 0", "", "| training | probe - tfidf (H1) | probe - embed (H2) | probe - native_lr | fusion - tfidf |", "|---|---|---|---|---|"]
    out = {}
    for tr, preds in preds_by_train.items():
        cells = []
        for a, b in (("probe", "tfidf"), ("probe", "embed"), ("probe", "native_lr"), ("fusion (post hoc)", "tfidf")):
            m, lo, hi = paired_bootstrap(y, preds[a], preds[b])
            cells.append(f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]{'*' if lo > 0 or hi < 0 else ''}")
            out[(tr, a, b)] = (m, lo, hi)
        L.append(f"| {tr} | " + " | ".join(cells) + " |")
    pc = {}
    for c in classes:
        mask = np.asarray(y) == c
        pc[c] = {tr: {s: float((preds[s][mask] == c).mean()) for s in ("probe", "tfidf")} for tr, preds in preds_by_train.items()}
    L += ["", "Per-class recall (probe / TF-IDF), n in brackets (post hoc, descriptive):", ""]
    for c in classes:
        n = int((np.asarray(y) == c).sum())
        L.append(f"- {c} [{n}]: " + "; ".join(f"{tr}: {pc[c][tr]['probe']:.2f} / {pc[c][tr]['tfidf']:.2f}" for tr in preds_by_train))
    return "\n".join(L), out


def verdict(results):
    """H1 tally: sources where probe - tfidf has a CI above 0, per training set."""
    lines = []
    for tr in ("HUM", "LLM"):
        wins = [s for s, r in results.items() if r[(tr, "probe", "tfidf")][1] > 0]
        losses = [s for s, r in results.items() if r[(tr, "probe", "tfidf")][2] < 0]
        lines.append(f"- trained on {tr}: probe > TF-IDF on {len(wins)}/{len(results)} sources {wins}; probe < TF-IDF on {len(losses)} {losses}")
    return lines


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Llama-3.2-3B-Instruct")
    ap.add_argument("--cache-dir", required=True); ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--pool", default="data/corpus/corpus_human_goemotions_pool.jsonl")
    ap.add_argument("--testsets", default="data/testsets")
    ap.add_argument("--sources", nargs="*", default=list(SOURCES))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from probekit_eval_v3 import load_states
    from probekit_external import load_source
    from probekit_frozen_eval import extract_cached, pool_new_rows
    from probekit_l0_baseline import load_embed
    eval_dir, cache = Path(a.eval_dir), Path(a.cache_dir)
    best = json.loads((eval_dir / "best_config.json").read_text())
    layer, skip, method = best["layer"], best["skip"], best["method"]
    cfg = cfg_for(eval_dir, method, layer)
    rows = load_jsonl(a.corpus); pool = load_jsonl(a.pool)
    Xc = load_states(rows, cache / "states_corpus.npz", [layer], skip, a.model)[layer]
    Ec = load_embed(rows, cache / "states_embed_corpus.npz", skip, a.model)
    new = pool_new_rows(pool, rows)
    Xn = extract_cached(new, cache / "states_pool_new.npz", EXTRACT_LAYERS, skip, a.model)
    cat = np.array([r["category"] for r in rows]); spl = np.array([r["split"] for r in rows])
    wr = np.array([r.get("writer", "llama-3.2-3b") for r in rows]); hard = np.array([bool(r.get("hard", True)) for r in rows])
    txt = np.array([r["text"] for r in rows], dtype=object)
    keep = cat != "calm"; human = wr == "reddit-goemotions"
    llm = (spl == "train") & keep & ~human
    va = (spl == "val") & hard & keep; cal = (spl == "val") & (cat == "neutral")
    byid = {r["id"]: i for i, r in enumerate(rows)}; nid = {r["id"]: i for i, r in enumerate(new)}
    pr = [r for r in pool if r["split"] == "train"]
    Xp = np.stack([Xc[byid[r["id"]]] if r["id"] in byid else Xn[layer][nid[r["id"]]] for r in pr])
    Ep = np.stack([Ec[byid[r["id"]]] if r["id"] in byid else Xn["E"][nid[r["id"]]] for r in pr])
    yp = np.array([r["category"] for r in pr], dtype=object); Tp = np.array([r["text"] for r in pr], dtype=object)

    trainings = {}
    X_h = np.vstack([Xc[llm], Xp]); E_h = np.vstack([Ec[llm], Ep]); T_h = np.concatenate([txt[llm], Tp]); y_h = np.concatenate([cat[llm], yp])
    trainings["HUM"] = fit_systems(X_h, E_h, T_h, y_h, Xc[va], Ec[va], cat[va], Xc[cal], layer, cfg, method)
    va_l = va & ~human
    trainings["LLM"] = fit_systems(Xc[llm], Ec[llm], txt[llm], cat[llm], Xc[va_l], Ec[va_l], cat[va_l], Xc[cal & ~human], layer, cfg, method)
    L = [f"# T0d external cross-domain evaluation (layer {layer}, {method}; pre-registered in PLAN_v2)", "",
         f"Training sizes: HUM n={len(y_h)} (tau {trainings['HUM'][1]:.2f}), LLM n={int(llm.sum())} (tau {trainings['LLM'][1]:.2f}). Nothing tuned on the external sources. Frozen GoEmotions results are in frozen_eval.md.", ""]
    results = {}; blocks = []
    for name in a.sources:
        items, manifest = load_source(name, a.testsets)
        Xe = extract_cached(items, cache / f"states_ext_{name}.npz", EXTRACT_LAYERS, skip, a.model)
        y = np.array([r["category"] for r in items], dtype=object); T = np.array([r["text"] for r in items], dtype=object)
        classes = sorted(set(y))
        pbt = {tr: evaluate_source(sys_[0], Xe[layer], Xe["E"], T, y, classes) for tr, sys_ in trainings.items()}
        md, res = render_source(name, manifest, y, pbt)
        blocks.append(md); results[name] = res
    L += ["## Pre-registered tally (H1: probe - TF-IDF, CI excluding 0)", ""] + verdict(results) + [""] + blocks
    Path(a.out).write_text("\n".join(L) + "\n")
    Path(a.out).with_suffix(".json").write_text(json.dumps({s: {f"{k[0]}|{k[1]}-{k[2]}": v for k, v in r.items()} for s, r in results.items()}, indent=1))
    print("\n".join(L))


if __name__ == "__main__":
    main()
