#!/usr/bin/env python3
"""
probekit_frozen_eval.py -- PLAN_v2 T0c (pre-registered): evaluate on the FROZEN human test set and run the human-data learning curve.

    python probekit_frozen_eval.py extract --cache-dir data/eval_merged2/<slug> --corpus data/corpus/corpus_merged2_scored.jsonl
    python probekit_frozen_eval.py eval    --cache-dir data/eval_merged2/<slug> --eval-dir data/eval_merged2/<slug>/v3 \
                                           --corpus data/corpus/corpus_merged2_scored.jsonl --out data/eval_merged2/<slug>/frozen_eval.md

Nothing here is selected on the frozen set: layer, estimator config and tau come from the pooled eval's VAL-hard result. Calm is
excluded from training and evaluation (it is LLM-only, i.e. writer-confounded), so the primary metric is 8-way macro-F1 over
7 emotions + neutral. The frozen set is read only through probekit_testset.load_frozen (hash-verified).
"""
import argparse
import json
from pathlib import Path

import numpy as np

from probekit_estimators import calibrate_pooled, fit_directions
from probekit_eval_v2 import f1_9, load_jsonl, pick_tau, probe_predict
from probekit_eval_v3 import paired_bootstrap
from probekit_l0_baseline import fit_native_lr, fit_tfidf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from probekit_eval_v2 import LR_CS
from probekit_probes import DEFAULT_EMOTIONS, NEUTRAL

EMOTIONS = [e for e in DEFAULT_EMOTIONS if e != "calm"]
CLASSES = EMOTIONS + [NEUTRAL]
VALENCE = {"joy": "positive", "love": "positive", "sadness": "negative", "fear": "negative", "anger": "negative",
           "disgust": "negative", "surprise": "ambiguous", "neutral": "neutral"}
FROZEN_MANIFEST = "data/testsets/goemotions_test_frozen.manifest.json"
EXTRACT_LAYERS = [0, 6, 8, 10, 12, 14, 16, 18]


# ---------------------------------------------------------------- extraction (needs the model)
def extract_cached(items, cache: Path, layers, skip, model):
    """{layer: (N,d)} plus 'E' = bag-of-embeddings, cached in one npz keyed by the id list."""
    ids = [i["id"] for i in items]
    if cache.exists():
        d = np.load(cache, allow_pickle=False)
        if list(d["ids"]) == ids and int(d["skip"]) == skip and all(f"L{l}" in d.files for l in layers) and "E" in d.files:
            print(f"  cache hit: {cache}")
            return {**{l: d[f"L{l}"] for l in layers}, "E": d["E"]}
    print(f"  extracting {len(items)} passages -> {cache.name}", flush=True)
    from probekit_capture import Extractor
    ex = Extractor(model)
    acc = {l: [] for l in layers}; emb = []
    for n, it in enumerate(items):
        for l, v in ex.pooled(it["text"], layers, skip).items():
            acc[l].append(v)
        emb.append(ex.embedded_pooled(it["text"], skip))
        if (n + 1) % 200 == 0:
            print(f"    {n + 1}/{len(items)}", flush=True)
    out = {l: np.stack(v) for l, v in acc.items()}; out["E"] = np.stack(emb)
    np.savez(cache, ids=np.array(ids), skip=skip, E=out["E"], **{f"L{l}": out[l] for l in layers})
    return out


def pool_new_rows(pool, corpus_rows):
    have = {r["id"] for r in corpus_rows}
    return [r for r in pool if r["id"] not in have]


# ---------------------------------------------------------------- fitting / prediction
def cfg_for(eval_dir: Path, method, layer):
    rep = json.loads((eval_dir / "report_v3.json").read_text())
    cell = rep["per_layer"][str(layer)][method]
    return {k: v for k, v in cell["cfg"].items() if k != "method"}


def fit_probe(Xtr, ytr, Xva, yva, Xcal, layer, method, cfg):
    ps = fit_directions(Xtr, ytr, EMOTIONS, layer, method=method, **cfg)
    calibrate_pooled(ps, Xcal)
    tau = pick_tau(ps, Xva, yva)
    return (lambda X: probe_predict(ps, X, tau)), tau


def fit_fusion(Xtr, ytr, Xva, yva, texts_tr):
    """POST HOC (added after the frozen result was seen): late fusion = mean of class probabilities from TF-IDF LR and
    native hidden-state LR (C chosen on VAL like fit_native_lr). Returns predict(X, texts). Asks: do hidden states add
    information on top of the lexical model?"""
    vec = TfidfVectorizer(sublinear_tf=True, min_df=2, stop_words="english")
    lex = LogisticRegression(max_iter=3000, C=5.0).fit(vec.fit_transform(texts_tr), ytr)
    sc = StandardScaler().fit(Xtr); best = None
    for C in LR_CS:
        clf = LogisticRegression(C=C, max_iter=3000).fit(sc.transform(Xtr), ytr)
        f = f1_9(yva, clf.predict(sc.transform(Xva)))
        if best is None or f > best[0]:
            best = (f, clf)
    clf = best[1]; assert list(clf.classes_) == list(lex.classes_)
    def predict(X, texts):
        pr = lex.predict_proba(vec.transform(texts)) + clf.predict_proba(sc.transform(X))
        return clf.classes_[pr.argmax(1)]
    return predict


def macro_f1(y, p):
    return f1_9(y, p)


def boot_ci(y, p, B=1000, seed=0):
    rng = np.random.default_rng(seed); y, p = np.asarray(y), np.asarray(p)
    v = [f1_9(y[i], p[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(B))]
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def shuffle_null_p95(y, p, B=500, seed=0):
    rng = np.random.default_rng(seed); y, p = np.asarray(y), np.asarray(p)
    return float(np.percentile([f1_9(rng.permutation(y), p) for _ in range(B)], 95))


def per_class(y, p):
    from sklearn.metrics import precision_recall_fscore_support
    pr, rc, f, n = precision_recall_fscore_support(y, p, labels=CLASSES, zero_division=0)
    return {c: {"recall": float(r), "f1": float(x), "n": int(k)} for c, r, x, k in zip(CLASSES, rc, f, n)}


def coarse(v):
    return np.array([VALENCE[x] for x in v])


def stratum_scores(y, preds, mask):
    return {s: f1_9(np.asarray(y)[mask], np.asarray(p)[mask]) for s, p in preds.items()} if mask.sum() >= 20 else None


# ---------------------------------------------------------------- learning curve
def sample_human(pool_y, pool_rank, k, rng):
    """Indices of k pool rows per class (random per seed; all if k >= available)."""
    idx = []
    for c in CLASSES:
        ci = np.where(pool_y == c)[0]
        idx += list(ci if k >= len(ci) else rng.choice(ci, k, replace=False))
    return np.array(sorted(idx), dtype=int)


def learning_curve(X_llm, y_llm, X_pool, y_pool, Xva, yva, Xcal, X_fr, y_fr, layer, method, cfg, ks, seeds, include_llm=True, texts=None):
    """texts = (T_llm, T_pool, T_fr) enables the post hoc TF-IDF and fusion columns at equal training data."""
    res = []
    for k in ks:
        for seed in (seeds if (k > 0 and k < min(np.sum(y_pool == c) for c in CLASSES)) else seeds[:1]):
            rng = np.random.default_rng(seed)
            hi = sample_human(y_pool, None, k, rng) if k > 0 else np.array([], dtype=int)
            Xtr = np.vstack([X_llm[layer] if include_llm else np.empty((0, X_llm[layer].shape[1])), X_pool[layer][hi]])
            ytr = np.concatenate([y_llm if include_llm else np.array([], dtype=object), y_pool[hi]])
            if len(set(ytr)) < len(CLASSES):
                continue
            pr, tau = fit_probe(Xtr, ytr, Xva, yva, Xcal, layer, method, cfg)
            lr, C = fit_native_lr(Xtr, ytr, Xva, yva)
            pp, lp = pr(X_fr[layer]), lr(X_fr[layer])
            entry = {"k": k, "seed": seed, "n_train": int(len(ytr)), "llm": include_llm,
                     "probe": macro_f1(y_fr, pp), "native_lr": macro_f1(y_fr, lp), "C": C, "_p": {"probe": pp, "native_lr": lp}}
            msg = f"probe {entry['probe']:.3f} native {entry['native_lr']:.3f}"
            if texts is not None:
                T_llm, T_pool, T_fr = texts
                Ttr = np.concatenate([T_llm if include_llm else np.array([], dtype=object), T_pool[hi]])
                tp = fit_tfidf(Ttr, ytr)(T_fr)
                fp = fit_fusion(Xtr, ytr, Xva, yva, Ttr)(X_fr[layer], T_fr)
                entry.update(tfidf=macro_f1(y_fr, tp), fusion=macro_f1(y_fr, fp)); entry["_p"].update(tfidf=tp, fusion=fp)
                msg += f" tfidf {entry['tfidf']:.3f} fusion {entry['fusion']:.3f}"
            res.append(entry)
            print(f"    k={k:4d} seed={seed} n_train={len(ytr)} llm={include_llm}: {msg}", flush=True)
    return res


def summarize_curve(res):
    out = {}
    for r in res:
        out.setdefault((r["llm"], r["k"]), []).append(r)
    extra = any("tfidf" in r for r in res)
    L = ["| training | k per class | n train | probe F1 | native LR F1 |" + (" TF-IDF F1 (post hoc) | fusion F1 (post hoc) |" if extra else ""),
         "|---|---|---|---|---|" + ("---|---|" if extra else "")]
    for (llm, k), rs in sorted(out.items(), key=lambda t: (not t[0][0], t[0][1])):
        f = lambda key: (np.mean([r[key] for r in rs]), np.std([r[key] for r in rs]))
        pm, ps = f("probe"); nm, ns = f("native_lr")
        row = (f"| {'LLM + human' if llm else 'human only'} | {k} | {int(np.mean([r['n_train'] for r in rs]))} | "
               f"{pm:.3f} ± {ps:.3f} | {nm:.3f} ± {ns:.3f} |")
        if extra:
            tm, ts_ = f("tfidf"); fm, fs = f("fusion")
            row += f" {tm:.3f} ± {ts_:.3f} | {fm:.3f} ± {fs:.3f} |"
        L.append(row)
    return "\n".join(L)


# ---------------------------------------------------------------- main
def run_eval(a):
    from probekit_testset import load_frozen
    eval_dir, cache = Path(a.eval_dir), Path(a.cache_dir)
    best = json.loads((eval_dir / "best_config.json").read_text())
    layer, skip, sel_method = best["layer"], best["skip"], best["method"]
    rows = load_jsonl(a.corpus)
    frozen, manifest = load_frozen(a.manifest)
    pool = load_jsonl(a.pool)
    from probekit_l0_baseline import load_embed
    from probekit_eval_v3 import load_states
    Xc = load_states(rows, cache / "states_corpus.npz", [0, layer], skip, a.model)
    Ec = load_embed(rows, cache / "states_embed_corpus.npz", skip, a.model)
    Xf = extract_cached(frozen, cache / "states_frozen.npz", EXTRACT_LAYERS, skip, a.model)
    new = pool_new_rows(pool, rows)
    Xn = extract_cached(new, cache / "states_pool_new.npz", EXTRACT_LAYERS, skip, a.model)

    cat = np.array([r["category"] for r in rows]); spl = np.array([r["split"] for r in rows])
    wr = np.array([r.get("writer", "llama-3.2-3b") for r in rows]); hard = np.array([bool(r.get("hard", True)) for r in rows])
    texts = np.array([r["text"] for r in rows])
    keep = cat != "calm"
    tr = (spl == "train") & keep; va = (spl == "val") & hard & keep; cal = (spl == "val") & (cat == NEUTRAL)
    y_fr = np.array([r["category"] for r in frozen]); t_fr = np.array([r["text"] for r in frozen])
    agree = np.array([r["agree"] if r["agree"] is not None else 0 for r in frozen]); nw = np.array([r["n_words"] for r in frozen])

    # ---- systems, all trained on pooled merged2 TRAIN without calm
    preds, extra = {}, {}
    for method in ("lda", "meandiff", "logreg"):
        pr, tau = fit_probe(Xc[layer][tr], cat[tr], Xc[layer][va], cat[va], Xc[layer][cal], layer, method, cfg_for(eval_dir, method, layer))
        preds[f"probe_{method}"] = pr(Xf[layer]); extra[f"probe_{method}_tau"] = tau
    lr, C = fit_native_lr(Xc[layer][tr], cat[tr], Xc[layer][va], cat[va]); preds["native_lr"] = lr(Xf[layer]); extra["native_C"] = C
    for name, Xtr_, Xva_, Xfr_ in (("embed", Ec, Ec, Xf["E"]), ("block0", Xc[0], Xc[0], Xf[0])):
        f_, _ = fit_native_lr(Xtr_[tr], cat[tr], Xva_[va], cat[va]); preds[name] = f_(Xfr_)
    preds["tfidf"] = fit_tfidf(texts[tr], cat[tr])(t_fr)
    preds["fusion_tfidf_native (post hoc)"] = fit_fusion(Xc[layer][tr], cat[tr], Xc[layer][va], cat[va], texts[tr])(Xf[layer], t_fr)
    primary = f"probe_{sel_method}"

    L = [f"# T0c frozen-set evaluation (n={len(frozen)}; manifest sha256 {manifest['testset_sha256'][:12]}...)", "",
         f"Layer {layer}; selected estimator {sel_method} (chosen on VAL-hard of the pooled eval, not on this set). Trained on pooled merged2 TRAIN "
         f"without calm (n={int(tr.sum())}); tau and C on VAL-hard (n={int(va.sum())}). 8-way macro-F1 over {', '.join(CLASSES)}; bootstrap 95% CI over test items.", "",
         "| system | macro-F1 [95% CI] | accuracy | shuffle-null p95 | coarse valence F1 |", "|---|---|---|---|---|"]
    for s, p in preds.items():
        lo, hi = boot_ci(y_fr, p)
        L.append(f"| {s}{' (selected)' if s == primary else ''} | {macro_f1(y_fr, p):.3f} [{lo:.3f}, {hi:.3f}] | {(y_fr == p).mean():.3f} | "
                 f"{shuffle_null_p95(y_fr, p):.3f} | {macro_f1(coarse(y_fr), coarse(p)):.3f} |")
    L += ["", "## Paired differences, macro-F1 (mean [95% interval]); * = excludes 0", "", "| comparison | difference |", "|---|---|"]
    for b in ("tfidf", "embed", "block0", "native_lr"):
        m, lo, hi = paired_bootstrap(y_fr, preds[primary], preds[b])
        L.append(f"| {primary} - {b} | {m:+.3f} [{lo:+.3f}, {hi:+.3f}]{'*' if lo > 0 or hi < 0 else ''} |")
    fz = "fusion_tfidf_native (post hoc)"
    for a_, b_ in ((fz, "tfidf"), (fz, "native_lr")):
        m, lo, hi = paired_bootstrap(y_fr, preds[a_], preds[b_])
        L.append(f"| {a_} - {b_} (post hoc) | {m:+.3f} [{lo:+.3f}, {hi:+.3f}]{'*' if lo > 0 or hi < 0 else ''} |")
    L += ["", "## Per class (selected probe)", "", "| class | n | recall | F1 | native LR recall |", "|---|---|---|---|---|"]
    pc, pn = per_class(y_fr, preds[primary]), per_class(y_fr, preds["native_lr"])
    for c in CLASSES:
        L.append(f"| {c} | {pc[c]['n']} | {pc[c]['recall']:.2f} | {pc[c]['f1']:.2f} | {pn[c]['recall']:.2f} |")
    L += ["", "## Strata (macro-F1, systems: " + ", ".join(["probe", "native_lr", "embed", "tfidf"]) + ")", ""]
    sub = {"probe": preds[primary], "native_lr": preds["native_lr"], "embed": preds["embed"], "tfidf": preds["tfidf"]}
    for name, mask in (("rater agreement >= 0.67", agree >= 0.67), ("rater agreement < 0.67", agree < 0.67),
                       ("<= 12 words", nw <= 12), ("> 12 words", nw > 12)):
        sc = stratum_scores(y_fr, sub, mask)
        L.append(f"- {name} (n={int(mask.sum())}): " + (", ".join(f"{k} {v:.3f}" for k, v in sc.items()) if sc else "n < 20, not reported"))
    st = manifest["subtype_counts"]
    L.append(f"- subtypes (too small to score; n only): {st}")

    # ---- learning curve: LLM rows + k human rows/class from the pool
    pool_rows = [r for r in pool if r["split"] == "train"]
    byid = {r["id"]: i for i, r in enumerate(rows)}; nid = {r["id"]: i for i, r in enumerate(new)}
    Xp = {layer: np.stack([(Xc[layer][byid[r["id"]]] if r["id"] in byid else Xn[layer][nid[r["id"]]]) for r in pool_rows])}
    y_pool = np.array([r["category"] for r in pool_rows])
    llm = tr & (wr != "reddit-goemotions")
    X_llm = {layer: Xc[layer][llm]}; y_llm = cat[llm]
    avail = min(int(np.sum(y_pool == c)) for c in CLASSES)
    ks = [k for k in (0, 20, 40, 80, 160, 320) if k < avail] + [avail]
    cfg = cfg_for(eval_dir, sel_method, layer)
    print("learning curve:", ks, flush=True)
    T_llm = texts[llm]; T_pool = np.array([r["text"] for r in pool_rows], dtype=object)
    res = learning_curve(X_llm, y_llm, Xp, y_pool, Xc[layer][va], cat[va], Xc[layer][cal], Xf, y_fr, layer, sel_method, cfg, ks, [0, 1, 2],
                         texts=(T_llm, T_pool, t_fr))
    res += learning_curve(X_llm, y_llm, Xp, y_pool, Xc[layer][va], cat[va], Xc[layer][cal], Xf, y_fr, layer, sel_method, cfg,
                          [k for k in ks if k >= 20], [0, 1, 2], include_llm=False, texts=(T_llm, T_pool, t_fr))
    L += ["", f"## Human-data learning curve (frozen-set macro-F1, mean ± sd over subsampling seeds; k=max is {avail} per class, fear-limited)", "",
          summarize_curve(res)]
    d2 = [r for r in res if r["llm"] and r["k"] == avail]; d80 = [r for r in res if r["llm"] and r["k"] == 80]
    if d2 and d80:
        L += ["", f"D2 check: native LR at k={avail} minus at k=80 = {np.mean([r['native_lr'] for r in d2]) - np.mean([r['native_lr'] for r in d80]):+.3f} "
                  f"(threshold 0.03; k={avail} has one fit, so no seed variance, k=80 mean over {len(d80)} seeds)."]
    if d2 and d80:                                  # paired CIs on the same frozen items (seed 0 fit at k=80)
        p80 = next(r for r in d80 if r["seed"] == 0)["_p"]; pm_ = d2[0]["_p"]
        L += ["", "Paired on the frozen items (post hoc CIs for the pre-registered D2 and for equal-data comparisons):", ""]
        for name, (x, y_) in {f"D2 native LR k={avail} - k=80": (pm_["native_lr"], p80["native_lr"]),
                              f"probe - TF-IDF at k={avail}": (pm_["probe"], pm_["tfidf"]),
                              f"native LR - TF-IDF at k={avail}": (pm_["native_lr"], pm_["tfidf"]),
                              f"fusion - TF-IDF at k={avail}": (pm_["fusion"], pm_["tfidf"])}.items():
            m, lo, hi = paired_bootstrap(y_fr, x, y_)
            L.append(f"- {name}: {m:+.3f} [{lo:+.3f}, {hi:+.3f}]{'*' if lo > 0 or hi < 0 else ''}")
    d0 = [r for r in res if r["llm"] and r["k"] == 0]
    if d0:                                          # zero human training rows: pure cross-writer transfer (post hoc CIs)
        p0 = d0[0]["_p"]
        L += ["", "Zero human training data (LLM-written rows only), paired on the frozen items (post hoc):", ""]
        for name, (x, y_) in {"probe - TF-IDF": (p0["probe"], p0["tfidf"]), "native LR - TF-IDF": (p0["native_lr"], p0["tfidf"])}.items():
            m, lo, hi = paired_bootstrap(y_fr, x, y_)
            L.append(f"- {name} at k=0: {m:+.3f} [{lo:+.3f}, {hi:+.3f}]{'*' if lo > 0 or hi < 0 else ''}")
        lo_, hi_ = boot_ci(y_fr, p0["probe"])
        L.append(f"- probe at k=0: macro-F1 {macro_f1(y_fr, p0['probe']):.3f} [{lo_:.3f}, {hi_:.3f}] vs shuffle-null p95 {shuffle_null_p95(y_fr, p0['probe']):.3f}; "
                 f"TF-IDF {macro_f1(y_fr, p0['tfidf']):.3f}")
    for r in res:
        r.pop("_p", None)
    md = "\n".join(L)
    Path(a.out).with_name("frozen_preds.json").write_text(json.dumps(    # ids, labels and predictions only; no text
        {"ids": [r["id"] for r in frozen], "labels": [str(x) for x in y_fr], "preds": {k: [str(x) for x in v] for k, v in preds.items()}}))
    Path(a.out).write_text(md + "\n"); Path(a.out).with_suffix(".json").write_text(json.dumps({"curve": res, "extra": extra}, indent=1))
    print(md)


def run_extract(a):
    from probekit_testset import load_frozen
    cache = Path(a.cache_dir); cache.mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(a.corpus)
    frozen, _ = load_frozen(a.manifest)
    skip = json.loads((Path(a.eval_dir) / "best_config.json").read_text())["skip"] if a.eval_dir else 4
    extract_cached(frozen, cache / "states_frozen.npz", EXTRACT_LAYERS, skip, a.model)
    extract_cached(pool_new_rows(load_jsonl(a.pool), rows), cache / "states_pool_new.npz", EXTRACT_LAYERS, skip, a.model)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    for name in ("extract", "eval"):
        p = sp.add_parser(name)
        p.add_argument("--model", default="mlx-community/Llama-3.2-3B-Instruct")
        p.add_argument("--cache-dir", required=True); p.add_argument("--corpus", required=True)
        p.add_argument("--eval-dir", default=None)
        p.add_argument("--manifest", default=FROZEN_MANIFEST)
        p.add_argument("--pool", default="data/corpus/corpus_human_goemotions_pool.jsonl")
        if name == "eval":
            p.add_argument("--out", required=True)
    a = ap.parse_args()
    {"extract": run_extract, "eval": run_eval}[a.cmd](a)


if __name__ == "__main__":
    main()
