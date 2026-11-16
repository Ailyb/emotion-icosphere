#!/usr/bin/env python3
"""
probekit_eval_v3.py -- step 1: which estimator, which layer, and how much do we lose vs a trained classifier?

Compares meandiff / lda / logreg directions (probekit_estimators) over the mid-layer band, with a WIDER nuisance
grid (v2 always picked its lowest cutoff, 0.8, so the grid now goes down to 0.0), against:
  * native hidden-state logistic regression (the ceiling from v2; not a single direction per emotion)
  * lexical TF-IDF (the shortcut to beat)
Everything uses the cached hidden states from probekit_eval_v2.py: no model load unless the cache is stale
(e.g. after you add own_voice.jsonl, only the small hard-set cache is recomputed).

Selection rule: per layer, hyperparameters and the neutral threshold are chosen on VAL-hard. The (method, layer)
winner is also chosen on VAL-hard. The hand-written set is NEVER used for selection.
Calibration (bias/sigma per concept) uses held-out VAL neutral passages.

    python probekit_eval_v3.py            # ~10-20 min (LDA/logreg fits are the slow part); --fast for a quick pass
Outputs (data/eval_v3/<model>/): report_v3.md, report_v3.json, errors_v3.md, layer_curve_v3.png, best_config.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from probekit_estimators import METHODS, calibrate_pooled, fit_directions
from probekit_eval_v2 import LR_CS, acc, f1_9, load_jsonl, pick_tau, probe_predict, wilson
from probekit_probes import DEFAULT_EMOTIONS, NEUTRAL

MID_BAND = [6, 8, 10, 12, 14, 16, 18]
GRIDS_FULL = {
    "meandiff": [dict(var_frac=v) for v in (0.0, 0.3, 0.5, 0.8)],
    "lda": [dict(var_frac=v, shrinkage="auto") for v in (0.5, 0.8)],
    "logreg": [dict(var_frac=v, C=c) for v in (0.5, 0.8) for c in (0.003, 0.03)],
}
GRIDS_FAST = {
    "meandiff": [dict(var_frac=v) for v in (0.3, 0.8)],
    "lda": [dict(var_frac=0.8, shrinkage="auto")],
    "logreg": [dict(var_frac=0.8, C=0.03)],
}


def paired_bootstrap(y, pa, pb, B=2000, seed=0):
    """Mean and 95% interval of macro-F1(a) - macro-F1(b), resampling passages with replacement."""
    rng = np.random.default_rng(seed)
    y, pa, pb = np.asarray(y), np.asarray(pa), np.asarray(pb)
    d = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        d.append(f1_9(y[i], pa[i]) - f1_9(y[i], pb[i]))
    d = np.array(d)
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def sweep(layers, Xc, rows, Xh, hrows, emotions, grids, train_on="all"):
    cat = np.array([r["category"] for r in rows]); spl = np.array([r["split"] for r in rows])
    hard = np.array([bool(r.get("hard", True)) for r in rows]); texts = np.array([r["text"] for r in rows])
    hcat = np.array([r["category"] for r in hrows]); htxt = np.array([r["text"] for r in hrows])
    tr = (spl == "train") & (hard if train_on == "hard" else True)
    va, te_h, te_a = (spl == "val") & hard, (spl == "test") & hard, (spl == "test")
    cal = (spl == "val") & (cat == NEUTRAL)                 # held-out neutral passages for calibration

    vec = TfidfVectorizer(sublinear_tf=True, min_df=2, stop_words="english")
    lex = LogisticRegression(max_iter=3000, C=5.0).fit(vec.fit_transform(texts[tr]), cat[tr])
    lp = lambda t: lex.predict(vec.transform(t))
    lexical = {"val": f1_9(cat[va], lp(texts[va])), "test_hard": f1_9(cat[te_h], lp(texts[te_h])),
               "test_all": f1_9(cat[te_a], lp(texts[te_a])), "hardset": f1_9(hcat, lp(htxt)), "hardset_pred": lp(htxt)}

    per_layer = {}
    for l in layers:
        Xtr, Xva = Xc[l][tr], Xc[l][va]
        entry = {}
        for method, cfgs in grids.items():
            best = None
            for cfg in cfgs:
                ps = fit_directions(Xtr, cat[tr], emotions, l, method=method, **cfg)
                calibrate_pooled(ps, Xc[l][cal])
                tau = pick_tau(ps, Xva, cat[va])
                f = f1_9(cat[va], probe_predict(ps, Xva, tau))
                if best is None or f > best[0]:
                    best = (f, ps, tau, dict(cfg, method=method))
            f, ps, tau, cfg = best
            ph = probe_predict(ps, Xh[l], tau)
            entry[method] = {"cfg": cfg, "tau": tau, "val": f,
                             "test_hard": f1_9(cat[te_h], probe_predict(ps, Xc[l][te_h], tau)),
                             "test_all": f1_9(cat[te_a], probe_predict(ps, Xc[l][te_a], tau)),
                             "hardset": f1_9(hcat, ph), "hardset_pred": ph, "_ps": ps}
        sc = StandardScaler().fit(Xtr); bl = None
        for C in LR_CS:
            clf = LogisticRegression(C=C, max_iter=3000).fit(sc.transform(Xtr), cat[tr])
            f = f1_9(cat[va], clf.predict(sc.transform(Xva)))
            if bl is None or f > bl[0]:
                bl = (f, clf, C)
        f, clf, C = bl
        pr = lambda X: clf.predict(sc.transform(X))
        entry["native_lr"] = {"cfg": {"method": "native_lr", "C": C}, "val": f,
                              "test_hard": f1_9(cat[te_h], pr(Xc[l][te_h])), "test_all": f1_9(cat[te_a], pr(Xc[l][te_a])),
                              "hardset": f1_9(hcat, pr(Xh[l])), "hardset_pred": pr(Xh[l])}
        per_layer[l] = entry
        print(f"  layer {l:2d}: " + " | ".join(f"{m} val {entry[m]['val']:.3f} hard {entry[m]['hardset']:.3f}" for m in entry))
    return {"per_layer": per_layer, "lexical": lexical, "hcat": hcat, "htxt": htxt,
            "hsub": np.array([r.get("subtype", "") for r in hrows]),
            "n": {"train": int(tr.sum()), "val_hard": int(va.sum()), "test_hard": int(te_h.sum()),
                  "test_all": int(te_a.sum()), "hardset": int(len(hcat)), "calib_neutral": int(cal.sum())}}


def select(res, layers, methods=METHODS):
    """Best (method, layer) by VAL only. Ties prefer simpler method (METHODS order), then earlier layer."""
    pl = res["per_layer"]
    key = lambda ml: (pl[ml[1]][ml[0]]["val"], -methods.index(ml[0]), -ml[1])
    m, L = max(((m, l) for l in layers for m in methods), key=key)
    best_val = pl[L][m]["val"]
    band = sorted({l for l in layers if max(pl[l][mm]["val"] for mm in methods) >= best_val - 0.03})
    return m, L, band


def write_outputs(out: Path, res, layers, model, emotions, Xh, hrows):
    pl, lex = res["per_layer"], res["lexical"]
    m, L, band = select(res, layers)
    S = pl[L][m]; hc = res["hcat"]; hs = res["hsub"]
    k = int((hc == S["hardset_pred"]).sum()); ci = wilson(k, len(hc))
    cmp_ = {"vs lexical": paired_bootstrap(hc, S["hardset_pred"], lex["hardset_pred"]),
            "vs native LR (ceiling)": paired_bootstrap(hc, S["hardset_pred"], pl[L]["native_lr"]["hardset_pred"])}
    if m != "meandiff":
        cmp_["vs meandiff (same layer)"] = paired_bootstrap(hc, S["hardset_pred"], pl[L]["meandiff"]["hardset_pred"])
    rows_ = [f"# Eval v3: {model}", "",
             f"n: {res['n']}. Selected **{m} @ layer {L}** by VAL-hard macro-F1 only. Config: `{S['cfg']}`, tau {S['tau']:.2f}.",
             f"Layers within 0.03 val-F1 of the best (the plateau): **{band}**", "",
             "## Headline (9-way macro-F1; chance ~0.11)", "",
             "| | VAL-hard | TEST-hard | TEST-all | HAND-WRITTEN set |", "|---|---|---|---|---|"]
    for name in METHODS + ("native_lr",):
        v = pl[L][name]; tag = " (ceiling)" if name == "native_lr" else ""
        rows_.append(f"| {name}{tag} @ L{L} | {v['val']:.3f} | {v['test_hard']:.3f} | {v['test_all']:.3f} | {v['hardset']:.3f} |")
    rows_.append(f"| lexical TF-IDF | {lex['val']:.3f} | {lex['test_hard']:.3f} | {lex['test_all']:.3f} | {lex['hardset']:.3f} |")
    rows_ += ["", f"Selected probe hand-written accuracy {k}/{len(hc)} (95% Wilson {ci[0]:.2f}-{ci[1]:.2f}).", "",
              "## Paired bootstrap on the hand-written set (macro-F1 difference, 95% interval)", "",
              "| comparison | mean diff | 95% interval | excludes 0? |", "|---|---|---|---|"]
    for name, (mean, lo, hi) in cmp_.items():
        rows_.append(f"| selected {name} | {mean:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {'yes' if lo > 0 or hi < 0 else 'no'} |")
    rows_ += ["", "## By hand-written subtype (accuracy at the selected layer)", "",
              "| subtype | n | selected | native LR | lexical |", "|---|---|---|---|---|"]
    for s in sorted(set(hs)):
        mk = hs == s
        rows_.append(f"| {s} | {int(mk.sum())} | {acc(hc[mk], S['hardset_pred'][mk]):.2f} | "
                     f"{acc(hc[mk], pl[L]['native_lr']['hardset_pred'][mk]):.2f} | {acc(hc[mk], lex['hardset_pred'][mk]):.2f} |")
    rows_ += ["", "## Layer x method (VAL-hard / HAND-WRITTEN macro-F1)", "",
              "| layer | " + " | ".join(METHODS + ("native_lr",)) + " |", "|---|" + "---|" * (len(METHODS) + 1)]
    for l in layers:
        rows_.append(f"| {l} | " + " | ".join(f"{pl[l][n]['val']:.2f} / {pl[l][n]['hardset']:.2f}" for n in METHODS + ("native_lr",)) + " |")
    chosen = {n: pl[L][n]["cfg"] for n in METHODS}
    rows_ += ["", "## Chosen hyperparameters at the selected layer", ""] + [f"- {n}: `{c}`" for n, c in chosen.items()]
    (out / "report_v3.md").write_text("\n".join(rows_) + "\n")

    # misclassified hand-written passages, so label judgment calls can be audited
    Z = S["_ps"].score(Xh[L], "pooled")
    err = ["# Hand-written passages the selected probe got wrong", "",
           "Audit these: some may be label judgment calls rather than model errors.", ""]
    for i in np.where(hc != S["hardset_pred"])[0]:
        top = np.argsort(Z[i])[::-1][:3]
        err.append(f"- **[{hs[i]}] true={hc[i]}, predicted={S['hardset_pred'][i]}**  top z: " +
                   ", ".join(f"{emotions[j]} {Z[i, j]:+.1f}" for j in top) + f"\n  > {res['htxt'][i]}")
    (out / "errors_v3.md").write_text("\n".join(err) + "\n")

    best_cfg = {"model": model, "method": m, "layer": L, "cfg": S["cfg"], "tau": S["tau"], "val_f1": S["val"],
                "plateau_layers": band, "skip": 4}
    (out / "best_config.json").write_text(json.dumps(best_cfg, indent=2))
    ser = {str(l): {n: {k2: v for k2, v in d.items() if not k2.startswith("_") and k2 != "hardset_pred"}
                    for n, d in e.items()} for l, e in pl.items()}
    (out / "report_v3.json").write_text(json.dumps(
        {"selected": best_cfg, "per_layer": ser, "lexical": {k2: v for k2, v in lex.items() if k2 != "hardset_pred"},
         "bootstrap": cmp_, "n": res["n"]}, indent=2, default=float))
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 4.5))
        for n in METHODS + ("native_lr",):
            plt.plot(layers, [pl[l][n]["hardset"] for l in layers], "o-" if n != "native_lr" else "s--", label=n)
        plt.axhline(lex["hardset"], c="red", ls=":", label="lexical"); plt.axhline(1 / 9, c="gray", ls="--", label="chance")
        for l in band: plt.axvspan(l - 0.5, l + 0.5, color="green", alpha=0.06)
        plt.xlabel("layer (after block)"); plt.ylabel("hand-written-set macro-F1"); plt.legend(fontsize=8)
        plt.tight_layout(); plt.savefig(out / "layer_curve_v3.png", dpi=140)
    except Exception as e:
        print(f"(plot skipped: {e})")
    return best_cfg


def load_states(items, cache: Path, layers, skip, model):
    ids = [i["id"] for i in items]
    if cache.exists():
        d = np.load(cache, allow_pickle=False); files = set(d.files)
        if list(d["ids"]) == ids and int(d["skip"]) == skip and all(f"L{l}" in files for l in layers):
            print(f"  cache hit: {cache}")
            return {l: d[f"L{l}"] for l in layers}
    print(f"  cache stale/missing for {cache.name}: extracting with the model")
    from probekit_capture import Extractor
    ex = Extractor(model)
    all_layers = sorted(set(layers) | set(range(0, 25, 2)) | {1, 3})
    acc_ = {l: [] for l in all_layers}
    for n, it in enumerate(items):
        for l, v in ex.pooled(it["text"], all_layers, skip).items():
            acc_[l].append(v)
        if (n + 1) % 100 == 0:
            print(f"    {n + 1}/{len(items)}")
    X = {l: np.stack(v) for l, v in acc_.items()}
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, ids=np.array(ids), skip=skip, **{f"L{l}": X[l] for l in all_layers})
    return {l: X[l] for l in layers}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Llama-3.2-3B-Instruct")
    ap.add_argument("--corpus", default="data/corpus/corpus_v2_scored.jsonl")
    ap.add_argument("--hardset", default="data/hardset/hardset.jsonl")
    ap.add_argument("--own-voice", default="data/hardset/own_voice.jsonl")
    ap.add_argument("--cache-dir", default=None, help="default: data/eval_v2/<model> (reuses v2 hidden states)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--layers", type=int, nargs="*", default=MID_BAND)
    ap.add_argument("--skip", type=int, default=4)
    ap.add_argument("--train-on", choices=["all", "hard"], default="all")
    ap.add_argument("--fast", action="store_true")
    a = ap.parse_args()

    from build_emotion_vectors import slug
    cache_dir = Path(a.cache_dir or f"data/eval_v2/{slug(a.model)}")
    out = Path(a.out or f"data/eval_v3/{slug(a.model)}"); out.mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(a.corpus); hrows = load_jsonl(a.hardset)
    if Path(a.own_voice).exists():
        own = [dict(r, id=f"own|{i}") for i, r in enumerate(load_jsonl(a.own_voice)) if r.get("text", "").strip()]
        print(f"own-voice passages: {len(own)}"); hrows += own
    emotions = [e for e in DEFAULT_EMOTIONS if any(r["category"] == e for r in rows)]
    Xc = load_states(rows, cache_dir / "states_corpus.npz", a.layers, a.skip, a.model)
    Xh = load_states(hrows, cache_dir / "states_hardset.npz", a.layers, a.skip, a.model)
    res = sweep(a.layers, Xc, rows, Xh, hrows, emotions, GRIDS_FAST if a.fast else GRIDS_FULL, a.train_on)
    best = write_outputs(out, res, a.layers, a.model, emotions, Xh, hrows)
    print(f"\nSelected {best['method']} @ layer {best['layer']} (val F1 {best['val_f1']:.3f}); plateau layers {best['plateau_layers']}")
    print(f"Report: {out/'report_v3.md'}   Errors: {out/'errors_v3.md'}   Config: {out/'best_config.json'}")


if __name__ == "__main__":
    main()
