#!/usr/bin/env python3
"""
probekit_eval_v2.py -- do the probes read something beyond vocabulary?

Everything is 9-way (8 emotions + neutral) so the harder sets can include neutral text. For each layer:
  * PROBE     mean-difference directions (probekit_probes), nuisance cutoff and a 'neutral' threshold chosen on VAL
  * HIDDEN-LR logistic regression on the same hidden states (fairer: a trained classifier, like the word baseline)
  * LEXICAL   TF-IDF unigram logistic regression (the shortcut we must beat)
Evaluated on: TEST-hard (adversarially filtered), TEST-all, the HAND-WRITTEN hard set (+ your own-voice set),
and a WORD-SHUFFLED copy of the test split (destroys syntax/negation, keeps the words).

Layer selection uses VAL-hard only. The hand-written set is never used for any selection.

    python probekit_eval_v2.py --corpus data/corpus/corpus_v2_scored.jsonl
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from probekit_probes import DEFAULT_EMOTIONS, NEUTRAL, fit_probes, null_distribution

LAYERS = [0, 1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]
VAR_FRACS = [0.8, 0.9, 0.95]
LR_CS = [0.003, 0.03, 0.3]


# ---------------------------------------------------------------- small helpers
def f1_9(y, p):
    return float(f1_score(y, p, labels=sorted(set(y)), average="macro", zero_division=0))


def acc(y, p):
    return float((np.asarray(y) == np.asarray(p)).mean())


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    ph = k / n
    d = 1 + z * z / n
    c = ph + z * z / (2 * n)
    r = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n))
    return ((c - r) / d, (c + r) / d)


def probe_predict(ps, X, tau):
    Z = ps.score(X, "pooled")
    pred = np.array(ps.emotions, dtype=object)[Z.argmax(1)]
    pred[Z.max(1) < tau] = NEUTRAL
    return pred


def pick_tau(ps, X, y):
    zmax = ps.score(X, "pooled").max(1)
    cands = np.unique(np.percentile(zmax, np.arange(0, 101, 5)))
    scores = [(f1_9(y, probe_predict(ps, X, t)), -t) for t in cands]   # tie -> larger tau
    return float(-max(scores)[1])


def shuffle_words(text, rng):
    w = text.split(); rng.shuffle(w); return " ".join(w)


# ---------------------------------------------------------------- core analysis (model-free, unit-testable)
def analyze(layers, Xc, rows, Xh, hrows, Xs, srows_idx, emotions, train_on="all", seed=0):
    """
    Xc[l]: (N,H) pooled states for corpus rows; Xh[l]: (M,H) for hard-set rows; Xs[l]: (K,H) for word-shuffled
    copies of corpus rows `srows_idx` (indices into rows). Returns a results dict.
    """
    cat = np.array([r["category"] for r in rows]); spl = np.array([r["split"] for r in rows])
    hard = np.array([bool(r.get("hard", True)) for r in rows])
    texts = np.array([r["text"] for r in rows])
    hcat = np.array([r["category"] for r in hrows]); hsub = np.array([r.get("subtype", "") for r in hrows])
    htxt = np.array([r["text"] for r in hrows])

    tr = (spl == "train") & (hard if train_on == "hard" else True)
    va = (spl == "val") & hard
    te_h = (spl == "test") & hard
    te_a = (spl == "test")
    srows_idx = np.asarray(srows_idx)
    s_is_hard = hard[srows_idx]

    # ---- lexical baseline (layer independent). Unigram bag-of-words is invariant to shuffling by construction.
    vec = TfidfVectorizer(sublinear_tf=True, min_df=2, stop_words="english")
    lex = LogisticRegression(max_iter=3000, C=5.0).fit(vec.fit_transform(texts[tr]), cat[tr])
    lp = lambda t: lex.predict(vec.transform(t))
    lexical = {"val": f1_9(cat[va], lp(texts[va])), "test_hard": f1_9(cat[te_h], lp(texts[te_h])),
               "test_all": f1_9(cat[te_a], lp(texts[te_a])), "hardset": f1_9(hcat, lp(htxt)),
               "hardset_pred": lp(htxt)}

    per_layer = {}
    for l in layers:
        Xtr, Xva = Xc[l][tr], Xc[l][va]
        # PROBE: choose nuisance cutoff + neutral threshold on VAL
        best = None
        for vf in VAR_FRACS:
            ps = fit_probes(Xtr, cat[tr], emotions, l, var_frac=vf)
            tau = pick_tau(ps, Xva, cat[va])
            f = f1_9(cat[va], probe_predict(ps, Xva, tau))
            if best is None or f > best[0]:
                best = (f, ps, tau, vf)
        f_val, ps, tau, vf = best
        pp = lambda X: probe_predict(ps, X, tau)
        # HIDDEN-LR
        sc = StandardScaler().fit(Xtr); bestlr = None
        for C in LR_CS:
            clf = LogisticRegression(C=C, max_iter=3000).fit(sc.transform(Xtr), cat[tr])
            f = f1_9(cat[va], clf.predict(sc.transform(Xva)))
            if bestlr is None or f > bestlr[0]:
                bestlr = (f, clf, C)
        lr_val, clf, C = bestlr
        lp_h = lambda X: clf.predict(sc.transform(X))

        ph, lh = pp(Xh[l]), lp_h(Xh[l])
        per_layer[l] = {
            "var_frac": vf, "tau": tau, "lr_C": C,
            "probe": {"val": f_val, "test_hard": f1_9(cat[te_h], pp(Xc[l][te_h])),
                      "test_all": f1_9(cat[te_a], pp(Xc[l][te_a])), "hardset": f1_9(hcat, ph),
                      "shuffled_test_hard": f1_9(cat[srows_idx][s_is_hard], pp(Xs[l][s_is_hard])),
                      "hardset_pred": ph},
            "hidden_lr": {"val": lr_val, "test_hard": f1_9(cat[te_h], lp_h(Xc[l][te_h])),
                          "test_all": f1_9(cat[te_a], lp_h(Xc[l][te_a])), "hardset": f1_9(hcat, lh),
                          "shuffled_test_hard": f1_9(cat[srows_idx][s_is_hard], lp_h(Xs[l][s_is_hard])),
                          "hardset_pred": lh},
            "_ps": ps,
        }

    L = max(layers, key=lambda l: (per_layer[l]["probe"]["val"], -l))     # select on VAL only; tie -> earlier
    ps = per_layer[L]["_ps"]
    sub = {}
    for s in sorted(set(hsub)):
        m = hsub == s
        sub[s] = {"n": int(m.sum()),
                  "probe": acc(hcat[m], per_layer[L]["probe"]["hardset_pred"][m]),
                  "hidden_lr": acc(hcat[m], per_layer[L]["hidden_lr"]["hardset_pred"][m]),
                  "lexical": acc(hcat[m], lexical["hardset_pred"][m])}
    k = int((hcat == per_layer[L]["probe"]["hardset_pred"]).sum())
    ci = wilson(k, len(hcat))
    emo_te = te_h & (cat != NEUTRAL)
    null = null_distribution(Xc[L][tr], cat[tr], Xc[L][emo_te], cat[emo_te], emotions, L, runs=15,
                             var_frac=per_layer[L]["var_frac"])
    for l in layers:
        for k2 in ("probe", "hidden_lr"):
            per_layer[l][k2].pop("hardset_pred")
        per_layer[l].pop("_ps")
    lexical.pop("hardset_pred")
    return {"layers": layers, "selected_layer": L, "per_layer": per_layer, "lexical": lexical, "subtypes": sub,
            "hardset_acc_probe": {"k": k, "n": len(hcat), "wilson95": ci}, "null_8way_test_hard": null,
            "n": {"train": int(tr.sum()), "val": int(va.sum()), "test_hard": int(te_h.sum()),
                  "test_all": int(te_a.sum()), "hardset": int(len(hcat))}, "train_on": train_on}


# ---------------------------------------------------------------- reporting
def write_report(out: Path, res: dict, model: str):
    L, pl, lex = res["selected_layer"], res["per_layer"], res["lexical"]
    P = pl[L]["probe"]; R = pl[L]["hidden_lr"]
    gap = P["hardset"] - lex["hardset"]; ci = res["hardset_acc_probe"]["wilson95"]
    best_hard = max(res["layers"], key=lambda l: pl[l]["probe"]["hardset"])
    lines = [f"# Eval v2: {model}", "",
             f"n: {res['n']} (train on `{res['train_on']}`); selected layer **{L}** by VAL-hard probe macro-F1 only.", "",
             "## Headline (9-way macro-F1; chance ~0.11)", "",
             "| | VAL-hard | TEST-hard | TEST-all | HAND-WRITTEN hard set | word-shuffled TEST-hard |", "|---|---|---|---|---|---|",
             f"| probe @ L{L} | {P['val']:.3f} | {P['test_hard']:.3f} | {P['test_all']:.3f} | **{P['hardset']:.3f}** | {P['shuffled_test_hard']:.3f} |",
             f"| hidden-state LR @ L{L} | {R['val']:.3f} | {R['test_hard']:.3f} | {R['test_all']:.3f} | **{R['hardset']:.3f}** | {R['shuffled_test_hard']:.3f} |",
             f"| lexical TF-IDF | {lex['val']:.3f} | {lex['test_hard']:.3f} | {lex['test_all']:.3f} | **{lex['hardset']:.3f}** | (same by construction) |", "",
             f"Hard-set probe accuracy {res['hardset_acc_probe']['k']}/{res['hardset_acc_probe']['n']} "
             f"(95% Wilson CI {ci[0]:.2f}-{ci[1]:.2f}). Label-shuffle null (8-way, test-hard): mean "
             f"{res['null_8way_test_hard']['mean']:.3f}, p95 {res['null_8way_test_hard']['p95']:.3f}.", "",
             "## By hard-set subtype (accuracy at the selected layer)", "",
             "| subtype | n | probe | hidden-LR | lexical |", "|---|---|---|---|---|"]
    for s, v in res["subtypes"].items():
        lines.append(f"| {s} | {v['n']} | {v['probe']:.2f} | {v['hidden_lr']:.2f} | {v['lexical']:.2f} |")
    lines += ["", "## Layer curve (macro-F1)", "",
              "| layer | probe val | probe test-hard | probe HARD SET | hidden-LR HARD SET | probe shuffled |", "|---|---|---|---|---|---|"]
    for l in res["layers"]:
        a = pl[l]
        lines.append(f"| {l} | {a['probe']['val']:.3f} | {a['probe']['test_hard']:.3f} | {a['probe']['hardset']:.3f} | "
                     f"{a['hidden_lr']['hardset']:.3f} | {a['probe']['shuffled_test_hard']:.3f} |")
    lines += ["", "## Reading this (mechanical, not a proof)", "",
              f"- Probe minus lexical on the hand-written set: **{gap:+.3f}** macro-F1 at the layer chosen WITHOUT looking at it. "
              f"With n={res['n']['hardset']}, differences under ~0.10 are within noise.",
              f"- Layer with the best hand-written-set probe score: {best_hard} (for inspection only; picking it would be test-set selection).",
              f"- Shuffle drop for the probe: {P['test_hard'] - P['shuffled_test_hard']:+.3f}. A large drop means the probe uses word order/context; "
              f"~0 means it behaves like a bag of words (not proof either way: a concept could also survive shuffling).",
              "- Weak spots to check first: `negation` and `emotion_word_neutral` rows. Lexical-style shortcuts fail there."]
    if gap > 0.10 and P["hardset"] > res["null_8way_test_hard"]["p95"] + 0.2:
        lines.append("- **Consistent with the probes carrying information beyond vocabulary.** Still needs: other writers, other models, steering tests.")
    elif gap < 0.05:
        lines.append("- **No evidence yet that the probes beat word statistics.** Treat the visualizer as an emotional-vocabulary detector until this changes.")
    else:
        lines.append("- **Ambiguous.** Probes edge out words but inside the noise; add more own-voice passages before drawing conclusions.")
    (out / "report_v2.md").write_text("\n".join(lines) + "\n")
    (out / "report_v2.json").write_text(json.dumps(res, indent=2, default=float))
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        xs = res["layers"]; plt.figure(figsize=(8, 4.5))
        plt.plot(xs, [pl[l]["probe"]["hardset"] for l in xs], "o-", label="probe: hand-written set")
        plt.plot(xs, [pl[l]["hidden_lr"]["hardset"] for l in xs], "s-", label="hidden-state LR: hand-written set")
        plt.plot(xs, [pl[l]["probe"]["test_hard"] for l in xs], "o--", alpha=.6, label="probe: test-hard")
        plt.axhline(lex["hardset"], c="red", ls=":", label="lexical: hand-written set")
        plt.axhline(1 / 9, c="gray", ls="--", label="chance"); plt.xlabel("layer (after block)"); plt.ylabel("macro-F1")
        plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(out / "layer_curve_v2.png", dpi=140)
    except Exception as e:
        print(f"(plot skipped: {e})")


# ---------------------------------------------------------------- extraction with a robust cache
def extract(ex, items, layers, skip, cache: Path):
    ids = [i["id"] for i in items]
    if cache.exists():
        d = np.load(cache, allow_pickle=False); files = set(d.files)
        if list(d["ids"]) == ids and int(d["skip"]) == skip and all(f"L{l}" in files for l in layers):
            print(f"  cache hit: {cache.name}"); return {l: d[f"L{l}"] for l in layers}
    acc_ = {l: [] for l in layers}
    for n, it in enumerate(items):
        for l, v in ex.pooled(it["text"], layers, skip).items():
            acc_[l].append(v)
        if (n + 1) % 100 == 0:
            print(f"  {cache.name}: {n + 1}/{len(items)}")
    X = {l: np.stack(v) for l, v in acc_.items()}
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, ids=np.array(ids), skip=skip, **{f"L{l}": X[l] for l in layers})
    return X


def load_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Llama-3.2-3B-Instruct")
    ap.add_argument("--corpus", default="data/corpus/corpus_v2_scored.jsonl")
    ap.add_argument("--hardset", default="data/hardset/hardset.jsonl")
    ap.add_argument("--own-voice", default="data/hardset/own_voice.jsonl")
    ap.add_argument("--layers", type=int, nargs="*", default=LAYERS)
    ap.add_argument("--skip", type=int, default=4)
    ap.add_argument("--train-on", choices=["all", "hard"], default="all")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from build_emotion_vectors import slug
    from probekit_capture import Extractor
    out = Path(a.out or f"data/eval_v2/{slug(a.model)}"); out.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(a.corpus)
    if "hard" not in rows[0]:
        raise SystemExit("corpus has no 'hard' field: run adversarial_filter.py first")
    hrows = load_jsonl(a.hardset)
    if Path(a.own_voice).exists():
        own = [dict(r, id=f"own|{i}") for i, r in enumerate(load_jsonl(a.own_voice)) if r.get("text", "").strip()]
        print(f"own-voice passages: {len(own)}"); hrows += own
    rng = np.random.default_rng(0)
    s_idx = [i for i, r in enumerate(rows) if r["split"] == "test"]
    srows = [{"id": "shuf|" + rows[i]["id"], "text": shuffle_words(rows[i]["text"], rng)} for i in s_idx]
    emotions = [e for e in DEFAULT_EMOTIONS if any(r["category"] == e for r in rows)]
    print(f"{len(rows)} corpus, {len(hrows)} hard-set, {len(srows)} shuffled; layers {a.layers}")

    ex = Extractor(a.model)
    Xc = extract(ex, rows, a.layers, a.skip, out / "states_corpus.npz")
    Xh = extract(ex, hrows, a.layers, a.skip, out / "states_hardset.npz")
    Xs = extract(ex, srows, a.layers, a.skip, out / "states_shuffled.npz")
    res = analyze(a.layers, Xc, rows, Xh, hrows, Xs, s_idx, emotions, a.train_on)
    write_report(out, res, a.model)
    L = res["selected_layer"]; P = res["per_layer"][L]["probe"]
    print(f"\nselected layer {L}: probe hard-set F1 {P['hardset']:.3f} | hidden-LR "
          f"{res['per_layer'][L]['hidden_lr']['hardset']:.3f} | lexical {res['lexical']['hardset']:.3f}")
    print(f"report: {out/'report_v2.md'}")


if __name__ == "__main__":
    main()
