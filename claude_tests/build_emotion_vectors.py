#!/usr/bin/env python3
"""
build_emotion_vectors.py (v2) -- build and HONESTLY validate emotion probes.

    python probekit_corpus.py                    # once: generate the labeled corpus
    python build_emotion_vectors.py              # build, sweep, validate, export
    python build_emotion_vectors.py --external   # also test on dair-ai/emotion (needs `pip install datasets`)

What it does
  1. Extracts mean-pooled residual states for every passage, ALL candidate layers, in one pass each (cached).
  2. Sweeps (layer x nuisance-variance cutoff), fitting on TRAIN topics and scoring on VAL topics.
  3. Refits the winner on TRAIN+VAL, then reports on TEST topics the sweep never saw:
       accuracy / macro-F1 (chance = 12.5% for 8 emotions), per-emotion AUC, confusion matrix,
       a label-shuffle NULL (real probes must beat its 95th percentile), and split-half reliability.
  4. Calibrates single-token z-scores for live streaming and writes probes.npz + report.json/.md + sweep.png.
  5. Exports legacy data/emotion_vectors/<emotion>.json so facet_mapping.py keeps working
     (re-run `python facet_mapping.py` afterward: the vectors changed).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from probekit_probes import (DEFAULT_EMOTIONS, NEUTRAL, evaluate, fit_probes, null_distribution,
                             split_half_reliability)

DAIR_LABELS = {0: "sadness", 1: "joy", 2: "love", 3: "anger", 4: "fear", 5: "surprise"}


def slug(name: str) -> str:
    return name.replace("/", "__")


def load_corpus(path):
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    if not rows:
        raise SystemExit(f"{path} is empty; run probekit_corpus.py first")
    return rows


def extract_all(ex, rows, layers, skip, cache: Path):
    ids = np.array([r["id"] for r in rows])
    if cache.exists():
        d = np.load(cache, allow_pickle=False)
        if list(d["ids"]) == list(ids) and all(f"L{l}" in d for l in layers) and int(d["skip"]) == skip:
            print(f"Using cached states: {cache}")
            return {l: d[f"L{l}"] for l in layers}
    acc = {l: [] for l in layers}
    for i, r in enumerate(rows):
        for l, v in ex.pooled(r["text"], layers, skip).items():
            acc[l].append(v)
        if (i + 1) % 50 == 0:
            print(f"  extracted {i + 1}/{len(rows)}")
    X = {l: np.stack(v) for l, v in acc.items()}
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, ids=ids, skip=skip, **{f"L{l}": X[l] for l in layers})
    return X


def external_eval(ex, ps, skip, n=600, seed=0):
    from datasets import load_dataset  # optional dependency
    ds = load_dataset("dair-ai/emotion", split="test").shuffle(seed=seed).select(range(n))
    cols = [ps.emotions.index(e) for e in DAIR_LABELS.values() if e in ps.emotions]
    names = [ps.emotions[c] for c in cols]
    X = np.stack([ex.pooled(t, [ps.layer], skip=min(skip, 2))[ps.layer] for t in ds["text"]])
    y = np.array([DAIR_LABELS[l] for l in ds["label"]])
    Z = ps.score(X, "pooled")[:, cols]
    pred = np.array(names)[Z.argmax(1)]
    from sklearn.metrics import f1_score
    return {"n": int(n), "emotions": names, "chance_accuracy": 1 / len(names),
            "accuracy": float((pred == y).mean()),
            "macro_f1": float(f1_score(y, pred, labels=names, average="macro", zero_division=0)),
            "note": "tweets of the form 'i feel ...' contain explicit emotion words: a domain-shift check, "
                    "not a like-for-like score"}


def write_report(out: Path, rep: dict):
    (out / "report.json").write_text(json.dumps(rep, indent=2))
    t, nl = rep["test"], rep["null"]
    L = [f"# Probe report: {rep['model']}", "",
         f"- layer **{rep['layer']}** (after block), nuisance cutoff {rep['nuisance_var_frac']} "
         f"({rep['nuisance_k']} PCs removed), contrast `{rep['contrast']}`, pooled tokens skip={rep['skip']}",
         f"- corpus: {rep['n_train']} train / {rep['n_val']} val / {rep['n_test']} test passages (split by TOPIC)",
         "", "## Held-out TEST topics (never used for layer or hyperparameter selection)", "",
         f"- accuracy **{t['accuracy']:.3f}** (chance {t['chance_accuracy']:.3f}); macro-F1 **{t['macro_f1']:.3f}**",
         f"- label-shuffle null macro-F1: mean {nl['mean']:.3f}, p95 {nl['p95']:.3f}, max {nl['max']:.3f}",
         f"- emotional-vs-neutral AUC: {t['auc_emotional_vs_neutral']:.3f}", "",
         "| emotion | n test | AUC (1-vs-rest) | split-half cosine |", "|---|---|---|---|"]
    for e in rep["emotions"]:
        L.append(f"| {e} | {t['n_per_emotion'][e]} | {t['auc_one_vs_rest'].get(e, float('nan')):.3f} | "
                 f"{rep['split_half'][e]:.2f} |")
    L += ["", "## Layer sweep (best val macro-F1 per layer)", "", "| layer | val macro-F1 |", "|---|---|"]
    L += [f"| {l} | {f:.3f} |" for l, f in rep["sweep_by_layer"].items()]
    if "external" in rep:
        x = rep["external"]
        L += ["", "## External check (dair-ai/emotion)", "",
              f"- accuracy {x['accuracy']:.3f} (chance {x['chance_accuracy']:.3f}), macro-F1 {x['macro_f1']:.3f}",
              f"- {x['note']}"]
    weak = [e for e in rep["emotions"] if t["auc_one_vs_rest"].get(e, 1) < 0.75 or rep["split_half"][e] < 0.5]
    L += ["", "## Verdict", ""]
    L.append(("Weak concepts (AUC < 0.75 or unstable): " + ", ".join(weak) + ". Treat these as unreliable, add data.")
             if weak else "All concepts clear the AUC/stability bar on held-out topics.")
    L.append("Probes are only meaningful if test macro-F1 is far above the null p95.")
    (out / "report.md").write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Llama-3.2-3B-Instruct")
    ap.add_argument("--corpus", default="data/corpus/corpus.jsonl")
    ap.add_argument("--out", default=None, help="default: data/probes/<model-slug>")
    ap.add_argument("--layers", type=int, nargs="*", default=None, help="default: every 2nd block from 6 to n-4")
    ap.add_argument("--skip", type=int, default=4, help="tokens dropped from the start of each passage when pooling")
    ap.add_argument("--contrast", choices=["mean_emotion", "neutral"], default="mean_emotion")
    ap.add_argument("--var-fracs", type=float, nargs="*", default=[0.5, 0.8, 0.9, 0.95])
    ap.add_argument("--null-runs", type=int, default=20)
    ap.add_argument("--external", action="store_true")
    ap.add_argument("--no-legacy-export", action="store_true")
    a = ap.parse_args()

    from probekit_capture import Extractor
    ex = Extractor(a.model)
    layers = a.layers or list(range(6, ex.n_layers - 3, 2))
    out = Path(a.out or f"data/probes/{slug(a.model)}"); out.mkdir(parents=True, exist_ok=True)

    rows = load_corpus(a.corpus)
    cats = np.array([r["category"] for r in rows]); split = np.array([r["split"] for r in rows])
    emotions = [e for e in DEFAULT_EMOTIONS if (cats == e).any()]
    print(f"{len(rows)} passages; emotions={emotions}; layers={layers}")
    tr, va, te = (split == "train"), (split == "val"), (split == "test")
    for nm, m in (("train", tr), ("val", va), ("test", te)):
        print(f"  {nm}: {int(m.sum())} passages, {int((m & (cats != NEUTRAL)).sum() // max(1, len(emotions)))} per emotion")

    X = extract_all(ex, rows, layers, a.skip, out / "states.npz")

    # ---- sweep layer x nuisance cutoff on VAL topics ---------------------------------------
    best, sweep, by_layer = None, [], {}
    for l in layers:
        for vf in a.var_fracs:
            ps = fit_probes(X[l][tr], cats[tr], emotions, l, var_frac=vf, contrast=a.contrast)
            f1 = evaluate(ps, X[l][va], cats[va])["macro_f1"]
            sweep.append({"layer": l, "var_frac": vf, "val_macro_f1": f1})
            by_layer[l] = max(by_layer.get(l, 0), f1)
            if best is None or f1 > best[0]:
                best = (f1, l, vf)
        print(f"  layer {l:2d}: best val macro-F1 {by_layer[l]:.3f}")
    _, L, VF = best
    print(f"Selected layer {L}, var_frac {VF} (val macro-F1 {best[0]:.3f})")

    # ---- final fit on train+val, report on untouched TEST ---------------------------------
    trva = tr | va
    fit_kw = dict(var_frac=VF, contrast=a.contrast)
    ps = fit_probes(X[L][trva], cats[trva], emotions, L, **fit_kw)
    test = evaluate(ps, X[L][te], cats[te])
    null = null_distribution(X[L][trva], cats[trva], X[L][te], cats[te], emotions, L, runs=a.null_runs, **fit_kw)
    half = split_half_reliability(X[L][trva], cats[trva], emotions, L, **fit_kw)

    # ---- single-token calibration for live streaming ---------------------------------------
    rng = np.random.default_rng(0)
    def sample_tokens(mask, k=60):
        idx = rng.permutation(np.where(mask)[0])[:k]
        return np.concatenate([ex.token_states(rows[i]["text"], L, a.skip) for i in idx])
    ps.calibrate_tokens(sample_tokens(trva & (cats == NEUTRAL)), sample_tokens(trva & (cats != NEUTRAL)))
    ps.meta.update({"model": a.model, "skip": a.skip, "z_full": ps.meta.get("z_full", 4.0)})
    ps.save(out / "probes.npz")

    rep = {"model": a.model, "layer": L, "skip": a.skip, "emotions": emotions, "contrast": a.contrast,
           "nuisance_var_frac": VF, "nuisance_k": ps.meta["nuisance_k"], "z_full": ps.meta["z_full"],
           "n_train": int(tr.sum()), "n_val": int(va.sum()), "n_test": int(te.sum()),
           "test": test, "null": null, "split_half": half, "sweep": sweep, "sweep_by_layer": by_layer}
    if a.external:
        rep["external"] = external_eval(ex, ps, a.skip)
    write_report(out, rep)

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 4)); plt.plot(list(by_layer), list(by_layer.values()), "o-", label="val macro-F1")
        plt.axhline(1 / len(emotions), ls="--", c="gray", label="chance"); plt.axhline(null["p95"], ls=":", c="red", label="null p95")
        plt.xlabel("layer (after block)"); plt.ylabel("macro-F1"); plt.legend(); plt.tight_layout(); plt.savefig(out / "sweep.png", dpi=140)
    except Exception as e:  # plotting is optional
        print(f"(skipped plot: {e})")

    if not a.no_legacy_export:
        legacy = Path("data/emotion_vectors"); legacy.mkdir(parents=True, exist_ok=True)
        for i, e in enumerate(emotions):
            (legacy / f"{e}.json").write_text(json.dumps({"emotion": e, "vector": ps.V[i].tolist(), "layer": L,
                                                          "source": "probekit v2"}))
        print(f"Legacy vectors exported to {legacy}/ -- re-run facet_mapping.py")

    print(f"\nTEST accuracy {test['accuracy']:.3f} | macro-F1 {test['macro_f1']:.3f} | null p95 {null['p95']:.3f} "
          f"| chance {test['chance_accuracy']:.3f}")
    print(f"Report: {out/'report.md'}")


if __name__ == "__main__":
    main()
