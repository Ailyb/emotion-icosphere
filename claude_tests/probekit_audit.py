#!/usr/bin/env python3
"""
probekit_audit.py -- PLAN_v2 T4: one command that audits whether emotion probes carry information beyond cheap baselines.

Model-agnostic: it reads texts + labels (JSONL) and CACHED activations (npz written by the extraction step), never a model.

    python probekit_audit.py --train data/corpus/corpus_merged2_scored.jsonl --train-cache data/eval_merged2/<slug>/states_corpus.npz \
        --test goemotions=data/testsets/goemotions_test_frozen.jsonl=data/eval_merged2/<slug>/states_frozen.npz \
        --exclude calm --layer 16 --out audit.md

Cache format: npz with `ids` (same order as the JSONL), `skip`, `L<layer>` (N,d) mean-pooled states, optional `E` (N,d) bag-of-embeddings,
and (recommended) a `<cache>.meta.json` sidecar (see write_meta). Train JSONL rows carry `split` (train/val) and `category`; `hard`
(optional) restricts validation to the hard passages.

Reported per test set: per-class n, macro-F1 [bootstrap CI] and shuffle-null p95 for the probe, native hidden-state LR, TF-IDF, and the
bag-of-embeddings baseline; paired differences; length strata; classes skipped and why. Everything is trained once on the train split;
tau and C are chosen on VAL only; nothing is tuned on the test sets.
"""
import argparse
import hashlib
import json
import subprocess
import time
from collections import Counter
from pathlib import Path

import numpy as np

from probekit_eval_v2 import f1_9, load_jsonl
from probekit_eval_v3 import paired_bootstrap
from probekit_external_eval import evaluate_source, fit_systems
from probekit_frozen_eval import boot_ci, shuffle_null_p95

MIN_TRAIN, MIN_TEST = 20, 10
DEFAULT_CFG = {"var_frac": 0.5, "shrinkage": "auto"}


def write_meta(cache_path, model, skip, layers, ids, pooling="mean over tokens[skip:]", **extra):
    """Provenance sidecar for an activation cache (PLAN_v2 T4 metadata)."""
    try:
        rev = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5).stdout.strip() or None
    except Exception:
        rev = None
    meta = {"model": model, "skip": skip, "layers": sorted(layers), "pooling": pooling, "n": len(ids),
            "ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(), "code_version": rev,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"), **extra}
    Path(str(cache_path) + ".meta.json").write_text(json.dumps(meta, indent=1))
    return meta


def load_cache(path, rows, embed_path=None):
    """`embed_path`: optional separate npz holding the bag-of-embeddings matrix `E` (some caches store it apart)."""
    d = np.load(path, allow_pickle=False)
    if list(d["ids"]) != [r["id"] for r in rows]:
        raise ValueError(f"{path}: cached ids do not match the JSONL order; re-extract")
    layers = sorted(int(k[1:]) for k in d.files if k.startswith("L") and k[1:].isdigit())
    meta = Path(str(path) + ".meta.json")
    E = d["E"] if "E" in d.files else None
    if embed_path:
        de = np.load(embed_path, allow_pickle=False)
        if list(de["ids"]) != [r["id"] for r in rows]:
            raise ValueError(f"{embed_path}: cached ids do not match the JSONL order; re-extract")
        E = de["E"]
    return {"X": {l: d[f"L{l}"] for l in layers}, "E": E,
            "meta": json.loads(meta.read_text()) if meta.exists() else None}


def class_table(y_train, tests):
    """Per-class counts for train and every test set, plus the evaluable class list per test set (skip-and-report)."""
    tr = Counter(map(str, y_train)); ok_train = {c for c, n in tr.items() if n >= MIN_TRAIN}
    out, skipped = {}, {}
    for name, y in tests.items():
        te = Counter(y)
        out[name] = {c: te.get(c, 0) for c in sorted(set(te) | set(tr))}
        ev = sorted(c for c, n in te.items() if n >= MIN_TEST and c in ok_train)
        skipped[name] = {c: ("not in train (or < %d)" % MIN_TRAIN if c not in ok_train else f"test n < {MIN_TEST}") for c in te if c not in ev}
        tests_ev = ev
        out[name]["_evaluated"] = tests_ev
    return tr, out, skipped


def audit(train, tests, layer, exclude=(), cfg=None, method="lda", min_words_strata=12):
    """train: dict(rows, feats); tests: {name: dict(rows, feats)}. Returns (report_md, results dict)."""
    cfg = cfg or DEFAULT_CFG
    rows, F = train["rows"], train["feats"]
    cat = np.array([str(r["category"]) for r in rows], dtype=object); spl = np.array([r.get("split", "train") for r in rows])
    hard = np.array([bool(r.get("hard", True)) for r in rows]); txt = np.array([r["text"] for r in rows], dtype=object)
    keep = ~np.isin(cat, list(exclude))
    tr = (spl == "train") & keep; va = (spl == "val") & hard & keep; cal = (spl == "val") & (cat == "neutral")
    if va.sum() == 0 or cal.sum() == 0 or "neutral" not in set(cat[tr]):
        raise ValueError("audit needs a 'val' split with neutral rows and a neutral class in train (probe calibration and tau)")
    emotions = sorted(str(c) for c in set(cat[tr]) if c != "neutral" and (cat[tr] == c).sum() >= MIN_TRAIN)
    keep_cls = set(emotions) | {"neutral"}
    tr, va = tr & np.isin(cat, list(keep_cls)), va & np.isin(cat, list(keep_cls))
    E = F["E"]
    systems, tau = fit_systems(F["X"][layer][tr], E[tr] if E is not None else None, txt[tr], cat[tr], F["X"][layer][va], E[va] if E is not None else None,
                               cat[va], F["X"][layer][cal], layer, cfg, method, emotions=emotions)
    ytr_all = list(cat[tr])
    tests_y = {n: [r["category"] for r in t["rows"]] for n, t in tests.items()}
    train_counts, per_class, skipped = class_table(ytr_all, tests_y)
    L = [f"# Probe audit (layer {layer}, {method}; trained on n={int(tr.sum())}, validated on n={int(va.sum())}; classes {sorted(map(str, keep_cls))})", "",
         f"Train classes (n): {dict(sorted((str(k), v) for k, v in train_counts.items()))}; excluded by request: {list(exclude) or 'none'}. tau (neutral threshold) = {tau:.2f}, chosen on VAL.", ""]
    meta = train["feats"].get("meta")
    L.append("Cache provenance: " + (f"model {meta['model']}, pooling {meta['pooling']}, skip {meta['skip']}, code {str(meta.get('code_version'))[:8]}" if meta else "NO META SIDECAR (provenance unknown)") + "\n")
    results = {}
    for name, t in tests.items():
        classes = per_class[name]["_evaluated"]
        if len(classes) < 2:
            L.append(f"## {name}: skipped (fewer than 2 evaluable classes). Skipped classes: {skipped[name]}\n"); continue
        keep_i = np.array([r["category"] in classes for r in t["rows"]])
        y = np.array([r["category"] for r in t["rows"]], dtype=object)[keep_i]
        X = t["feats"]["X"][layer][keep_i]; Et = t["feats"]["E"][keep_i] if t["feats"]["E"] is not None else None
        T = np.array([r["text"] for r in t["rows"]], dtype=object)[keep_i]
        preds = evaluate_source(systems, X, Et, T, y, classes)
        L += [f"## {name}: n={len(y)}, evaluated classes {classes}; chance {1 / len(classes):.3f}",
              "Per-class n: " + ", ".join(f"{c} {per_class[name][c]}" for c in classes) + ("; skipped: " + ", ".join(f"{c} ({why})" for c, why in skipped[name].items()) if skipped[name] else ""), "",
              "| system | macro-F1 [95% CI] | accuracy | shuffle-null p95 |", "|---|---|---|---|"]
        for s, p in preds.items():
            lo, hi = boot_ci(y, p); L.append(f"| {s} | {f1_9(y, p):.3f} [{lo:.3f}, {hi:.3f}] | {(y == p).mean():.3f} | {shuffle_null_p95(y, p):.3f} |")
        L += ["", "Paired differences (macro-F1, mean [95% CI]); * = excludes 0:", ""]
        pairs = [("probe", "tfidf"), ("probe", "embed"), ("probe", "native_lr")]
        diffs = {}
        for a_, b_ in pairs:
            if a_ in preds and b_ in preds:
                m, lo, hi = paired_bootstrap(y, preds[a_], preds[b_]); diffs[f"{a_} - {b_}"] = (m, lo, hi)
                L.append(f"- {a_} - {b_}: {m:+.3f} [{lo:+.3f}, {hi:+.3f}]{'*' if lo > 0 or hi < 0 else ''}")
        nw = np.array([len(x.split()) for x in T])
        for nm, mask in ((f"<= {min_words_strata} words", nw <= min_words_strata), (f"> {min_words_strata} words", nw > min_words_strata)):
            L.append(f"- stratum {nm} (n={int(mask.sum())}): " + (", ".join(f"{s} {f1_9(y[mask], p[mask]):.3f}" for s, p in preds.items()) if mask.sum() >= 20 else "n < 20, not reported"))
        L.append("")
        results[name] = {"f1": {s: f1_9(y, p) for s, p in preds.items()}, "diffs": diffs, "n": int(len(y)), "classes": classes, "y": y, "preds": preds}
    return "\n".join(L), results


DEMO_CLASSES = ["joy", "sadness", "fear", "anger", "surprise", "disgust", "love", "neutral"]
DEMO_LAYER, DEMO_DIM = 5, 48


def make_demo(out_dir, seed=0):
    """Write a small SYNTHETIC audit example (texts + cached 'activations' + meta sidecars) and return the audit arguments.
    Nothing here comes from a model or from human data: each class has its own direction in a random space, and each class has cue
    words. The `shifted` test set keeps the directions but replaces the cue words, so lexical features should lose and a hidden-state
    probe should not: a toy version of the in-domain / cross-domain contrast the audit is built to measure. Numbers mean nothing about real models."""
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True); r = np.random.RandomState(seed)
    cues = {"train": {c: [f"{c[:3]}{k}" for k in range(4)] for c in DEMO_CLASSES}, "in": None, "shifted": {c: [f"{c[:3]}x{k}" for k in range(4)] for c in DEMO_CLASSES}}
    cues["in"] = cues["train"]
    fill = [f"w{k}" for k in range(40)]

    def build(name, n_per, with_split):
        rows, X, E = [], [], []
        for i, c in enumerate(DEMO_CLASSES):
            for j in range(n_per):
                v = r.randn(DEMO_DIM); v[i] += 2.0
                words = list(r.choice(fill, 8)) + ([str(r.choice(cues[name][c]))] if r.rand() < 0.8 else [])
                r.shuffle(words)
                row = {"id": f"{name}-{c}-{j}", "text": " ".join(words), "category": c}
                if with_split:
                    row["split"] = "train" if j < int(0.75 * n_per) else "val"
                rows.append(row); X.append(v); E.append(r.randn(DEMO_DIM) + 0.1 * v)
        jl = out / f"demo_{name}.jsonl"
        jl.write_text("\n".join(json.dumps(x) for x in rows) + "\n")
        npz = out / f"demo_{name}.npz"
        np.savez(npz, ids=np.array([x["id"] for x in rows]), skip=4, **{f"L{DEMO_LAYER}": np.array(X)}, E=np.array(E))
        write_meta(npz, "synthetic-demo (no model)", 4, [DEMO_LAYER], [x["id"] for x in rows])
        return jl, npz
    tr = build("train", 80, True); te = [("in_domain", *build("in", 40, False)), ("shifted", *build("shifted", 40, False))]
    return tr, te


def main():
    ap = argparse.ArgumentParser(description="Audit emotion probes against cheap baselines. `probekit-audit --demo` runs a bundled synthetic example end to end.")
    ap.add_argument("--demo", action="store_true", help="generate a small synthetic example (no model, no data download) and audit it")
    ap.add_argument("--demo-dir", default=None, help="keep the generated demo files here (default: a temporary directory)")
    ap.add_argument("--train"); ap.add_argument("--train-cache")
    ap.add_argument("--train-embed", default=None, help="separate npz with E for the train rows")
    ap.add_argument("--test", action="append", help="name=jsonl=cache (repeatable)")
    ap.add_argument("--layer", type=int); ap.add_argument("--method", default="lda", choices=["lda", "meandiff", "logreg"])
    ap.add_argument("--exclude", nargs="*", default=[]); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.demo:
        import tempfile
        d = a.demo_dir or tempfile.mkdtemp(prefix="probekit_demo_")
        (trj, trc), tests_ = make_demo(d)
        a.train, a.train_cache, a.layer = str(trj), str(trc), DEMO_LAYER
        a.test = [f"{n}={j}={c}" for n, j, c in tests_]; a.out = a.out or "audit_demo.md"
    elif not (a.train and a.train_cache and a.test and a.layer is not None and a.out):
        ap.error("give --demo, or all of --train --train-cache --test --layer --out")
    rows = load_jsonl(a.train); train = {"rows": rows, "feats": load_cache(a.train_cache, rows, a.train_embed)}
    tests = {}
    for spec in a.test:
        name, jl, cache = spec.split("=", 2); r = load_jsonl(jl); tests[name] = {"rows": r, "feats": load_cache(cache, r)}
    md, _ = audit(train, tests, a.layer, tuple(a.exclude), method=a.method)
    if a.demo:
        md = ("> **SYNTHETIC DEMO.** Random vectors and made-up words, no model and no human data. It shows the report format and the in-domain vs shifted contrast the "
              "audit measures; the numbers say nothing about real models.\n\n") + md
    Path(a.out).write_text(md + "\n"); print(md); print(f"\nreport written to {a.out}")


if __name__ == "__main__":
    main()
