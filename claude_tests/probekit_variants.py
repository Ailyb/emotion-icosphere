#!/usr/bin/env python3
"""
probekit_variants.py -- PLAN_v2 T9 (pre-registered): do other poolings / a prompted read-out give a better linear representation?

    python probekit_variants.py extract --cache-dir data/eval_merged2/<slug> --eval-dir data/eval_merged2/<slug>/v3 --corpus data/corpus/corpus_merged2_scored.jsonl
    python probekit_variants.py eval    --cache-dir ... --eval-dir ... --corpus ... --out data/eval_merged2/<slug>/variants.md

Variants (all at the baseline's selected layer, trained on HUM, same flat-LR recipe so only the representation changes):
V0 mean-pool (baseline) | V1 last token | V2 max-pool | V3 mean of last 4 tokens | V4 prompted last token | V5 concat of plateau-layer mean-pools.
A variant is a finding only if variant - V0 has a CI above 0 on >= 3 of the 4 test sets (GoEmotions frozen, dair, ISEAR, DailyDialog).
"""
import argparse
import json
from pathlib import Path

import numpy as np

from probekit_eval_v2 import f1_9, load_jsonl
from probekit_eval_v3 import paired_bootstrap
from probekit_external_eval import argmax_classes, columns, fit_lr

VARIANTS = ["V0 mean", "V1 last", "V2 max", "V3 last4", "V4 prompted", "V5 concat"]


# ---------------------------------------------------------------- extraction (needs the model)
def extract_variants(items, cache: Path, layers, skip, model):
    ids = [i["id"] for i in items]
    if cache.exists():
        d = np.load(cache, allow_pickle=False)
        if list(d["ids"]) == ids and int(d["skip"]) == skip and all(f"prompted_L{l}" in d.files for l in layers):
            print(f"  cache hit: {cache}"); return {k: d[k] for k in d.files if k not in ("ids", "skip")}
    print(f"  extracting variants for {len(items)} passages -> {cache.name}", flush=True)
    from probekit_capture import Extractor
    ex = Extractor(model)
    acc = {f"{v}_L{l}": [] for l in layers for v in ("last", "max", "last4", "prompted")}
    for n, it in enumerate(items):
        for l, d in ex.variant_states(it["text"], layers, skip).items():
            for v, x in d.items():
                acc[f"{v}_L{l}"].append(x)
        for l, x in ex.prompted_last(it["text"], layers).items():
            acc[f"prompted_L{l}"].append(x)
        if (n + 1) % 250 == 0:
            print(f"    {n + 1}/{len(items)}", flush=True)
    out = {k: np.stack(v) for k, v in acc.items()}
    np.savez(cache, ids=np.array(ids), skip=skip, **out)
    return out


# ---------------------------------------------------------------- evaluation (pure functions)
def variant_matrix(name, mean_by_layer, var, layer, plateau):
    """Feature matrix of one variant for one item set. mean_by_layer: {layer: (N,d)} mean-pool; var: dict from extract_variants."""
    if name == "V0 mean":
        return mean_by_layer[layer]
    if name == "V5 concat":
        return np.hstack([mean_by_layer[l] for l in plateau])
    key = {"V1 last": "last", "V2 max": "max", "V3 last4": "last4", "V4 prompted": "prompted"}[name]
    return var[f"{key}_L{layer}"]


def evaluate_variants(train, val, tests, variants=VARIANTS):
    """train/val: (feature_fn(name)->X, y); tests: [(set_name, feature_fn, y)]. Returns per-set preds and the report text."""
    (ftr, ytr), (fva, yva) = train, val
    models = {}
    for v in variants:
        sc, clf = fit_lr(ftr(v), ytr, fva(v), yva); models[v] = (sc, clf)
    rows, preds = [], {}
    for name, fte, y in tests:
        cl = sorted(set(y)); preds[name] = {}
        for v in variants:
            sc, clf = models[v]
            preds[name][v] = argmax_classes(columns(clf, clf.predict_proba(sc.transform(fte(v))), cl), cl)
    return preds


def tally(preds, tests_y, base="V0 mean", variants=VARIANTS):
    out = {}
    for v in variants:
        if v == base:
            continue
        diffs = {s: paired_bootstrap(tests_y[s], preds[s][v], preds[s][base]) for s in preds}
        out[v] = {"diffs": diffs, "wins": sum(d[1] > 0 for d in diffs.values()), "losses": sum(d[2] < 0 for d in diffs.values())}
    return out


def render(preds, tests_y, tl, layer):
    sets = list(preds)
    L = [f"# T9 pooling / prompting variants (layer {layer}; trained on HUM; pre-registered in PLAN_v2)", "",
         "Macro-F1 per set (predictions restricted to the source's classes); paired differences vs V0 are [95% CI]; * = excludes 0.", "",
         "| variant | " + " | ".join(f"{s} (n={len(tests_y[s])})" for s in sets) + " |", "|---|" + "---|" * len(sets)]
    for v in VARIANTS:
        L.append(f"| {v} | " + " | ".join(f"{f1_9(tests_y[s], preds[s][v]):.3f}" for s in sets) + " |")
    L += ["", "| variant - V0 | " + " | ".join(sets) + " | sets with CI > 0 (claim needs >= 3) |", "|---|" + "---|" * (len(sets) + 1)]
    for v, r in tl.items():
        L.append(f"| {v} | " + " | ".join(f"{d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}]{'*' if d[1] > 0 or d[2] < 0 else ''}" for d in r["diffs"].values()) + f" | {r['wins']}/{len(sets)} |")
    good = [v for v, r in tl.items() if r["wins"] >= 3]
    L += ["", "Pre-registered reading: " + (f"variants with a consistent gain: {good}." if good else "no variant improves on mean-pooling on >= 3 of 4 sets; the representation is not limited by pooling or prompt framing at this layer.")]
    return "\n".join(L)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    for name in ("extract", "eval"):
        p = sp.add_parser(name)
        p.add_argument("--model", default="mlx-community/Llama-3.2-3B-Instruct")
        p.add_argument("--cache-dir", required=True); p.add_argument("--eval-dir", required=True); p.add_argument("--corpus", required=True)
        p.add_argument("--pool", default="data/corpus/corpus_human_goemotions_pool.jsonl"); p.add_argument("--testsets", default="data/testsets")
        if name == "eval":
            p.add_argument("--out", required=True)
    a = ap.parse_args()
    from probekit_eval_v3 import load_states
    from probekit_external import load_source
    from probekit_frozen_eval import EXTRACT_LAYERS, extract_cached, pool_new_rows
    from probekit_testset import load_frozen
    eval_dir, cache = Path(a.eval_dir), Path(a.cache_dir)
    best = json.loads((eval_dir / "best_config.json").read_text()); layer, skip = best["layer"], best["skip"]
    plateau = sorted(set(best.get("plateau_layers", [])) | {layer})
    rows = load_jsonl(a.corpus); pool = load_jsonl(a.pool); new = pool_new_rows(pool, rows)
    frozen, _ = load_frozen(f"{a.testsets}/goemotions_test_frozen.manifest.json")
    ext = {s: load_source(s, a.testsets)[0] for s in ("dair", "isear", "dailydialog")}
    sets = {"corpus": rows, "pool_new": new, "frozen": frozen, **ext}
    if a.cmd == "extract":
        for k, items in sets.items():
            extract_variants(items, cache / f"states_var_{k}.npz", plateau, skip, a.model)
        return
    var = {k: extract_variants(items, cache / f"states_var_{k}.npz", plateau, skip, a.model) for k, items in sets.items()}
    means = {"corpus": load_states(rows, cache / "states_corpus.npz", plateau, skip, a.model),
             "pool_new": extract_cached(new, cache / "states_pool_new.npz", EXTRACT_LAYERS, skip, a.model),
             "frozen": extract_cached(frozen, cache / "states_frozen.npz", EXTRACT_LAYERS, skip, a.model)}
    for s, it in ext.items():
        means[s] = extract_cached(it, cache / f"states_ext_{s}.npz", EXTRACT_LAYERS, skip, a.model)
    cat = np.array([r["category"] for r in rows]); spl = np.array([r["split"] for r in rows])
    wr = np.array([r.get("writer", "llama-3.2-3b") for r in rows]); hard = np.array([bool(r.get("hard", True)) for r in rows])
    keep = cat != "calm"; llm = (spl == "train") & keep & (wr != "reddit-goemotions"); va = (spl == "val") & hard & keep
    byid = {r["id"]: i for i, r in enumerate(rows)}; nid = {r["id"]: i for i, r in enumerate(new)}
    pr = [r for r in pool if r["split"] == "train"]
    def feat(set_key, v):
        return variant_matrix(v, {l: means[set_key][l] for l in plateau}, var[set_key], layer, plateau)
    def train_feat(v):
        Xc = feat("corpus", v); Xn = feat("pool_new", v)
        Xp = np.stack([Xc[byid[r["id"]]] if r["id"] in byid else Xn[nid[r["id"]]] for r in pr])
        return np.vstack([Xc[llm], Xp])
    ytr = np.concatenate([cat[llm], np.array([r["category"] for r in pr], dtype=object)])
    val_feat = lambda v: feat("corpus", v)[va]
    tests = [(("goemotions" if k == "frozen" else k), (lambda v, k=k: feat(k, v)), np.array([r["category"] for r in sets[k]], dtype=object))
             for k in ("frozen", "dair", "isear", "dailydialog")]
    preds = evaluate_variants((train_feat, ytr), (val_feat, cat[va]), tests)
    tests_y = {n: y for n, _, y in tests}
    md = render(preds, tests_y, tally(preds, tests_y), layer)
    Path(a.out).write_text(md + "\n"); print(md)


if __name__ == "__main__":
    main()
