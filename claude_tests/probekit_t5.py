#!/usr/bin/env python3
"""
probekit_t5.py -- PLAN_v2 T5/C2 (pre-registered 2026-11-22, 318cf6a): audit of a published model-written emotion corpus
(ryancodrai/emotion-probes, CC-BY-4.0, Gemini-written stories). Builds the pre-registered topic-disjoint subsample and extracts Llama states.

    python probekit_t5.py build     # data/external/emotion_probes/t5_{train,test}.jsonl (gitignored raw text) + data/corpus/t5_manifest.json (ids only)
    python probekit_t5.py extract   # states caches next to the Llama caches
Then probekit_audit.py (see run commands in PLAN_v2 / the report header).
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

RAW = Path("data/external/emotion_probes")
MAP = {"joyful": "joy", "sad": "sadness", "afraid": "fear", "angry": "anger", "surprised": "surprise", "disgusted": "disgust", "loving": "love"}
PER = 2   # stories per class and topic
SEED = 0


def split_of(topic_index):
    return "test" if topic_index % 5 == 0 else "val" if topic_index % 5 == 1 else "train"


def build_rows():
    import pandas as pd
    s = pd.read_parquet(RAW / "expression/stories.parquet"); n = pd.read_parquet(RAW / "expression/neutral_stories.parquet")
    s = s[s.emotion.isin(MAP)].assign(category=lambda d: d.emotion.map(MAP)); n = n.assign(category="neutral", emotion="neutral")
    d = pd.concat([s[["category", "topic", "story"]], n[["category", "topic", "story"]]], ignore_index=True)
    topics = sorted(d.topic.unique()); tix = {t: i for i, t in enumerate(topics)}
    rng = np.random.default_rng(SEED); rows = []
    for (cat, top), g in sorted(d.groupby(["category", "topic"])):
        take = rng.permutation(len(g))[:PER]
        for k, j in enumerate(sorted(take)):
            rows.append({"id": f"t5|{cat}|{tix[top]:03d}|{k}", "category": cat, "topic": top, "split": split_of(tix[top]), "text": g.iloc[j].story.strip(), "hard": True})
    return rows


def cmd_build(a):
    rows = build_rows()
    for sp in ("train", "val", "test"):
        pass
    trv = [r for r in rows if r["split"] in ("train", "val")]; te = [r for r in rows if r["split"] == "test"]
    for name, rr in (("t5_trainval", trv), ("t5_test", te)):
        with (RAW / f"{name}.jsonl").open("w") as f:
            for r in rr:
                f.write(json.dumps(r) + "\n")
    counts = {sp: sum(r["split"] == sp for r in rows) for sp in ("train", "val", "test")}
    man = {"source": "ryancodrai/emotion-probes (CC-BY-4.0, doi 10.57967/hf/8303)", "mapping": MAP, "per_class_topic": PER, "seed": SEED, "counts": counts,
           "ids_sha256": hashlib.sha256("\n".join(r["id"] for r in rows).encode()).hexdigest(), "ids": [r["id"] for r in rows]}
    Path("data/corpus/t5_manifest.json").write_text(json.dumps(man))
    print(counts, "->", RAW)


def cmd_extract(a):
    from probekit_frozen_eval import EXTRACT_LAYERS, extract_cached
    out = Path(a.cache_dir)
    for name in ("t5_trainval", "t5_test"):
        rows = [json.loads(l) for l in (RAW / f"{name}.jsonl").open()]
        extract_cached(rows, out / f"states_{name}.npz", EXTRACT_LAYERS, 4, a.model)


def cmd_audit(a):
    """P1/P2 (trained on T5) and P3 (trained on merged2, tested on T5-TEST, paired against the T5-trained probe) with the T4 audit code path."""
    from probekit_audit import audit, load_cache
    from probekit_eval_v2 import load_jsonl
    from probekit_eval_v3 import paired_bootstrap
    from probekit_frozen_eval import boot_ci
    c = Path(a.cache_dir); T = Path("data/testsets"); LAYER = 16
    ld = lambda jl, cache, emb=None: (lambda r: {"rows": r, "feats": load_cache(cache, r, emb)})(load_jsonl(str(jl)))
    t5te = ld(RAW / "t5_test.jsonl", c / "states_t5_test.npz")
    ext = {"goemotions": ld(T / "goemotions_test_frozen.jsonl", c / "states_frozen.npz"), "dair": ld(T / "dair_test_frozen.jsonl", c / "states_ext_dair.npz"),
           "isear": ld(T / "isear_test_frozen.jsonl", c / "states_ext_isear.npz"), "dailydialog": ld(T / "dailydialog_test_frozen.jsonl", c / "states_ext_dailydialog.npz")}
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    tr5 = ld(RAW / "t5_trainval.jsonl", c / "states_t5_trainval.npz")
    md1, r1 = audit(tr5, {"t5_test": t5te, **ext}, LAYER)
    trm = ld("data/corpus/corpus_merged2_scored.jsonl", c / "states_corpus.npz", c / "states_embed_corpus.npz")
    md2, r2 = audit(trm, {"t5_test": t5te}, LAYER, exclude=("calm",))
    y = r1["t5_test"]["y"]; assert list(y) == list(r2["t5_test"]["y"])
    L = ["## P3 paired: T5-trained minus merged2-trained, on T5-TEST (n = %d)" % len(y), ""]
    for s in ("probe", "native_lr", "tfidf", "embed"):
        if s in r1["t5_test"]["preds"] and s in r2["t5_test"]["preds"]:
            m, lo, hi = paired_bootstrap(y, r1["t5_test"]["preds"][s], r2["t5_test"]["preds"][s]); L.append(f"- {s}: {m:+.3f} [{lo:+.3f}, {hi:+.3f}]{'*' if lo > 0 or hi < 0 else ''}")
    (out / "t5_audit_trained_on_t5.md").write_text(md1 + "\n"); (out / "t5_audit_trained_on_merged2.md").write_text(md2 + "\n\n" + "\n".join(L) + "\n")
    print(md1); print(md2); print("\n".join(L))


def main():
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("build"); au = sp.add_parser("audit"); au.add_argument("--cache-dir", default="data/eval_merged2/mlx-community__Llama-3.2-3B-Instruct"); au.add_argument("--out-dir", default="data/eval_merged2/mlx-community__Llama-3.2-3B-Instruct"); e = sp.add_parser("extract")
    e.add_argument("--model", default="mlx-community/Llama-3.2-3B-Instruct"); e.add_argument("--cache-dir", default="data/eval_merged2/mlx-community__Llama-3.2-3B-Instruct")
    a = ap.parse_args(); {"build": cmd_build, "extract": cmd_extract, "audit": cmd_audit}[a.cmd](a)


if __name__ == "__main__":
    main()
