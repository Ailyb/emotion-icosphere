#!/usr/bin/env python3
"""
probekit_corpus_hf.py -- add HUMAN-written training text from public datasets, in the probekit JSONL schema.

Why: every passage so far was written by an LLM. Probes trained only on that text lose a lot on human writing
(your own-voice set: ~0.49 even for a fully trained classifier). Public human text widens the training distribution
while own_voice.jsonl and the hand-written set stay untouched as the final test.

Sources (needs `pip install datasets` and internet; check each dataset card's license before redistributing anything):
  goemotions    Reddit comments, 27 emotions + neutral (Google; Apache-2.0 as far as I know: verify).
                Mapped to: joy, sadness, fear, anger, surprise, disgust, love, neutral. NO 'calm' exists here.
  dair_emotion  tweets labeled sadness/joy/love/anger/fear/surprise (license: check the card). Mostly "i feel ..." text.

What this does (and why):
  * keeps only comments with EXACTLY ONE label that maps to our set (multi-label rows are ambiguous)
  * 10-80 words (very short text pools over too few tokens: see the "The sky is blue." failure)
  * replaces anonymization tags ([NAME] -> Sam, [RELIGION] -> church, other [TAGS] removed)
  * dedupes, samples a fixed number per (category, split), balanced across categories
  * uses the dataset's own train/validation/test splits; every comment is its own `topic` group, so topic-disjointness
    is NOT guaranteed here (same subreddit/thread themes can appear in several splits). Keep that in mind.
  * does NOT filter emotion words: real people use them. adversarial_filter.py will mark the passages that are hard
    for a word model; the lexical baseline will look strong on this data and that is expected.
  * Contains profanity and insults (esp. anger/disgust). Keep it local.

    python probekit_corpus_hf.py --source goemotions --out data/corpus/corpus_human_goemotions.jsonl
"""
import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

GOEMO_MAP = {n: n for n in ("joy", "sadness", "fear", "anger", "surprise", "disgust", "love", "neutral")}
DAIR_MAP = {n: n for n in ("joy", "sadness", "fear", "anger", "surprise", "love")}

SOURCES = {
    "goemotions": {"hf_ids": ["google-research-datasets/go_emotions", "go_emotions"], "config": "simplified",
                   "labels": "labels", "multi": True, "map": GOEMO_MAP, "writer": "reddit-goemotions",
                   "style": "informal social-media comment (reddit)"},
    "dair_emotion": {"hf_ids": ["dair-ai/emotion"], "config": None, "labels": "label", "multi": False,
                     "map": DAIR_MAP, "writer": "twitter-dair", "style": "informal social-media post (twitter)"},
}
SPLIT_NAMES = {"train": "train", "validation": "val", "test": "test"}


def clean_text(t: str) -> str:
    t = t.replace("[NAME]", "Sam").replace("[RELIGION]", "church")
    t = re.sub(r"\[[A-Z_]+\]", "", t)
    return re.sub(r"\s+", " ", t).strip()


def convert(examples, label_names, mapping, split, source, writer, style, sizes, rng, min_words=10, max_words=80):
    """examples: iterable of dicts with `text`, `labels` (list[int]) or `label` (int), optional `id`.
    Returns (rows, stats). Pure function (no network): unit-tested."""
    pool, stats, seen = {}, Counter(), set()
    for i, ex in enumerate(examples):
        labs = ex.get("labels", ex.get("label"))
        labs = [labs] if isinstance(labs, int) else list(labs)
        if len(labs) != 1:
            stats["dropped_multi_or_no_label"] += 1; continue
        cat = mapping.get(label_names[labs[0]])
        if cat is None:
            stats["dropped_unmapped_label"] += 1; continue
        text = clean_text(ex["text"])
        n = len(text.split())
        if not (min_words <= n <= max_words):
            stats["dropped_length"] += 1; continue
        key = text.lower()
        if key in seen:
            stats["dropped_duplicate"] += 1; continue
        seen.add(key)
        pool.setdefault(cat, []).append((ex.get("id", i), text))
    rows = []
    for cat, items in pool.items():
        rng.shuffle(items)
        for rid, text in items[:sizes[split]]:
            rows.append({"id": f"hf|{source}|{split}|{rid}", "category": cat, "topic": f"{source}|{split}|{rid}",
                         "style": style, "split": split, "writer": writer, "source": source, "text": text})
    return rows, stats


def load_hf(src):
    from datasets import load_dataset
    last = None
    for hid in src["hf_ids"]:
        try:
            ds = load_dataset(hid, src["config"]) if src["config"] else load_dataset(hid)
            break
        except Exception as e:
            last = e
    else:
        raise SystemExit(f"could not load any of {src['hf_ids']}: {last}")
    feat = ds["train"].features[src["labels"]]
    names = feat.feature.names if src["multi"] else feat.names
    return ds, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=list(SOURCES) + ["both"], default="goemotions")
    ap.add_argument("--out", default="data/corpus/corpus_human.jsonl")
    ap.add_argument("--sizes", type=int, nargs=3, default=[80, 20, 20], metavar=("TRAIN", "VAL", "TEST"),
                    help="passages per category per split")
    ap.add_argument("--min-words", type=int, default=10)
    ap.add_argument("--max-words", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    sizes = dict(zip(("train", "val", "test"), a.sizes))
    rng = random.Random(a.seed)
    rows = []
    for name in (list(SOURCES) if a.source == "both" else [a.source]):
        src = SOURCES[name]; ds, names = load_hf(src)
        print(f"{name}: label names = {names}")
        for hf_split, split in SPLIT_NAMES.items():
            if hf_split not in ds:
                continue
            r, stats = convert(ds[hf_split], names, src["map"], split, name, src["writer"], src["style"], sizes, rng,
                               a.min_words, a.max_words)
            rows += r
            print(f"  {hf_split:10s} kept {len(r):4d} | dropped: {dict(stats)}")
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    c = Counter((r["split"], r["category"]) for r in rows)
    print(f"\nwrote {len(rows)} passages -> {out}")
    for sp in ("train", "val", "test"):
        print(f"  {sp:5s}: " + ", ".join(f"{k[1][:5]}={v}" for k, v in sorted(c.items()) if k[0] == sp))
    missing = {"joy", "sadness", "fear", "anger", "surprise", "disgust", "calm", "love", "neutral"} - {r["category"] for r in rows}
    if missing:
        print(f"NOTE: no examples for: {sorted(missing)} (expected: 'calm' is not in these datasets)")
    thin = [k for k, v in c.items() if v < min(a.sizes) * 0.5]
    if thin:
        print(f"WARNING: some cells have far fewer passages than requested: {thin}")


if __name__ == "__main__":
    main()
