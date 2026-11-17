#!/usr/bin/env python3
"""
probekit_overlap.py -- does any TEST-ONLY text (hand-written hard set, own_voice) overlap a training corpus?

Prints AGGREGATE counts only: never passage text, ids or categories of individual matches.

    python probekit_overlap.py --test data/hardset/hardset.jsonl data/hardset/own_voice.jsonl \
        --corpus data/corpus/corpus_v2.jsonl data/corpus/corpus_human_goemotions.jsonl

For each test passage: exact match after normalization (lowercase, alphanumerics only) and the maximum word 5-gram
containment, |shingles(test) & shingles(corpus passage)| / |shingles(test)|, over all corpus passages. Containment >= 0.5
means at least half of the test passage's 5-grams occur in one corpus passage (near-duplicate).
"""
import argparse
import json
import re
from collections import defaultdict

THRESHOLDS = (0.2, 0.5, 0.8)


def norm(t):
    return re.sub(r"[^a-z0-9 ]+", "", t.lower().replace("\n", " ")).split()


def shingles(words, n=5):
    return {" ".join(words[i:i + n]) for i in range(max(0, len(words) - n + 1))}


def load(paths):
    out = []
    for p in paths:
        for l in open(p):
            if l.strip():
                out.append(json.loads(l)["text"])
    return out


def containments(test_texts, corpus_texts, n=5):
    """Per test text: max over corpus passages of |shingles(test) & shingles(corpus)| / |shingles(test)| (0 if < n words)."""
    cw = [norm(t) for t in corpus_texts]
    index = defaultdict(set)                      # shingle -> corpus passage indices
    for i, w in enumerate(cw):
        for s in shingles(w, n):
            index[s].add(i)
    out = []
    for t in test_texts:
        sh = shingles(norm(t), n)
        hits = defaultdict(int)
        for s in sh:
            for i in index.get(s, ()):
                hits[i] += 1
        out.append(max(hits.values()) / len(sh) if hits and sh else 0.0)
    return out


def overlap(test_texts, corpus_texts, n=5):
    """Return aggregate stats: exact matches and max-containment histogram. No texts leave this function."""
    exact_set = {" ".join(norm(t)) for t in corpus_texts}
    n_exact = sum(" ".join(norm(t)) in exact_set for t in test_texts)
    maxc = containments(test_texts, corpus_texts, n)
    return {"n_test": len(test_texts), "n_corpus": len(corpus_texts), "exact": n_exact,
            **{f"containment>={t}": sum(c >= t for c in maxc) for t in THRESHOLDS},
            "max_containment": round(max(maxc), 3) if maxc else 0.0, "any_shared_5gram": sum(c > 0 for c in maxc)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", nargs="+", required=True)
    ap.add_argument("--corpus", nargs="+", required=True)
    a = ap.parse_args()
    test = load(a.test)
    print(f"test passages: {len(test)} (from {len(a.test)} file(s)); aggregates only, no text printed")
    for c in a.corpus:
        r = overlap(test, load([c]))
        print(f"{c}: " + ", ".join(f"{k}={v}" for k, v in r.items() if k != "n_test"))


if __name__ == "__main__":
    main()
