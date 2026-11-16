#!/usr/bin/env python3
"""
adversarial_filter.py -- mark the passages that plain word statistics find hard.

For every passage we compute the probability a TF-IDF + logistic-regression model assigns to the TRUE label,
out-of-fold with folds grouped by TOPIC (the model never sees the passage's own topic). Within each
(category, split) cell we keep the `keep_frac` passages with the LOWEST such probability: the ones where
vocabulary alone doesn't give the answer away. Output keeps ALL rows, adding `lex_p` and `hard` fields.

    python adversarial_filter.py --corpus data/corpus/corpus_v2.jsonl --out data/corpus/corpus_v2_scored.jsonl

Known pitfall (so you can read results correctly): this filter also tends to retain off-target or ambiguous
passages, because those are 'hard' for any model. That is why the hand-written hard set is the cleaner measure
and the filtered test split is only a supporting one.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold


def lexical_true_prob(rows, n_splits=5):
    texts = np.array([r["text"] for r in rows]); y = np.array([r["category"] for r in rows])
    groups = np.array([r["topic"] for r in rows])
    p = np.zeros(len(rows))
    for tr, te in GroupKFold(n_splits=min(n_splits, len(set(groups)))).split(texts, y, groups):
        vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, stop_words="english")
        clf = LogisticRegression(max_iter=2000, C=5.0).fit(vec.fit_transform(texts[tr]), y[tr])
        proba = clf.predict_proba(vec.transform(texts[te]))
        cls = list(clf.classes_)
        for k, i in enumerate(te):
            p[i] = proba[k, cls.index(y[i])] if y[i] in cls else 0.0
    return p


def mark_hard(rows, p, keep_frac=0.5):
    hard = np.zeros(len(rows), dtype=bool)
    cells = {}
    for i, r in enumerate(rows):
        cells.setdefault((r["category"], r["split"]), []).append(i)
    for idx in cells.values():
        idx = np.array(idx)
        k = max(1, int(round(keep_frac * len(idx))))
        hard[idx[np.argsort(p[idx])[:k]]] = True       # lowest lexical confidence in the true class
    return hard


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus/corpus_v2.jsonl")
    ap.add_argument("--out", default="data/corpus/corpus_v2_scored.jsonl")
    ap.add_argument("--keep-frac", type=float, default=0.5)
    a = ap.parse_args()
    rows = [json.loads(l) for l in Path(a.corpus).read_text().splitlines() if l.strip()]
    p = lexical_true_prob(rows)
    hard = mark_hard(rows, p, a.keep_frac)
    for r, pi, h in zip(rows, p, hard):
        r["lex_p"], r["hard"] = round(float(pi), 4), bool(h)
    Path(a.out).write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(f"{len(rows)} passages; lexical model mean p(true) all={p.mean():.3f}, kept-hard={p[hard].mean():.3f}")
    c = Counter((r["split"], r["category"]) for r in rows if r["hard"])
    for sp in ("train", "val", "test"):
        print(f"  {sp:5s} hard per category: " + ", ".join(f"{k[1][:4]}={v}" for k, v in sorted(c.items()) if k[0] == sp))
    thin = [k for k, v in c.items() if k[0] == "test" and v < 6]
    if thin:
        print("WARNING: some test cells have <6 hard passages; regenerate with --samples 4 or 5 for tighter estimates.")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
