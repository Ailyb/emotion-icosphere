#!/usr/bin/env python3
"""
probekit_tfidf_gate.py -- model-free "is this corpus lexically solvable?" gate (PLAN_v2 T0a).

TF-IDF (1-2 grams) + logistic regression, evaluated on topic-DISJOINT folds: for each seed the topics (not passages)
are randomly partitioned into fit/test, so no topic is ever on both sides. Reports macro-F1 per seed and a bootstrap CI
over the pooled test predictions (passages resampled), for two label sets:
  * 8-way, neutral excluded (comparable to the v1 number 0.953 from lexical_baseline.py)
  * 9-way, neutral included
Then the 20 most predictive features per class, flagging any that match the emotion-word filter (probekit_corpus_v2.STEM_RE2),
and the most common passage openers (stock-opener check).

    python probekit_tfidf_gate.py --corpus data/corpus/corpus_v3_qwen3.8-27b.jsonl [--seeds 0 1 2] [--gate 0.7]

Exit status 0 if the upper 95% bound of the 8-way macro-F1 is below --gate, else 1.
"""
import argparse
import json
import sys
from collections import Counter, defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from probekit_corpus_v2 import STEM_RE2


def check_topics_disjoint(rows):
    """Every topic must sit in exactly one of the corpus's own splits (it is the unit of disjointness)."""
    sp = defaultdict(set)
    for r in rows:
        sp[r["topic"]].add(r["split"])
    bad = [t for t, s in sp.items() if len(s) > 1]
    return len(sp), bad


def fit_predict(fit, test, ngram=(1, 2)):
    vec = TfidfVectorizer(ngram_range=ngram, sublinear_tf=True, min_df=2, stop_words="english")
    X = vec.fit_transform([r["text"] for r in fit])
    clf = LogisticRegression(max_iter=2000, C=5.0).fit(X, [r["category"] for r in fit])
    return clf.predict(vec.transform([r["text"] for r in test]))


def gate(rows, seeds=(0, 1, 2), test_frac=0.3, n_boot=1000, drop_neutral=True):
    """Return per-seed macro-F1, pooled bootstrap mean/CI, and n. Topic-disjoint random partitions per seed."""
    if drop_neutral:
        rows = [r for r in rows if r["category"] != "neutral"]
    topics = sorted({r["topic"] for r in rows})
    labels = sorted({r["category"] for r in rows})
    per_seed, yt, yp, seed_of = [], [], [], []
    for s in seeds:
        rng = np.random.RandomState(s)
        held = set(rng.permutation(topics)[: max(1, int(round(len(topics) * test_frac)))])
        fit = [r for r in rows if r["topic"] not in held]
        test = [r for r in rows if r["topic"] in held]
        assert not ({r["topic"] for r in fit} & {r["topic"] for r in test})
        pred = fit_predict(fit, test)
        y = [r["category"] for r in test]
        per_seed.append(float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)))
        yt += y; yp += list(pred); seed_of += [s] * len(y)
    yt, yp = np.array(yt), np.array(yp)
    rng = np.random.RandomState(1234)
    boots = []
    for _ in range(n_boot):
        i = rng.randint(0, len(yt), len(yt))
        boots.append(f1_score(yt[i], yp[i], labels=labels, average="macro", zero_division=0))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"per_seed": per_seed, "mean": float(np.mean(per_seed)), "ci": (float(lo), float(hi)),
            "n_test_pooled": len(yt), "n_topics": len(topics), "classes": labels}


def top_features(rows, k=20):
    cats = sorted({r["category"] for r in rows})
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, stop_words="english")
    X = vec.fit_transform([r["text"] for r in rows])
    clf = LogisticRegression(max_iter=2000, C=5.0).fit(X, [r["category"] for r in rows])
    vocab = np.array(vec.get_feature_names_out())
    out = {}
    for i, c in enumerate(clf.classes_):
        out[c] = [(w, bool(STEM_RE2.search(w))) for w in vocab[np.argsort(clf.coef_[i])[::-1][:k]]]
    return out


def openers(rows, n=2, top=3):
    """Most common first-n-word openers per category with their share (stock-opener check)."""
    res = {}
    for c in sorted({r["category"] for r in rows}):
        cnt = Counter(" ".join(r["text"].lower().split()[:n]) for r in rows if r["category"] == c)
        tot = sum(cnt.values())
        res[c] = [(o, round(v / tot, 3)) for o, v in cnt.most_common(top)]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--gate", type=float, default=0.7)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.corpus) if l.strip()]
    nt, bad = check_topics_disjoint(rows)
    print(f"{len(rows)} passages, {nt} topics; topics appearing in more than one split: {len(bad)}")
    ok = True
    for name, dn in (("8-way, neutral excluded (v1-comparable)", True), ("9-way, neutral included", False)):
        r = gate(rows, a.seeds, drop_neutral=dn)
        print(f"\n[{name}] topic-disjoint, seeds {a.seeds}: per-seed macro-F1 {[round(x, 3) for x in r['per_seed']]}, "
              f"mean {r['mean']:.3f}, bootstrap 95% CI [{r['ci'][0]:.3f}, {r['ci'][1]:.3f}] "
              f"(pooled test n={r['n_test_pooled']}, {r['n_topics']} topics; chance {1 / len(r['classes']):.3f})")
        if dn:
            ok = r["ci"][1] < a.gate
            print(f"GATE (upper CI < {a.gate}): {'PASS' if ok else 'FAIL'}")
    print("\nTop 20 TF-IDF features per class (* = matches the emotion-word filter):")
    for c, feats in top_features(rows).items():
        print(f"  {c:9s} " + ", ".join(w + ("*" if f else "") for w, f in feats))
    print("\nMost common 2-word openers per class (share of that class):")
    for c, o in openers(rows).items():
        print(f"  {c:9s} " + ", ".join(f"'{w}' {s}" for w, s in o))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
