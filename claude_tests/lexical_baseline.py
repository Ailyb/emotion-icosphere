#!/usr/bin/env python3
"""
lexical_baseline.py -- how well can plain word statistics classify the corpus?

Trains TF-IDF + logistic regression on the TRAIN+VAL passages and scores the TEST passages, using the
same topic-disjoint split and the same metrics as build_emotion_vectors.py (8-way, neutral excluded).

Reading the result:
  * baseline ~= probe score  -> the corpus is lexically solvable; the probe score says little about
                                 abstraction. Build a harder corpus (implicit emotion, other writers).
  * baseline << probe score  -> the probes capture something beyond word counts (necessary, not sufficient).
Also prints the most predictive words per emotion, which shows the emotional "dialect" of the generator.

    python lexical_baseline.py [--corpus data/corpus/corpus.jsonl] [--probe-f1 0.857]
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


def run(rows, ngram=(1, 2), top=8):
    emo = [r for r in rows if r["category"] != "neutral"]
    fit = [r for r in emo if r["split"] in ("train", "val")]
    test = [r for r in emo if r["split"] == "test"]
    labels = sorted({r["category"] for r in emo})
    vec = TfidfVectorizer(ngram_range=ngram, sublinear_tf=True, min_df=2, stop_words="english")
    Xf = vec.fit_transform([r["text"] for r in fit])
    clf = LogisticRegression(max_iter=2000, C=5.0).fit(Xf, [r["category"] for r in fit])
    y = [r["category"] for r in test]
    pred = clf.predict(vec.transform([r["text"] for r in test]))
    res = {"n_fit": len(fit), "n_test": len(test), "chance": 1 / len(labels),
           "accuracy": float(accuracy_score(y, pred)),
           "macro_f1": float(f1_score(y, pred, labels=labels, average="macro", zero_division=0))}
    vocab = np.array(vec.get_feature_names_out())
    res["top_words"] = {c: list(vocab[np.argsort(clf.coef_[i])[::-1][:top]]) for i, c in enumerate(clf.classes_)}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus/corpus.jsonl")
    ap.add_argument("--probe-f1", type=float, default=None, help="your probe's test macro-F1, for side-by-side")
    a = ap.parse_args()
    rows = [json.loads(l) for l in Path(a.corpus).read_text().splitlines() if l.strip()]
    print(f"{len(rows)} passages")
    for name, ng in (("unigrams", (1, 1)), ("unigrams+bigrams", (1, 2))):
        r = run(rows, ng)
        print(f"\n[{name}] fit n={r['n_fit']}, test n={r['n_test']}: accuracy {r['accuracy']:.3f} "
              f"(chance {r['chance']:.3f}), macro-F1 {r['macro_f1']:.3f}")
        if name == "unigrams+bigrams":
            print("Most predictive words:")
            for c, w in r["top_words"].items():
                print(f"  {c:9s} {', '.join(w)}")
            if a.probe_f1 is not None:
                gap = a.probe_f1 - r["macro_f1"]
                print(f"\nProbe macro-F1 {a.probe_f1:.3f} vs lexical {r['macro_f1']:.3f} (gap {gap:+.3f}).")
                print("Gap near 0: corpus is lexically solvable." if gap < 0.05 else
                      "Probes beat word statistics, but that alone doesn't prove concept-level reading.")


if __name__ == "__main__":
    main()
