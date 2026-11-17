#!/usr/bin/env python3
"""
probekit_lexicon_subtypes.py -- lexicon-based subtype strata for a frozen test set, written as a SIDECAR (the frozen file and its
hash are never touched), plus the stratified comparison of systems from a saved predictions file.

Lexicon: NRC Word-Emotion Association Lexicon v0.92 (Mohammad & Turney; free for non-commercial research or educational use; do not
redistribute; cite Mohammad & Turney 2013). An "emotion word" is any word the lexicon associates with at least one of its eight
emotions (anger, anticipation, disgust, fear, joy, sadness, surprise, trust). Exact lower-case token match, no stemming.

Subtypes (per item):
  no_lexicon_match      emotion-labelled item with zero emotion words      (is anything left once the words are gone?)
  lexicon_match         emotion-labelled item with >= 1 emotion word
  emotion_word_neutral  neutral-labelled item with >= 1 emotion word
  plain_neutral         neutral-labelled item with zero emotion words
  negation              an emotion word within 3 tokens after a negator (any label)

    python probekit_lexicon_subtypes.py build  --nrc data/external/nrc/NRC-Emotion-Lexicon/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt \
        --testset data/testsets/goemotions_test_frozen.jsonl --out data/testsets/goemotions_test_frozen.subtypes_nrc.json
    python probekit_lexicon_subtypes.py report --sidecar <that file> --preds data/eval_merged2/<slug>/frozen_preds.json --out <md>
"""
import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

from probekit_eval_v2 import f1_9
from probekit_eval_v3 import paired_bootstrap

EIGHT = {"anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust"}
NEGATORS = {"not", "no", "never", "without", "hardly", "neither", "nor", "cannot", "cant", "dont", "wont", "isnt", "arent", "wasnt",
            "werent", "didnt", "doesnt", "couldnt", "wouldnt", "shouldnt", "hasnt", "havent"}


def load_nrc(path):
    words = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) == 3 and parts[1] in EIGHT and parts[2].strip() == "1":
            words.add(parts[0])
    return words


def toks(text):
    return re.findall(r"[a-z]+(?:['’][a-z]+)?", text.lower().replace("’", "'"))


def is_negator(t):
    return t in NEGATORS or t.endswith("n't")


def subtypes(text, label, words):
    tk = toks(text)
    hits = [i for i, t in enumerate(tk) if t in words]
    sub = []
    if label == "neutral":
        sub.append("emotion_word_neutral" if hits else "plain_neutral")
    else:
        sub.append("lexicon_match" if hits else "no_lexicon_match")
    if any(any(is_negator(x) for x in tk[max(0, i - 3):i]) for i in hits):
        sub.append("negation")
    return sub


def build(a):
    words = load_nrc(a.nrc)
    rows = [json.loads(l) for l in open(a.testset, encoding="utf-8") if l.strip()]
    side = {r["id"]: subtypes(r["text"], r["category"], words) for r in rows}
    manifest = {"lexicon": "NRC Word-Emotion Association Lexicon v0.92 (non-commercial research use; not redistributed)",
                "lexicon_sha256": hashlib.sha256(Path(a.nrc).read_bytes()).hexdigest(), "n_emotion_words": len(words),
                "testset": str(a.testset), "testset_sha256": hashlib.sha256(Path(a.testset).read_bytes()).hexdigest(),
                "counts": dict(Counter(s for v in side.values() for s in v)),
                "rules": "see module docstring; exact lower-case token match; negator within 3 tokens before an emotion word"}
    Path(a.out).write_text(json.dumps({"manifest": manifest, "subtypes": side}, indent=1))
    print(json.dumps(manifest["counts"]), "emotion words:", len(words))


def stratified(labels, preds, side_ids, ids, systems=("probe_lda", "native_lr", "embed", "tfidf", "fusion_tfidf_native (post hoc)"), min_n=50):
    """Return {stratum: {n, f1 per system, probe - tfidf paired CI (if n >= min_n)}}."""
    sub = {i: s for i, s in zip(ids, side_ids)}
    out = {}
    for name in ("no_lexicon_match", "lexicon_match", "emotion_word_neutral", "plain_neutral", "negation"):
        m = np.array([name in s for s in sub.values()]); y = np.asarray(labels)[m]
        if m.sum() < 20:
            out[name] = {"n": int(m.sum()), "note": "n < 20, not scored"}; continue
        e = {"n": int(m.sum()), "f1": {s: f1_9(y, np.asarray(preds[s])[m]) for s in systems if s in preds}}
        if m.sum() >= min_n and "probe_lda" in preds and "tfidf" in preds:
            e["probe_minus_tfidf"] = paired_bootstrap(y, np.asarray(preds["probe_lda"])[m], np.asarray(preds["tfidf"])[m])
        out[name] = e
    return out


def report(a):
    side = json.loads(Path(a.sidecar).read_text()); pr = json.loads(Path(a.preds).read_text())
    ids = pr["ids"]; labels = np.array(pr["labels"])
    res = stratified(labels, pr["preds"], [side["subtypes"][i] for i in ids], ids)
    L = ["# Lexicon-subtype strata on the GoEmotions frozen set (NRC EmoLex sidecar; exploratory, post hoc)", "",
         "Hypothesis stated before running: probe - TF-IDF is larger (less negative) in `no_lexicon_match` than in `lexicon_match`. "
         "Macro-F1 over the classes present in each stratum; a stratum with a single class is not meaningful.", "",
         "| stratum | n | " + " | ".join(next(iter(r["f1"] for r in res.values() if "f1" in r)).keys()) + " | probe - TF-IDF [95% CI] |", "|---|---|---|---|---|---|---|---|"]
    for k, r in res.items():
        if "f1" not in r:
            L.append(f"| {k} | {r['n']} | {r['note']} |"); continue
        d = r.get("probe_minus_tfidf")
        L.append(f"| {k} | {r['n']} | " + " | ".join(f"{v:.3f}" for v in r["f1"].values()) + " | " +
                 (f"{d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}]{'*' if d[1] > 0 or d[2] < 0 else ''}" if d else "n < 50") + " |")
    L += ["", f"Sidecar manifest: {json.dumps(side['manifest']['counts'])}; lexicon sha256 {side['manifest']['lexicon_sha256'][:12]}..."]
    Path(a.out).write_text("\n".join(L) + "\n"); print("\n".join(L))


def main():
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    b = sp.add_parser("build"); b.add_argument("--nrc", required=True); b.add_argument("--testset", required=True); b.add_argument("--out", required=True)
    r = sp.add_parser("report"); r.add_argument("--sidecar", required=True); r.add_argument("--preds", required=True); r.add_argument("--out", required=True)
    a = ap.parse_args(); {"build": build, "report": report}[a.cmd](a)


if __name__ == "__main__":
    main()
