#!/usr/bin/env python3
"""
probekit_corpus_audit.py -- quality audit of a generated corpus (AGENT_BRIEF Task B step 4). Reads ONLY --corpus
(and optionally the generator's run log); never touches the hard set or own_voice.

    python probekit_corpus_audit.py --corpus data/corpus/corpus_v3_qwen3.8-27b.jsonl [--log run.log] [--per-cat 3] [--seed 0]

Prints: passages per (category, split); rejection reasons (parsed from the generator log if given, and re-applied to the
saved passages); format adherence per requested style (share of passages with quotation marks, share with line breaks);
word-length stats per category; duplicate ids/texts; then `--per-cat` random passages per category for a human read.
The "off-target" judgment (e.g. calm text that is actually anxious) is for a person; this script only samples.
"""
import argparse
import ast
import json
import random
import re
from collections import Counter, defaultdict

from probekit_corpus_v2 import reject_reason

QUOTE = re.compile(r"[\"“”]")


def parse_log_reasons(path):
    for l in open(path):
        if l.startswith("Top rejection reasons:"):
            return dict(ast.literal_eval(l.split(":", 1)[1].strip()))   # literal list of (reason, n) tuples
    return {}


def audit(rows, log_reasons=None, per_cat=3, seed=0):
    out = []
    cats = sorted({r["category"] for r in rows})
    splits = sorted({r["split"] for r in rows})
    c = Counter((r["category"], r["split"]) for r in rows)
    out.append(f"## Passages per (category, split): n={len(rows)}\n")
    out.append("| category | " + " | ".join(splits) + " | total |\n|---|" + "---|" * (len(splits) + 1))
    for k in cats:
        out.append(f"| {k} | " + " | ".join(str(c[(k, s)]) for s in splits) + f" | {sum(c[(k, s)] for s in splits)} |")
    out.append("\n## Rejection reasons\n")
    if log_reasons is not None:
        out.append(f"During generation (from log): {log_reasons or 'none recorded'}")
    again = Counter(reject_reason(r["text"]) for r in rows)
    out.append(f"Saved passages re-checked with the current filter: {dict(again)} (None = passes)")
    out.append("\n## Format adherence by requested style\n")
    by = defaultdict(list)
    for r in rows:
        by[r.get("style", "?")].append(r["text"])
    out.append("| style | n | has quotes | has line breaks |\n|---|---|---|---|")
    for s, ts in sorted(by.items()):
        q = sum(bool(QUOTE.search(t)) for t in ts) / len(ts)
        nl = sum("\n" in t.strip() for t in ts) / len(ts)
        out.append(f"| {s[:60]} | {len(ts)} | {q:.2f} | {nl:.2f} |")
    out.append("\n## Length (words) per category\n")
    out.append("| category | mean | min | max |\n|---|---|---|---|")
    for k in cats:
        w = [len(r["text"].split()) for r in rows if r["category"] == k]
        out.append(f"| {k} | {sum(w) / len(w):.0f} | {min(w)} | {max(w)} |")
    ids, texts = Counter(r["id"] for r in rows), Counter(r["text"] for r in rows)
    out.append(f"\nDuplicate ids: {sum(v > 1 for v in ids.values())}; duplicate texts: {sum(v > 1 for v in texts.values())}")
    rng = random.Random(seed)
    out.append(f"\n## Random samples ({per_cat} per category; judge off-target by reading)\n")
    for k in cats:
        pool = [r for r in rows if r["category"] == k]
        for r in rng.sample(pool, min(per_cat, len(pool))):
            body = r["text"].replace("\n", "\n> ")
            out.append(f"**[{k}] {r['topic']} / {r.get('style', '?')[:40]} / {r['split']}**\n> {body}\n")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--log", default=None, help="generator stdout log (for rejection counts)")
    ap.add_argument("--per-cat", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.corpus) if l.strip()]
    print(audit(rows, parse_log_reasons(a.log) if a.log else None, a.per_cat, a.seed))


if __name__ == "__main__":
    main()
