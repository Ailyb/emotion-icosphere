#!/usr/bin/env python3
"""
probekit_judge.py -- the fixed zero-shot emotion judge for the steering tests (PLAN_v2 T10).

A local OpenAI-compatible server (Ollama) answers one word from LABELS for a text. Fixed prompt, temperature 0, reasoning off. The judge
is itself validated before any steering result is looked at: macro-F1 >= GATE (0.50) on the GoEmotions frozen set (n=840).

    python probekit_judge.py validate --out data/steering/judge_validation.md
"""
import argparse
import json
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

LABELS = ["joy", "sadness", "fear", "anger", "surprise", "disgust", "love", "neutral"]
GATE = 0.50
MODEL, BASE = "granite4.2:8b", "http://localhost:11434/v1"
PROMPT = ("Classify the emotion expressed by the writer of the text. Answer with exactly one word from this list: "
          "joy, sadness, fear, anger, surprise, disgust, love, neutral.\n\nText: \"{t}\"\nAnswer:")
_RE = re.compile(r"\b(" + "|".join(LABELS) + r")\b")


def parse_label(answer):
    """First label word in the answer (lower-cased), or None if the judge said something else."""
    m = _RE.search((answer or "").lower())
    return m.group(1) if m else None


def ask(text, model=MODEL, base=BASE, tries=4, opener=None):
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": PROMPT.format(t=text.replace('"', "'"))}],
                       "temperature": 0, "max_tokens": 8, "reasoning_effort": "none"}).encode()
    for k in range(tries):
        try:
            req = urllib.request.Request(base.rstrip("/") + "/chat/completions", data=body, headers={"Content-Type": "application/json"})
            with (opener or urllib.request.urlopen)(req, timeout=120) as r:
                return parse_label(json.loads(r.read())["choices"][0]["message"]["content"])
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(2 ** k)


def judge_texts(texts, model=MODEL, base=BASE, workers=4, opener=None):
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(lambda t: ask(t, model, base, opener=opener), texts))


def validate(out):
    import numpy as np
    from probekit_eval_v2 import f1_9
    from probekit_testset import load_frozen
    rows, m = load_frozen("data/testsets/goemotions_test_frozen.manifest.json")
    pred = judge_texts([r["text"] for r in rows])
    y = np.array([r["category"] for r in rows], dtype=object); p = np.array([x or "unparsed" for x in pred], dtype=object)
    f1 = f1_9(y, p); acc = float((y == p).mean())
    conf = {t: {q: int(((y == t) & (p == q)).sum()) for q in LABELS + ["unparsed"] if ((y == t) & (p == q)).sum()} for t in LABELS}
    L = [f"# Judge validation ({MODEL}; PLAN_v2 T10 gate: macro-F1 >= {GATE})", "",
         f"GoEmotions frozen set n={len(y)} (manifest {m['testset_sha256'][:12]}...): macro-F1 {f1:.3f}, accuracy {acc:.3f}, unparsed {int((p == 'unparsed').sum())}. "
         f"**Gate: {'PASS' if f1 >= GATE else 'FAIL'}**", "", "Confusion (true -> predicted counts):", ""]
    L += [f"- {t}: {c}" for t, c in conf.items()]
    Path(out).parent.mkdir(parents=True, exist_ok=True); Path(out).write_text("\n".join(L) + "\n"); print("\n".join(L))
    return f1


def main():
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    v = sp.add_parser("validate"); v.add_argument("--out", default="data/steering/judge_validation.md")
    a = ap.parse_args(); validate(a.out)


if __name__ == "__main__":
    main()
