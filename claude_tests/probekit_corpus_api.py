#!/usr/bin/env python3
"""
probekit_corpus_api.py -- generate corpus passages with ANY OpenAI-compatible chat endpoint (stdlib only).

Why: (1) a different, stronger WRITER than the 3B model we probe removes the self-generation confound and enables
leave-one-writer-out tests; (2) it can run in a separate process or machine from MLX (llama.cpp server, Ollama,
LM Studio, `mlx_lm.server`, vLLM, or a hosted API).

Uses the same prompts, modes, leakage filter and JSONL schema as probekit_corpus_v2.py, plus 24 NEW topics
(16 train / 4 val / 4 test) so the test split gets bigger and more varied. Every row records `writer`.
Resumable. Merge with v2 by simple concatenation, then run adversarial_filter.py on the merged file:

    python probekit_corpus_api.py --api-base http://localhost:8080/v1 --api-model gemma-27b --writer gemma-27b \
        --topics all --samples 2 --out data/corpus/corpus_v3_gemma27b.jsonl
    cat data/corpus/corpus_v2.jsonl data/corpus/corpus_v3_gemma27b.jsonl > data/corpus/corpus_merged.jsonl
    python adversarial_filter.py --corpus data/corpus/corpus_merged.jsonl --out data/corpus/corpus_merged_scored.jsonl
"""
import argparse
import json
import os
import time
import urllib.request
from collections import Counter
from pathlib import Path

from probekit_corpus import EMOTIONS, SPLITS, clean
from probekit_corpus_v2 import MODES, emotion_prompt, neutral_prompt, reject_reason

TOPICS_EXT = {
    "train": ["a bus ride home", "a quiet library", "an unexpected rainstorm", "a conversation with a coworker",
              "a family reunion", "moving furniture", "a late delivery", "a first day at a new job",
              "a neighborhood barbecue", "a crowded elevator", "a long phone queue", "repairing a bicycle",
              "a bake sale", "a visit to the vet", "a camping trip", "a school play"],
    "val": ["a flea market", "a night shift", "a parking ticket", "a moving truck"],
    "test": ["an airport security line", "a community garden", "a concert crowd", "an office relocation"],
}
SPLITS_EXT = {t: sp for sp, ts in TOPICS_EXT.items() for t in ts}


def chat(api_base, model, user, temperature, max_tokens, api_key=None, timeout=300, tries=4, no_think=False):
    payload = {"model": model, "messages": [{"role": "user", "content": user}],
               "temperature": temperature, "top_p": 0.95, "max_tokens": max_tokens}
    if no_think:
        payload["reasoning_effort"] = "none"
    body = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    for k in range(tries):
        try:
            req = urllib.request.Request(api_base.rstrip("/") + "/chat/completions", data=body, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())["choices"][0]["message"]["content"]
        except Exception as e:
            if k == tries - 1:
                raise
            time.sleep(2 ** k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-base", required=True, help="e.g. http://localhost:8080/v1")
    ap.add_argument("--api-model", required=True)
    ap.add_argument("--writer", default=None, help="label stored in each row (default: api-model)")
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    ap.add_argument("--out", default="data/corpus/corpus_v3_api.jsonl")
    ap.add_argument("--topics", choices=["base", "ext", "all"], default="all")
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--temp", type=float, default=0.95)
    ap.add_argument("--no-think", action="store_true", help="send reasoning_effort=none (reasoning models spend max_tokens on hidden thinking and return empty text)")
    ap.add_argument("--max-tokens", type=int, default=300, help="completion token cap per request (reasoning models need more unless --no-think)")
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--max-new", type=int, default=2000, help="stop after generating this many new passages (smoke tests)")
    a = ap.parse_args()
    writer = a.writer or a.api_model
    splits = {}
    if a.topics in ("base", "all"):
        splits.update(SPLITS)
    if a.topics in ("ext", "all"):
        splits.update(SPLITS_EXT)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["id"] for l in out.read_text().splitlines() if l.strip()} if out.exists() else set()
    cats, reasons, made = EMOTIONS + ["neutral"], Counter(), 0
    with out.open("a") as f:
        for ci, cat in enumerate(cats):
            for ti, (topic, sp) in enumerate(splits.items()):
                for j in range(a.samples):
                    rid = f"v3|{writer}|{cat}|{topic}|{j}"
                    if rid in done:
                        continue
                    if made >= a.max_new:
                        break
                    mode = MODES[(ci + ti + j) % len(MODES)]
                    user = neutral_prompt(topic, mode) if cat == "neutral" else emotion_prompt(cat, topic, mode)
                    text = None
                    for _ in range(a.retries):
                        cand = clean(chat(a.api_base, a.api_model, user, a.temp, a.max_tokens, a.api_key, no_think=a.no_think))
                        why = reject_reason(cand)
                        if why is None:
                            text = cand; break
                        reasons[why] += 1
                    if text is None:
                        print(f"  [skip] {rid}"); continue
                    f.write(json.dumps({"id": rid, "category": cat, "topic": topic, "style": mode, "split": sp,
                                        "writer": writer, "text": text}) + "\n"); f.flush(); made += 1
                    if made % 25 == 0:
                        print(f"  {made} new passages")
    print(f"Done: {made} new passages -> {out}\nTop rejection reasons: {reasons.most_common(10)}")


if __name__ == "__main__":
    main()
