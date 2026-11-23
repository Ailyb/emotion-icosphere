#!/usr/bin/env python3
"""
probekit_viz_gen.py -- viz_plan Option 1: a larger SYNTHETIC passage set for measuring the display's hit and false-alarm rates per length bucket.
Written by a local model (granite4.2:8b via Ollama; not the probe model, no dataset text in the prompt). Pre-registered before generation (2026-11-23):
  * 6 length buckets (1-2, 3-4, 5-7, 8-11, 12-19, 20+ words) x {neutral x 120, each of 7 emotions x 20}; passages outside the bucket, duplicates, or
    near-duplicates (same lower-cased text) are dropped; no filtering on probe output, and nothing here is used to tune any threshold.
  * Intended labels are the writer's request, NOT validated; report hit rate vs intended label as such.
    python probekit_viz_gen.py [--out data/viz/synth.jsonl]
"""
import argparse
import json
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from probekit_length_calibration import BUCKETS, bucket_name, bucket_of
from probekit_testset import n_words

EMO = ["joy", "sadness", "fear", "anger", "surprise", "disgust", "love"]
NEUTRAL_N, EMO_N, PER_CALL = 120, 20, 10
MODEL, BASE = "granite4.2:8b", "http://localhost:11434/v1"
DESC = {"neutral": "plain factual or logistical statements with no emotion (schedules, instructions, facts, errands, descriptions)",
        **{e: f"text in which the writer clearly expresses {e}" for e in EMO}}


def prompt(cat, bi, k, seed):
    lo, hi = BUCKETS[bi]; span = f"at least {lo} words" if hi >= 10 ** 5 else f"{lo} to {hi} words" if lo != hi else f"{lo} words"
    return (f"Write {k} different short passages, each {span} long, each in a different topic and voice. Type: {DESC[cat]}. "
            f"Vary topics widely (batch {seed}). Output a JSON list of {k} strings and nothing else.")


def call(p, tries=4):
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": p}], "temperature": 1.0, "max_tokens": 2500, "reasoning_effort": "none"}).encode()
    for t in range(tries):
        try:
            req = urllib.request.Request(BASE + "/chat/completions", data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.loads(r.read())["choices"][0]["message"]["content"]
        except Exception:
            if t == tries - 1:
                return ""
            time.sleep(2 ** t)


def parse(ans):
    m = re.search(r"\[.*\]", ans or "", re.S)
    try:
        return [s.strip() for s in json.loads(m.group(0)) if isinstance(s, str)] if m else []
    except Exception:
        return []


def collect(cat, bi, want, seen):
    out, seed = [], 0
    while len(out) < want and seed < 12:
        seed += 1
        for s in parse(call(prompt(cat, bi, PER_CALL, seed))):
            if bucket_of(n_words(s)) == bi and s.lower() not in seen:
                seen.add(s.lower()); out.append(s)
                if len(out) == want:
                    break
    return out


TOPICS = ["weather", "trains", "cooking", "office supplies", "geography", "chemistry", "football scores", "parking", "libraries", "recipes", "software updates", "maps",
          "banking forms", "gardening", "furniture assembly", "time zones", "bus routes", "elevators", "calendars", "laundry", "traffic", "museum hours", "plumbing",
          "tax deadlines", "shipping", "hotel check-in", "car maintenance", "astronomy", "grocery lists", "printers", "bicycles", "road signs", "packaging", "math facts",
          "city history", "paint colours", "shelving", "pharmacy hours", "airport gates", "meeting rooms"]


def topup(a):
    """Deviation from the pre-registered loop (documented in the commit): short NEUTRAL buckets fell short of 120 (the writer repeats itself), so top them up
    with topic-seeded prompts. Same filters (bucket range, exact-duplicate drop); no use of probe output."""
    rows = [json.loads(l) for l in open(a.out) if l.strip()]; seen = {r["text"].lower() for r in rows}
    for bi in range(len(BUCKETS)):
        bn = bucket_name(bi); have = sum(1 for r in rows if r["intended"] == "neutral" and r["bucket"] == bn)
        for t in TOPICS:
            if have >= NEUTRAL_N:
                break
            for s in parse(call(prompt("neutral", bi, 20, 0) + f" Every passage must be about: {t}.")):
                if bucket_of(n_words(s)) == bi and s.lower() not in seen and have < NEUTRAL_N:
                    seen.add(s.lower()); have += 1
                    rows.append({"id": f"synth{len(rows):05d}", "intended": "neutral", "text": s, "n_words": n_words(s), "bucket": bn})
        print(f"topup neutral {bn}: {have}", flush=True)
    Path(a.out).write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="data/viz/synth.jsonl"); ap.add_argument("--topup", action="store_true"); a = ap.parse_args()
    if a.topup:
        return topup(a)
    jobs = [(c, bi, NEUTRAL_N if c == "neutral" else EMO_N) for bi in range(len(BUCKETS)) for c in ["neutral"] + EMO]
    seen = set(); rows = []
    def run(j):
        c, bi, n = j
        return j, collect(c, bi, n, seen)
    with ThreadPoolExecutor(2) as ex:
        for (c, bi, n), got in ex.map(run, jobs):
            for s in got:
                rows.append({"id": f"synth{len(rows):05d}", "intended": c, "text": s, "n_words": n_words(s), "bucket": bucket_name(bi)})
            print(f"{bucket_name(bi):>6} {c:9} {len(got)}/{n}", flush=True)
    Path(a.out).write_text("\n".join(json.dumps(r) for r in rows) + "\n"); print("wrote", len(rows), a.out)


if __name__ == "__main__":
    main()
