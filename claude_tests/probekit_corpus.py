#!/usr/bin/env python3
"""
probekit_corpus.py -- generate a labeled corpus for building and validating concept probes.

Design choices (each one addresses a flaw in the single-sentence-pair approach):
  * many passages per emotion across many topics, in several writing styles
  * a matched NEUTRAL passage for every (topic, style) -- baseline and calibration
  * train / val / test splits are DISJOINT BY TOPIC, so validation measures generalization to
    new situations, not memorization of scenery
  * passages containing emotion words are rejected, so probes can't win by spotting the word "afraid"
  * resumable: re-running appends only what's missing

Default: 32 topics x 9 categories x 2 samples = 576 passages (~20-30 min on an M-series Mac with a 3B 4-bit model).
You can swap in text from a stronger model: just write the same JSONL schema.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

EMOTIONS = ["joy", "sadness", "fear", "anger", "surprise", "disgust", "calm", "love"]

BRIEF = {
    "joy": "joy: delighted, elated, buoyant",
    "sadness": "sadness: heavy-hearted, grieving, low",
    "fear": "fear: frightened, tense, in danger",
    "anger": "anger: furious, indignant, seething",
    "surprise": "surprise: caught completely off guard, astonished",
    "disgust": "disgust: repulsed, nauseated, appalled",
    "calm": "calm: serene, unhurried, at ease",
    "love": "love: tender, devoted, deeply attached to someone",
}

# Stems (matched at word starts). Applied to emotional AND neutral passages.
BANNED = {
    "joy": ["joy", "happy", "happi", "delight", "elat", "thrill", "ecsta", "glee", "cheer"],
    "sadness": ["sad\\b", "sadn", "sadl", "sadden", "sorrow", "grief", "griev", "mourn", "heartbr", "depress", "melanchol", "miserab"],
    "fear": ["fear", "afraid", "scare", "scary", "terrif", "terror", "frighten", "dread", "panic", "anxi"],
    "anger": ["anger", "angry", "angri", "furi", "fury", "rage", "irate", "outrag", "livid", "annoy"],
    "surprise": ["surpris", "astonish", "amaz", "shock", "stunn", "startl", "unexpected"],
    "disgust": ["disgust", "revolt", "repuls", "nause", "gross", "sicken", "vile"],
    "calm": ["calm", "peace", "serene", "tranquil", "relax"],
    "love": ["love", "adore", "affection", "devot", "cherish", "beloved"],
}
ALL_STEMS = sorted({s for v in BANNED.values() for s in v})
# stems are regex fragments matched at word starts (e.g. "sad\\b" bans "sad" but not "saddle"; "happi" bans
# "happiness" but NOT "happened")
STEM_RE = re.compile(r"\b(" + "|".join(ALL_STEMS) + ")")

TRAIN_TOPICS = [
    "a job interview", "moving to a new city", "a family dinner", "a long drive at night", "waiting for test results",
    "a school exam", "a neighbor's dog", "a delayed flight", "a hiking trip", "a wedding", "cooking dinner",
    "a software deployment", "a phone call from an old friend", "a garden in spring", "a crowded train",
    "a lost package", "a birthday party", "a visit to the dentist", "a group project", "a late-night phone call",
]
VAL_TOPICS = ["a farmers market", "a power outage", "a school reunion", "a swim at the lake", "a broken laptop", "a new roommate"]
TEST_TOPICS = ["a hospital waiting room", "a road trip with friends", "a restaurant kitchen", "a thunderstorm",
               "an apartment viewing", "a town council meeting"]
SPLITS = {**{t: "train" for t in TRAIN_TOPICS}, **{t: "val" for t in VAL_TOPICS}, **{t: "test" for t in TEST_TOPICS}}

STYLES = ["first-person diary entry", "third-person short narrative", "message written to a friend"]


def emotion_prompt(emotion, topic, style):
    return (f"Write a {style} of 3-4 sentences about {topic}, in which the writer clearly experiences {BRIEF[emotion]}. "
            f"Do not use the word '{emotion}' or any direct emotion label; convey it through situation, behavior, and "
            f"physical sensation. Output only the text, no title or commentary.")


def neutral_prompt(topic, style):
    return (f"Write a {style} of 3-4 sentences about {topic}. Keep it neutral and matter-of-fact, with no emotional "
            f"language and no strong feelings. Output only the text, no title or commentary.")


_QUOTES = "\"\u201c\u201d"


def clean(text: str) -> str:
    """Strip commentary and a quote pair that wraps the WHOLE output, but keep dialogue quotes and line breaks.
    (Earlier versions stripped the first/last quote of every passage and joined lines, flattening the dialogue and
    text-message styles; corpus.jsonl, corpus_v2.jsonl and corpus_v3_qwen3.8-27b.jsonl were generated with that
    behaviour and are NOT regenerated.)"""
    text = text.strip()
    if len(text) > 1 and text[0] in _QUOTES and text[-1] in _QUOTES and sum(ch in _QUOTES for ch in text) == 2:
        text = text[1:-1].strip()                       # one wrapping pair, no inner quotes => not dialogue
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if len(lines) > 1 and lines[0].endswith(":") and len(lines[0].split()) >= 3:   # "Here is a diary entry:" preamble
        lines = lines[1:]                               # (one-word "Mara:" speaker labels are kept)
    return "\n".join(lines)


def reject_reason(text: str):
    if len(text) < 120:
        return "too_short"
    if len(text) > 900:
        return "too_long"
    m = STEM_RE.search(text.lower())
    return f"word:{m.group(1)}" if m else None


def acceptable(text: str) -> bool:
    return reject_reason(text) is None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Llama-3.2-3B-Instruct")
    ap.add_argument("--out", default="data/corpus/corpus.jsonl")
    ap.add_argument("--samples", type=int, default=2, help="samples per (category, topic); raise for a bigger test set")
    ap.add_argument("--temp", type=float, default=0.9)
    ap.add_argument("--retries", type=int, default=3)
    a = ap.parse_args()

    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    model, tok = load(a.model)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(l)["id"] for l in out.read_text().splitlines() if l.strip()}
    print(f"{len(done)} passages already present")

    from collections import Counter
    reasons = Counter()
    cats = EMOTIONS + ["neutral"]
    topics = list(SPLITS)
    total = len(cats) * len(topics) * a.samples
    n = 0
    with out.open("a") as f:
        for ci, cat in enumerate(cats):
            for ti, topic in enumerate(topics):
                for j in range(a.samples):
                    n += 1
                    rid = f"{cat}|{topic}|{j}"
                    if rid in done:
                        continue
                    style = STYLES[(ci + ti + j) % len(STYLES)]
                    user = neutral_prompt(topic, style) if cat == "neutral" else emotion_prompt(cat, topic, style)
                    prompt = tok.apply_chat_template([{"role": "user", "content": user}], add_generation_prompt=True, tokenize=False)
                    text = None
                    for _ in range(a.retries):
                        cand = clean(generate(model, tok, prompt=prompt, max_tokens=160, verbose=False,
                                              sampler=make_sampler(temp=a.temp, top_p=0.95)))
                        why = reject_reason(cand)
                        if why is None:
                            text = cand; break
                        reasons[why] += 1
                    if text is None:
                        print(f"  [skip] {rid}: no acceptable sample after {a.retries} tries"); continue
                    f.write(json.dumps({"id": rid, "category": cat, "topic": topic, "style": style,
                                        "split": SPLITS[topic], "text": text}) + "\n")
                    f.flush()
                    if n % 20 == 0:
                        print(f"  {n}/{total}")
    print(f"Done -> {out}")
    print("Top rejection reasons:", reasons.most_common(8))


if __name__ == "__main__":
    main()