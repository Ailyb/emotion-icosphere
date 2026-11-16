#!/usr/bin/env python3
"""
probekit_corpus_v2.py -- a harder labeled corpus.

What changed vs. v1 (v1 was solvable by word counts: TF-IDF scored 0.953 vs probes 0.857):
  * the prompt no longer asks for "physical sensation" (that produced 'clenched fists', 'heart racing', ...)
  * six writing MODES (dialogue, texts, report-style, reasoning, understated, ...) so there is no single dialect
  * emotion is shown through what people say, choose, notice, or avoid
  * banned list now covers near-synonyms (frustration, excitement, revulsion...) AND the body-cliché phrases
  * same topic-disjoint train/val/test split and JSONL schema as v1, so the rest of the toolkit works unchanged

Default 3 samples per (category, topic) = 864 passages, because adversarial_filter.py will discard ~half.
Resumable. Output: data/corpus/corpus_v2.jsonl
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

from probekit_corpus import ALL_STEMS, EMOTIONS, SPLITS, clean

MODES = [
    "short dialogue between two people (use quotation marks)",
    "text-message exchange between two people",
    "plain report-style account where the feeling shows only in what the person decides or does",
    "reflective paragraph in which the writer reasons about what to do next",
    "dry, understated first-person account",
    "third-person narrative focused on one person's actions and choices",
]

# Concept-level descriptions (deliberately NOT the stock emotion vocabulary or body language)
BRIEF2 = {
    "joy": "very pleased and uplifted by how things turned out",
    "sadness": "grieving a loss, quietly low and withdrawn",
    "fear": "sure that something bad may happen and unable to relax",
    "anger": "wronged and determined to confront whoever caused it",
    "surprise": "completely caught off guard by something they had not expected at all",
    "disgust": "repelled by something and wanting to get away from it",
    "calm": "settled and unbothered, taking things at an easy pace",
    "love": "deeply caring about another person and showing it through small acts",
}

# Additional stems (regex fragments, matched at word starts) on top of probekit_corpus.ALL_STEMS
EXTRA = [
    # near-synonyms
    "excit", "jubil", "overjoy", "euphor", "merry", "blissful", "pleased", "unhappy", "gloom", "tearful",
    "terror", "horror", "alarmed", "nervous", "uneasy", "apprehens", "petrif", "frustrat", "fum", "irritat",
    "resent", "infuriat", "seeth", "enrag", "hostil", "bewilder", "dumbfound", "flabbergast", "speechless",
    "taken aback", "revuls", "repugn", "loath", "queasy", "gagg", "foul", "soothing", "stillness", "tranquil",
    "unhurried", "tender", "fond", "adore",
    # body-language clichés
    r"heart (rac|pound|hammer)", "pounding", "clench", "fist", "stomach", "knot", "jackhammer", "trembl",
    "shiver", "goosebump", "butterflies", "lump in", "tears", "wept", "weep", "sobbed", r"cry\b", "cried",
    "crying", "gasp", "stench", "nausea", "bile", r"ache\b", "aching",
]
STEM_RE2 = re.compile(r"\b(" + "|".join(sorted(set(ALL_STEMS + EXTRA))) + ")")


def reject_reason(text: str):
    if len(text) < 100:
        return "too_short"
    if len(text) > 1000:
        return "too_long"
    m = STEM_RE2.search(text.lower())
    return f"word:{m.group(1)}" if m else None


def emotion_prompt(emotion, topic, mode):
    return (f"Write a {mode} of 3-5 sentences about {topic}, in which the main person is {BRIEF2[emotion]}. "
            f"Show it through what they say, choose, notice, or avoid doing. Do NOT name the emotion or close "
            f"synonyms, and do NOT use clichéd body-sensation phrases (racing heart, clenched fists, churning "
            f"stomach, trembling, tears, goosebumps, butterflies, lump in the throat). Output only the text.")


def neutral_prompt(topic, mode):
    return (f"Write a {mode} of 3-5 sentences about {topic}. Keep it matter-of-fact: people go about the task with "
            f"no emotional language and no strong feelings. Output only the text.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Llama-3.2-3B-Instruct")
    ap.add_argument("--out", default="data/corpus/corpus_v2.jsonl")
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--temp", type=float, default=0.95)
    ap.add_argument("--retries", type=int, default=4)
    a = ap.parse_args()

    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler
    model, tok = load(a.model)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["id"] for l in out.read_text().splitlines() if l.strip()} if out.exists() else set()
    print(f"{len(done)} passages already present")

    cats, topics, reasons, n = EMOTIONS + ["neutral"], list(SPLITS), Counter(), 0
    total = len(cats) * len(topics) * a.samples
    with out.open("a") as f:
        for ci, cat in enumerate(cats):
            for ti, topic in enumerate(topics):
                for j in range(a.samples):
                    n += 1
                    rid = f"v2|{cat}|{topic}|{j}"
                    if rid in done:
                        continue
                    mode = MODES[(ci + ti + j) % len(MODES)]
                    user = neutral_prompt(topic, mode) if cat == "neutral" else emotion_prompt(cat, topic, mode)
                    prompt = tok.apply_chat_template([{"role": "user", "content": user}],
                                                     add_generation_prompt=True, tokenize=False)
                    text = None
                    for _ in range(a.retries):
                        cand = clean(generate(model, tok, prompt=prompt, max_tokens=200, verbose=False,
                                              sampler=make_sampler(temp=a.temp, top_p=0.95)))
                        why = reject_reason(cand)
                        if why is None:
                            text = cand; break
                        reasons[why] += 1
                    if text is None:
                        print(f"  [skip] {rid}"); continue
                    f.write(json.dumps({"id": rid, "category": cat, "topic": topic, "style": mode,
                                        "split": SPLITS[topic], "text": text}) + "\n"); f.flush()
                    if n % 25 == 0:
                        print(f"  {n}/{total}")
    print(f"Done -> {out}\nTop rejection reasons: {reasons.most_common(10)}")


if __name__ == "__main__":
    main()
