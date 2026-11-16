#!/usr/bin/env python3
"""Model-free tests for the v2 bundle: filters, hard set, adversarial filter, and the full analyze() pipeline
on SYNTHETIC hidden states. Verifies plumbing, not that real LLM states behave this way."""
import json, tempfile, random
from pathlib import Path
import numpy as np

import probekit_corpus_v2 as c2
import adversarial_filter as af
import probekit_hardset as hs
import probekit_eval_v2 as ev
from probekit_probes import DEFAULT_EMOTIONS as E, NEUTRAL

# ---- corpus v2 filter
pad = " " + "word " * 25
cases = {"I was so excited": False, "his stomach dropped": False, "her heart racing": False, "she wept quietly": False,
         "the stock market happened to rise": True, "a plain walk to the shop": True, "he cried out": False,
         "a cryptic note": True, "the knot in the rope": False, "an achievement": True}
for t, ok in cases.items():
    assert c2.reject_reason(t + pad) is None if ok else c2.reject_reason(t + pad) is not None, t
print("corpus_v2 filter ok")

# ---- hard set writer
with tempfile.TemporaryDirectory() as d:
    hs.main(d)
    rows = [json.loads(l) for l in open(Path(d) / "hardset.jsonl")]
    assert len(rows) == 47 and {r["subtype"] for r in rows} == {"implicit", "negation", "emotion_word_neutral", "plain_neutral"}
    assert sum(r["category"] == "neutral" for r in rows) == 14
    assert len(open(Path(d) / "own_voice_TEMPLATE.jsonl").read().splitlines()) == 45
print("hard set ok (47 passages)")

# ---- adversarial filter on a synthetic lexical corpus
rng = random.Random(0); nrng = np.random.default_rng(0)
cats = E + [NEUTRAL]
cue = {c: [f"{c}cue{i}" for i in range(4)] for c in cats}
filler = [f"w{i}" for i in range(60)]
topics = {"train": [f"t{i}" for i in range(14)], "val": ["v1", "v2", "v3", "v4"], "test": ["x1", "x2", "x3", "x4"]}
rows = []
for sp, ts in topics.items():
    for t in ts:
        for c in cats:
            for j in range(3):
                easy = rng.random() < 0.5                      # half the passages carry obvious cue words
                words = rng.choices(filler, k=18) + ([rng.choice(cue[c]) for _ in range(3)] if easy else [])
                rng.shuffle(words)
                rows.append({"id": f"{c}|{t}|{j}", "category": c, "topic": t, "split": sp, "text": " ".join(words), "_easy": easy})
p = af.lexical_true_prob(rows); hard = af.mark_hard(rows, p, 0.5)
easy = np.array([r["_easy"] for r in rows])
print(f"adversarial filter: P(easy | kept-hard) = {easy[hard].mean():.2f} vs overall {easy.mean():.2f}")
assert easy[hard].mean() < easy.mean() - 0.2
for r, h in zip(rows, hard): r["hard"] = bool(h)

# ---- analyze() on synthetic hidden states
H = 96; layers = [2, 5]
dirs = {e: (lambda v: v / np.linalg.norm(v))(nrng.normal(size=H)) for e in E}
offset = nrng.normal(size=H) * 8
axes = nrng.normal(size=(5, H))
def state(c, strength=3.5):
    x = offset + 6 * (nrng.normal(size=5) @ axes) + 0.7 * nrng.normal(size=H)
    return x + (strength * dirs[c] if c != NEUTRAL else 0)
def states(cat_list): return {l: np.stack([state(c) + l * 0.0 for c in cat_list]) for l in layers}
Xc = states([r["category"] for r in rows])
hrows = [{"id": f"h{i}", "category": c, "subtype": st, "text": f"alpha beta {c} gamma"} for i, (c, st) in
         enumerate([(c, s) for c in cats for s in ("implicit", "negation")] * 2)]
Xh = states([r["category"] for r in hrows])
s_idx = [i for i, r in enumerate(rows) if r["split"] == "test"]
Xs = states([rows[i]["category"] for i in s_idx])
res = ev.analyze(layers, Xc, rows, Xh, hrows, Xs, s_idx, E, "all")
L = res["selected_layer"]; P = res["per_layer"][L]["probe"]
print(f"analyze: selected layer {L}; probe val {P['val']:.2f}, test-hard {P['test_hard']:.2f}, hardset {P['hardset']:.2f}, "
      f"hidden-LR hardset {res['per_layer'][L]['hidden_lr']['hardset']:.2f}")
assert P["test_hard"] > 0.8 and P["hardset"] > 0.8 and res["null_8way_test_hard"]["p95"] < 0.35
with tempfile.TemporaryDirectory() as d:
    ev.write_report(Path(d), res, "synthetic")
    md = (Path(d) / "report_v2.md").read_text()
    assert "Headline" in md and "Layer curve" in md and "Wilson" in md
    json.loads((Path(d) / "report_v2.json").read_text())
print("report written ok")
print("ALL BUNDLE TESTS PASSED")
