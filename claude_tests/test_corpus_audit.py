"""Model-free tests for probekit_corpus_audit.py and probekit_overlap.py (synthetic text only)."""
import json
import tempfile
from pathlib import Path

import probekit_corpus_audit as au
import probekit_overlap as ov

# ---- overlap: exact, near-duplicate, unrelated; output must be aggregate-only
corpus = ["Priya sorted the invoices by date and filed them in the grey cabinet while the printer warmed up slowly.",
          "The bus pulled away from the curb and Tom counted the stops left before his street came into view."]
test = ["priya sorted the invoices by date and filed them in the grey cabinet while the printer warmed up slowly",   # exact (normalized)
        "Priya sorted the invoices by date and filed them in the grey cabinet before lunch that day.",               # partial overlap
        "A completely different sentence about gardening tools left out in the rain overnight again."]               # none
r = ov.overlap(test, corpus)
assert r["exact"] == 1 and r["n_test"] == 3 and r["any_shared_5gram"] == 2, r
assert r["containment>=0.5"] == 2 and r["containment>=0.8"] == 1 and r["max_containment"] == 1.0, r
assert not any(isinstance(v, str) for v in r.values()), "overlap must return counts only"
assert ov.overlap([], corpus)["max_containment"] == 0.0 and ov.overlap(test, [])["exact"] == 0

# ---- audit
rows = []
for i, (cat, sp) in enumerate([(c, s) for c in ("joy", "calm", "neutral") for s in ("train", "test")] * 2):
    rows.append({"id": f"x|{cat}|{sp}|{i}", "category": cat, "split": sp, "topic": "a bus ride home",
                 "style": "short dialogue between two people (use quotation marks)" if i % 2 else "plain report-style account",
                 "text": ('"We made it," she said. ' if i % 2 else "He filed the forms. ") + "The line moved forward and nobody spoke much "
                         f"until the doors opened for passenger number {i}, who carried a blue bag of oranges home."})
with tempfile.TemporaryDirectory() as d:
    log = Path(d) / "run.log"
    log.write_text("  25 new passages\nDone: 12 new passages -> x\nTop rejection reasons: [('word:knot', 3), ('too_short', 1)]\n")
    assert au.parse_log_reasons(log) == {"word:knot": 3, "too_short": 1}
md = au.audit(rows, {"word:knot": 3}, per_cat=2, seed=1)
for needle in ("Passages per (category, split): n=12", "| joy | 2 | 2 | 4 |", "word:knot", "Duplicate ids: 0",
               "has quotes", "**[calm]", "**[joy]", "**[neutral]"):
    assert needle in md, needle
assert md.count("**[joy]") == 2                          # per_cat respected
assert au.audit(rows, None, 2, 1) == au.audit(rows, None, 2, 1)   # deterministic for a fixed seed
q = [l for l in md.splitlines() if l.startswith("| short dialogue")][0]
assert "| 1.00 |" in q                                    # every dialogue row contains quotation marks

# ---- TF-IDF gate (T0a): lexically solvable corpus must FAIL the gate; label-shuffled corpus must sit near chance
import random as _r
import probekit_tfidf_gate as gt
KW = {"joy": "zorp", "anger": "blik", "fear": "quen", "calm": "tarv"}
FILL = "the line moved forward and nobody spoke much until the doors opened for the next passenger today".split()
def mk(lexical, seed=0):
    rng, out = _r.Random(seed), []
    for t in range(20):                      # 20 topics, each wholly inside one split
        for c, kw in KW.items():
            for j in range(3):
                words = rng.sample(FILL, 8) + [f"topic{t}w{rng.randint(0, 3)}"]
                if lexical:
                    words += [kw, kw + "s"]
                else:
                    words += [rng.choice(list(KW.values()))]
                rng.shuffle(words)
                out.append({"id": f"{t}|{c}|{j}", "category": c, "topic": f"topic{t}", "split": "train" if t < 14 else "test",
                            "text": " ".join(words)})
    return out
easy, noise = mk(True), mk(False)
n_t, bad = gt.check_topics_disjoint(easy); assert n_t == 20 and not bad
easy2 = easy + [dict(easy[0], id="dup", split="test")]; assert gt.check_topics_disjoint(easy2)[1] == ["topic0"]
ge, gn = gt.gate(easy, n_boot=50), gt.gate(noise, n_boot=50)
assert ge["mean"] > 0.9 and ge["ci"][0] > 0.8, ge          # keywords make it lexically solvable
assert gn["mean"] < 0.6, gn                                 # random keywords: well below
assert ge["n_topics"] == 20 and len(ge["per_seed"]) == 3
feats = gt.top_features(easy, k=5); assert any(w.startswith("zorp") for w, _ in feats["joy"])
assert gt.openers(easy)["joy"][0][1] <= 1.0
print("CORPUS AUDIT / OVERLAP / TFIDF-GATE TESTS PASSED")
