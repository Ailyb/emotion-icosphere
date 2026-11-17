"""Model-free tests for probekit_lexicon_subtypes on a tiny synthetic lexicon."""
import json
import tempfile
from pathlib import Path

import numpy as np

import probekit_lexicon_subtypes as ls

with tempfile.TemporaryDirectory() as d:
    d = Path(d)
    (d / "nrc.txt").write_text("\n".join(["happy\tjoy\t1", "happy\tfear\t0", "awful\tfear\t1", "table\tanger\t0", "trust\ttrust\t1", "great\tpositive\t1"]))
    words = ls.load_nrc(d / "nrc.txt")
    assert words == {"happy", "awful", "trust"}                                    # positive/negative-only and 0-association rows excluded
    assert ls.subtypes("I am happy today", "joy", words) == ["lexicon_match"]
    assert ls.subtypes("The table is wooden", "anger", words) == ["no_lexicon_match"]
    assert ls.subtypes("I am not happy at all", "sadness", words) == ["lexicon_match", "negation"]
    assert ls.subtypes("It isn't awful", "neutral", words) == ["emotion_word_neutral", "negation"]
    assert ls.subtypes("The meeting is at noon", "neutral", words) == ["plain_neutral"]
    assert ls.subtypes("no way too far from awful", "fear", words) == ["lexicon_match"]       # negator more than 3 tokens before
    rows = [{"id": f"i{k}", "text": t, "category": c} for k, (t, c) in enumerate(
        [("so happy now", "joy")] * 30 + [("the wooden table", "anger")] * 30 + [("plain notice here", "neutral")] * 30)]
    (d / "t.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    class A: pass
    a = A(); a.nrc, a.testset, a.out = str(d / "nrc.txt"), str(d / "t.jsonl"), str(d / "side.json"); ls.build(a)
    side = json.loads((d / "side.json").read_text()); assert side["manifest"]["counts"] == {"lexicon_match": 30, "no_lexicon_match": 30, "plain_neutral": 30}
    ids = [r["id"] for r in rows]; labels = [r["category"] for r in rows]
    preds = {"probe_lda": labels, "tfidf": ["joy"] * 90, "native_lr": labels, "embed": labels}
    res = ls.stratified(labels, preds, [side["subtypes"][i] for i in ids], ids)
    assert res["lexicon_match"]["n"] == 30 and res["negation"]["note"].startswith("n < 20")
    assert "probe_minus_tfidf" not in res["lexicon_match"]                           # n=30 < 50: no CI reported
    big = ls.stratified(labels * 2, {k: v * 2 for k, v in preds.items()}, [side["subtypes"][i] for i in ids] * 2, ids + [i + "b" for i in ids])
    assert big["lexicon_match"]["n"] == 60 and "probe_minus_tfidf" in big["lexicon_match"]
print("LEXICON SUBTYPE TESTS PASSED")
