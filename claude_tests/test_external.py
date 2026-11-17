"""Model-free tests for probekit_external.py using an injected fake datasets-server."""
import json
import tempfile
from pathlib import Path

import probekit_external as ex

# ---- paging: fetch_rows follows offsets until num_rows_total
data = [{"row": {"i": i}} for i in range(250)]
calls = []
def fake_get(url):
    calls.append(url)
    q = dict(p.split("=", 1) for p in url.split("?", 1)[1].split("&"))
    off, ln = int(q["offset"]), int(q["length"])
    return {"rows": data[off:off + ln], "num_rows_total": len(data)}
rows = ex.fetch_rows("x/y", "c", "s", get=fake_get, pause=0)
assert len(rows) == 250 and len(calls) == 3 and "/rows?" in calls[0]
assert "/filter?" in (ex.fetch_rows("x/y", "c", "s", where="\"a\"='b'", get=fake_get, pause=0) and calls[-1])

# ---- adapters
assert ex.adapt_dair([{"text": "i feel joyful today", "label": 1}])[0][2:] == ("joy", "joy")
isr = ex.adapt_isear([{"ID": 1, "SIT": "When I passed the exam.", "Field1": "joy"}, {"ID": 2, "SIT": "[ No response. ]", "Field1": "fear"},
                      {"ID": 3, "SIT": "When I lied.", "Field1": "guilt"}])
assert [(t[2], t[3]) for t in isr] == [("joy", "joy"), (None, "guilt")]                      # no-response dropped, guilt unmapped
dd = ex.adapt_dailydialog([{"dialogue_id": "d0", "turns": [{"emotion": "no emotion", "utterance": "Hello there my friend", "utt_idx": 0},
                                                           {"emotion": "happiness", "utterance": "That is great news indeed", "utt_idx": 1}]}])
assert [t[2] for t in dd] == ["neutral", "joy"] and dd[1][0] == "dd|d0|1"

# ---- build_items: unmapped/short/duplicate drops, per-class cap, neutral capped at the largest other class
items = [(f"a{i}", f"joy sentence number {i} with words", "joy", "joy") for i in range(10)] + \
        [(f"n{i}", f"plain neutral statement number {i} here", "neutral", "no emotion") for i in range(30)] + \
        [(f"f{i}", f"fear sentence number {i} with words", "fear", "fear") for i in range(4)] + \
        [("u", "this one has no mapped label at all", None, "shame"), ("s", "too short", "joy", "joy"),
         ("d", "joy sentence number 0 with words", "joy", "joy")]
rows, drops = ex.build_items(items, cap_per_class=8)
cc = {c: sum(r["category"] == c for r in rows) for c in ("joy", "neutral", "fear")}
assert cc == {"joy": 8, "neutral": 8, "fear": 4}, cc                                          # per-class cap 8; neutral <= largest other (8)
assert drops["unmapped_label"] == 1 and drops["too_short"] == 1 and drops["duplicate"] == 1 and drops["over_cap_joy"] == 2
assert ex.build_items(items, 8)[0] == rows                                                    # deterministic

# ---- near-duplicate drop and full build/verify round trip with the fake server
r2, nd = ex.drop_near_duplicates(rows, ["joy sentence number 3 with words and more"])
assert nd >= 0 and len(r2) == len(rows) - nd
def fake_get2(url):
    raw = [{"text": f"i feel joyful and calm about thing number {i}", "label": 1} for i in range(40)] + \
          [{"text": f"i feel sad and tired about thing number {i}", "label": 0} for i in range(40)]
    return {"rows": [{"row": r} for r in raw], "num_rows_total": len(raw)}
with tempfile.TemporaryDirectory() as d:
    d = Path(d)
    m = ex.build_source("dair", str(d / "out"), str(d / "raw"), train_corpora=(), get=fake_get2)
    assert m["n"] == 80 and m["class_counts"] == {"joy": 40, "sadness": 40} and m["classes_present"] == ["joy", "sadness"]
    rows3, m3 = ex.load_source("dair", str(d / "out")); assert len(rows3) == 80 and all(r["writer"] == "human-dair" for r in rows3)
    p = d / "out" / "dair_test_frozen.jsonl"; p.write_text(p.read_text() + " ")
    try:
        ex.load_source("dair", str(d / "out")); raise AssertionError("expected refusal")
    except RuntimeError as e:
        assert "changed" in str(e)
# ---- start offset + row filter (DailyDialog tail fetch)
tail = ex.fetch_rows("x/y", "c", "s", get=fake_get, pause=0, start=200)
assert len(tail) == 50 and tail[0] == {"i": 200}
# ---- zip route (DailyDialog) with a fake download: split filter + utterance expansion + neutral cap
import io, zipfile
dlg = [{"dialogue_id": f"dailydialog-test-{i}", "data_split": "test", "turns": [
          {"emotion": "no emotion", "utterance": f"Well hello there number {i} my friend", "utt_idx": 0},
          {"emotion": "happiness", "utterance": f"What a lovely surprise to see you {i}", "utt_idx": 1},
          {"emotion": "anger", "utterance": f"I am really not pleased about this {i}", "utt_idx": 2}]} for i in range(6)]
dlg.append({"dialogue_id": "dailydialog-train-0", "data_split": "train", "turns": [{"emotion": "fear", "utterance": "A training utterance that must not appear here", "utt_idx": 0}]})
buf = io.BytesIO()
with zipfile.ZipFile(buf, "w") as z:
    z.writestr("data/dialogues.json", json.dumps(dlg))
with tempfile.TemporaryDirectory() as d:
    d = Path(d)
    m = ex.build_source("dailydialog", str(d / "o"), str(d / "r"), train_corpora=(), get_bytes=lambda url: buf.getvalue())
    assert m["class_counts"] == {"neutral": 6, "joy": 6, "anger": 6} and m["raw_rows"] == 6, m
    rows4, _ = ex.load_source("dailydialog", str(d / "o")); assert all("train" not in r["id"] for r in rows4)
print("EXTERNAL SOURCE TESTS PASSED")
