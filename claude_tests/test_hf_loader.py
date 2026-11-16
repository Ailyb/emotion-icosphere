import random
import probekit_corpus_hf as hf
names = ["admiration", "joy", "sadness", "fear", "anger", "surprise", "disgust", "love", "neutral", "relief"]
W = lambda n: " ".join(["word"] * n)
ex = []
for i in range(40):
    ex.append({"id": f"a{i}", "text": f"[NAME] said {W(12)} number{i} [RELIGION] [FOO_BAR]", "labels": [names.index("joy")]})
for i in range(5):
    ex.append({"id": f"b{i}", "text": f"multi label {W(12)} {i}", "labels": [1, 2]})                # multi-label: dropped
for i in range(5):
    ex.append({"id": f"c{i}", "text": f"unmapped relief {W(12)} {i}", "labels": [names.index("relief")]})  # unmapped: dropped
ex += [{"id": "d0", "text": "too short", "labels": [2]}, {"id": "d1", "text": W(200), "labels": [2]}]      # length: dropped
ex += [{"id": "e0", "text": f"dup {W(12)}", "labels": [3]}, {"id": "e1", "text": f"DUP {W(12)}", "labels": [3]}]  # duplicate
ex += [{"id": f"n{i}", "text": f"neutral thing {W(12)} {i}", "labels": [names.index("neutral")]} for i in range(30)]
rows, st = hf.convert(ex, names, hf.GOEMO_MAP, "train", "goemotions", "reddit-goemotions", "style", {"train": 25, "val": 5, "test": 5}, random.Random(0))
cnt = {c: sum(r["category"] == c for r in rows) for c in {r["category"] for r in rows}}
print("kept:", cnt, "| dropped:", dict(st))
assert cnt == {"joy": 25, "fear": 1, "neutral": 25}, cnt
assert st["dropped_multi_or_no_label"] == 5 and st["dropped_unmapped_label"] == 5 and st["dropped_length"] == 2 and st["dropped_duplicate"] == 1
j = [r for r in rows if r["category"] == "joy"][0]
assert "Sam said" in j["text"] and "church" in j["text"] and "[" not in j["text"], j["text"]
assert set(j) >= {"id", "category", "topic", "style", "split", "writer", "text"} and j["split"] == "train" and j["id"].startswith("hf|goemotions|train|")
assert len({r["id"] for r in rows}) == len(rows) and len({r["topic"] for r in rows}) == len(rows)
r2, _ = hf.convert([{"text": W(12), "label": 3}], ["sadness", "joy", "love", "anger"], hf.DAIR_MAP, "val", "dair_emotion", "w", "s", {"val": 5}, random.Random(0))
assert r2 and r2[0]["category"] == "anger"
print("HF LOADER TEST PASSED")
