"""Model-free end-to-end test of probekit_audit on synthetic data, including skip-and-report of missing classes."""
import tempfile
from pathlib import Path

import numpy as np

import probekit_audit as pa

rng = np.random.RandomState(0)
CL = ["joy", "sadness", "fear", "anger", "neutral"]
D = 24
def make(n_per, classes, seed, split=None):
    r = np.random.RandomState(seed); rows, X, E = [], [], []
    for i, c in enumerate(CL):
        if c not in classes:
            continue
        for j in range(n_per):
            v = r.randn(D); v[i] += 3.0; X.append(v); E.append(r.randn(D) + 0.2 * v)
            rows.append({"id": f"{seed}{c}{j}", "text": f"{c} word{j % 4} filler text number {j}", "category": c, **({"split": split(j)} if split else {})})
    return {"rows": rows, "feats": {"X": {5: np.array(X)}, "E": np.array(E), "meta": None}}
train = make(60, CL, 1, split=lambda j: "train" if j < 45 else "val")
tests = {"in": make(30, CL, 2), "partial": make(30, ["joy", "anger"], 3), "tiny": make(4, CL, 4)}
md, res = pa.audit(train, tests, 5)
assert "Probe audit" in md and "NO META SIDECAR" in md
assert set(res) == {"in", "partial"}                                         # 'tiny' (n<10 per class) skipped entirely and reported
assert "## tiny: skipped" in md and "test n < 10" in md
assert res["in"]["classes"] == sorted(CL) and res["partial"]["classes"] == ["anger", "joy"]
assert res["in"]["f1"]["probe"] > 0.5 and res["in"]["f1"]["tfidf"] > 0.9       # class word in the text: lexical is easy; probe finds the signal
assert "probe - tfidf" in res["in"]["diffs"] and "probe - embed" in res["in"]["diffs"] and "stratum" in md
# an exclusion removes the class from training and from the evaluated set
md2, res2 = pa.audit(train, {"in": tests["in"]}, 5, exclude=("fear",))
assert "fear" not in res2["in"]["classes"] and "fear (not in train" in md2
# missing neutral in train -> clear error, not a silent wrong answer
try:
    pa.audit(make(60, ["joy", "anger", "fear"], 5, split=lambda j: "train" if j < 45 else "val"), {"in": tests["in"]}, 5); raise AssertionError("expected ValueError")
except ValueError as e:
    assert "neutral" in str(e)
# cache round trip with meta + id-order guard
with tempfile.TemporaryDirectory() as d:
    d = Path(d); rows = train["rows"]; ids = [r["id"] for r in rows]
    np.savez(d / "c.npz", ids=np.array(ids), skip=4, L5=train["feats"]["X"][5], E=train["feats"]["E"])
    pa.write_meta(d / "c.npz", "toy-model", 4, [5], ids)
    c = pa.load_cache(d / "c.npz", rows); assert c["meta"]["model"] == "toy-model" and c["meta"]["n"] == len(ids) and 5 in c["X"]
    try:
        pa.load_cache(d / "c.npz", rows[::-1]); raise AssertionError("expected ValueError")
    except ValueError as e:
        assert "do not match" in str(e)
print("AUDIT TESTS PASSED")
