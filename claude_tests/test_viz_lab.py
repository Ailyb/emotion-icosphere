#!/usr/bin/env python3
"""Model-free test of probekit_viz_lab helpers (synthetic generations)."""
import json
import tempfile
from pathlib import Path

import probekit_viz_lab as L

d = Path(tempfile.mkdtemp())
gens = [dict(gid="base|0", cond="base", emotion=None, rid=None, strength=0.0, pid=0, topic="t", text="plain", nll=1.0),
        dict(gid="base|1", cond="base", emotion=None, rid=None, strength=0.0, pid=1, topic="t2", text="plain2", nll=1.0),
        dict(gid="p|0", cond="probe", emotion="joy", rid=None, strength=0.1, pid=0, topic="t", text="happy!", nll=1.2),
        dict(gid="p|1", cond="probe", emotion="joy", rid=None, strength=0.1, pid=1, topic="t2", text="glad", nll=1.0)]
(d / "generations.jsonl").write_text("\n".join(map(json.dumps, gens))); (d / "judged.jsonl").write_text("\n".join(json.dumps(dict(gid=g["gid"], label="joy" if g["cond"] == "probe" else "neutral")) for g in gens))
g, lab, rows = L.load(d)
r = [x for x in rows if x["cond"] == "probe" and x["pid"] == 0][0]
assert abs(r["ppl_ratio"] - 2.718281828 ** 0.2) < 1e-6 and r["label"] == "joy"
s = L.samples(g, "probe", "joy", 0.1); assert [x["steered"] for x in s] == ["happy!", "glad"] and s[0]["base"] == "plain"
assert L.samples(g, "probe", "joy", 0.5) == []
page = L.PAGE
assert "__DATA__" in page and "NOT FLUENT" in page and "model-dependent" not in page.lower()   # the note is injected by main(), not hard-coded in the template
print("PASS test_viz_lab")
