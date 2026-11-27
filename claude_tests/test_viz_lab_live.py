#!/usr/bin/env python3
import probekit_viz_lab_live as L
ok, r = L.plan_request(dict(topic="a bake sale", emotion="joy", dtype="probe", strength=0.2, n_random=3), 20); assert ok and r["ei"] == 0 and r["n_random"] == 3
bad = [dict(topic="", emotion="joy", strength=0.2), dict(topic="x" * 81, emotion="joy", strength=0.2), dict(topic="a", emotion="rage", strength=0.2), dict(topic="a", emotion="joy", dtype="x", strength=0.2),
       dict(topic="a", emotion="joy", strength=0), dict(topic="a", emotion="joy", strength=1.5), dict(topic="a", emotion="joy", strength="abc"), dict(topic="a", emotion="joy", strength=0.2, n_random=99), None]
assert not any(L.plan_request(b, 20)[0] for b in bad)
assert L.plan_request(dict(topic="a", emotion="joy", strength=0.2, n_random=8), 3)[1]["n_random"] == 3
s = L.summarize("neutral", "joy", ["joy", None, "neutral"], "joy", 1.2); assert s["target_in_steered"] and not s["target_in_base"] and s["random_with_target"] == 1 and s["n_random_labelled"] == 2 and s["fluent"]
assert not L.summarize("a", "b", [], "joy", 1.6)["fluent"]
print("PASS test_viz_lab_live")
