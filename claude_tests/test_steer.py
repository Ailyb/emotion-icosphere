"""Model-free tests for the steering analysis (probekit_steer.analyze and friends) on synthetic judge labels."""
import numpy as np

import probekit_steer as st

rng = np.random.default_rng(0)
N, STR = 40, [0.1, 0.2]
def build(effect, random_effect=0.0, ppl=1.1, n_random=20, random_pushes=False):
    """effect: dict emotion -> prob that steered text is labelled with that emotion; base is 'neutral'."""
    g = []
    for pid in range(N):
        g.append({"cond": "base", "emotion": None, "rid": None, "strength": 0.0, "pid": pid, "label": "neutral", "ppl_ratio": 1.0})
    for s in STR:
        for e in st.EMOTIONS:
            for pid in range(N):
                lab = e if rng.random() < effect.get(e, 0.0) else "neutral"
                g.append({"cond": "probe", "emotion": e, "rid": None, "strength": s, "pid": pid, "label": lab, "ppl_ratio": ppl})
                g.append({"cond": "v1", "emotion": e, "rid": None, "strength": s, "pid": pid, "label": e if rng.random() < 0.2 else "neutral", "ppl_ratio": ppl})
        for r in range(n_random):
            for pid in range(N):
                target = st.EMOTIONS[r % 7] if random_pushes else rng.choice(st.EMOTIONS)       # each random direction pushes toward ONE emotion
                lab = target if rng.random() < random_effect else "neutral"
                g.append({"cond": "random", "emotion": None, "rid": r, "strength": s, "pid": pid, "label": lab, "ppl_ratio": ppl})
    return g

# strong, specific steering on 5 emotions, nothing on 2, random directions do little -> claim
res = st.analyze(build({e: 0.8 for e in st.EMOTIONS[:5]}, random_effect=0.05))
assert res["n_prompts"] == N and res["n_random"] == 20 and res["claim"] and set(res["steerable"]) == set(st.EMOTIONS[:5]), res["steerable"]
c = res["emotions"]["joy"][0.2]
assert c["delta"] > 0.5 and c["ci_excludes_0"] and c["beats_all_random"] and c["fluent"] and c["specific"] and c["probe_minus_v1"][0] > 0.3
assert not res["emotions"]["love"][0.2]["specific"] and res["emotions"]["joy"][0.2]["off_target"]["joy"] == c["delta"]
# if random directions shift labels just as much the claim must fail even though probe directions "work"
res2 = st.analyze(build({e: 0.5 for e in st.EMOTIONS}, random_effect=0.9, random_pushes=True)); assert not res2["claim"] and not res2["steerable"]
# ...but scattering random directions (13% per emotion) do not block a 50% probe effect
assert st.analyze(build({e: 0.5 for e in st.EMOTIONS}, random_effect=0.9))["claim"]
# broken (non-fluent) steering never counts
res3 = st.analyze(build({e: 0.9 for e in st.EMOTIONS}, random_effect=0.0, ppl=3.0)); assert not res3["claim"] and not any(c["fluent"] for m in res3["emotions"].values() for c in m.values())
# no effect at all
assert not st.analyze(build({}, 0.0))["claim"]
# helpers
assert st.delta(["neutral"] * 10, ["joy"] * 5 + ["neutral"] * 5, "joy") == 0.5
lo, hi = st.boot_delta(["neutral"] * 40, ["joy"] * 40, "joy"); assert lo == hi == 1.0
d, lo2, hi2 = st.boot_diff(["neutral"] * 40, ["joy"] * 40, ["neutral"] * 40, "joy"); assert d == 1.0 and lo2 == 1.0
md = st.render(res); assert "specifically steerable" in md and "Pre-registered conclusion" in md and "**yes**" in md
main, pilot = st.topics(); assert len(main) == 40 and len(pilot) == 8 and not set(main) & set(pilot)
assert st.prompt_text("a bus ride") == "Write three sentences about a bus ride."
print("STEER TESTS PASSED")
