"""Model-free tests for probekit_variants evaluation logic (synthetic features)."""
import numpy as np

import probekit_variants as pv

CL = ["joy", "sadness", "fear", "anger"]
def data(n, seed, good):
    r = np.random.RandomState(seed); y = np.array([c for c in CL for _ in range(n)], dtype=object)
    base = r.randn(len(y), 12)
    for i, c in enumerate(CL):
        base[y == c, i] += 2.0                                  # signal in the 'good' representation
    noise = r.randn(len(y), 12)                                  # a representation with no signal
    return {"V0 mean": base, "V5 concat": np.hstack([base, base]), "V1 last": noise, "V2 max": base * 0.5, "V3 last4": noise, "V4 prompted": base + 3 * (y == "joy")[:, None]}, y
Ftr, ytr = data(60, 0, True); Fva, yva = data(20, 1, True); Fte, yte = data(60, 2, True); Fte2, yte2 = data(60, 3, True)
preds = pv.evaluate_variants((lambda v: Ftr[v], ytr), (lambda v: Fva[v], yva), [("A", lambda v: Fte[v], yte), ("B", lambda v: Fte2[v], yte2)])
assert set(preds) == {"A", "B"} and set(preds["A"]) == set(pv.VARIANTS)
f = lambda s, v, y: pv.f1_9(y, preds[s][v])
assert f("A", "V0 mean", yte) > 0.7 and f("A", "V1 last", yte) < 0.5                  # no-signal representation is near chance (0.25)
tl = pv.tally(preds, {"A": yte, "B": yte2})
assert tl["V1 last"]["losses"] == 2 and tl["V1 last"]["wins"] == 0                    # clearly worse on both sets
assert tl["V5 concat"]["wins"] <= 2 and "V0 mean" not in tl
md = pv.render(preds, {"A": yte, "B": yte2}, tl, 16)
assert "variant - V0" in md and "claim needs" in md and "no variant improves" in md
# variant_matrix routing
means = {14: np.zeros((3, 4)), 16: np.ones((3, 4))}
var = {"last_L16": np.full((3, 4), 2.0), "max_L16": np.full((3, 4), 3.0), "last4_L16": np.full((3, 4), 4.0), "prompted_L16": np.full((3, 4), 5.0)}
assert pv.variant_matrix("V0 mean", means, var, 16, [14, 16]).mean() == 1.0 and pv.variant_matrix("V5 concat", means, var, 16, [14, 16]).shape == (3, 8)
assert [pv.variant_matrix(v, means, var, 16, [14, 16]).mean() for v in ("V1 last", "V2 max", "V3 last4", "V4 prompted")] == [2.0, 3.0, 4.0, 5.0]
print("VARIANTS TESTS PASSED")
