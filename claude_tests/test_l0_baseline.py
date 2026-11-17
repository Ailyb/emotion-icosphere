"""Model-free test of probekit_l0_baseline.run/render on synthetic features."""
import numpy as np

import probekit_l0_baseline as l0

rng = np.random.RandomState(0)
CATS = ["joy", "anger", "fear", "calm", "neutral"]
rows, X_sig, X_noise, texts = [], [], [], {}
for sp, per in (("train", 40), ("val", 20), ("test", 20)):
    for c_i, c in enumerate(CATS):
        for j in range(per):
            rows.append({"id": f"{sp}{c}{j}", "category": c, "split": sp, "hard": True,
                         "text": f"word{c_i} common{rng.randint(50)} filler text number {j}"})
            v = rng.randn(16); v_sig = v.copy(); v_sig[c_i] += 4.0       # class signal on one coordinate
            X_sig.append(v_sig); X_noise.append(rng.randn(16))
X_sig, X_noise = np.array(X_sig), np.array(X_noise)
hrows = [dict(r, id="h" + r["id"]) for r in rows if r["split"] == "test"]
Xh_sig = X_sig[[i for i, r in enumerate(rows) if r["split"] == "test"]]
Xh_noise = X_noise[[i for i, r in enumerate(rows) if r["split"] == "test"]]

cat = np.array([r["category"] for r in rows]); spl = np.array([r["split"] for r in rows])
def probe_pred(k, sets):                       # an oracle-ish probe: perfect on every set
    return np.array([r["category"] for r in hrows]) if k == "hand-written" else cat[sets[k]]
preds, truth = l0.run({"embed": X_noise, "block0": X_sig, "native_lr_L": X_sig},
                      {"embed": Xh_noise, "block0": Xh_sig, "native_lr_L": Xh_sig}, rows, hrows, probe_pred)
assert set(preds) == {"VAL-hard", "TEST-hard", "TEST-all", "hand-written"}
for k in preds:
    assert set(preds[k]) == {"embed", "block0", "native_lr_L", "tfidf", "probe"}, preds[k].keys()
    assert all(len(p) == len(truth[k]) for p in preds[k].values())
f = lambda k, s: l0.f1_9(truth[k], preds[k][s])
assert f("TEST-hard", "probe") == 1.0
assert f("TEST-hard", "embed") < 0.45 and f("TEST-hard", "block0") > 0.8, (f("TEST-hard", "embed"), f("TEST-hard", "block0"))
md = l0.render(preds, truth, 14, "lda")
assert "probe - embed (the T1 test)" in md and "*" in md      # perfect probe vs noise embedding: interval excludes 0
assert md.count("\n| ") >= 12
print("L0 BASELINE TESTS PASSED")
