"""Model-free tests for probekit_external_eval restricted prediction and reporting (synthetic)."""
import numpy as np

import probekit_external_eval as ee
from probekit_frozen_eval import CLASSES, EMOTIONS

D = 20
def make(n, seed, noise=1.0):
    r = np.random.RandomState(seed); X, y, T = [], [], []
    for i, c in enumerate(CLASSES):
        for j in range(n):
            v = r.randn(D) * noise; v[i] += 3.0; X.append(v); y.append(c); T.append(f"{c} word{j % 5} text")
    return np.array(X), np.array(y, dtype=object), np.array(T, dtype=object)
Xtr, ytr, Ttr = make(40, 0); Xva, yva, _ = make(15, 1); Xcal = Xva[yva == "neutral"]
systems, tau = ee.fit_systems(Xtr, Xtr, Ttr, ytr, Xva, Xva, yva, Xcal, 5, {"var_frac": 0.5, "shrinkage": "auto"})
Xte, yte, Tte = make(20, 2)
# restricted prediction: classes without neutral -> never neutral; every prediction stays inside the allowed set
for classes in (["joy", "sadness", "love", "anger", "fear", "surprise"], list(CLASSES), ["joy", "anger", "disgust", "neutral"]):
    for n, f in systems.items():
        p = f(Xte, Xte, Tte, classes)
        assert len(p) == len(Xte) and set(p) <= set(classes), (n, set(p) - set(classes))
assert "neutral" not in set(systems["probe"](Xte, Xte, Tte, ["joy", "sadness", "love", "anger", "fear", "surprise"]))
# on in-class data the lexical model (class word in text) and the probe are well above chance
mask = np.isin(yte, ["joy", "sadness", "love", "anger", "fear", "surprise"])
p = systems["tfidf"](Xte[mask], Xte[mask], Tte[mask], ["joy", "sadness", "love", "anger", "fear", "surprise"])
assert ee.f1_9(yte[mask], p) > 0.9
pp = systems["probe"](Xte[mask], Xte[mask], Tte[mask], ["joy", "sadness", "love", "anger", "fear", "surprise"])
assert ee.f1_9(yte[mask], pp) > 0.5
# columns(): zeros for unseen class
import sklearn.linear_model as lm
clf = lm.LogisticRegression().fit(np.array([[0.], [1.], [2.]]), ["a", "b", "c"])
P = ee.columns(clf, clf.predict_proba(np.array([[0.], [2.]])), ["a", "z", "c"])
assert P.shape == (2, 3) and (P[:, 1] == 0).all() and P[0, 0] > P[0, 2]
# report + verdict
class M(dict): pass
man = {"license_note": "test", "label_provenance": "synthetic"}
md, res = ee.render_source("toy", man, mask_y := yte[mask], {"HUM": {s: f(Xte[mask], Xte[mask], Tte[mask], sorted(set(mask_y))) for s, f in systems.items()},
                                                               "LLM": {s: f(Xte[mask], Xte[mask], Tte[mask], sorted(set(mask_y))) for s, f in systems.items()}})
assert "H1" in md and ("HUM", "probe", "tfidf") in res and "Per-class recall" in md
v = ee.verdict({"toy": res}); assert len(v) == 2 and "trained on HUM" in v[0]
print("EXTERNAL EVAL TESTS PASSED")
