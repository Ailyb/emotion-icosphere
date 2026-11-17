"""Model-free tests for probekit_t8 building blocks (synthetic)."""
import numpy as np

import probekit_t8 as t8

D = 24
CL = ["joy", "love", "sadness", "fear", "anger", "disgust", "surprise", "neutral"]
def make(n, seed):
    r = np.random.RandomState(seed); X, y = [], []
    for i, c in enumerate(CL):
        for _ in range(n):
            v = r.randn(D); v[i] += 3.0; X.append(v); y.append(c)
    return np.array(X), np.array(y, dtype=object)
Xtr, ytr = make(40, 0); Xva, yva = make(15, 1); Xte, yte = make(20, 2)
h = t8.fit_hierarchical(Xtr, ytr, Xva, yva)
p = h.predict(Xte, CL); assert set(p) <= set(CL) and t8.f1_9(yte, p) > 0.8
sub = ["joy", "sadness", "anger"]; ps_ = h.predict(Xte, sub); assert set(ps_) <= set(sub)           # restriction to a class subset
P = h.proba(Xte, CL); assert P.shape == (len(Xte), 8) and np.allclose(P.sum(1), 1.0, atol=1e-6)    # soft composition is a distribution
sc, mlp, alpha = t8.fit_mlp(Xtr, ytr, Xva, yva); assert alpha in (1e-3, 1e-2, 1e-1) and (mlp.predict(sc.transform(Xte)) == yte).mean() > 0.7

# similarity test: identical geometry -> rho = 1 and a small exact p; shuffled geometry -> large p
rng = np.random.RandomState(0); W = rng.randn(7, 3)
E = np.linalg.norm(W[:, None] - W[None], axis=2)
rho, p = t8.rsa_permutation(E.copy(), E); assert abs(rho - 1) < 1e-9 and p <= 1 / 5040 + 1e-12
perm = rng.permutation(7); rho2, p2 = t8.rsa_permutation(E[np.ix_(perm, perm)], E)
assert rho2 < 0.95 and p2 > 0.001
V = rng.randn(7, 50); emos = ["joy", "sadness", "fear", "anger", "surprise", "disgust", "love"]
vad = {e: W[i] for i, e in enumerate(emos)}
Dm, Em = t8.dist_matrices(V, vad, emos); assert Dm.shape == Em.shape == (7, 7) and np.allclose(np.diag(Dm), 0, atol=1e-9)
import tempfile, pathlib
with tempfile.TemporaryDirectory() as d:
    f = pathlib.Path(d) / "vad.txt"; f.write_text("term\tvalence\tarousal\tdominance\njoy\t0.9\t0.6\t0.5\nlove apple\t0.7\t-0.2\t0.1\nanger\t-0.7\t0.7\t0.3\n")
    got = t8.load_vad(f, {"joy", "anger", "love"}); assert set(got) == {"joy", "anger"} and abs(got["joy"][0] - 0.9) < 1e-9   # multiword terms ignored
print("T8 TESTS PASSED")
