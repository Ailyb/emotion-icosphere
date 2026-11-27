#!/usr/bin/env python3
import numpy as np
import probekit_stream_display as D

s = "It rained all day.  Then the sun came out! Was it over? yes"
sp = D.split_sentences(s); assert [x[2] for x in sp] == ["It rained all day.", "Then the sun came out!", "Was it over?", "yes"], sp
assert all(s[a:b] == t for a, b, t in sp)
rng = np.random.default_rng(0); V = rng.normal(size=(7, 40)); V /= np.linalg.norm(V, axis=1, keepdims=True)
A = D.anchors_from_directions(V); assert A.shape == (7, 3) and np.allclose(np.linalg.norm(A, axis=1), 1)
assert np.allclose(A, D.anchors_from_directions(V))
V2 = V.copy(); V2[1] = V[0] + 0.01 * rng.normal(size=40)                                       # near-duplicate directions land near each other
A2 = D.anchors_from_directions(V2); assert A2[0] @ A2[1] > 0.9

class PS:                                                                                      # stub probe set: z = first 7 coordinates
    def score(self, X, mode): return np.asarray(X)[:, :7]
eng = D.StreamEngine(PS(), (np.zeros(7), np.ones(7)), 3.0, skip=2, window=4)
st = np.zeros((12, 8)); st[6:, 3] = 10.0
fr = eng.positions(st)
assert fr[0]["pos"] == 2 and fr[0]["warmup"] and not fr[3]["warmup"]
assert [f["flag"] for f in fr[:3]] == [False] * 3
late = fr[-1]; assert late["flag"] and late["top"] == "anger" and not late["warmup"]
I = D.face_intensity(late["z"], True, "anger"); assert I["anger"] == 1.0 and sum(v > 0 for v in I.values()) == 1
assert max(D.face_intensity(late["z"], False, None).values()) == 0.0
print("PASS test_stream_display")
