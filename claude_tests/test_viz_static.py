#!/usr/bin/env python3
"""Model-free test of probekit_viz_static helpers (synthetic)."""
import numpy as np
import probekit_viz_static as V

rng = np.random.default_rng(0)
Z = rng.normal(2.0, 3.0, size=(600, 7)); w = np.array([1] * 300 + [30] * 300)
par = V.recal_params(Z, w)
for bi in par:
    m = V.bucket_of(1) == bi
    sel = (w == 1) if m else (w == 30)
    z = (Z[sel] - par[bi][0]) / par[bi][1]
    assert abs(z.mean()) < 1e-6 and abs(z.std() - 1) < 1e-6
rows = [dict(intended="neutral", bucket="5-7", probe="neutral"), dict(intended="neutral", bucket="5-7", probe="joy"), dict(intended="joy", bucket="5-7", probe="joy")]
assert V.neutral_rate(rows) == {"5-7": [1, 2]}
html = V.page([dict(id="a", text="x <b>", intended="joy", n_words=2, bucket="1-2", z_recal={"joy": 1.0}, threshold=3.0, low_confidence=True, probe="neutral", tfidf="joy", native_lr="joy", encoder="joy")],
              dict(model="m", layer=1, method="lda", n_train=1), {})
assert "&lt;b&gt;" in html and "What this is not" in html and "low confidence" in html
rows = [dict(intended="neutral", bucket="5-7", probe="neutral", tfidf="joy", native_lr="neutral", encoder="neutral"),
        dict(intended="joy", bucket="5-7", probe="neutral", tfidf="joy", native_lr="fear", encoder="joy")]
sm = V.summarize(rows)["5-7"]
assert sm["tfidf"]["fa"]["k"] == 1 and sm["probe"]["fa"]["k"] == 0 and sm["probe"]["miss"]["k"] == 1
assert sm["tfidf"]["hit"]["k"] == 1 and sm["native_lr"]["hit"]["k"] == 0 and sm["native_lr"]["miss"]["k"] == 0
print("PASS test_viz_static")
