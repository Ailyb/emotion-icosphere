#!/usr/bin/env python3
"""
Sanity tests for probekit_probes using SYNTHETIC activations (no model required).

These verify the plumbing -- centering, nuisance removal, calibration, null control,
save/load -- NOT that real LLM emotion directions behave this way. Run: python test_probes_synthetic.py
"""
import tempfile
from pathlib import Path

import numpy as np

from probekit_probes import (DEFAULT_EMOTIONS, NEUTRAL, ProbeSet, evaluate, fit_probes,
                             null_distribution, split_half_reliability)

rng = np.random.default_rng(0)
H, E = 128, DEFAULT_EMOTIONS
cats_all = E + [NEUTRAL]

# Ingredients that make real residual streams awkward:
offset = rng.normal(size=H); offset *= 30 / np.linalg.norm(offset)      # big shared mean
dirs = {e: (lambda v: v / np.linalg.norm(v))(rng.normal(size=H)) for e in E}
# Nuisance (topic/style/length) lives in a SHARED low-dim subspace, as in real models; each topic is a
# random mix of those axes plus a little idiosyncratic noise. Unseen test topics reuse the same axes.
axes = rng.normal(size=(6, H)); axes /= np.linalg.norm(axes, axis=1, keepdims=True)
topics = {t: 8.0 * (rng.normal(size=6) @ axes) + 0.3 * rng.normal(size=H) for t in range(32)}
outlier = np.zeros(H); outlier[0] = 1.0                                  # a huge-variance 'massive activation' dim


def make(topic_ids, per_cell=2, strength=3.5, noise=0.7):
    X, y = [], []
    for t in topic_ids:
        for c in cats_all:
            for _ in range(per_cell):
                x = offset + topics[t] + noise * rng.normal(size=H) + 15 * rng.normal() * outlier
                if c != NEUTRAL:
                    x = x + strength * dirs[c]
                X.append(x); y.append(c)
    return np.array(X), np.array(y)


Xtr, ytr = make(range(0, 20))
Xte, yte = make(range(26, 32))

ps0 = fit_probes(Xtr, ytr, E, layer=18, var_frac=0.0)   # no nuisance removal (ablation)
res0 = evaluate(ps0, Xte, yte)
ps = fit_probes(Xtr, ytr, E, layer=18)
res = evaluate(ps, Xte, yte)
print(f"ablation: no nuisance removal acc {res0['accuracy']:.3f} -> with removal {res['accuracy']:.3f}")
assert res["accuracy"] > res0["accuracy"] + 0.15, "nuisance removal should help when nuisance is low-dim"

print(f"held-out accuracy {res['accuracy']:.3f} (chance {res['chance_accuracy']:.3f}), macro-F1 {res['macro_f1']:.3f}, "
      f"nuisance PCs removed: {ps.meta['nuisance_k']}")
assert res["accuracy"] > 0.9, "probes should recover planted directions on held-out topics"
assert res["auc_emotional_vs_neutral"] > 0.9

null = null_distribution(Xtr, ytr, Xte, yte, E, 18, runs=15)
print(f"null macro-F1 mean {null['mean']:.3f}, p95 {null['p95']:.3f}")
assert null["p95"] < 0.35, "shuffled labels must score near chance"
assert res["macro_f1"] > null["p95"] + 0.3

sh = split_half_reliability(Xtr, ytr, E, 18, reps=5)
print("split-half cosine:", {k: round(v, 2) for k, v in sh.items()})
assert min(sh.values()) > 0.5

# neutral z-scores should be ~N(0,1) on held-out neutral text
z = ps.score(Xte[yte == NEUTRAL], "pooled")
print(f"neutral held-out z: mean {z.mean():.2f}, std {z.std():.2f}")
assert abs(z.mean()) < 0.5 and 0.5 < z.std() < 2.0

# token calibration + save/load round trip
tok_neutral = Xtr[ytr == NEUTRAL] + rng.normal(size=(sum(ytr == NEUTRAL), H)) * 2.0
tok_emo = Xtr[ytr != NEUTRAL] + rng.normal(size=(sum(ytr != NEUTRAL), H)) * 2.0
ps.calibrate_tokens(tok_neutral, tok_emo)
assert (ps.sigma_tok > ps.sigma_pool).all() and ps.meta["z_full"] > 0
with tempfile.TemporaryDirectory() as d:
    p = Path(d) / "probes.npz"; ps.save(p); ps2 = ProbeSet.load(p)
    assert np.allclose(ps.score(Xte, "token"), ps2.score(Xte, "token"), atol=1e-5)
    assert ps2.emotions == ps.emotions and ps2.layer == 18
print("ALL SYNTHETIC TESTS PASSED")
