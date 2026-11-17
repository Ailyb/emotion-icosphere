"""Model-free test of the frozen-eval building blocks on synthetic features."""
import numpy as np

import probekit_frozen_eval as fe

rng = np.random.RandomState(0)
CL = fe.CLASSES                                  # 7 emotions + neutral
D = 24
def make(n_per, shift=0.0, seed=0):
    r = np.random.RandomState(seed); X, y = [], []
    for i, c in enumerate(CL):
        for _ in range(n_per):
            v = r.randn(D); v[i] += 3.0 - shift; X.append(v); y.append(c)
    return np.array(X), np.array(y, dtype=object)
Xl, yl = make(40, 0, 1); Xp, yp = make(60, 0.5, 2); Xva, yva = make(15, 0, 3); Xcal = Xva[yva == "neutral"]; Xfr, yfr = make(30, 0.5, 4)

# sample_human: k per class, deterministic per seed, capped at availability
a = fe.sample_human(yp, None, 20, np.random.default_rng(0)); b = fe.sample_human(yp, None, 20, np.random.default_rng(0))
assert (a == b).all() and all((yp[a] == c).sum() == 20 for c in CL)
assert len(fe.sample_human(yp, None, 500, np.random.default_rng(0))) == len(yp)

# learning curve runs end to end and returns the expected rows; more human data should not hurt on this easy synthetic task
res = fe.learning_curve({5: Xl}, yl, {5: Xp}, yp, Xva, yva, Xcal, {5: Xfr}, yfr, 5, "lda",
                        {"var_frac": 0.5, "shrinkage": "auto"}, [0, 20, 60], [0, 1])
ks = sorted({(r["llm"], r["k"]) for r in res}); assert (True, 0) in ks and (True, 20) in ks and (True, 60) in ks, ks
assert sum(r["k"] == 60 for r in res) == 1                       # k = all available: single fit, no seed repeats
assert all(0.0 <= r["probe"] <= 1.0 and 0.0 <= r["native_lr"] <= 1.0 for r in res)
assert max(r["probe"] for r in res if r["k"] == 60) > 0.4 and max(r["native_lr"] for r in res if r["k"] == 60) > 0.7   # well above chance (0.125)
hum = fe.learning_curve({5: Xl}, yl, {5: Xp}, yp, Xva, yva, Xcal, {5: Xfr}, yfr, 5, "lda", {"var_frac": 0.5, "shrinkage": "auto"}, [20, 60], [0], include_llm=False)
assert hum and not any(r["llm"] for r in hum)
T_llm = np.array([f"{c} filler w{i % 7}" for i, c in enumerate(yl)], dtype=object); T_pool = np.array([f"{c} filler w{i % 5}" for i, c in enumerate(yp)], dtype=object)
T_fr = np.array([f"{c} filler w{i % 3}" for i, c in enumerate(yfr)], dtype=object)
rt = fe.learning_curve({5: Xl}, yl, {5: Xp}, yp, Xva, yva, Xcal, {5: Xfr}, yfr, 5, "lda", {"var_frac": 0.5, "shrinkage": "auto"}, [20], [0], texts=(T_llm, T_pool, T_fr))
assert "tfidf" in rt[0] and "fusion" in rt[0] and set(rt[0]["_p"]) == {"probe", "native_lr", "tfidf", "fusion"} and rt[0]["tfidf"] > 0.9   # class word in text: lexical is easy
assert "TF-IDF F1 (post hoc)" in fe.summarize_curve(rt)
pf = fe.fit_fusion(Xl, yl, Xva, yva, T_llm)(Xfr, T_fr); assert len(pf) == len(yfr) and (pf == yfr).mean() > 0.8
md = fe.summarize_curve(res + hum); assert "LLM + human" in md and "human only" in md

# metrics helpers
p = yfr.copy(); assert fe.macro_f1(yfr, p) == 1.0
lo, hi = fe.boot_ci(yfr, p, B=50); assert lo == hi == 1.0
assert fe.shuffle_null_p95(yfr, p, B=50) < 0.3                    # a permuted-label null is near chance (1/8)
pc = fe.per_class(yfr, p); assert all(pc[c]["recall"] == 1.0 and pc[c]["n"] == 30 for c in CL)
assert set(fe.coarse(np.array(["joy", "surprise", "neutral", "fear"]))) == {"positive", "ambiguous", "neutral", "negative"}
m = np.zeros(len(yfr), bool); m[:10] = True
assert fe.stratum_scores(yfr, {"a": p}, m) is None                # n < 20 not reported
m[:40] = True; assert fe.stratum_scores(yfr, {"a": p}, m)["a"] > 0.9
print("FROZEN EVAL TESTS PASSED")
