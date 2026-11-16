import numpy as np, random, json
from probekit_probes import DEFAULT_EMOTIONS as E, NEUTRAL
import probekit_confusion as pc
nrng = np.random.default_rng(3); rng = random.Random(3); H = 64; cats = E + [NEUTRAL]
dirs = {e: (lambda v: v/np.linalg.norm(v))(nrng.normal(size=H)) for e in E}
def st(c): return 5*nrng.normal(size=H)*0 + nrng.normal(size=H)*0.7 + (3.5*dirs[c] if c != NEUTRAL else 0)
rows = [{"id": f"{c}{t}{j}", "category": c, "topic": t, "split": sp, "text": "x", "hard": True}
        for sp, ts in {"train": range(12), "val": range(12, 16)}.items() for t in ts for c in cats for j in range(3)
        for t in [f"{sp}{t}"]]
Xl = np.stack([st(r["category"]) for r in rows])
hrows = [{"id": f"h{i}", "category": c, "subtype": s, "text": " ".join(["w"] * n)} for i, (c, s, n) in
         enumerate([(c, s, n) for c in cats for s in ("own_voice", "implicit") for n in (5, 30)])]
Xhl = np.stack([st(r["category"]) for r in hrows])
cfgs = {m: {"cfg": {"method": m, "var_frac": 0.5, **({"C": 0.03} if m == "logreg" else {}), **({"shrinkage": "auto"} if m == "lda" else {})}, "tau": 1.0} for m in ("meandiff", "lda", "logreg")}
report = {"per_layer": {"5": {**cfgs, "native_lr": {"cfg": {"C": 0.03}}}}}
preds = pc.predictions(Xl, Xhl, rows, E, 5, report)
assert set(preds) == {"meandiff", "lda", "logreg", "native_lr"} and all(len(p) == len(hrows) for p in preds.values())
md = pc.report(preds, hrows, 5)
for s in ("Accuracy by estimator", "by length", "Own-voice confusion: lda", "coarse"): assert s in md, s
assert list(pc.coarse(["joy", "fear", "surprise", "neutral"])) == ["positive", "negative", "surprise", "neutral"]
print("pc ok; sample accuracy row:", [l for l in md.splitlines() if l.startswith("| meandiff")][0][:110])
print("CONFUSION TEST PASSED")
