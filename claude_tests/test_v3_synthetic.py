#!/usr/bin/env python3
"""Model-free tests for step-1 scripts on SYNTHETIC data: estimators, calibration, sweep/select/report, build,
stream-check helpers, robustness, and the API corpus generator (against a local fake server)."""
import json, sys, tempfile, threading, random
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import numpy as np

from probekit_probes import DEFAULT_EMOTIONS as E, NEUTRAL
import probekit_estimators as est
import probekit_eval_v3 as ev3
import probekit_build_v3 as b3
import probekit_stream_check as sc
import probekit_robustness as rb
from probekit_eval_v2 import f1_9, probe_predict, pick_tau

nrng = np.random.default_rng(1); rng = random.Random(1)
H = 96; cats = E + [NEUTRAL]
dirs = {e: (lambda v: v / np.linalg.norm(v))(nrng.normal(size=H)) for e in E}
w = {e: (lambda v: v / np.linalg.norm(v))(nrng.normal(size=H)) for e in E}
n_dir = {e: (dirs[e] + w[e]) / np.sqrt(2) for e in E}          # noise axis partly aligned with each signal direction
offset = nrng.normal(size=H) * 6

def state(c, strength=3.5, noise_scale=3.5):
    x = offset + 0.5 * nrng.normal(size=H)
    for e in E:
        x = x + noise_scale * nrng.normal() * n_dir[e]          # cross-talk noise that hurts naive mean-difference
    return x + (strength * dirs[c] if c != NEUTRAL else 0)

# ---- 1. estimators: LDA should beat meandiff when noise is correlated with signal and nuisance removal is OFF
topics = {"train": [f"t{i}" for i in range(14)], "val": ["v1", "v2", "v3", "v4"], "test": ["x1", "x2", "x3", "x4"]}
cue = {c: [f"{c}cue{i}" for i in range(4)] for c in cats}; filler = [f"w{i}" for i in range(50)]
styles = ["dialogue", "report", "diary"]; writers = ["w3b", "w27b"]
rows = []
for sp, ts in topics.items():
    for t in ts:
        for c in cats:
            for j in range(4):
                words = rng.choices(filler, k=16) + [rng.choice(cue[c]) for _ in range(2)]; rng.shuffle(words)
                rows.append({"id": f"{c}|{t}|{j}", "category": c, "topic": t, "split": sp, "text": " ".join(words),
                             "hard": True, "style": styles[j % 3], "writer": writers[(j // 2) % 2]})
Xc = {5: np.stack([state(r["category"]) for r in rows])}
cat = np.array([r["category"] for r in rows]); spl = np.array([r["split"] for r in rows])
tr, te = spl == "train", spl == "test"
cal = (spl == "val") & (cat == NEUTRAL); va = spl == "val"
res = {}
for m in ("meandiff", "lda", "logreg"):
    ps = est.fit_directions(Xc[5][tr], cat[tr], E, 5, method=m, var_frac=0.0)
    est.calibrate_pooled(ps, Xc[5][cal]); tau = pick_tau(ps, Xc[5][va], cat[va])
    res[m] = f1_9(cat[te], probe_predict(ps, Xc[5][te], tau))
print("no-nuisance-removal test F1:", {k: round(v, 3) for k, v in res.items()})
assert res["lda"] > res["meandiff"] + 0.05, "LDA should handle correlated noise better than mean-difference"
# calibration: held-out neutral z ~ N(0,1)
z = ps.score(Xc[5][(spl == "test") & (cat == NEUTRAL)], "pooled")
print(f"held-out neutral z (logreg): mean {z.mean():.2f}, sd {z.std():.2f}")

# ---- 2. sweep + select + outputs
hrows = [{"id": f"h{i}", "category": c, "subtype": s, "text": f"alpha {c} cue{i}"} for i, (c, s) in
         enumerate([(c, s) for c in cats for s in ("implicit", "negation")] * 2)]
Xh = {5: np.stack([state(r["category"]) for r in hrows])}
grids = {"meandiff": [dict(var_frac=0.0), dict(var_frac=0.5)], "lda": [dict(var_frac=0.5, shrinkage="auto")],
         "logreg": [dict(var_frac=0.5, C=0.03)]}
S = ev3.sweep([5], Xc, rows, Xh, hrows, E, grids)
m, L, band = ev3.select(S, [5])
print(f"selected {m} @ {L}, plateau {band}")
with tempfile.TemporaryDirectory() as d:
    best = ev3.write_outputs(Path(d), S, [5], "synthetic", E, Xh, hrows)
    for f in ("report_v3.md", "errors_v3.md", "best_config.json", "report_v3.json"):
        assert (Path(d) / f).exists(), f
    json.loads((Path(d) / "report_v3.json").read_text())
    assert "Paired bootstrap" in (Path(d) / "report_v3.md").read_text()
mean, lo, hi = ev3.paired_bootstrap(["a", "b"] * 10, ["a", "b"] * 10, ["a", "b"] * 10, B=200)
assert mean == 0 and lo == 0 and hi == 0

# ---- 3. build
ps, tau = b3.build(Xc[5], rows, E, 5, dict(best["cfg"], method=best["method"]))
assert ps.layer == 5 and np.isfinite(tau)

# ---- 4. stream-check helpers
items = []
for i, c in enumerate(cats * 2):
    n = 14; zt = nrng.normal(size=(n, len(E)))
    if c != NEUTRAL: zt[4:, E.index(c)] += 4.0
    items.append({"label": c, "subtype": "implicit", "text": "x", "tokens": [f"t{k}" for k in range(n + 4)],
                  "z_all": np.vstack([np.zeros((4, len(E))), zt]), "z_tok": zt, "pooled_z": zt.mean(0), "pooled_pred": c})
met = sc.analyze_passages(items, E)
print("stream metrics:", {k: (round(v["emotion_accuracy_8way"], 2)) for k, v in met.items() if isinstance(v, dict)})
assert met["mean_z"]["emotion_accuracy_8way"] > 0.9 and met["mean_z"]["emotional_vs_neutral_auc"] > 0.9
assert 0.2 < met["onset_fraction_median"] < 0.6
h = sc.build_html(items, E, 4.0); assert "<span" in h and "Token-level" in h
assert sc.onset(np.array([0, 0, 3.0, 3.0])) == 2 / 3

# ---- 5. robustness
style = rb.leave_one_group_out(Xc[5], Xh[5], rows, np.array([r["category"] for r in hrows]), E, 5, dict(best["cfg"], method=best["method"]), "style")
writer = rb.leave_one_group_out(Xc[5], Xh[5], rows, np.array([r["category"] for r in hrows]), E, 5, dict(best["cfg"], method=best["method"]), "writer")
curve = rb.learning_curve(Xc[5], Xh[5], rows, np.array([r["category"] for r in hrows]), E, 5, dict(best["cfg"], method=best["method"]), fracs=(0.3, 1.0), reps=2)
print(f"robustness: {len(style)} style folds, {len(writer)} writer folds, curve {[round(v['test_f1_mean'],2) for v in curve.values()]}")
assert len(style) == 3 and len(writer) == 2
assert "Learning curve" in rb.render(best, style, writer, curve)

# ---- 5b. regression: a held-out group owns the ONLY training examples of one emotion
import copy
rows2 = copy.deepcopy(rows)
for i, r in enumerate(rows2):
    if (r["split"] == "train" and r["category"] == "calm") or (r["split"] != "train" and i % 2 == 0):
        r["writer"] = "w_calm"
cfg_ = dict(best["cfg"], method=best["method"]); hc_ = np.array([r["category"] for r in hrows])
sk = {}
got = rb.leave_one_group_out(Xc[5], Xh[5], rows2, hc_, E, 5, cfg_, "writer", skipped=sk)
assert "w_calm" in sk and "calm" in sk["w_calm"] and "w_calm" not in got, (sk, list(got))
assert len(got) >= 1, "valid folds must still run"
sk2 = {}
got2 = rb.leave_one_group_out(Xc[5], Xh[5], rows2, hc_, E, 5, cfg_, "writer", skipped=sk2, allow_partial=True)
assert "w_calm" in got2 and "calm" in got2["w_calm"]["note"] and "w_calm" not in sk2
md2 = rb.render(best, {}, got2, curve, {}, sk)
assert "Folds NOT run" in md2 and "partial" in md2
for m_ in ("meandiff", "lda", "logreg"):
    try:
        est.fit_directions(Xc[5][tr & (cat != "calm")], cat[tr & (cat != "calm")], E, 5, method=m_)
        raise AssertionError("expected ValueError for " + m_)
    except ValueError as e_:
        assert "calm" in str(e_), e_
print("missing-class regression ok (skipped:", sk["w_calm"][:60], ")")

# ---- 6. API corpus generator against a fake OpenAI-compatible server
class H_(BaseHTTPRequestHandler):
    n = 0
    last = {}
    def do_POST(self):
        H_.last = json.loads(self.rfile.read(int(self.headers["Content-Length"]))); H_.n += 1
        txt = (f"Priya sorted the invoices by date and filed them in the grey cabinet while the printer warmed up. "
               f"She checked the totals twice, labeled folder number {H_.n}, and carried the stack to the front desk.")
        body = json.dumps({"choices": [{"message": {"content": txt}}]}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(body)
    def log_message(self, *a): pass
srv = HTTPServer(("127.0.0.1", 0), H_); threading.Thread(target=srv.serve_forever, daemon=True).start()
import probekit_corpus_api as api
with tempfile.TemporaryDirectory() as d:
    out = Path(d) / "c.jsonl"
    sys.argv = ["x", "--api-base", f"http://127.0.0.1:{srv.server_port}/v1", "--api-model", "fake", "--topics", "ext",
                "--samples", "1", "--max-new", "5", "--out", str(out)]
    api.main()
    got = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(got) == 5 and all(g["writer"] == "fake" and g["split"] in ("train", "val", "test") for g in got)
    api.main()   # resume: should add 5 more, not duplicate
    got2 = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(got2) == 10 and len({g["id"] for g in got2}) == 10
    assert H_.last["max_tokens"] == 300 and "reasoning_effort" not in H_.last   # defaults unchanged
    out2 = Path(d) / "c2.jsonl"
    sys.argv = ["x", "--api-base", f"http://127.0.0.1:{srv.server_port}/v1", "--api-model", "fake", "--topics", "ext",
                "--samples", "1", "--max-new", "1", "--max-tokens", "1234", "--no-think", "--out", str(out2)]
    api.main()
    assert H_.last["max_tokens"] == 1234 and H_.last["reasoning_effort"] == "none", H_.last
srv.shutdown()
assert len(api.SPLITS_EXT) == 24 and not (set(api.SPLITS_EXT) & set(api.SPLITS))
print("ALL V3 TESTS PASSED")
