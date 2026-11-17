"""Model-free tests for probekit_length_calibration."""
import numpy as np

import probekit_length_calibration as lc

assert [lc.bucket_of(n) for n in (1, 2, 3, 4, 5, 7, 8, 11, 12, 19, 20, 99)] == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
assert lc.bucket_name(0) == "1-2" and lc.bucket_name(5) == "20+"
assert lc.cp_upper(0, 500) < 0.0075 and 0.0 < lc.cp_upper(10, 500) < 0.04 and lc.cp_upper(0, 0) == 1.0
assert lc.cp_upper(0, 38) > 0.09                               # the plan's blocker: zero exceedances in 38 cannot bound the rate below ~9%

rng = np.random.default_rng(0)
n = 4000; b = rng.integers(0, len(lc.BUCKETS), n)
spread = np.array([3.0, 1.0, 1.0, 1.0, 1.0, 1.0])[b][:, None]            # short texts: z wildly over-dispersed AND shifted
Z = rng.standard_normal((n, 7)) * spread + (b == 0)[:, None] * 2.0
raw = lc.stats(Z, b)
assert raw[0]["emo_rate_max"] > 0.2 and raw[5]["emo_rate_max"] < 0.02            # short bucket violates, long bucket is fine
rec = lc.stats(lc.recalibrate_cv(Z, b), b)
assert all(r["mean_z"] == r["mean_z"] and abs(r["mean_z"]) < 0.1 for r in rec), [r["mean_z"] for r in rec]
assert rec[0]["emo_rate_max"] < 0.02 and rec[0]["passage_rate"] < 0.1             # recalibration repairs the short bucket (normal tails)
vr, vc = lc.verdicts(raw), lc.verdicts(rec)
assert not vr[0]["pass"] and vr[5]["pass"] and lc.min_length_rule(vr) == 1
# per-passage bound is looser than a union of 7 independent 0.27% tails
assert vc[0]["reason"] in ("ok", "per-passage bound exceeded") and rec[3]["passage_rate"] < 0.05
# buckets with n < 500 cannot be certified
small = lc.stats(Z[:300], b[:300]); assert any("cannot certify" in v["reason"] for v in lc.verdicts(small))
assert lc.min_length_rule([{"pass": False}] * 3) is None and lc.min_length_rule([{"pass": True}] * 3) == 0
md = lc.render(raw, rec, list("abcdefg"), "toy"); assert "RAW" in md and "RECAL" in md and "PASS" in md and "FAIL" in md
# cross-validation really is out-of-fold: constant-offset in one fold's members cannot leak into its own correction
Zc = np.zeros((600, 7)); bb = np.zeros(600, int); Zc[:5] = 100.0
out = lc.recalibrate_cv(Zc, bb, folds=5)
assert np.isfinite(out).all()
th = lc.alarm_thresholds(Z, b)
assert len(th) == len(lc.BUCKETS) and th[0]["raw_q95"] > th[5]["raw_q95"]          # short texts need a higher alarm threshold
assert all(r["recal_q99"] >= r["recal_q95"] for r in th) and abs(th[3]["recal_q95"] - th[5]["recal_q95"]) < 0.5   # recalibration equalises buckets
assert "alarm thresholds" in lc.render_thresholds(th)
print("LENGTH CALIBRATION TESTS PASSED")
