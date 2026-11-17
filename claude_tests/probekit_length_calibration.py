#!/usr/bin/env python3
"""
probekit_length_calibration.py -- PLAN_v2 T3 (pre-registered): are the probe's z-scores valid at every text length?

    python probekit_length_calibration.py reserve   # build reserve B + the calibration text file (model-free)
    python probekit_length_calibration.py eval --cache-dir data/eval_merged2/<slug> --eval-dir data/eval_merged2/<slug>/v3 \
        --corpus data/corpus/corpus_merged2_scored.jsonl --out data/eval_merged2/<slug>/length_calibration.md

On held-out NEUTRAL human text the share of |z| > 3 should be small at every length. RAW uses the probe's existing calibration;
RECAL re-estimates mean and sd per word-length bucket with K-fold cross-validation (each passage scored by a calibration that never
saw it). Acceptance: Clopper-Pearson 95% upper bound <= 2% per emotion and <= 5% per passage in every bucket with n >= 500.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import beta

BUCKETS = [(1, 2), (3, 4), (5, 7), (8, 11), (12, 19), (20, 10 ** 6)]
EMO_LIMIT, PASSAGE_LIMIT, MIN_N, THRESH = 0.02, 0.05, 500, 3.0


def bucket_of(n_words):
    for i, (lo, hi) in enumerate(BUCKETS):
        if lo <= n_words <= hi:
            return i
    return 0 if n_words < 1 else len(BUCKETS) - 1


def bucket_name(i):
    lo, hi = BUCKETS[i]
    return f"{lo}+" if hi >= 10 ** 5 else f"{lo}-{hi}" if lo != hi else f"{lo}"


def cp_upper(k, n, conf=0.95):
    """Clopper-Pearson one-sided-style upper bound of the 2-sided `conf` interval."""
    return 1.0 if n == 0 else float(beta.ppf(1 - (1 - conf) / 2, k + 1, n - k)) if k < n else 1.0


def recalibrate_cv(Z, b, folds=5, seed=0):
    """Per-bucket (mean, sd) per emotion estimated on the other folds; returns Z' (same shape) scored out-of-fold."""
    rng = np.random.default_rng(seed); out = np.zeros_like(Z)
    fold = np.zeros(len(Z), int)
    for bi in np.unique(b):                                    # stratified assignment: each bucket spread over the folds
        idx = np.where(b == bi)[0]; rng.shuffle(idx); fold[idx] = np.arange(len(idx)) % folds
    for f in range(folds):
        te = fold == f
        for bi in np.unique(b[te]):
            tr = (~te) & (b == bi)
            mu = Z[tr].mean(0) if tr.sum() >= 20 else Z[~te].mean(0)
            sd = Z[tr].std(0) if tr.sum() >= 20 else Z[~te].std(0)
            m = te & (b == bi); out[m] = (Z[m] - mu) / np.maximum(sd, 1e-6)
    return out


def stats(Z, b, thresh=THRESH):
    """Per bucket: n, mean/sd of z (averaged over emotions), per-emotion exceedance (worst emotion) and per-passage exceedance with upper bounds."""
    res = []
    for bi in range(len(BUCKETS)):
        m = b == bi; n = int(m.sum())
        if n == 0:
            res.append({"bucket": bucket_name(bi), "n": 0}); continue
        ex = np.abs(Z[m]) > thresh
        per_em = ex.sum(0); worst = int(per_em.argmax())
        pp = int(ex.any(1).sum())
        res.append({"bucket": bucket_name(bi), "n": n, "mean_z": float(Z[m].mean()), "sd_z": float(Z[m].std(0).mean()),
                    "emo_rate_max": float(per_em.max() / n), "emo_upper_max": max(cp_upper(int(k), n) for k in per_em), "worst_emotion_idx": worst,
                    "passage_rate": pp / n, "passage_upper": cp_upper(pp, n)})
    return res


def verdicts(res):
    out = []
    for r in res:
        if r["n"] < MIN_N:
            out.append({"bucket": r["bucket"], "pass": False, "reason": f"n={r['n']} < {MIN_N}: cannot certify"}); continue
        ok_e, ok_p = r["emo_upper_max"] <= EMO_LIMIT, r["passage_upper"] <= PASSAGE_LIMIT
        out.append({"bucket": r["bucket"], "pass": ok_e and ok_p,
                    "reason": "ok" if ok_e and ok_p else f"{'per-emotion ' if not ok_e else ''}{'per-passage' if not ok_p else ''} bound exceeded".strip()})
    return out


def min_length_rule(verdict):
    """Smallest bucket such that it and every longer bucket pass; None if even the longest fails."""
    last_fail = max([i for i, v in enumerate(verdict) if not v["pass"]], default=-1)
    return None if last_fail == len(verdict) - 1 else last_fail + 1


def alarm_thresholds(Z, b, folds=5, seed=0, q=(0.95, 0.99)):
    """POST HOC / descriptive: per bucket, the quantile of max|z| over emotions on neutral text, estimated out of fold (the threshold a
    display would need so that only ~(1-q) of neutral passages raise an alarm). RAW = existing calibration, RECAL = per-bucket CV."""
    rec = recalibrate_cv(Z, b, folds, seed)
    out = []
    for bi in range(len(BUCKETS)):
        m = b == bi
        if m.sum() == 0:
            continue
        out.append({"bucket": bucket_name(bi), "n": int(m.sum()),
                    **{f"raw_q{int(x * 100)}": float(np.quantile(np.abs(Z[m]).max(1), x)) for x in q},
                    **{f"recal_q{int(x * 100)}": float(np.quantile(np.abs(rec[m]).max(1), x)) for x in q}})
    return out


def render_thresholds(rows):
    L = ["", "## Post hoc, descriptive: alarm thresholds that would hold per bucket", "",
         "Quantile of max |z| over the 7 emotions on neutral text. To keep false alarms at 5% (or 1%) of neutral passages, only highlight "
         "a passage when its max |z| exceeds the value shown. RAW = existing calibration; RECAL = per-bucket recalibrated z (out of fold).", "",
         "| words | n | RAW q95 | RAW q99 | RECAL q95 | RECAL q99 |", "|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['bucket']} | {r['n']} | {r['raw_q95']:.1f} | {r['raw_q99']:.1f} | {r['recal_q95']:.1f} | {r['recal_q99']:.1f} |")
    return "\n".join(L)


def render(raw, rec, emotions, model_tag):
    vr, vc = verdicts(raw), verdicts(rec)
    L = [f"# T3 length-aware calibration ({model_tag}; pre-registered in PLAN_v2)", "",
         f"Held-out neutral GoEmotions-TRAIN comments; |z| > {THRESH:g}; Clopper-Pearson 95% upper bounds; acceptance: <= {EMO_LIMIT:.0%} per emotion (worst of {len(emotions)}), "
         f"<= {PASSAGE_LIMIT:.0%} per passage, in every bucket with n >= {MIN_N}.", "",
         "| words | n | mean z (RAW) | sd z (RAW) | worst-emotion rate (upper) RAW | per-passage rate (upper) RAW | RAW | worst-emotion rate (upper) RECAL | per-passage rate (upper) RECAL | RECAL |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for a, c, x, y in zip(raw, rec, vr, vc):
        if a["n"] == 0:
            continue
        L.append(f"| {a['bucket']} | {a['n']} | {a['mean_z']:+.2f} | {a['sd_z']:.2f} | {a['emo_rate_max']:.1%} ({a['emo_upper_max']:.1%}) | {a['passage_rate']:.1%} ({a['passage_upper']:.1%}) | "
                 f"{'PASS' if x['pass'] else 'FAIL'} | {c['emo_rate_max']:.1%} ({c['emo_upper_max']:.1%}) | {c['passage_rate']:.1%} ({c['passage_upper']:.1%}) | {'PASS' if y['pass'] else 'FAIL'} |")
    for name, v in (("RAW", vr), ("RECAL", vc)):
        k = min_length_rule(v)
        L.append(f"\n{name}: " + ("no bucket qualifies" if k is None else f"scores valid from {bucket_name(k)} words upward" if k > 0 else "valid at every length") +
                 "; " + "; ".join(f"{x['bucket']}: {x['reason']}" for x in v if not x["pass"]))
    return "\n".join(L)


# ---------------------------------------------------------------- data (model-free part)
def cmd_reserve(a):
    import probekit_testset as ts
    from probekit_corpus_hf import clean_text
    from probekit_overlap import norm
    d = Path(a.data_dir); out = Path(a.out_dir)
    names = (d / "emotions.txt").read_text().split()
    train = ts.read_tsv(d / "train.tsv", names)
    A = json.loads((out / "goemotions_calibration_ids.json").read_text())["ids"]
    pool = {json.loads(l)["id"].split("|")[-1] for l in open(a.pool) if l.strip()}
    test_keys = {" ".join(norm(json.loads(l)["text"])) for l in open(out / "goemotions_test_frozen.jsonl")}
    used = set(A) | pool
    rng = np.random.default_rng(0)
    free = {i: [] for i in range(len(BUCKETS))}
    for it in train:
        if it["label"] != "neutral" or it["id"] in used:
            continue
        t = clean_text(it["text"]); k = " ".join(norm(t))
        if k and k not in test_keys:
            free[bucket_of(ts.n_words(t))].append(it["id"])
    cur = np.zeros(len(BUCKETS), int)
    byid = {it["id"]: it for it in train}
    for i in A:
        cur[bucket_of(ts.n_words(clean_text(byid[i]["text"])))] += 1
    B = []
    for bi in range(len(BUCKETS)):
        need = max(0, a.target - cur[bi]); pick = list(rng.permutation(free[bi])[:need]); B += pick
        print(f"bucket {bucket_name(bi)}: reserve A {cur[bi]}, free {len(free[bi])}, added {len(pick)}")
    ids = sorted(set(A) | set(B))
    texts = [{"id": i, "text": clean_text(byid[i]["text"]), "n_words": ts.n_words(clean_text(byid[i]["text"])), "category": "neutral", "split": "calibration"} for i in ids]
    ts.write_jsonl(out / "goemotions_calibration.jsonl", texts)
    (out / "goemotions_calibration_ids_B.json").write_text(json.dumps({"ids_B": sorted(B), "n_B": len(B), "n_total": len(ids), "target_per_bucket": a.target,
        "buckets": [bucket_name(i) for i in range(len(BUCKETS))], "calibration_file_sha256": ts.sha256_file(out / "goemotions_calibration.jsonl")}, indent=1))
    print(f"calibration set: {len(ids)} passages (A {len(A)} + B {len(B)})")


def cmd_eval(a):
    from probekit_estimators import calibrate_pooled, fit_directions
    from probekit_eval_v2 import load_jsonl
    from probekit_eval_v3 import load_states
    from probekit_frozen_eval import EMOTIONS, EXTRACT_LAYERS, cfg_for, extract_cached
    eval_dir, cache = Path(a.eval_dir), Path(a.cache_dir)
    best = json.loads((eval_dir / "best_config.json").read_text()); layer, skip, method = best["layer"], best["skip"], best["method"]
    rows = load_jsonl(a.corpus)
    Xc = load_states(rows, cache / "states_corpus.npz", [layer], skip, a.model)[layer]
    cat = np.array([r["category"] for r in rows]); spl = np.array([r["split"] for r in rows])
    tr = (spl == "train") & (cat != "calm"); cal = (spl == "val") & (cat == "neutral")
    ps = fit_directions(Xc[tr], cat[tr], EMOTIONS, layer, method=method, **cfg_for(eval_dir, method, layer))
    calibrate_pooled(ps, Xc[cal])
    items = load_jsonl(a.calib)
    Xk = extract_cached(items, cache / "states_calib.npz", EXTRACT_LAYERS, skip, a.model)[layer]
    Z = ps.score(Xk, "pooled"); b = np.array([bucket_of(r["n_words"]) for r in items])
    raw = stats(Z, b); rec = stats(recalibrate_cv(Z, b), b)
    thr = alarm_thresholds(Z, b)
    md = render(raw, rec, EMOTIONS, f"{a.model}, layer {layer}, {method}; n={len(items)}") + "\n" + render_thresholds(thr)
    Path(a.out).write_text(md + "\n"); Path(a.out).with_suffix(".json").write_text(json.dumps({"raw": raw, "recal": rec, "alarm_thresholds": thr, "emotions": EMOTIONS}, indent=1))
    print(md)


def main():
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("reserve"); r.add_argument("--data-dir", default="data/external/goemotions"); r.add_argument("--out-dir", default="data/testsets")
    r.add_argument("--pool", default="data/corpus/corpus_human_goemotions_pool.jsonl"); r.add_argument("--target", type=int, default=900)
    e = sp.add_parser("eval"); e.add_argument("--model", default="mlx-community/Llama-3.2-3B-Instruct"); e.add_argument("--cache-dir", required=True)
    e.add_argument("--eval-dir", required=True); e.add_argument("--corpus", required=True); e.add_argument("--calib", default="data/testsets/goemotions_calibration.jsonl")
    e.add_argument("--out", required=True)
    a = ap.parse_args(); {"reserve": cmd_reserve, "eval": cmd_eval}[a.cmd](a)


if __name__ == "__main__":
    main()
