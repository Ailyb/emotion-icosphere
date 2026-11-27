#!/usr/bin/env python3
"""
probekit_viz_stream.py -- viz_plan T15 (pre-registered 2026-11-27, commit 2cbb564): can the stream display colour token by token?
Splice test: neutral | emotional | neutral streams scored by the pooled probe over causal windows; onset/offset latency, hit rate, false alarms.

    python probekit_viz_stream.py --model M --cache-dir D --eval-dir E --corpus C --out <md>
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

EMO = ["joy", "sadness", "fear", "anger", "surprise", "disgust", "love"]
WINDOWS = [8, 16, 32, "prefix"]
THR_Q, FA_MAX_SEL, OFF_MAX = 0.99, 0.02, 12
RULE = dict(hit=0.60, onset=12, offset=12, fa=0.05)
TAIL = 8


def window_means(st, skip, W):
    """(T,H) layer states -> {t: mean over the causal window ending at t}, as an array for t = skip..T-1 (window starts no earlier than `skip`)."""
    T = len(st); cs = np.vstack([np.zeros((1, st.shape[1])), np.cumsum(st.astype(np.float64), 0)])
    ts = np.arange(skip, T); lo = np.full(len(ts), skip) if W == "prefix" else np.maximum(skip, ts - W + 1)
    return ((cs[ts + 1] - cs[lo]) / (ts + 1 - lo)[:, None]).astype(np.float32), ts


def splice_metrics(flag, top, ts, e_start, e_end, intended, tail=TAIL):
    """flag/top: per position (aligned with ts). Returns dict(hit, onset, offset, offset_run); onset/offset in tokens (None if undefined)."""
    inside = (ts >= e_start) & (ts <= e_end + tail)
    ok = inside & flag & (top == intended)
    hit = bool(ok.any()); onset = int(ts[ok].min() - e_start) if hit else None
    after = flag & (ts > e_end); offset = int(ts[after].max() - e_end) if after.any() else 0
    run = 0                                                    # contiguous flagged run continuing past the end of the emotional sentence (post hoc variant)
    for t in range(e_end + 1, int(ts.max()) + 1):
        i = t - ts[0]
        if flag[i]:
            run += 1
        else:
            break
    return dict(hit=hit, onset=onset, offset=offset, offset_run=run)


def summarize(ms, fa_share):
    on = [m["onset"] for m in ms if m["onset"] is not None]
    return dict(n=len(ms), hit=float(np.mean([m["hit"] for m in ms])), onset_median=float(np.median(on)) if on else None,
                offset_median=float(np.median([m["offset"] for m in ms])), offset_run_median=float(np.median([m["offset_run"] for m in ms])), fa_share=float(fa_share))


def boot(ms, key, B=1000, seed=0):
    rng = np.random.default_rng(seed); n = len(ms); v = [m[key] for m in ms]
    out = []
    for _ in range(B):
        s = [v[i] for i in rng.integers(0, n, n)]; s = [x for x in s if x is not None]
        out.append(np.median(s) if key != "hit" and s else (np.mean(s) if s else np.nan))
    return float(np.nanpercentile(out, 2.5)), float(np.nanpercentile(out, 97.5))


def build_streams(synth, part, seed=0):
    """Neutral and splice streams from 8-19 word synthetic passages; partition by within-class index parity (VAL = even, TEST = odd)."""
    rng = np.random.default_rng(seed + (0 if part == "val" else 1))
    pool = {}
    for r in synth:
        if r["bucket"] in ("8-11", "12-19"):
            pool.setdefault(r["intended"], []).append(r["text"])
    par = 0 if part == "val" else 1
    sel = {k: [t for i, t in enumerate(v) if i % 2 == par] for k, v in pool.items()}
    neu = sel["neutral"]; pick = lambda n: [neu[i] for i in rng.integers(0, len(neu), n)]
    streams = [dict(kind="neutral", sents=pick(4), emotion=None, k=None) for _ in range(140)]
    for e in EMO:
        for t in sel[e][:20]:
            streams.append(dict(kind="splice", sents=pick(2) + [t] + pick(2), emotion=e, k=2))
    return streams


def score_stream(ex, ps, layer, skip, s):
    """-> (st layer states, [token start index of each sentence] + [end])"""
    bounds = [0]
    for k in range(1, len(s["sents"]) + 1):
        bounds.append(len(ex.encode(" ".join(s["sents"][:k]))))
    ids = ex.encode(" ".join(s["sents"]))
    return ex.states_from_ids(ids, [layer])[layer], bounds


def run_part(streams, ex, ps, layer, skip):
    """Per window: raw pooled-probe z at every position for every stream."""
    out = {W: [] for W in WINDOWS}; meta = []
    for s in streams:
        st, b = score_stream(ex, ps, layer, skip, s)
        for W in WINDOWS:
            M, ts = window_means(st, skip, W); out[W].append((ps.score(M, "pooled"), ts))
        meta.append(dict(kind=s["kind"], emotion=s["emotion"], e_start=b[s["k"]] if s["k"] is not None else None, e_end=(b[s["k"] + 1] - 1) if s["k"] is not None else None))
    return out, meta


def evaluate(Z, meta, W, norm, thr, shuffle_seed=None):
    """norm=(mu,sd) per emotion, thr = flag threshold on max z'. Returns (summary, per-splice metrics)."""
    mu, sd = norm; fa_n = fa_d = 0; ms = []; labs = [m["emotion"] for m in meta if m["kind"] == "splice"]
    if shuffle_seed is not None:
        labs = list(np.random.default_rng(shuffle_seed).permutation(labs))
    li = 0
    for (z, ts), m in zip(Z[W], meta):
        zz = (z - mu) / sd; flag = zz.max(1) > thr; top = zz.argmax(1)
        if m["kind"] == "neutral":
            fa_n += int(flag.sum()); fa_d += len(flag)
        else:
            ms.append(splice_metrics(flag, top, ts, m["e_start"], m["e_end"], EMO.index(labs[li]))); li += 1
    return summarize(ms, fa_n / max(1, fa_d)), ms


def calib(Z, meta, W):
    zs = np.vstack([z for (z, ts), m in zip(Z[W], meta) if m["kind"] == "neutral"])
    mu, sd = zs.mean(0), np.maximum(zs.std(0), 1e-6)
    return (mu, sd), float(np.quantile(((zs - mu) / sd).max(1), THR_Q))


def main():
    import probekit_t11 as T
    from probekit_capture import Extractor
    from probekit_estimators import calibrate_pooled, fit_directions
    ap = argparse.ArgumentParser()
    for k in ("model", "cache-dir", "eval-dir", "corpus", "out"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--synth", default="data/viz/synth.jsonl"); ap.add_argument("--pool", default="data/corpus/corpus_human_goemotions_pool.jsonl")
    ap.add_argument("--testsets", default="data/testsets"); ap.add_argument("--calib", default="data/testsets/goemotions_calibration.jsonl")
    a = ap.parse_args()
    c = T.setup(SimpleNamespace(model=a.model, cache_dir=a.cache_dir, eval_dir=a.eval_dir, corpus=a.corpus, pool=a.pool, testsets=a.testsets, calib=a.calib))
    layer, skip = c["layer"], c["skip"]
    X_h, _, y_h = T.hum_training(c)
    ps = fit_directions(X_h, y_h, T.EMOTIONS, layer, method=c["method"], **c["cfg"]); calibrate_pooled(ps, c["Xc"][c["cal"]])
    synth = [json.loads(l) for l in open(a.synth)]; ex = Extractor(a.model)
    V = build_streams(synth, "val"); Tt = build_streams(synth, "test")
    ZV, mV = run_part(V, ex, ps, layer, skip)
    L = [f"# T15 stream splice test (pre-registered 2026-11-27, commit 2cbb564; {a.model}, layer {layer}, {c['method']}, HUM-trained; synthetic streams, intended labels unvalidated)", "",
         "## VAL (window selection)", "", "| window | neutral FA share | hit | onset median | offset median | offset (contiguous run, post hoc) | qualifies |", "|---|---|---|---|---|---|---|"]
    sel, cal = None, {}
    for W in WINDOWS:
        norm, thr = calib(ZV, mV, W); cal[W] = (norm, thr); s, _ = evaluate(ZV, mV, W, norm, thr)
        q = s["fa_share"] <= FA_MAX_SEL and s["offset_median"] <= OFF_MAX
        L.append(f"| {W} | {s['fa_share']:.3f} | {s['hit']:.3f} | {s['onset_median']} | {s['offset_median']} | {s['offset_run_median']} | {'yes' if q else 'no'} |")
        if q and (sel is None or s["hit"] > sel[1] + 1e-12):
            sel = (W, s["hit"])
    result = dict(val_selected=None if sel is None else sel[0])
    if sel is None:
        L += ["", "No window qualifies on VAL (neutral false-alarm share <= 2% and median offset <= 12 tokens): **token-level display rejected without a TEST run**; use sentence-level animation."]
        usable = False
    else:
        W = sel[0]; ZT, mT = run_part(Tt, ex, ps, layer, skip); norm, thr = cal[W]
        s, ms = evaluate(ZT, mT, W, norm, thr)
        chance = [evaluate(ZT, mT, W, norm, thr, shuffle_seed=i)[0]["hit"] for i in range(200)]
        ci = {k: boot(ms, k) for k in ("hit", "onset", "offset", "offset_run")}
        per = {e: float(np.mean([m["hit"] for m, mm in zip(ms, [x for x in mT if x["kind"] == "splice"]) if mm["emotion"] == e])) for e in EMO}
        usable = s["hit"] >= RULE["hit"] and s["onset_median"] is not None and s["onset_median"] <= RULE["onset"] and s["offset_median"] <= RULE["offset"] and s["fa_share"] <= RULE["fa"]
        L += ["", f"## TEST (window {W} chosen on VAL; threshold {thr:.2f} on recalibrated max z)", "",
              f"- n splice streams = {s['n']}, neutral streams = 140.",
              f"- Neutral per-position false-alarm share: {s['fa_share']:.3f} (rule <= {RULE['fa']}).",
              f"- Hit rate: {s['hit']:.3f} [{ci['hit'][0]:.3f}, {ci['hit'][1]:.3f}] (rule >= {RULE['hit']}); shuffled-label chance {np.mean(chance):.3f} (95th pct {np.percentile(chance, 95):.3f}).",
              f"- Onset latency, tokens, median: {s['onset_median']} [{ci['onset'][0]:.1f}, {ci['onset'][1]:.1f}] (rule <= {RULE['onset']}).",
              f"- Offset latency, tokens, median: {s['offset_median']} [{ci['offset'][0]:.1f}, {ci['offset'][1]:.1f}] (rule <= {RULE['offset']}); contiguous-run variant (post hoc): {s['offset_run_median']} [{ci['offset_run'][0]:.1f}, {ci['offset_run'][1]:.1f}].",
              "- Hit rate per emotion (descriptive): " + ", ".join(f"{e} {v:.2f}" for e, v in per.items()), "",
              f"**Rule {'MET: token-level colour usable' if usable else 'NOT MET: token-level colour not supported; animate at the sentence level'}.**"]
        result.update(test=s, ci=ci, chance_mean=float(np.mean(chance)), per_emotion=per, thr=thr)
    result["usable"] = bool(usable)
    Path(a.out).write_text("\n".join(L) + "\n"); Path(a.out).with_suffix(".json").write_text(json.dumps(result, indent=1, default=str)); print("\n".join(L))


if __name__ == "__main__":
    main()
