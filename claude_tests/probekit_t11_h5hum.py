#!/usr/bin/env python3
"""
probekit_t11_h5hum.py -- POST HOC (2026-11-22, requested in review; not pre-registered): the H5 human-neutral-threshold analysis repeated with the probe trained on the HUM set
(the paper's primary training set) instead of the first-run merged2 TRAIN set that the pre-registered H5 (probekit_t11.h5) used. Same tau rule (tau on LLM-written VAL-hard),
same tau_h (90th percentile of max z on the even-index half of the human neutral reserve, checked on the other half), same frozen set; cached states only.

    python probekit_t11_h5hum.py --model M --cache-dir D --eval-dir E --corpus C --out <md>
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import probekit_t11 as T
from probekit_estimators import calibrate_pooled, fit_directions
from probekit_eval_v2 import f1_9, pick_tau
from probekit_eval_v3 import paired_bootstrap


def analyse(c):
    X_h, _, y_h = T.hum_training(c)
    ps = fit_directions(X_h, y_h, T.EMOTIONS, c["layer"], method=c["method"], **c["cfg"])
    calibrate_pooled(ps, c["Xc"][c["cal"]]); tau = pick_tau(ps, c["Xc"][c["va"]], c["cat"][c["va"]])
    zk = ps.score(c["calib"]["X"], "pooled").max(1); even = T.parity_split(c["calib"]["ids"]); tau_h = float(np.percentile(zk[even], 90))
    f = c["sets"]["frozen"]; y = f["y"]
    p0, p1 = T.probe_predict(ps, f["X"], tau), T.probe_predict(ps, f["X"], tau_h)
    m, lo, hi = paired_bootstrap(y, p1, p0)
    neu = y == "neutral"; classes = sorted(set(y))
    rec = {k: (float((p0[y == k] == k).mean()), float((p1[y == k] == k).mean())) for k in classes}
    emo = ~neu & (p0 != "neutral")
    return dict(n_train=int(len(y_h)), tau=float(tau), tau_h=tau_h, f1_orig=f1_9(y, p0), f1_tauh=f1_9(y, p1), diff=[m, lo, hi],
                recall_odd_orig=float((zk[~even] < tau).mean()), recall_odd_tauh=float((zk[~even] < tau_h).mean()),
                frozen_neutral_recall_orig=float((p0[neu] == "neutral").mean()), frozen_neutral_recall_tauh=float((p1[neu] == "neutral").mean()),
                per_class_recall=rec, share_emotional_to_neutral=float((p1[emo] == "neutral").mean()), n_frozen=int(len(y)))


def main():
    ap = argparse.ArgumentParser()
    for k in ("model", "cache-dir", "eval-dir", "corpus", "out"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--pool", default="data/corpus/corpus_human_goemotions_pool.jsonl"); ap.add_argument("--testsets", default="data/testsets")
    ap.add_argument("--calib", default="data/testsets/goemotions_calibration.jsonl")
    a = ap.parse_args(); a.cache_dir, a.eval_dir = a.cache_dir, a.eval_dir
    c = T.setup(SimpleNamespace(model=a.model, cache_dir=a.cache_dir, eval_dir=a.eval_dir, corpus=a.corpus, pool=a.pool, testsets=a.testsets, calib=a.calib))
    r = analyse(c)
    L = [f"# H5 repeated with the HUM-trained probe (POST HOC, not pre-registered; {a.model}, layer {c['layer']}, {c['method']}; training n = {r['n_train']})", "",
         f"tau (LLM-written VAL-hard) = {r['tau']:.2f}; tau_h (90th percentile of max z, human neutral reserve) = {r['tau_h']:.2f}.", "",
         f"- Held-out human neutral recall (odd half, n check): original tau {r['recall_odd_orig']:.3f} -> tau_h {r['recall_odd_tauh']:.3f}.",
         f"- Frozen set (n = {r['n_frozen']}) neutral recall: {r['frozen_neutral_recall_orig']:.3f} -> {r['frozen_neutral_recall_tauh']:.3f}.",
         f"- Frozen 8-way macro-F1: {r['f1_orig']:.3f} -> {r['f1_tauh']:.3f}; difference {r['diff'][0]:+.3f} [{r['diff'][1]:+.3f}, {r['diff'][2]:+.3f}].",
         f"- Share of emotional frozen items predicted as an emotion under tau that become neutral under tau_h: {r['share_emotional_to_neutral']:.3f}.", "",
         "| class | recall at tau | recall at tau_h | change |", "|---|---|---|---|"]
    L += [f"| {k} | {o:.3f} | {h:.3f} | {h - o:+.3f} |" for k, (o, h) in r["per_class_recall"].items()]
    Path(a.out).write_text("\n".join(L) + "\n"); Path(a.out).with_suffix(".json").write_text(json.dumps(r, indent=1)); print("\n".join(L))


if __name__ == "__main__":
    main()
