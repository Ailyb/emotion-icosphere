#!/usr/bin/env python3
"""
probekit_build_v3.py -- turn the winning config from probekit_eval_v3.py into a deployable probes.npz.

Ships EXACTLY what was evaluated: fit on TRAIN, calibrate on held-out VAL neutral passages, threshold tau chosen on
VAL-hard. (We don't refit on train+val: that would invalidate the calibration we just measured.)
Adds single-token calibration for live streaming from VAL passages (needs the model), and z_full for display.

    python probekit_build_v3.py            # reads data/eval_v3/<model>/best_config.json
The backend can use the result directly:  PROBE_DIR=data/probes_v3/<model> python backend.py
"""
import argparse
import json
from pathlib import Path

import numpy as np

from probekit_estimators import calibrate_pooled, fit_from_config
from probekit_eval_v2 import load_jsonl, pick_tau
from probekit_eval_v3 import load_states
from probekit_probes import DEFAULT_EMOTIONS, NEUTRAL


def build(Xc_layer, rows, emotions, layer, cfg):
    cat = np.array([r["category"] for r in rows]); spl = np.array([r["split"] for r in rows])
    hard = np.array([bool(r.get("hard", True)) for r in rows])
    tr, va = spl == "train", (spl == "val") & hard
    ps = fit_from_config(Xc_layer[tr], cat[tr], emotions, layer, cfg)
    calibrate_pooled(ps, Xc_layer[(spl == "val") & (cat == NEUTRAL)])
    tau = pick_tau(ps, Xc_layer[va], cat[va])
    return ps, tau


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Llama-3.2-3B-Instruct")
    ap.add_argument("--corpus", default="data/corpus/corpus_v2_scored.jsonl")
    ap.add_argument("--config", default=None)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--layer", type=int, default=None, help="override layer (e.g. another plateau layer)")
    a = ap.parse_args()

    from build_emotion_vectors import slug
    cfg_path = Path(a.config or f"data/eval_v3/{slug(a.model)}/best_config.json")
    best = json.loads(cfg_path.read_text())
    layer = a.layer or best["layer"]
    cache_dir = Path(a.cache_dir or f"data/eval_v2/{slug(a.model)}")
    out = Path(a.out or f"data/probes_v3/{slug(a.model)}"); out.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(a.corpus)
    emotions = [e for e in DEFAULT_EMOTIONS if any(r["category"] == e for r in rows)]
    Xc = load_states(rows, cache_dir / "states_corpus.npz", [layer], best["skip"], a.model)
    ps, tau = build(Xc[layer], rows, emotions, layer, dict(best["cfg"], method=best["method"]))

    # single-token calibration from VAL passages (held out from the fit)
    from probekit_capture import Extractor
    ex = Extractor(a.model)
    spl = np.array([r["split"] for r in rows]); cat = np.array([r["category"] for r in rows])
    rng = np.random.default_rng(0)
    def tokens(mask, k=60):
        idx = rng.permutation(np.where(mask)[0])[:k]
        return np.concatenate([ex.token_states(rows[i]["text"], layer, best["skip"]) for i in idx])
    ps.calibrate_tokens(tokens((spl == "val") & (cat == NEUTRAL)), tokens((spl == "val") & (cat != NEUTRAL)))
    ps.meta.update({"model": a.model, "skip": best["skip"], "tau": tau, "z_full": ps.meta.get("z_full", 4.0),
                    "source": "probekit_build_v3", "config": best["cfg"], "method": best["method"]})
    ps.save(out / "probes.npz")
    (out / "report.json").write_text(json.dumps({"model": a.model, "layer": layer, "method": best["method"],
                                                 "config": best["cfg"], "tau": tau, "val_f1": best["val_f1"],
                                                 "emotions": emotions, "z_full": ps.meta["z_full"]}, indent=2))
    print(f"saved {out/'probes.npz'}: {best['method']} @ layer {layer}, tau {tau:.2f}, z_full {ps.meta['z_full']:.2f}")
    print(f"Run the backend with:  PROBE_DIR={out} python backend.py")


if __name__ == "__main__":
    main()
