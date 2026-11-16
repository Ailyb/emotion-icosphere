#!/usr/bin/env python3
"""
make_config.py -- write a config file for ANY (method, layer) from an eval_v3 report, using the hyperparameters
that probekit_eval_v3.py already chose on VAL for that cell. Lets you run build / stream-check / robustness for
each estimator without hand-editing JSON.

    python make_config.py --method logreg --layer 14      # -> data/eval_v3/<model>/best_config_logreg_L14.json
Then e.g.:  python probekit_robustness.py --config data/eval_v3/<model>/best_config_logreg_L14.json
            python probekit_build_v3.py --config <that file> --out data/probes_v3/<model>_logreg_L14
"""
import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="mlx-community/Llama-3.2-3B-Instruct")
ap.add_argument("--method", required=True, choices=["meandiff", "lda", "logreg"])
ap.add_argument("--layer", type=int, default=None, help="default: the layer in best_config.json")
ap.add_argument("--eval-dir", default=None)
a = ap.parse_args()
d = Path(a.eval_dir or f"data/eval_v3/{a.model.replace('/', '__')}")
best = json.loads((d / "best_config.json").read_text())
rep = json.loads((d / "report_v3.json").read_text())
L = a.layer or best["layer"]
cell = rep["per_layer"][str(L)][a.method]
cfg = {k: v for k, v in cell["cfg"].items() if k != "method"}
out = dict(best, method=a.method, layer=L, cfg=cfg, tau=cell["tau"], val_f1=cell["val"])
p = d / f"best_config_{a.method}_L{L}.json"
p.write_text(json.dumps(out, indent=2))
print(f"wrote {p}: {a.method} @ layer {L}, cfg {cfg}, val F1 {cell['val']:.3f}")
