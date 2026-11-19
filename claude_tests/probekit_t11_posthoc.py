#!/usr/bin/env python3
"""
probekit_t11_posthoc.py -- POST HOC analyses requested in docs/REVIEW_v3.md (A5, B2); not pre-registered, labelled as such everywhere.
  A5: trained classifier on the same hidden states (native LR) minus the frozen sentence encoder, paired bootstrap, frozen set + 3 external sources;
      bag of embeddings on the frozen set under HUM training (for the primary in-domain table).
  B2: per-class recall of the probe on the frozen set with the original tau and with the human-neutral tau_h (H5).
Same HUM training, layer, estimator and restricted predictors as probekit_t11.py (the H3/H5 verdicts are unchanged).

    python probekit_t11_posthoc.py --model <hf id> --cache-dir ... --eval-dir ... --corpus ... --out data/eval_merged2/<slug>/t11_posthoc.md
"""
import argparse
import json
from pathlib import Path

import numpy as np

from probekit_estimators import calibrate_pooled, fit_directions
from probekit_eval_v2 import f1_9, load_jsonl, pick_tau, probe_predict
from probekit_eval_v3 import paired_bootstrap
from probekit_external_eval import fit_systems
from probekit_frozen_eval import EMOTIONS, boot_ci
from probekit_t11 import SETS, encoder_predictor, fmt, hum_training, parity_split, setup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--cache-dir", required=True); ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--corpus", required=True); ap.add_argument("--pool", default="data/corpus/corpus_human_goemotions_pool.jsonl")
    ap.add_argument("--testsets", default="data/testsets"); ap.add_argument("--calib", default="data/testsets/goemotions_calibration.jsonl")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    c = setup(a)
    X_h, T_h, y_h = hum_training(c)
    li = np.where(c["llm"])[0]; E_h = np.vstack([c["Ec"][li], c["Ep"]])
    va, cal = c["va"], c["cal"]
    systems, tau = fit_systems(X_h, E_h, T_h, y_h, c["Xc"][va], c["Ec"][va], c["cat"][va], c["Xc"][cal], c["layer"], c["cfg"], c["method"])
    enc = encoder_predictor(c, T_h, y_h)
    L = [f"# T11 post hoc analyses ({a.model}; layer {c['layer']}, {c['method']}; NOT pre-registered; requested in docs/REVIEW_v3.md)", "",
         "## A5: trained classifier on the same hidden states (native LR) vs the frozen sentence encoder (HUM training; paired bootstrap)", "",
         "| set | n | native LR | encoder | native LR - encoder | probe - native LR | probe - bag of embeddings |", "|---|---|---|---|---|---|---|"]
    res = {"sets": {}}
    for name in SETS:
        s = c["sets"][name]; y = s["y"]; classes = sorted(set(y))
        P = {k: f(s["X"], s["E"], s["T"], classes) for k, f in systems.items()}; P["encoder"] = enc(s["T"], classes)
        d_ne = paired_bootstrap(y, P["native_lr"], P["encoder"]); d_pn = paired_bootstrap(y, P["probe"], P["native_lr"]); d_pe = paired_bootstrap(y, P["probe"], P["embed"])
        res["sets"][name] = {"native_lr": f1_9(y, P["native_lr"]), "encoder": f1_9(y, P["encoder"]), "embed": f1_9(y, P["embed"]),
                             "native_minus_encoder": list(d_ne), "probe_minus_native": list(d_pn), "probe_minus_embed": list(d_pe)}
        lo1, hi1 = boot_ci(y, P["native_lr"]); lo2, hi2 = boot_ci(y, P["encoder"])
        L.append(f"| {name} | {len(y)} | {f1_9(y, P['native_lr']):.3f} [{lo1:.3f}, {hi1:.3f}] | {f1_9(y, P['encoder']):.3f} [{lo2:.3f}, {hi2:.3f}] | {fmt(*d_ne)} | {fmt(*d_pn)} | {fmt(*d_pe)} |")
    # B2
    tr = c["tr_all"]
    ps = fit_directions(c["Xc"][tr], c["cat"][tr], EMOTIONS, c["layer"], method=c["method"], **c["cfg"])
    calibrate_pooled(ps, c["Xc"][cal]); tau0 = pick_tau(ps, c["Xc"][va], c["cat"][va])
    zk = ps.score(c["calib"]["X"], "pooled").max(1); tau_h = float(np.percentile(zk[parity_split(c["calib"]["ids"])], 90))
    f = c["sets"]["frozen"]; y = f["y"]; p0, p1 = probe_predict(ps, f["X"], tau0), probe_predict(ps, f["X"], tau_h)
    L += ["", "## B2: per-class recall on the frozen set, original tau vs human-neutral tau_h (H5 model; post hoc breakdown)", "",
          f"tau {tau0:.2f} -> tau_h {tau_h:.2f}", "", "| class | n | recall (tau) | recall (tau_h) | change |", "|---|---|---|---|---|"]
    rec = {}
    for k in sorted(set(y)):
        m = y == k; r0, r1 = float((p0[m] == k).mean()), float((p1[m] == k).mean()); rec[k] = [r0, r1]
        L.append(f"| {k} | {int(m.sum())} | {r0:.2f} | {r1:.2f} | {r1 - r0:+.2f} |")
    emo = y != "neutral"; moved = float(((p0[emo] != "neutral") & (p1[emo] == "neutral")).sum() / emo.sum())
    L += ["", f"Share of emotional (non-neutral) test items whose prediction changed from an emotion to neutral when tau became tau_h: {moved:.3f}."]
    res["per_class_recall"] = rec; res["emotional_to_neutral_share"] = moved
    Path(a.out).write_text("\n".join(L) + "\n"); Path(a.out).with_suffix(".json").write_text(json.dumps(res, indent=1)); print("\n".join(L))


if __name__ == "__main__":
    main()
