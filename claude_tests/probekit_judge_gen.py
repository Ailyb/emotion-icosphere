#!/usr/bin/env python3
"""
probekit_judge_gen.py -- viz_plan T16 (pre-registered 2026-11-27, commit 2cbb564): is the steering judge (granite4.2:8b) consistent with a second judge
(qwen3.8:27b) on GENERATED text? Not a validation against humans.

    python probekit_judge_gen.py run --out data/eval_merged2/judge_gen   # judge samples + synthetic set with qwen, write report
Existing granite labels (judged.jsonl) are used for the steering samples; granite is asked fresh only for the synthetic set.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

import probekit_judge as J

SETS = {"llama": Path("data/steering"), "gemma": Path("data/steering_gemma")}
SECOND = "qwen3.8:27b"


def kappa(a, b):
    """Cohen's kappa over rows where both labels are present."""
    a, b = np.asarray(a, dtype=object), np.asarray(b, dtype=object)
    m = np.array([x is not None and y is not None for x, y in zip(a, b)])
    a, b = a[m], b[m]
    if len(a) == 0:
        return float("nan")
    cls = sorted(set(a) | set(b)); po = float((a == b).mean())
    pe = sum(float((a == c).mean()) * float((b == c).mean()) for c in cls)
    return float("nan") if pe == 1 else (po - pe) / (1 - pe)


def boot_kappa(a, b, B=1000, seed=0):
    rng = np.random.default_rng(seed); n = len(a); a, b = np.asarray(a, dtype=object), np.asarray(b, dtype=object)
    ks = [kappa(a[i], b[i]) for i in (rng.integers(0, n, n) for _ in range(B))]
    return float(np.nanpercentile(ks, 2.5)), float(np.nanpercentile(ks, 97.5))


def sample(model_dir, seed=0):
    """Stratified sample (see viz_plan T16): 40 base, 120 probe@max, 120 random@max, 120 probe@other strengths. Returns rows with gid, text, stratum, granite."""
    gen = [json.loads(l) for l in open(model_dir / "generations.jsonl")]
    lab = {r["gid"]: r["label"] for r in map(json.loads, open(model_dir / "judged.jsonl"))}
    rng = np.random.default_rng(seed)
    pr = sorted({g["strength"] for g in gen if g["cond"] == "probe"}); mx = pr[-1]
    pick = {"base": [g for g in gen if g["cond"] == "base"],
            "probe_max": [g for g in gen if g["cond"] == "probe" and g["strength"] == mx],
            "random_max": [g for g in gen if g["cond"] == "random" and g["strength"] == mx],
            "probe_other": [g for g in gen if g["cond"] == "probe" and g["strength"] != mx]}
    want = {"base": 40, "probe_max": 120, "random_max": 120, "probe_other": 120}
    out = []
    for k, rows in pick.items():
        rows = [g for g in rows if g["gid"] in lab]
        idx = rng.permutation(len(rows))[:want[k]]
        out += [dict(gid=rows[i]["gid"], text=rows[i]["text"], stratum=k, granite=lab[rows[i]["gid"]]) for i in sorted(idx)]
    return out


def macro_f1(y, p):
    from probekit_eval_v2 import f1_9
    return float(f1_9(np.array(y, dtype=object), np.array([x or "unparsed" for x in p], dtype=object)))


def run(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    res = {}
    for name, d in SETS.items():
        rows = sample(d); q = J.judge_texts([r["text"] for r in rows], model=SECOND, workers=2)
        for r, l in zip(rows, q):
            r["qwen"] = l
        res[name] = rows; (out / f"sample_{name}.json").write_text(json.dumps(rows, indent=1))
        print(name, "judged", len(rows), flush=True)
    syn = [json.loads(l) for l in open("data/viz/synth.jsonl")]
    g = J.judge_texts([r["text"] for r in syn], workers=2); q = J.judge_texts([r["text"] for r in syn], model=SECOND, workers=2)
    y = [r["intended"] for r in syn]
    S = dict(n=len(syn), granite_f1=macro_f1(y, g), qwen_f1=macro_f1(y, q), granite_unparsed=sum(x is None for x in g), qwen_unparsed=sum(x is None for x in q),
             kappa_granite_qwen=kappa(g, q))
    L = ["# T16 judge consistency on generated text (pre-registered 2026-11-27, commit 2cbb564)", "",
         f"Judges: granite4.2:8b (existing labels) vs {SECOND} (fresh). Cohen's kappa over rows where both parsed; bootstrap 95% CI over rows. **Not a validation against humans.**", "",
         "| set | stratum | n | agreement | kappa [95% CI] | unparsed granite / qwen |", "|---|---|---|---|---|---|"]
    ok = {}
    for name, rows in res.items():
        for st in [None, "base", "probe_max", "random_max", "probe_other"]:
            rr = [r for r in rows if st is None or r["stratum"] == st]
            if not rr:
                continue
            ga, qa = [r["granite"] for r in rr], [r["qwen"] for r in rr]
            both = [(x, y_) for x, y_ in zip(ga, qa) if x is not None and y_ is not None]
            k = kappa(ga, qa); lo, hi = boot_kappa(ga, qa)
            L.append(f"| {name} | {st or 'all'} | {len(rr)} | {np.mean([x == y_ for x, y_ in both]):.3f} | {k:.3f} [{lo:.3f}, {hi:.3f}] | {sum(x is None for x in ga)} / {sum(x is None for x in qa)} |")
            ok[(name, st or "all")] = k
    probe_k = [ok[(n, "probe_max")] for n in res] + [ok[(n, "probe_other")] for n in res]
    allk = [ok[(n, "all")] for n in res]
    verdict = all(k >= 0.60 for k in allk) and all(k >= 0.60 for k in probe_k)
    L += ["", f"Synthetic set (n = {S['n']}; intended label = the writer's request, writer was granite4.2:8b so granite may be favoured): macro-F1 vs intended: granite {S['granite_f1']:.3f}, qwen {S['qwen_f1']:.3f}; "
          f"granite-qwen kappa {S['kappa_granite_qwen']:.3f}; unparsed granite {S['granite_unparsed']}, qwen {S['qwen_unparsed']}.", "",
          f"**Decision rule (overall kappa >= 0.60 on both models AND kappa >= 0.60 in the probe-steered strata of both models): "
          f"{'MET: judge labels are consistent across judges on generated text' if verdict else 'NOT MET: judge labels on steered text are not validated'}.** This is agreement between two models, not accuracy."]
    (out / "judge_gen.md").write_text("\n".join(L) + "\n"); (out / "judge_gen.json").write_text(json.dumps(dict(synthetic=S, kappas={f'{k[0]}|{k[1]}': v for k, v in ok.items()}, rule_met=bool(verdict)), indent=1))
    # annotation sheet for the user: judge labels withheld
    rng = np.random.default_rng(1); pool = [dict(r, set=n) for n, rows in res.items() for r in rows]
    pick = [pool[i] for i in rng.permutation(len(pool))[:120]]
    sheet = Path("docs/viz/judge_annotation_sample.csv"); sheet.parent.mkdir(parents=True, exist_ok=True)
    with open(sheet, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["id", "text", "your_label (joy/sadness/fear/anger/surprise/disgust/love/neutral)"])
        for i, r in enumerate(pick):
            w.writerow([f"{r['set']}-{r['gid']}", r["text"].replace("\n", " "), ""])
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run"); r.add_argument("--out", default="data/eval_merged2/judge_gen")
    a = ap.parse_args(); {"run": run}[a.cmd](a)


if __name__ == "__main__":
    main()
