#!/usr/bin/env python3
"""
probekit_steer_c1b.py -- PLAN_v2 T13/C1b (pre-registered 2026-11-22, 85e7e53): was the Llama steering null (T10) an artifact of the strength grid?
Thin wrapper over probekit_steer_c1.py: same self-test, helpers and decision rules, pointed at Llama-3.2-3B (layer 16), the T10 directions, and
regime F only (fluency-matched strengths on a finer grid between T10's 0.4 and 0.8).

    python probekit_steer_c1b.py setup | selftest | pilot | generate | judge | analyze
Outputs in data/steering_llama_f/ (jsonl and npz git-ignored); report data/eval_merged2/<llama slug>/steering_c1b_results.md.
"""
import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np

import probekit_steer as S
import probekit_steer_c1 as C

MODEL = "mlx-community/Llama-3.2-3B-Instruct"
LAYER = 16
DIR = Path("data/steering_llama_f")
GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
LIMIT = 1.3
SRC = Path("data/steering/directions.npz")


def configure():
    C.MODEL, C.LAYER, C.DIR = MODEL, LAYER, DIR
    C.GRID_F, C.LIMIT_F = GRID, LIMIT
    C.configure()


def plan(s_f):
    """{strength: set(types)}: for type t, {s_F(t)/2, s_F(t)}; random directions at every strength."""
    needs = {}
    for t, sf in s_f.items():
        for s in (sf / 2, sf):
            needs.setdefault(C.KEY(s), set()).add(t)
    return needs


def cmd_setup(a):
    configure()
    shutil.copy(SRC, DIR / "directions.npz"); print("copied T10 directions ->", DIR / "directions.npz")


def cmd_pilot(a):
    configure()
    from probekit_capture import Extractor
    ex = Extractor(a.model); sb = S.install(ex.model, LAYER); d = S.load_dirs(); _, pilot = S.topics()
    j = S.EMOTIONS.index("joy")
    dirs = {"joy_probe": d["probe"][j], "joy_v1": d["v1"][j], "rand0": d["random"][0], "rand1": d["random"][1], "rand2": d["random"][2]}
    base = {t: S.gen(ex.model, ex.tok, sb, None, S.prompt_text(t)) for t in pilot}
    base_nll = {t: S.unsteered_nll(ex.model, ex.tok, sb, S.prompt_text(t), base[t]) for t in pilot}
    table = {}
    for s in GRID:
        for name, v in dirs.items():
            ratios = [float(np.exp(S.unsteered_nll(ex.model, ex.tok, sb, S.prompt_text(t), S.gen(ex.model, ex.tok, sb, s * d["mean_norm"] * v, S.prompt_text(t))) - base_nll[t])) for t in pilot]
            table[f"{s}|{name}"] = float(np.median(ratios)); print(f"  s={s:<5} {name:<9} median ppl ratio {table[f'{s}|{name}']:.2f}", flush=True)
    rand = ["rand0", "rand1", "rand2"]
    s_f = {"probe": C.pilot_rule(table, ["joy_probe"] + rand, GRID, LIMIT), "v1": C.pilot_rule(table, ["joy_v1"] + rand, GRID, LIMIT)}
    out = {"s_F": s_f, "needs": {str(k): sorted(v) for k, v in plan(s_f).items()}, "table": table, "mean_norm": d["mean_norm"], "grid": GRID, "limit": LIMIT}
    (DIR / "pilot.json").write_text(json.dumps(out, indent=1)); print("s_F =", s_f, "; strengths:", out["needs"])


def cmd_generate(a):
    configure()
    pl = json.loads((DIR / "pilot.json").read_text())
    # reuse the C1 generator: it reads pilot.json 'needs' (same format)
    C.cmd_generate(a)


def cmd_analyze(a):
    configure()
    gens = {json.loads(l)["gid"]: json.loads(l) for l in (DIR / "generations.jsonl").open()}
    lab = {json.loads(l)["gid"]: json.loads(l)["label"] for l in (DIR / "judged.jsonl").open()}
    base_nll = {g["pid"]: g["nll"] for g in gens.values() if g["cond"] == "base"}
    rows = [{**{k: g[k] for k in ("cond", "emotion", "rid", "strength", "pid")}, "label": lab[gid],
             "ppl_ratio": float(np.exp(g["nll"] - base_nll[g["pid"]])) if g["pid"] in base_nll and np.isfinite(g["nll"]) else None} for gid, g in gens.items() if gid in lab]
    pl = json.loads((DIR / "pilot.json").read_text()); s_f = pl["s_F"]
    res = C.h7(rows, s_f["probe"], s_f)
    a_, b_ = res["analysis_probe"], res["analysis_v1"]
    L = [f"# T13/C1b fluency-matched steering on Llama-3.2-3B (layer {LAYER}; pre-registered in PLAN_v2, 85e7e53); {a_['n_prompts']} prompts, {a_['n_random']} random directions", "",
         f"Pilot (perplexity only): s_F = {s_f} (grid {GRID}, limit {LIMIT}, every tested direction); mean token norm {pl['mean_norm']:.1f}. T10 strengths (0.1, 0.2, 0.4) are not pooled. Pilot table: `pilot.json`.", "",
         "## Pre-registered decisions (T13/C1b)", "",
         f"- **H8a:** LDA probe directions specifically steerable at >= 1 of the two new strengths: {a_['steerable'] or 'none'} ({len(a_['steerable'])}/7; claim needs >= 4) -> **{'claim holds' if len(a_['steerable']) >= 4 else 'not established'}**.",
         "- **H8b:** paired Delta(mean-difference at s_F) - Delta(LDA at s_F), per emotion [95% CI over prompts]:", "",
         "| emotion | difference [CI] | CI above 0 | mean-difference specific at s_F |", "|---|---|---|---|"]
    for e, (m, lo, hi) in res["H7b_pairs"].items():
        L.append(f"| {e} | {m:+.3f} [{lo:+.3f}, {hi:+.3f}] | {'yes' if lo > 0 else 'no'} | {'yes' if res['md_specific_at_sF'][e] else 'no'} |")
    L += ["", f"Emotions with the interval above 0: {res['H7b_above'] or 'none'} ({len(res['H7b_above'])}/7; need >= 3); of those, mean-difference specific: {res['H7b_above_and_md_specific'] or 'none'} (need >= 2) -> "
          f"**{'claim holds' if res['H7b_claim'] else 'not established'}**.",
          f"- Mean-difference directions specifically steerable at any new fluent strength (descriptive): {res['md_steerable'] or 'none'}.",
          f"- Descriptive (joy / anger mean-difference, any new strength): {json.dumps({e: v['F'] for e, v in res['replication_joy_anger'].items()})}.", "",
          S.render(a_, "T13/C1b LDA probe directions (T10 rule)"), "", S.render(b_, "T13/C1b mean-difference directions (T10 rule)"), ""]
    ur = {}
    for r in rows:
        k = (r["cond"], r["strength"]); ur.setdefault(k, [0, 0]); ur[k][0] += r["label"] == "unparsed"; ur[k][1] += 1
    L.append("Unparsed judge labels by condition (post hoc): " + "; ".join(f"{c}@{x:g}: {u}/{n}" for (c, x), (u, n) in sorted(ur.items())) + "\n")
    Path(a.out).write_text("\n".join(L)); Path(a.out).with_suffix(".json").write_text(json.dumps(res, indent=1, default=float)); print("\n".join(L))


def main():
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    for n in ("setup", "selftest", "pilot", "generate", "judge", "analyze"):
        p = sp.add_parser(n); p.add_argument("--model", default=MODEL)
        if n == "analyze":
            p.add_argument("--out", default=f"data/eval_merged2/{MODEL.replace('/', '__')}/steering_c1b_results.md")
    a = ap.parse_args()
    {"setup": cmd_setup, "selftest": lambda a: (configure(), C.cmd_selftest(a)), "pilot": cmd_pilot, "generate": cmd_generate,
     "judge": lambda a: (configure(), S.cmd_judge(a)), "analyze": cmd_analyze}[a.cmd](a)


if __name__ == "__main__":
    main()
