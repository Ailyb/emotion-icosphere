#!/usr/bin/env python3
"""
probekit_steer_c1.py -- PLAN_v2 T12/C1 (pre-registered 2026-11-20, a2da166): does the T10 steering pattern replicate on Gemma-2-2B, and is the
LDA vs mean-difference contrast an effect-size artifact? Generalises probekit_steer.py (Llama, layer 16) to another model and layer without
touching the T10 code path: it reuses its helpers and sets their layer and output directory.

    python probekit_steer_c1.py prepare    # directions (LDA probe, v1-style mean-difference, 20 random) + mean token norm at layer 14
    python probekit_steer_c1.py selftest   # acceptance: steering adds exactly s*norm*direction at the block; zero strength changes nothing
    python probekit_steer_c1.py pilot      # perplexity-only pilot (never the judge): S* (regime N, <= 1.5) and s_F per type (regime F, <= 1.3)
    python probekit_steer_c1.py generate   # all conditions, resumable
    python probekit_steer_c1.py judge      # label with the fixed judge (granite4.2:8b), resumable
    python probekit_steer_c1.py analyze --out <md>

Outputs in data/steering_gemma/ (jsonl and npz git-ignored; only aggregate reports are committed).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

import probekit_steer as S

MODEL = "mlx-community/gemma-2-2b-it-4bit"
LAYER = 14
DIR = Path("data/steering_gemma")
GRID_N = S.GRID                                            # [0.05, 0.1, 0.2, 0.4, 0.8, 1.6]; limit 1.5 (the T10 rule)
GRID_F = [0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.2]          # finer grid; limit 1.3 per direction type
LIMIT_F = 1.3
TYPES = {"probe": "LDA probe", "v1": "mean-difference"}
KEY = lambda s: round(float(s), 6)


def configure():
    S.LAYER = LAYER; S.DIR = DIR; DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------- pure functions (model-free, unit-tested)
def pilot_rule(table, names, grid, limit):
    """Largest grid value at which EVERY listed direction has median perplexity ratio <= limit (the T10 'every tested direction' reading);
    the smallest grid value if none qualifies."""
    ok = [s for s in grid if all(table[f"{s}|{n}"] <= limit for n in names)]
    return max(ok) if ok else grid[0]


def plan_strengths(s_star, s_f):
    """{strength: set(of direction types to generate)}; random directions are generated at every strength. Regime N: {S*/4, S*/2, S*} for both types;
    regime F: {s_F(t)/2, s_F(t)} for type t. Keys are rounded so that coinciding strengths are shared."""
    needs = {}
    for s in (s_star / 4, s_star / 2, s_star):
        needs.setdefault(KEY(s), set()).update(TYPES)
    for t, sf in s_f.items():
        for s in (sf / 2, sf):
            needs.setdefault(KEY(s), set()).add(t)
    return needs


def h7(rows, s_star, s_f):
    """Pre-registered decision rules.
    H7a: LDA probe directions shift expressed emotion beyond random controls only if >= 4 of 7 emotions are specifically steerable (T10 rule) at >= 1
         fluent strength in regime N or F.
    H7b: mean-difference beats LDA at matched fluency only if, in regime F at s_F(t), for >= 3 of 7 emotions the paired bootstrap interval of
         Delta(mean-diff at s_F) - Delta(LDA at s_F) is above 0, AND the mean-difference direction is specifically steerable for >= 2 of those emotions.
    Descriptive: whether mean-difference is specific for joy and/or anger, per regime."""
    a = S.analyze(rows, cond="probe"); b = S.analyze(rows, cond="v1")
    by = {}
    for r in rows:
        by.setdefault((r["cond"], r.get("emotion"), r.get("rid"), KEY(r["strength"])), {})[r["pid"]] = r
    pids = sorted(next(v for k, v in by.items() if k[0] == "base")); base = [by[("base", None, None, 0.0)][p]["label"] for p in pids]
    sf_p, sf_m = KEY(s_f["probe"]), KEY(s_f["v1"]); pairs = {}
    for e in S.EMOTIONS:
        lda = by.get(("probe", e, None, sf_p)); md = by.get(("v1", e, None, sf_m))
        if lda and md:
            pairs[e] = S.boot_diff(base, [md[p]["label"] for p in pids], [lda[p]["label"] for p in pids], e)
    above = [e for e, (m, lo, hi) in pairs.items() if lo > 0]
    md_specific = lambda e, s: bool(b["emotions"].get(e, {}).get(s, {}).get("specific"))
    above_and_specific = [e for e in above if md_specific(e, sf_m)]
    n_strengths = set(plan_strengths(s_star, s_f))
    regime_n = {KEY(s_star / 4), KEY(s_star / 2), KEY(s_star)}; regime_f_md = {KEY(s_f["v1"] / 2), KEY(s_f["v1"])}
    desc = {e: {"N": any(md_specific(e, s) for s in regime_n), "F": any(md_specific(e, s) for s in regime_f_md)} for e in ("joy", "anger")}
    md_at_sf = {e: md_specific(e, sf_m) for e in pairs}
    spec_in = lambda res, e, ss: any(bool(res["emotions"].get(e, {}).get(x, {}).get("specific")) for x in ss)
    regime_f_probe = {KEY(s_f["probe"] / 2), KEY(s_f["probe"])}
    sensitivity = {"probe_N_only": [e for e in S.EMOTIONS if spec_in(a, e, regime_n)], "probe_F_only": [e for e in S.EMOTIONS if spec_in(a, e, regime_f_probe)],
                   "v1_N_only": [e for e in S.EMOTIONS if spec_in(b, e, regime_n)], "v1_F_only": [e for e in S.EMOTIONS if spec_in(b, e, regime_f_md)]}
    return {"sensitivity": sensitivity, "md_specific_at_sF": md_at_sf, "H7a_steerable": a["steerable"], "H7a_claim": len(a["steerable"]) >= 4, "H7b_pairs": {e: list(v) for e, v in pairs.items()},
            "H7b_above": above, "H7b_above_and_md_specific": above_and_specific, "H7b_claim": len(above) >= 3 and len(above_and_specific) >= 2,
            "md_steerable": b["steerable"], "replication_joy_anger": desc, "analysis_probe": a, "analysis_v1": b, "n_strengths": sorted(n_strengths)}


def render_h7(res, s_star, s_f):
    L = ["## Pre-registered decisions (T12/C1)", "",
         f"Strengths: regime N S* = {s_star:g} (x mean token norm; strengths {s_star / 4:g}, {s_star / 2:g}, {s_star:g}); regime F s_F = {s_f['probe']:g} (LDA) and {s_f['v1']:g} (mean-difference), strengths s_F/2 and s_F.", "",
         f"- **H7a:** LDA probe directions specifically steerable (>= 1 fluent strength, regimes N and F): {res['H7a_steerable'] or 'none'} ({len(res['H7a_steerable'])}/7; claim needs >= 4) -> "
         f"**{'claim holds' if res['H7a_claim'] else 'not established'}**.",
         f"- **H7b:** paired Delta(mean-difference at s_F) - Delta(LDA at s_F), per emotion [95% CI over prompts]:", "",
         "| emotion | difference [CI] | CI above 0 | mean-difference specific at s_F |", "|---|---|---|---|"]
    for e, (m, lo, hi) in res["H7b_pairs"].items():
        sp = res["md_specific_at_sF"][e]
        L.append(f"| {e} | {m:+.3f} [{lo:+.3f}, {hi:+.3f}] | {'yes' if lo > 0 else 'no'} | {'yes' if sp else 'no'} |")
    L += ["", f"Emotions with the interval above 0: {res['H7b_above'] or 'none'} ({len(res['H7b_above'])}/7; need >= 3); of those, mean-difference specifically steerable: "
              f"{res['H7b_above_and_md_specific'] or 'none'} (need >= 2) -> **{'claim holds' if res['H7b_claim'] else 'not established'}**.",
          f"- Descriptive replication (mean-difference specific for joy / anger): {json.dumps(res['replication_joy_anger'])}.",
          f"- Mean-difference directions specifically steerable at any fluent strength (descriptive): {res['md_steerable'] or 'none'}.",
          "- Sensitivity by regime (post hoc, descriptive; specifically steerable at some strength of the regime): " +
          "; ".join(f"{k}: {v or 'none'} ({len(v)}/7)" for k, v in res["sensitivity"].items()) + ".", ""]
    return "\n".join(L)


# ---------------------------------------------------------------- model side (MLX)
def cmd_prepare(a):
    configure(); a.model = a.model or MODEL
    S.cmd_prepare(a)


def cmd_selftest(a):
    configure()
    import mlx.core as mx
    from probekit_capture import Extractor
    ex = Extractor(a.model); sb = S.install(ex.model, LAYER); d = S.load_dirs()
    f = ex.fidelity_check("The bus arrived at noon and everyone got on.")
    ids = mx.array([ex.encode("The bus arrived at noon and everyone got on.")]); sb.capture = True
    S.set_vec(sb, None); ex.model(ids); base = np.array(sb.last.astype(mx.float32))
    S.set_vec(sb, None); ex.model(ids); again = np.array(sb.last.astype(mx.float32))
    v = 0.4 * d["mean_norm"] * d["probe"][0]; S.set_vec(sb, v); ex.model(ids); steered = np.array(sb.last.astype(mx.float32)); S.set_vec(sb, None)
    d1 = (steered - base)[0][1:]; u = v / np.linalg.norm(v)
    proj = (d1 @ u) / np.linalg.norm(v); rel = np.linalg.norm(d1 - v[None, :], axis=1) / np.linalg.norm(v)
    zero_ok = bool(np.array_equal(base, again))
    ok = 0.9 <= proj.mean() <= 1.1 and rel.mean() < 0.5 and zero_ok
    print(f"selftest (layer {LAYER}): projection / norm mean {proj.mean():.3f} (min {proj.min():.3f}, max {proj.max():.3f}); relative error mean {rel.mean():.3f}; "
          f"repeat of the unsteered pass identical: {zero_ok}; mean token norm {d['mean_norm']:.1f}  ->  {'PASS' if ok else 'FAIL'}")
    if f:
        print("extractor fidelity:", f)
    print("unsteered sample:", S.gen(ex.model, ex.tok, sb, None, "Write three sentences about a quiet library.")[:140].replace("\n", " "))
    if not ok:
        raise SystemExit("self-test failed: nothing may be generated (pre-registration)")


def cmd_pilot(a):
    configure()
    from probekit_capture import Extractor
    ex = Extractor(a.model); sb = S.install(ex.model, LAYER); d = S.load_dirs(); _, pilot = S.topics()
    j = S.EMOTIONS.index("joy")
    dirs = {"joy_probe": d["probe"][j], "joy_v1": d["v1"][j], "rand0": d["random"][0], "rand1": d["random"][1], "rand2": d["random"][2]}
    base = {t: S.gen(ex.model, ex.tok, sb, None, S.prompt_text(t)) for t in pilot}
    base_nll = {t: S.unsteered_nll(ex.model, ex.tok, sb, S.prompt_text(t), base[t]) for t in pilot}
    table = {}
    for s in sorted(set(GRID_N) | set(GRID_F)):
        for name, v in dirs.items():
            ratios = []
            for t in pilot:
                txt = S.gen(ex.model, ex.tok, sb, s * d["mean_norm"] * v, S.prompt_text(t))
                ratios.append(float(np.exp(S.unsteered_nll(ex.model, ex.tok, sb, S.prompt_text(t), txt) - base_nll[t])))
            table[f"{s}|{name}"] = float(np.median(ratios)); print(f"  s={s:<5} {name:<9} median ppl ratio {table[f'{s}|{name}']:.2f}", flush=True)
    rand = ["rand0", "rand1", "rand2"]
    s_star = pilot_rule(table, ["joy_probe"] + rand, GRID_N, S.PPL_LIMIT)
    s_f = {"probe": pilot_rule(table, ["joy_probe"] + rand, GRID_F, LIMIT_F), "v1": pilot_rule(table, ["joy_v1"] + rand, GRID_F, LIMIT_F)}
    out = {"S_star": s_star, "s_F": s_f, "needs": {str(k): sorted(v) for k, v in plan_strengths(s_star, s_f).items()}, "table": table, "mean_norm": d["mean_norm"],
           "grid_N": GRID_N, "grid_F": GRID_F, "limit_N": S.PPL_LIMIT, "limit_F": LIMIT_F}
    (DIR / "pilot.json").write_text(json.dumps(out, indent=1)); print("S* =", s_star, "; s_F =", s_f, "; strengths:", out["needs"])


def cmd_generate(a):
    configure()
    from probekit_capture import Extractor
    ex = Extractor(a.model); sb = S.install(ex.model, LAYER); d = S.load_dirs(); main, _ = S.topics()
    pl = json.loads((DIR / "pilot.json").read_text()); needs = {float(k): set(v) for k, v in pl["needs"].items()}
    path = DIR / "generations.jsonl"; done = {json.loads(l)["gid"] for l in path.open()} if path.exists() else set()
    conds = [("base", None, None, 0.0, None)]
    for s in sorted(needs):
        if "probe" in needs[s]:
            conds += [("probe", e, None, s, d["probe"][i]) for i, e in enumerate(S.EMOTIONS)]
        if "v1" in needs[s]:
            conds += [("v1", e, None, s, d["v1"][i]) for i, e in enumerate(S.EMOTIONS)]
        conds += [("random", None, r, s, d["random"][r]) for r in range(S.N_RANDOM)]
    print(f"{len(conds)} conditions x {len(main)} prompts = {len(conds) * len(main)} generations", flush=True)
    f = path.open("a"); n = 0; t0 = time.time()
    for cond, e, r, s, v in conds:
        for pid, t in enumerate(main):
            gid = f"{cond}|{e}|{r}|{s}|{pid}"
            if gid in done:
                continue
            txt = S.gen(ex.model, ex.tok, sb, None if v is None else s * d["mean_norm"] * v, S.prompt_text(t))
            nll = S.unsteered_nll(ex.model, ex.tok, sb, S.prompt_text(t), txt)
            f.write(json.dumps({"gid": gid, "cond": cond, "emotion": e, "rid": r, "strength": s, "pid": pid, "topic": t, "text": txt, "nll": nll}) + "\n"); f.flush(); n += 1
            if n % 200 == 0:
                print(f"  {n} generated ({time.time() - t0:.0f}s)", flush=True)
    print("generation done:", n, "new")


def cmd_judge(a):
    configure(); S.cmd_judge(a)


def cmd_analyze(a):
    configure()
    gens = {json.loads(l)["gid"]: json.loads(l) for l in (DIR / "generations.jsonl").open()}
    lab = {json.loads(l)["gid"]: json.loads(l)["label"] for l in (DIR / "judged.jsonl").open()}
    base_nll = {g["pid"]: g["nll"] for g in gens.values() if g["cond"] == "base"}
    rows = [{**{k: g[k] for k in ("cond", "emotion", "rid", "strength", "pid")}, "label": lab[gid],
             "ppl_ratio": float(np.exp(g["nll"] - base_nll[g["pid"]])) if g["pid"] in base_nll and np.isfinite(g["nll"]) else None} for gid, g in gens.items() if gid in lab]
    pl = json.loads((DIR / "pilot.json").read_text()); s_star, s_f = pl["S_star"], pl["s_F"]
    res = h7(rows, s_star, s_f)
    md = [f"# T12/C1 steering replication on Gemma-2-2B (layer {LAYER}; pre-registered in PLAN_v2, a2da166); {res['analysis_probe']['n_prompts']} prompts, {res['analysis_probe']['n_random']} random directions", "",
          f"Pilot (perplexity only): S* = {s_star:g} (limit {S.PPL_LIMIT}); s_F = {s_f} (limit {LIMIT_F}, every tested direction); mean token norm {pl['mean_norm']:.1f}. Pilot table: `pilot.json`.", "",
          render_h7(res, s_star, s_f),
          S.render(res["analysis_probe"], "T12/C1 LDA probe directions (T10 rule)"), "",
          S.render(res["analysis_v1"], "T12/C1 mean-difference directions (T10 rule)"), ""]
    unparsed = sum(r["label"] == "unparsed" for r in rows); base = [r["label"] for r in rows if r["cond"] == "base"]
    ur = {}
    for r in rows:
        k = (r["cond"], r["strength"]); ur.setdefault(k, [0, 0]); ur[k][0] += r["label"] == "unparsed"; ur[k][1] += 1
    md.append("Unparsed judge labels by condition (post hoc): " + "; ".join(f"{c}@{x:g}: {u}/{n}" for (c, x), (u, n) in sorted(ur.items())) + "\n")
    md.append(f"Judge: granite4.2:8b; unparsed labels {unparsed}/{len(rows)}. Unsteered label distribution: " + ", ".join(f"{l} {base.count(l)}" for l in sorted(set(base))) + "\n")
    Path(a.out).write_text("\n".join(md)); Path(a.out).with_suffix(".json").write_text(json.dumps(res, indent=1, default=float)); print("\n".join(md))


def main():
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    for name in ("prepare", "selftest", "pilot", "generate", "judge", "analyze"):
        p = sp.add_parser(name); p.add_argument("--model", default=MODEL)
        if name == "prepare":
            slug = MODEL.replace("/", "__")
            p.add_argument("--cache-dir", default=f"data/eval_merged2/{slug}"); p.add_argument("--eval-dir", default=f"data/eval_merged2/{slug}/v3")
            p.add_argument("--corpus", default="data/corpus/corpus_merged2_scored.jsonl")
        if name == "analyze":
            p.add_argument("--out", default=f"data/eval_merged2/{MODEL.replace('/', '__')}/steering_c1_results.md")
    a = ap.parse_args()
    {"prepare": cmd_prepare, "selftest": cmd_selftest, "pilot": cmd_pilot, "generate": cmd_generate, "judge": cmd_judge, "analyze": cmd_analyze}[a.cmd](a)


if __name__ == "__main__":
    main()
