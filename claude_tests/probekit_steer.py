#!/usr/bin/env python3
"""
probekit_steer.py -- PLAN_v2 T10 (pre-registered): do probe directions steer expressed emotion beyond random directions?

    python probekit_steer.py prepare   # directions (probe, v1-style, 20 random) + mean token norm       -> data/steering/directions.npz
    python probekit_steer.py selftest  # steering adds exactly s*norm*direction at the chosen block
    python probekit_steer.py pilot     # fluency pilot (perplexity only, never the judge) -> S*          -> data/steering/pilot.json
    python probekit_steer.py generate  # all conditions, resumable                                       -> data/steering/generations.jsonl
    python probekit_steer.py judge     # label every generation with the fixed judge, resumable          -> data/steering/judged.jsonl
    python probekit_steer.py analyze --out <md>

Generated text and judge labels stay in data/steering/*.jsonl (git-ignored); only aggregate reports are committed.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

LAYER = 16
EMOTIONS = ["joy", "sadness", "fear", "anger", "surprise", "disgust", "love"]
LABELS = EMOTIONS + ["neutral"]
GRID = [0.05, 0.1, 0.2, 0.4, 0.8, 1.6]
PPL_LIMIT = 1.5
N_RANDOM = 20
MAX_NEW = 60
DIR = Path("data/steering")


# ---------------------------------------------------------------- topics / prompts
def topics():
    from probekit_corpus import SPLITS
    from probekit_corpus_api import SPLITS_EXT
    base_train = sorted(t for t, s in SPLITS.items() if s == "train")[:16]
    main = sorted(SPLITS_EXT) + base_train
    pilot = sorted(t for t, s in SPLITS.items() if s in ("val", "test"))[:8]
    assert len(main) == 40 and len(pilot) == 8 and not set(main) & set(pilot)
    return main, pilot


def prompt_text(topic):
    return f"Write three sentences about {topic}."


# ---------------------------------------------------------------- analysis (pure functions)
def delta(base_labels, steered_labels, emotion):
    """Rate(steered == emotion) - rate(base == emotion) over the same prompts (aligned lists)."""
    b = np.array([x == emotion for x in base_labels], float); s = np.array([x == emotion for x in steered_labels], float)
    return float(s.mean() - b.mean())


def boot_delta(base_labels, steered_labels, emotion, B=2000, seed=0):
    """Bootstrap CI over prompts of the paired difference."""
    b = np.array([x == emotion for x in base_labels], float); s = np.array([x == emotion for x in steered_labels], float)
    rng = np.random.default_rng(seed); d = []
    for _ in range(B):
        i = rng.integers(0, len(b), len(b)); d.append(s[i].mean() - b[i].mean())
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def boot_diff(base, a_labels, b_labels, emotion, B=2000, seed=0):
    """Paired bootstrap of Delta(a) - Delta(b) (same prompts, same baseline)."""
    be = np.array([x == emotion for x in base], float); a = np.array([x == emotion for x in a_labels], float); b = np.array([x == emotion for x in b_labels], float)
    rng = np.random.default_rng(seed); d = []
    for _ in range(B):
        i = rng.integers(0, len(be), len(be)); d.append((a[i].mean() - be[i].mean()) - (b[i].mean() - be[i].mean()))
    return float((a.mean() - be.mean()) - (b.mean() - be.mean())), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def analyze(gens, cond="probe"):
    """gens: list of dicts {cond: 'base'|'probe'|'v1'|'random', emotion|None, rid|None, strength, pid, label, ppl_ratio}. Returns a results dict."""
    by = {}
    for g in gens:
        by.setdefault((g["cond"], g.get("emotion"), g.get("rid"), g["strength"]), {})[g["pid"]] = g
    pids = sorted(next(v for k, v in by.items() if k[0] == "base"))
    base = [by[("base", None, None, 0.0)][p]["label"] for p in pids]
    strengths = sorted({k[3] for k in by if k[0] == cond})
    rids = sorted({k[2] for k in by if k[0] == "random"})
    res = {"strengths": strengths, "emotions": {}, "n_prompts": len(pids), "n_random": len(rids)}
    for s in strengths:
        for e in EMOTIONS:
            cell = by.get((cond, e, None, s))
            if not cell:
                continue
            lab = [cell[p]["label"] for p in pids]
            d = delta(base, lab, e); lo, hi = boot_delta(base, lab, e)
            ratios = [cell[p]["ppl_ratio"] for p in pids if cell[p].get("ppl_ratio") is not None]
            med = float(np.median(ratios)) if ratios else float("nan")
            rand = [delta(base, [by[("random", None, r, s)][p]["label"] for p in pids], e) for r in rids if ("random", None, r, s) in by]
            v1c = by.get(("v1", e, None, s)); v1 = None
            if v1c and cond == "probe":
                v1 = boot_diff(base, lab, [v1c[p]["label"] for p in pids], e)
            off = {l: delta(base, lab, l) for l in LABELS}
            res["emotions"].setdefault(e, {})[s] = {
                "delta": d, "ci": (lo, hi), "median_ppl_ratio": med, "fluent": bool(med <= PPL_LIMIT),
                "random_max": max(rand) if rand else None, "random_deltas": rand,
                "beats_all_random": bool(rand and d > max(rand)), "ci_excludes_0": bool(lo > 0),
                "specific": bool(rand and d > max(rand) and lo > 0 and med <= PPL_LIMIT), "probe_minus_v1": v1, "off_target": off}
    steerable = [e for e, m in res["emotions"].items() if any(c["specific"] for c in m.values())]
    res["steerable"] = steerable
    res["claim"] = len(steerable) >= 4
    return res


def render(res, title="T10 steering results (pre-registered in PLAN_v2"):
    ss = res["strengths"]
    L = [f"# {title}; {res['n_prompts']} prompts, {res['n_random']} random directions)", "",
         "Delta = change in the judge's rate of the target emotion vs the unsteered text of the same prompts; * = bootstrap 95% CI over prompts excludes 0; "
         f"fluent = median perplexity ratio <= {PPL_LIMIT}; specific = fluent AND Delta above ALL random directions AND CI excludes 0.", "",
         "| emotion | strength | Delta [95% CI] | random max | ppl ratio | fluent | specific | probe - v1 [CI] |", "|---|---|---|---|---|---|---|---|"]
    for e, m in res["emotions"].items():
        for s in ss:
            c = m.get(s)
            if not c:
                continue
            v1 = c["probe_minus_v1"]
            L.append(f"| {e} | {s:.3g} | {c['delta']:+.3f} [{c['ci'][0]:+.3f}, {c['ci'][1]:+.3f}]{'*' if c['ci_excludes_0'] else ''} | "
                     f"{c['random_max'] if c['random_max'] is None else format(c['random_max'], '+.3f')} | {c['median_ppl_ratio']:.2f} | {'yes' if c['fluent'] else 'no'} | "
                     f"{'**yes**' if c['specific'] else 'no'} | " + (f"{v1[0]:+.3f} [{v1[1]:+.3f}, {v1[2]:+.3f}]" if v1 else "-") + " |")
    L += ["", f"Emotions that are specifically steerable at >= 1 fluent strength: {res['steerable'] or 'none'} ({len(res['steerable'])}/7; claim needs >= 4).",
          "**Pre-registered conclusion: " + ("probe directions shift expressed emotion beyond random-direction controls." if res["claim"] else "NOT established that probe directions shift expressed emotion beyond random controls.") + "**", ""]
    return "\n".join(L)


# ---------------------------------------------------------------- model side (needs MLX)
def make_wrapper():
    import mlx.core as mx
    import mlx.nn as nn

    class SteeredBlock(nn.Module):
        def __init__(self, block, dim):
            super().__init__()
            self.block = block; self.vec = mx.zeros((dim,)); self.on = False; self.capture = False; self.last = None
            self.use_sliding = getattr(block, "use_sliding", False)      # mlx-lm's Llama reads this attribute from every layer

        def __call__(self, x, mask=None, cache=None):
            out = self.block(x, mask, cache)
            if self.on:
                out = out + self.vec.astype(out.dtype)
            if self.capture:
                self.last = out
            return out
    return SteeredBlock


def install(model, layer=LAYER):
    import mlx.core as mx
    SteeredBlock = make_wrapper()
    blocks = model.model.layers
    if not isinstance(blocks[layer], SteeredBlock):
        blocks[layer] = SteeredBlock(blocks[layer], model.model.args.hidden_size)
    return blocks[layer]


def set_vec(sb, vec):
    import mlx.core as mx
    if vec is None:
        sb.on = False
    else:
        sb.vec = mx.array(np.asarray(vec, dtype=np.float32)); sb.on = True


def chat_prompt(tok, text):
    return tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False)


def gen(model, tok, sb, vec, text):
    from mlx_lm import generate
    set_vec(sb, vec)
    ids = tok.encode(chat_prompt(tok, text), add_special_tokens=False)          # the template already carries BOS; same ids as unsteered_nll
    out = generate(model, tok, prompt=ids, max_tokens=MAX_NEW, verbose=False)
    set_vec(sb, None)
    return out.strip()


def unsteered_nll(model, tok, sb, prompt, text):
    """Mean negative log-likelihood per generated token under the UNSTEERED model."""
    import mlx.core as mx
    set_vec(sb, None)
    p = tok.encode(chat_prompt(tok, prompt), add_special_tokens=False); g = tok.encode(text, add_special_tokens=False)
    if not g:
        return float("nan")
    ids = mx.array([p + g]); logits = model(ids)[0].astype(mx.float32)
    lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    tgt = mx.array(p[1:] + g); sel = mx.take_along_axis(lp[:-1], tgt[:, None], axis=1)[:, 0]
    mx.eval(sel)
    return float(-np.array(sel)[len(p) - 1:].mean())


def load_dirs():
    d = np.load(DIR / "directions.npz", allow_pickle=False)
    return {"probe": d["probe"], "v1": d["v1"], "random": d["random"], "mean_norm": float(d["mean_norm"])}


def cmd_prepare(a):
    from probekit_capture import Extractor
    from probekit_estimators import fit_directions
    from probekit_eval_v2 import load_jsonl
    from probekit_eval_v3 import load_states
    from probekit_frozen_eval import cfg_for
    DIR.mkdir(parents=True, exist_ok=True)
    ev = Path(a.eval_dir); cache = Path(a.cache_dir)
    best = json.loads((ev / "best_config.json").read_text()); assert best["layer"] == LAYER, "the pre-registration fixes layer 16"
    rows = load_jsonl(a.corpus)
    Xc = load_states(rows, cache / "states_corpus.npz", [LAYER], best["skip"], a.model)[LAYER]
    cat = np.array([r["category"] for r in rows]); spl = np.array([r["split"] for r in rows]); tr = (spl == "train") & (cat != "calm")
    ps = fit_directions(Xc[tr], cat[tr], EMOTIONS, LAYER, method="lda", **cfg_for(ev, "lda", LAYER))
    probe = np.stack([ps.V[ps.emotions.index(e)] for e in EMOTIONS])
    ex = Extractor(a.model)
    v1rows = load_jsonl("data/corpus/corpus.jsonl")
    X1 = np.stack([ex.pooled(r["text"], [LAYER], best["skip"])[LAYER] for r in v1rows])
    c1 = np.array([r["category"] for r in v1rows])
    ps1 = fit_directions(X1, c1, [e for e in EMOTIONS], LAYER, method="meandiff", var_frac=0.0)
    v1 = np.stack([ps1.V[ps1.emotions.index(e)] for e in EMOTIONS])
    neutral = [r["text"] for r in rows if r["category"] == "neutral" and r["split"] == "train"][:120]
    norms = np.concatenate([np.linalg.norm(ex.states(t, [LAYER])[LAYER][best["skip"]:], axis=1) for t in neutral])
    rnd = np.random.default_rng(0).standard_normal((N_RANDOM, probe.shape[1])); rnd /= np.linalg.norm(rnd, axis=1, keepdims=True)
    unit = lambda M: M / np.linalg.norm(M, axis=1, keepdims=True)
    np.savez(DIR / "directions.npz", probe=unit(probe), v1=unit(v1), random=rnd, mean_norm=float(norms.mean()), emotions=np.array(EMOTIONS))
    cos = (unit(probe) * unit(v1)).sum(1)
    print(f"mean token norm at layer {LAYER}: {norms.mean():.2f}; cos(probe, v1) per emotion: {dict(zip(EMOTIONS, np.round(cos, 2)))}; "
          f"mean |cos(probe, random)| {np.abs(unit(probe) @ rnd.T).mean():.3f}")


def cmd_selftest(a):
    import mlx.core as mx
    from probekit_capture import Extractor
    ex = Extractor(a.model); sb = install(ex.model); d = load_dirs()
    ids = mx.array([ex.encode("The bus arrived at noon and everyone got on.")]); sb.capture = True
    set_vec(sb, None); ex.model(ids); base = np.array(sb.last.astype(mx.float32))
    v = 0.4 * d["mean_norm"] * d["probe"][0]; set_vec(sb, v); ex.model(ids); steered = np.array(sb.last.astype(mx.float32)); set_vec(sb, None)
    diff = (steered - base)[0]                                          # (T, H); position 0 (BOS) carries massive activations, so it is excluded
    d1 = diff[1:]; u = v / np.linalg.norm(v)
    proj = (d1 @ u) / np.linalg.norm(v)                                 # should be ~1 at every position: the vector is added unbiased
    rel = np.linalg.norm(d1 - v[None, :], axis=1) / np.linalg.norm(v)  # norm-wise error; residuals are 16-bit, so rounding error is expected
    print(f"selftest: projection of (steered - base) on the added vector / its norm: mean {proj.mean():.3f} (min {proj.min():.3f}, max {proj.max():.3f}); "
          f"norm-wise relative error mean {rel.mean():.3f} (16-bit rounding)  ->  {'PASS' if 0.9 <= proj.mean() <= 1.1 and rel.mean() < 0.5 else 'FAIL'}")
    out = gen(ex.model, ex.tok, sb, None, "Write three sentences about a quiet library.")
    print("unsteered sample:", out[:120].replace("\n", " "))


def cmd_pilot(a):
    from probekit_capture import Extractor
    ex = Extractor(a.model); sb = install(ex.model); d = load_dirs(); _, pilot = topics()
    dirs = {"joy": d["probe"][EMOTIONS.index("joy")], "rand0": d["random"][0], "rand1": d["random"][1], "rand2": d["random"][2]}
    base = {t: gen(ex.model, ex.tok, sb, None, prompt_text(t)) for t in pilot}
    base_nll = {t: unsteered_nll(ex.model, ex.tok, sb, prompt_text(t), base[t]) for t in pilot}
    table = {}
    for s in GRID:
        for name, v in dirs.items():
            ratios = []
            for t in pilot:
                txt = gen(ex.model, ex.tok, sb, s * d["mean_norm"] * v, prompt_text(t))
                nll = unsteered_nll(ex.model, ex.tok, sb, prompt_text(t), txt)
                ratios.append(float(np.exp(nll - base_nll[t])))
            table[f"{s}|{name}"] = float(np.median(ratios)); print(f"  s={s:<5} {name:<6} median ppl ratio {table[f'{s}|{name}']:.2f}", flush=True)
    ok = [s for s in GRID if all(table[f"{s}|{n}"] <= PPL_LIMIT for n in dirs)]
    sstar = max(ok) if ok else GRID[0]
    out = {"S_star": sstar, "strengths": [sstar / 4, sstar / 2, sstar], "table": table, "mean_norm": d["mean_norm"], "grid": GRID}
    (DIR / "pilot.json").write_text(json.dumps(out, indent=1)); print("S* =", sstar, "-> strengths", out["strengths"])


def cmd_generate(a):
    from probekit_capture import Extractor
    ex = Extractor(a.model); sb = install(ex.model); d = load_dirs(); main, _ = topics()
    st = json.loads((DIR / "pilot.json").read_text())["strengths"]; path = DIR / "generations.jsonl"
    done = {json.loads(l)["gid"] for l in path.open()} if path.exists() else set()
    conds = [("base", None, None, 0.0, None)]
    for s in st:
        conds += [("probe", e, None, s, d["probe"][i]) for i, e in enumerate(EMOTIONS)] + [("v1", e, None, s, d["v1"][i]) for i, e in enumerate(EMOTIONS)]
        conds += [("random", None, r, s, d["random"][r]) for r in range(N_RANDOM)]
    f = path.open("a"); base_nll = {}; n = 0; t0 = time.time()
    for cond, e, r, s, v in conds:
        for pid, t in enumerate(main):
            gid = f"{cond}|{e}|{r}|{s}|{pid}"
            if gid in done:
                continue
            txt = gen(ex.model, ex.tok, sb, None if v is None else s * d["mean_norm"] * v, prompt_text(t))
            nll = unsteered_nll(ex.model, ex.tok, sb, prompt_text(t), txt)
            if cond == "base":
                base_nll[pid] = nll
            f.write(json.dumps({"gid": gid, "cond": cond, "emotion": e, "rid": r, "strength": s, "pid": pid, "topic": t, "text": txt, "nll": nll}) + "\n"); f.flush(); n += 1
            if n % 100 == 0:
                print(f"  {n} generated ({time.time() - t0:.0f}s)", flush=True)
    print("generation done:", n, "new")


def cmd_judge(a):
    from probekit_judge import judge_texts
    src = [json.loads(l) for l in (DIR / "generations.jsonl").open()]; path = DIR / "judged.jsonl"
    done = {json.loads(l)["gid"] for l in path.open()} if path.exists() else set()
    todo = [g for g in src if g["gid"] not in done]
    f = path.open("a")
    for i in range(0, len(todo), 200):
        chunk = todo[i:i + 200]; labels = judge_texts([g["text"] for g in chunk])
        for g, lab in zip(chunk, labels):
            f.write(json.dumps({"gid": g["gid"], "label": lab or "unparsed"}) + "\n")
        f.flush(); print(f"  judged {min(i + 200, len(todo))}/{len(todo)}", flush=True)


def cmd_analyze(a):
    gens = {json.loads(l)["gid"]: json.loads(l) for l in (DIR / "generations.jsonl").open()}
    lab = {json.loads(l)["gid"]: json.loads(l)["label"] for l in (DIR / "judged.jsonl").open()}
    base_nll = {g["pid"]: g["nll"] for g in gens.values() if g["cond"] == "base"}
    rows = []
    for gid, g in gens.items():
        if gid in lab:
            rows.append({**{k: g[k] for k in ("cond", "emotion", "rid", "strength", "pid")}, "label": lab[gid],
                         "ppl_ratio": float(np.exp(g["nll"] - base_nll[g["pid"]])) if g["pid"] in base_nll and np.isfinite(g["nll"]) else None})
    res = analyze(rows)
    unparsed = sum(r["label"] == "unparsed" for r in rows)
    md = render(res) + f"\nJudge: granite4.2:8b; unparsed labels {unparsed}/{len(rows)}; see judge_validation.md for the judge's accuracy on human text.\n"
    # POST HOC (not pre-registered): the same rule applied to the v1-style directions, and where the judge failed to produce a label
    v1 = analyze(rows, cond="v1")
    md += "\n" + render(v1, "POST HOC (not pre-registered): the same rule applied to the v1-style mean-difference directions (PLAN_v2 T10 (ii)") + "\n"
    ur = {}
    for r in rows:
        k = (r["cond"], r["strength"]); ur.setdefault(k, [0, 0]); ur[k][0] += r["label"] == "unparsed"; ur[k][1] += 1
    md += "Unparsed judge labels by condition (post hoc): " + "; ".join(f"{c}@{s:g}: {u}/{n}" for (c, s), (u, n) in sorted(ur.items())) + "\n"
    base = [r["label"] for r in rows if r["cond"] == "base"]
    md += "Unsteered label distribution: " + ", ".join(f"{l} {base.count(l)}" for l in sorted(set(base))) + "\n"
    Path(a.out).write_text(md); Path(a.out).with_suffix(".json").write_text(json.dumps(res, indent=1, default=float)); print(md)


def main():
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    for name in ("prepare", "selftest", "pilot", "generate", "judge", "analyze"):
        p = sp.add_parser(name); p.add_argument("--model", default="mlx-community/Llama-3.2-3B-Instruct")
        if name == "prepare":
            p.add_argument("--cache-dir", default="data/eval_merged2/mlx-community__Llama-3.2-3B-Instruct")
            p.add_argument("--eval-dir", default="data/eval_merged2/mlx-community__Llama-3.2-3B-Instruct/v3")
            p.add_argument("--corpus", default="data/corpus/corpus_merged2_scored.jsonl")
        if name == "analyze":
            p.add_argument("--out", default="data/eval_merged2/mlx-community__Llama-3.2-3B-Instruct/steering_results.md")
    a = ap.parse_args()
    {"prepare": cmd_prepare, "selftest": cmd_selftest, "pilot": cmd_pilot, "generate": cmd_generate, "judge": cmd_judge, "analyze": cmd_analyze}[a.cmd](a)


if __name__ == "__main__":
    main()
