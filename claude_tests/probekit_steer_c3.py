#!/usr/bin/env python3
"""
probekit_steer_c3.py -- PLAN_v2 T14/C3 (pre-registered 2026-11-22, f4cabfe): mean-difference directions fitted on the SAME data as the LDA directions (Gemma-2-2B, layer 14).
Wrapper over probekit_steer_c1.py. The saved LDA, random and base generations of T12 are reused; the old 'v1' arm is replaced by 'md-same' (stored under the type key 'v1').

    python probekit_steer_c3.py setup | selftest | pilot | generate | judge | analyze
Outputs in data/steering_gemma_md/ (jsonl/npz git-ignored); report data/eval_merged2/<gemma slug>/steering_c3_results.md.
"""
import argparse
import json
from pathlib import Path

import numpy as np

import probekit_steer as S
import probekit_steer_c1 as C

DIR = Path("data/steering_gemma_md")
SRC = Path("data/steering_gemma")
SLUG = C.MODEL.replace("/", "__")


def configure():
    C.DIR = DIR; C.configure()


def fit_md_same(corpus="data/corpus/corpus_merged2_scored.jsonl"):
    """Unit mean-difference directions (var_frac 0) per emotion on the merged2 TRAIN states used for the LDA probe."""
    from probekit_estimators import fit_directions
    from probekit_eval_v2 import load_jsonl
    cache = np.load(f"data/eval_merged2/{SLUG}/states_corpus.npz", allow_pickle=False)
    rows = load_jsonl(corpus); assert list(cache["ids"]) == [r["id"] for r in rows]
    X = cache[f"L{C.LAYER}"]; cat = np.array([r["category"] for r in rows]); spl = np.array([r["split"] for r in rows]); tr = (spl == "train") & (cat != "calm")
    ps = fit_directions(X[tr], cat[tr], S.EMOTIONS, C.LAYER, method="meandiff", var_frac=0.0)
    V = np.stack([ps.V[ps.emotions.index(e)] for e in S.EMOTIONS]); return V / np.linalg.norm(V, axis=1, keepdims=True)


def cmd_setup(a):
    configure(); d = np.load(SRC / "directions.npz", allow_pickle=False); md = fit_md_same()
    unit = lambda M: M / np.linalg.norm(M, axis=1, keepdims=True)
    cos_lda = (unit(d["probe"]) * md).sum(1); cos_v1 = (unit(d["v1"]) * md).sum(1)
    np.savez(DIR / "directions.npz", probe=d["probe"], v1=md, random=d["random"], mean_norm=float(d["mean_norm"]), emotions=d["emotions"])
    for name in ("generations.jsonl", "judged.jsonl"):
        keep = [l for l in (SRC / name).open() if json.loads(l)["gid"].split("|")[0] != "v1"]
        (DIR / name).write_text("".join(keep)); print(name, "kept", len(keep))
    (DIR / "cosines.json").write_text(json.dumps({"cos_md_same_vs_lda": dict(zip(S.EMOTIONS, map(float, cos_lda))), "cos_md_same_vs_v1": dict(zip(S.EMOTIONS, map(float, cos_v1)))}, indent=1))
    print("cos(md-same, LDA):", dict(zip(S.EMOTIONS, np.round(cos_lda, 2)))); print("cos(md-same, v1):", dict(zip(S.EMOTIONS, np.round(cos_v1, 2))))


def cmd_analyze(a):
    configure(); C.cmd_analyze(a)
    p = Path(a.out); body = p.read_text(); cs = json.loads((DIR / "cosines.json").read_text())
    head = ("> **T14/C3 (pre-registered f4cabfe).** In this report the type labelled 'mean-difference' / 'v1' / 'H7b' is **md-same**: mean-difference directions fitted on the merged2 TRAIN states "
            "that the LDA directions were fitted on (not the v1 corpus). H7a/H7b lines below are the H9 contrast. cos(md-same, LDA): " + ", ".join(f"{e} {v:.2f}" for e, v in cs["cos_md_same_vs_lda"].items()) + ".\n\n")
    p.write_text(head + body)


def main():
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    for n in ("setup", "selftest", "pilot", "generate", "judge", "analyze"):
        p = sp.add_parser(n); p.add_argument("--model", default=C.MODEL)
        if n == "analyze":
            p.add_argument("--out", default=f"data/eval_merged2/{SLUG}/steering_c3_results.md")
    a = ap.parse_args()
    {"setup": cmd_setup, "selftest": lambda a: (configure(), C.cmd_selftest(a)), "pilot": lambda a: (configure(), C.cmd_pilot(a)), "generate": lambda a: (configure(), C.cmd_generate(a)),
     "judge": lambda a: (configure(), S.cmd_judge(a)), "analyze": cmd_analyze}[a.cmd](a)


if __name__ == "__main__":
    main()
