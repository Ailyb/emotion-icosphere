#!/usr/bin/env python3
"""
probekit_t11.py -- PLAN_v2 T11 (pre-registered in commit 0d74f80 before any of these runs):
  H3 frozen sentence-encoder baseline, H4 NRC EmoLex zero-training baseline, H5 human-neutral threshold, H6 training-seed robustness.

    python probekit_t11.py run --cache-dir data/eval_merged2/<slug> --eval-dir data/eval_merged2/<slug>/v3 \
        --corpus data/corpus/corpus_merged2_scored.jsonl --model <hf id> --out data/eval_merged2/<slug>/t11.md
    python probekit_t11.py summary --dirs data/eval_merged2/<slug1> ... --out data/eval_merged2/t11_summary.md

Training set for H3/H4/H6 is HUM exactly as in probekit_external_eval (LLM-written rows + human GoEmotions-train pool);
predictions are restricted to the classes present in each test source. The frozen GoEmotions set is evaluated the same way
(all 8 classes). NRC EmoLex is read from data/external/nrc (non-commercial research use; never committed).
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np

from probekit_estimators import calibrate_pooled, fit_directions
from probekit_eval_v2 import f1_9, load_jsonl, pick_tau, probe_predict
from probekit_eval_v3 import paired_bootstrap
from probekit_external_eval import SOURCES, argmax_classes, columns, fit_lr, fit_systems
from probekit_frozen_eval import EMOTIONS, EXTRACT_LAYERS, cfg_for

ENCODER = "sentence-transformers/all-mpnet-base-v2"
NRC = "data/external/nrc/NRC-Emotion-Lexicon/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"
LEX_EMOTIONS = ["joy", "sadness", "fear", "anger", "surprise", "disgust"]       # no 'love' entry exists in EmoLex
ENC_CACHE = Path("data/eval_merged2/encoder_cache.npz")
SEEDS = [0, 1, 2, 3, 4]
SETS = ("frozen",) + SOURCES


# ---------------------------------------------------------------- H3: frozen encoder
def _key(t):
    return hashlib.sha1(t.encode("utf-8")).hexdigest()


def encoder_meta():
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(ENCODER)
    rev = getattr(m[0].auto_model.config, "_commit_hash", None)
    return m, {"name": ENCODER, "revision": rev, "parameters": int(sum(p.numel() for p in m.parameters())),
               "embedding_dim": int(m.get_sentence_embedding_dimension())}


def encode(texts, meta_out=None):
    """Frozen encoder embeddings, cached by text hash in a shared (gitignored) npz; model-independent."""
    keys = [_key(t) for t in texts]
    cache = {}
    if ENC_CACHE.exists():
        z = np.load(ENC_CACHE, allow_pickle=False); cache = dict(zip(z["keys"].tolist(), z["E"]))
    miss = [i for i, k in enumerate(keys) if k not in cache]
    model, meta = encoder_meta()
    if meta_out is not None:
        meta_out.update(meta)
    if miss:
        uniq = {}
        for i in miss:
            uniq.setdefault(keys[i], texts[i])
        E = model.encode(list(uniq.values()), batch_size=64, show_progress_bar=False, convert_to_numpy=True)
        cache.update(dict(zip(uniq.keys(), E)))
        ks = list(cache); np.savez(ENC_CACHE, keys=np.array(ks), E=np.stack([cache[k] for k in ks]).astype(np.float32))
    return np.stack([cache[k] for k in keys])


# ---------------------------------------------------------------- H4: EmoLex
def load_lexicon(path=NRC):
    lex = {e: set() for e in LEX_EMOTIONS}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        p = line.split("\t")
        if len(p) == 3 and p[1] in lex and p[2].strip() == "1":
            lex[p[1]].add(p[0])
    return lex


def toks(t):
    return re.findall(r"[a-z]+(?:'[a-z]+)?", t.lower().replace("’", "'"))


def emolex_predict(texts, lex, classes):
    """Most word hits among the source's classes (ties: fixed order joy, sadness, fear, anger, surprise, disgust); 'neutral' if no hits.
    A source without a neutral class still gets 'neutral' for hit-less texts (always wrong); a class without a lexicon entry is never predicted."""
    emo = [e for e in LEX_EMOTIONS if e in classes]
    out = []
    for t in texts:
        w = toks(t); h = [sum(x in lex[e] for x in w) for e in emo]
        out.append("neutral" if not emo or max(h) == 0 else emo[int(np.argmax(h))])
    return np.array(out, dtype=object)


# ---------------------------------------------------------------- setup (same data path as probekit_external_eval)
def setup(a):
    from probekit_eval_v3 import load_states
    from probekit_external import load_source
    from probekit_frozen_eval import extract_cached, pool_new_rows
    from probekit_l0_baseline import load_embed
    from probekit_testset import load_frozen
    eval_dir, cache = Path(a.eval_dir), Path(a.cache_dir)
    best = json.loads((eval_dir / "best_config.json").read_text())
    layer, skip, method = best["layer"], best["skip"], best["method"]
    cfg = cfg_for(eval_dir, method, layer)
    rows = load_jsonl(a.corpus); pool = load_jsonl(a.pool)
    Xc = load_states(rows, cache / "states_corpus.npz", [layer], skip, a.model)[layer]
    Ec = load_embed(rows, cache / "states_embed_corpus.npz", skip, a.model)
    new = pool_new_rows(pool, rows)
    Xn = extract_cached(new, cache / "states_pool_new.npz", EXTRACT_LAYERS, skip, a.model)
    cat = np.array([r["category"] for r in rows]); spl = np.array([r["split"] for r in rows])
    wr = np.array([r.get("writer", "llama-3.2-3b") for r in rows]); hard = np.array([bool(r.get("hard", True)) for r in rows])
    txt = np.array([r["text"] for r in rows], dtype=object)
    keep = cat != "calm"; human = wr == "reddit-goemotions"
    c = dict(layer=layer, skip=skip, method=method, cfg=cfg, rows=rows, Xc=Xc, cat=cat, txt=txt, keep=keep, human=human,
             tr_all=(spl == "train") & keep, llm=(spl == "train") & keep & ~human, va=(spl == "val") & hard & keep,
             cal=(spl == "val") & (cat == "neutral"))
    byid = {r["id"]: i for i, r in enumerate(rows)}; nid = {r["id"]: i for i, r in enumerate(new)}
    pr = [r for r in pool if r["split"] == "train"]
    c["Xp"] = np.stack([Xc[byid[r["id"]]] if r["id"] in byid else Xn[layer][nid[r["id"]]] for r in pr])
    c["Ec"] = Ec; c["Ep"] = np.stack([Ec[byid[r["id"]]] if r["id"] in byid else Xn["E"][nid[r["id"]]] for r in pr])
    c["yp"] = np.array([r["category"] for r in pr], dtype=object); c["Tp"] = np.array([r["text"] for r in pr], dtype=object)
    sets = {}
    fr, _ = load_frozen("data/testsets/goemotions_test_frozen.manifest.json")
    items = {"frozen": fr}
    for s in SOURCES:
        items[s] = load_source(s, a.testsets)[0]
    for name, it in items.items():
        Xe = extract_cached(it, cache / ("states_frozen.npz" if name == "frozen" else f"states_ext_{name}.npz"), EXTRACT_LAYERS, skip, a.model)
        sets[name] = dict(X=Xe[layer], E=Xe["E"], y=np.array([r["category"] for r in it], dtype=object), T=np.array([r["text"] for r in it], dtype=object))
    c["sets"] = sets
    cal_items = load_jsonl(a.calib)
    c["calib"] = dict(ids=[r["id"] for r in cal_items], X=extract_cached(cal_items, cache / "states_calib.npz", EXTRACT_LAYERS, skip, a.model)[layer],
                      n_words=np.array([r["n_words"] for r in cal_items]))
    return c


def hum_training(c, llm_idx=None, pool_idx=None):
    """HUM training arrays (LLM-written rows + human pool rows), optionally subsampled by index."""
    li = np.where(c["llm"])[0] if llm_idx is None else llm_idx
    pi = np.arange(len(c["yp"])) if pool_idx is None else pool_idx
    return (np.vstack([c["Xc"][li], c["Xp"][pi]]), np.concatenate([c["txt"][li], c["Tp"][pi]]), np.concatenate([c["cat"][li], c["yp"][pi]]))


def encoder_predictor(c, T_h, y_h, meta=None):
    Etr = encode(list(T_h), meta); Eva = encode(list(c["txt"][c["va"]]))
    sc, lr = fit_lr(Etr, y_h, Eva, c["cat"][c["va"]])
    def pred(T, classes):
        return argmax_classes(columns(lr, lr.predict_proba(sc.transform(encode(list(T)))), classes), classes)
    return pred


def fit_all(c, X_h, T_h, y_h, meta=None):
    """probe, native LR, TF-IDF (restricted predictors from fit_systems) + frozen encoder, all on one training set."""
    va, cal = c["va"], c["cal"]
    systems, tau = fit_systems(X_h, None, T_h, y_h, c["Xc"][va], None, c["cat"][va], c["Xc"][cal], c["layer"], c["cfg"], c["method"])
    enc = encoder_predictor(c, T_h, y_h, meta)
    out = {k: v for k, v in systems.items() if k in ("probe", "native_lr", "tfidf")}
    return out, enc, tau


def predict_sets(c, systems, enc, lex=None):
    P = {}
    for name, s in c["sets"].items():
        classes = sorted(set(s["y"]))
        p = {k: f(s["X"], None, s["T"], classes) for k, f in systems.items()}
        p["encoder"] = enc(s["T"], classes)
        if lex is not None:
            p["emolex"] = emolex_predict(s["T"], lex, classes)
        P[name] = p
    return P


# ---------------------------------------------------------------- H5
def parity_split(ids):
    """'Split by id parity' with string ids: sorted-id index parity (even index fits tau_h, odd index checks it)."""
    order = np.argsort(np.array(ids)); even = np.zeros(len(ids), bool); even[order[0::2]] = True
    return even


def h5(c):
    tr, va, cal = c["tr_all"], c["va"], c["cal"]
    ps = fit_directions(c["Xc"][tr], c["cat"][tr], EMOTIONS, c["layer"], method=c["method"], **c["cfg"])
    calibrate_pooled(ps, c["Xc"][cal]); tau = pick_tau(ps, c["Xc"][va], c["cat"][va])
    zk = ps.score(c["calib"]["X"], "pooled").max(1); even = parity_split(c["calib"]["ids"])
    tau_h = float(np.percentile(zk[even], 90))
    f = c["sets"]["frozen"]; y = f["y"]
    p0, p1 = probe_predict(ps, f["X"], tau), probe_predict(ps, f["X"], tau_h)
    m, lo, hi = paired_bootstrap(y, p1, p0)
    rec_orig, rec_h = float((zk[~even] < tau).mean()), float((zk[~even] < tau_h).mean())
    neu = y == "neutral"
    r = dict(tau=float(tau), tau_h=tau_h, n_fit=int(even.sum()), n_check=int((~even).sum()),
             recall_odd_orig=rec_orig, recall_odd_tauh=rec_h, f1_orig=f1_9(y, p0), f1_tauh=f1_9(y, p1), diff=[m, lo, hi],
             frozen_neutral_recall_orig=float((p0[neu] == "neutral").mean()), frozen_neutral_recall_tauh=float((p1[neu] == "neutral").mean()))
    r["fixes"] = bool(abs(rec_h - 0.90) <= 0.05 and lo > -0.01)
    return r


# ---------------------------------------------------------------- H6
def h6(c, lex_unused=None):
    out = []
    li_all = np.where(c["llm"])[0]; classes_p = sorted(set(c["yp"]))
    for s in SEEDS:
        rng = np.random.default_rng(s)
        li = rng.choice(li_all, int(round(0.8 * len(li_all))), replace=False)
        pi = np.concatenate([rng.choice(np.where(c["yp"] == k)[0], int(round(0.8 * (c["yp"] == k).sum())), replace=False) for k in classes_p])
        X_h, T_h, y_h = hum_training(c, li, pi)
        systems, enc, tau = fit_all(c, X_h, T_h, y_h)
        P = predict_sets(c, systems, enc)
        out.append({"seed": s, "n": int(len(y_h)), "tau": float(tau),
                    "f1": {name: {k: f1_9(c["sets"][name]["y"], p) for k, p in P[name].items()} for name in P}})
        print(f"seed {s}: " + ", ".join(f"{n} probe-tfidf {out[-1]['f1'][n]['probe'] - out[-1]['f1'][n]['tfidf']:+.3f}" for n in SETS), flush=True)
    return out


# ---------------------------------------------------------------- report
def fmt(m, lo, hi):
    return f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]{'*' if lo > 0 or hi < 0 else ''}"


def run(a):
    c = setup(a); lex = load_lexicon(); meta = {}
    X_h, T_h, y_h = hum_training(c)
    systems, enc, tau = fit_all(c, X_h, T_h, y_h, meta)
    P = predict_sets(c, systems, enc, lex)
    res = {"layer": c["layer"], "method": c["method"], "tau_hum": float(tau), "n_train": int(len(y_h)), "encoder": meta, "sets": {}}
    L = [f"# T11 results ({a.model}; layer {c['layer']}, {c['method']}; pre-registered in PLAN_v2, commit 0d74f80)", "",
         f"Training HUM n={len(y_h)} (tau {tau:.2f}); encoder {meta['name']} revision {meta['revision']}, {meta['parameters'] / 1e6:.1f}M parameters, "
         f"dim {meta['embedding_dim']} (frozen, LR on top, C on VAL, same recipe as native LR); EmoLex = NRC v0.92 word hits, no training.", "",
         "## H3 / H4: macro-F1 [95% CI] and paired differences", "",
         "| set | n | probe | TF-IDF | native LR | encoder | EmoLex | probe - encoder (H3) | probe - EmoLex (H4) | probe - TF-IDF |", "|---|---|---|---|---|---|---|---|---|---|"]
    from probekit_frozen_eval import boot_ci
    for name in SETS:
        y = c["sets"][name]["y"]; p = P[name]; cells = {}
        for k in ("probe", "tfidf", "native_lr", "encoder", "emolex"):
            lo, hi = boot_ci(y, p[k]); cells[k] = f"{f1_9(y, p[k]):.3f} [{lo:.3f}, {hi:.3f}]"
        d = {k: paired_bootstrap(y, p["probe"], p[k]) for k in ("encoder", "emolex", "tfidf")}
        res["sets"][name] = {"n": int(len(y)), "f1": {k: f1_9(y, v) for k, v in p.items()}, "probe_minus": {k: list(v) for k, v in d.items()}}
        L.append(f"| {name} | {len(y)} | {cells['probe']} | {cells['tfidf']} | {cells['native_lr']} | {cells['encoder']} | {cells['emolex']} | "
                 f"{fmt(*d['encoder'])} | {fmt(*d['emolex'])} | {fmt(*d['tfidf'])} |")
    ext_wins = [s for s in SOURCES if res["sets"][s]["probe_minus"]["encoder"][1] > 0]
    res["h3_model_pass"] = len(ext_wins) >= 2
    L += ["", f"H3 for this model: probe - encoder has a CI above 0 on {len(ext_wins)}/3 external sources {ext_wins} -> "
              f"{'meets' if res['h3_model_pass'] else 'does not meet'} the per-model condition (>= 2 of 3). Frozen GoEmotions is reported without a rule.", ""]
    L += ["## H5: human-neutral threshold", ""]
    r = h5(c); res["h5"] = r
    L += [f"Reserve = {r['n_fit'] + r['n_check']} human neutral passages (calibration A+B, sorted-id parity): even index (n={r['n_fit']}) fits "
          f"tau_h = 90th percentile of max z = {r['tau_h']:.2f}; odd index (n={r['n_check']}) checks recall. Original tau (LLM-validation) = {r['tau']:.2f}.", "",
          f"- neutral recall on the held-out odd half: original tau {r['recall_odd_orig']:.3f}; tau_h {r['recall_odd_tauh']:.3f} (target 0.90 +/- 0.05)",
          f"- frozen-set neutral recall: original {r['frozen_neutral_recall_orig']:.3f}; tau_h {r['frozen_neutral_recall_tauh']:.3f}",
          f"- frozen 8-way macro-F1: original {r['f1_orig']:.3f}; tau_h {r['f1_tauh']:.3f}; difference (tau_h - original) {fmt(*r['diff'])}",
          f"- rule (recall within 0.05 of 0.90 AND difference CI above -0.01): **{'tau_h fixes neutral detection' if r['fixes'] else 'tau_h does not meet the rule'}**", ""]
    L += ["## H6: training-seed robustness (5 seeds; 80% subsamples of LLM rows and of each human-pool class)", ""]
    seeds = h6(c); res["h6"] = seeds
    L += ["| set | " + " | ".join(f"s{s['seed']}" for s in seeds) + " | range | stable (same sign >= 4/5) |", "|---|" + "---|" * (len(seeds) + 2)]
    stab = {}
    for cmp_ in ("tfidf", "encoder"):
        for name in SETS:
            d = np.array([s["f1"][name]["probe"] - s["f1"][name][cmp_] for s in seeds])
            ok = bool(max((d > 0).sum(), (d < 0).sum()) >= 4); stab[f"{name}|{cmp_}"] = {"stable": ok, "sign": "+" if (d > 0).sum() >= 4 else "-" if (d < 0).sum() >= 4 else "mixed", "diffs": d.tolist()}
            L.append(f"| probe - {cmp_} on {name} | " + " | ".join(f"{x:+.3f}" for x in d) + f" | [{d.min():+.3f}, {d.max():+.3f}] | {'yes (' + stab[f'{name}|{cmp_}']['sign'] + ')' if ok else 'no'} |")
    res["h6_stability"] = stab
    Path(a.out).write_text("\n".join(L) + "\n"); Path(a.out).with_suffix(".json").write_text(json.dumps(res, indent=1))
    print("\n".join(L))


def summary(a):
    L = ["# T11 summary across models (H3 rule: probe - encoder CI > 0 on >= 2 of 3 external sources, on >= 2 of 3 models)", "",
         "| model | external sources with probe - encoder CI > 0 | per-model condition |", "|---|---|---|"]
    n = 0
    for d in a.dirs:
        r = json.loads((Path(d) / "t11.json").read_text())
        w = [s for s in SOURCES if r["sets"][s]["probe_minus"]["encoder"][1] > 0]
        n += r["h3_model_pass"]; L.append(f"| {Path(d).name} | {len(w)}/3 {w} | {'met' if r['h3_model_pass'] else 'not met'} |")
    L += ["", f"Models meeting the condition: {n} of {len(a.dirs)} -> H3 {'SUPPORTED (variant A)' if n >= 2 else 'NOT SUPPORTED (variant B)'}.",
          "", "| model | tau_h | odd-half neutral recall (orig -> tau_h) | frozen F1 diff [CI] | fixes? |", "|---|---|---|---|---|"]
    for d in a.dirs:
        h = json.loads((Path(d) / "t11.json").read_text())["h5"]
        L.append(f"| {Path(d).name} | {h['tau_h']:.2f} | {h['recall_odd_orig']:.2f} -> {h['recall_odd_tauh']:.2f} | {fmt(*h['diff'])} | {'yes' if h['fixes'] else 'no'} |")
    Path(a.out).write_text("\n".join(L) + "\n"); print("\n".join(L))


def main():
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run"); r.add_argument("--model", required=True); r.add_argument("--cache-dir", required=True); r.add_argument("--eval-dir", required=True)
    r.add_argument("--corpus", required=True); r.add_argument("--pool", default="data/corpus/corpus_human_goemotions_pool.jsonl")
    r.add_argument("--testsets", default="data/testsets"); r.add_argument("--calib", default="data/testsets/goemotions_calibration.jsonl"); r.add_argument("--out", required=True)
    s = sp.add_parser("summary"); s.add_argument("--dirs", nargs="+", required=True); s.add_argument("--out", required=True)
    a = ap.parse_args(); {"run": run, "summary": summary}[a.cmd](a)


if __name__ == "__main__":
    main()
