#!/usr/bin/env python3
"""
probekit_viz_static.py -- viz_plan Option 1 (static): score bundled SYNTHETIC passages with the audited pipeline and write one self-contained page.

    python probekit_viz_static.py --model M --cache-dir D --eval-dir E --corpus C [--out-dir docs/viz]

Probe = HUM-trained probe (the paper's primary training set), z-scores re-centred per word-length bucket on the neutral calibration set (WRITEUP 2.6).
A passage is shown as "emotional" only if its max recalibrated z exceeds the out-of-fold 95th percentile of max|z| on human neutral text of that length
(data/.../length_calibration.json). Baselines (TF-IDF, native LR, sentence encoder) are trained on the SAME HUM data. Nothing here is tuned on the passages.
"""
import argparse
import html
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import probekit_t11 as T
from probekit_estimators import fit_directions, calibrate_pooled
from probekit_frozen_eval import EXTRACT_LAYERS, extract_cached
from probekit_length_calibration import bucket_of, bucket_name
from probekit_testset import n_words

NOT = "What this is not: a reading of what the model feels. These are linear read-outs of emotion-concept text patterns, and the paper finds a TF-IDF classifier often does as well in-domain."


def recal_params(Zc, wc):
    b = np.array([bucket_of(w) for w in wc]); out = {}
    for bi in np.unique(b):
        Z = Zc[b == bi]; out[int(bi)] = (Z.mean(0), np.maximum(Z.std(0), 1e-6))
    return out


def score(passages, c, lc, cache, model):
    items = [{"id": p["id"], "text": p["text"]} for p in passages]
    X = extract_cached(items, cache, EXTRACT_LAYERS, c["skip"], model)[c["layer"]]
    X_h, T_h, y_h = T.hum_training(c)
    ps = fit_directions(X_h, y_h, T.EMOTIONS, c["layer"], method=c["method"], **c["cfg"]); calibrate_pooled(ps, c["Xc"][c["cal"]])
    systems, enc, _ = T.fit_all(c, X_h, T_h, y_h)
    classes = sorted(set(y_h))
    Zc = ps.score(c["calib"]["X"], "pooled"); par = recal_params(Zc, c["calib"]["n_words"])
    thr = {r["bucket"]: r["recal_q95"] for r in lc["alarm_thresholds"]}
    low = {r["bucket"]: (r["n"] < 500 or r["passage_rate"] > 0.05) for r in lc["recal"] if r["n"] > 0}
    Z = ps.score(X, "pooled"); texts = np.array([p["text"] for p in passages], dtype=object)
    base = {k: f(X, None, texts, classes) for k, f in systems.items() if k in ("tfidf", "native_lr")}
    base["encoder"] = enc(texts, classes)
    rows = []
    for i, p in enumerate(passages):
        nw = n_words(p["text"]); bi = bucket_of(nw); bn = bucket_name(bi); mu, sd = par.get(bi, (Zc.mean(0), Zc.std(0)))
        z = (Z[i] - mu) / sd; t = thr[bn]; j = int(z.argmax())
        rows.append(dict(id=p["id"], text=p["text"], intended=p["intended"], n_words=nw, bucket=bn, z_recal=dict(zip(T.EMOTIONS, map(float, z))),
                         threshold=float(t), low_confidence=bool(low.get(bn, True)), probe="neutral" if z.max() <= t else T.EMOTIONS[j],
                         tfidf=str(base["tfidf"][i]), native_lr=str(base["native_lr"][i]), encoder=str(base["encoder"][i])))
    return rows


def neutral_rate(rows):
    """Share of the bundled neutral synthetic passages shown as emotional, per bucket (descriptive; the bundle is small)."""
    out = {}
    for r in rows:
        if r["intended"] == "neutral":
            o = out.setdefault(r["bucket"], [0, 0]); o[1] += 1; o[0] += r["probe"] != "neutral"
    return out


def page(rows, meta, rate):
    def cell(lab, intended):
        return f'<td class="{"n" if lab == "neutral" else "e"}">{html.escape(lab)}</td>'
    body = []
    for r in rows:
        zs = "".join(f'<span class="z{" hi" if v > r["threshold"] else ""}" title="z {v:+.1f}">{e[:3]} {v:+.1f}</span>' for e, v in r["z_recal"].items())
        lc = '<span class="lc">low confidence: very short</span>' if r["low_confidence"] else ""
        body.append(f'<tr><td class="t">{html.escape(r["text"])}<div class="m">{r["n_words"]} words (bucket {r["bucket"]}); intended: {r["intended"]} {lc}</div></td>'
                    f'<td class="zs">{zs}<div class="m">threshold |z| &gt; {r["threshold"]:.1f}</div></td>{cell(r["probe"], 0)}{cell(r["tfidf"], 0)}{cell(r["native_lr"], 0)}{cell(r["encoder"], 0)}</tr>')
    rt = "; ".join(f"{b}: {k}/{n}" for b, (k, n) in sorted(rate.items()))
    return f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Reading the probe honestly</title>
<style>:root{{--bg:#fff;--ink:#222;--mute:#666;--line:#ddd;--e:#0072B2;--hi:#E69F00}}@media(prefers-color-scheme:dark){{:root{{--bg:#16181c;--ink:#e6e6e6;--mute:#9aa;--line:#333;--e:#56B4E9;--hi:#F0C060}}}}
body{{background:var(--bg);color:var(--ink);font:15px/1.45 system-ui,sans-serif;margin:0;padding:1rem 1rem 3rem}}main{{max-width:1100px;margin:auto}}
.note{{border-left:4px solid var(--hi);padding:.5rem .8rem;margin:1rem 0}}.scroll{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;min-width:760px}}
th,td{{border-bottom:1px solid var(--line);padding:.45rem;text-align:left;vertical-align:top}}.m{{color:var(--mute);font-size:12px}}.n{{color:var(--mute)}}.e{{color:var(--e);font-weight:600}}
.z{{display:inline-block;margin:0 .5rem .1rem 0;font-size:12px;color:var(--mute);font-variant-numeric:tabular-nums}}.z.hi{{color:var(--ink);font-weight:700}}.lc{{color:var(--hi)}}</style>
<main><h1>Reading the probe honestly</h1>
<p>Static scores for synthetic example passages ({html.escape(meta["model"])}, layer {meta["layer"]}, {meta["method"]} probe trained on the paper's primary HUM set; n = {meta["n_train"]} training passages).</p>
<div class="note"><b>{NOT}</b> Passages are new text written for this demo and were not used to tune anything. Grey means below the false-alarm threshold for that length. Grey is the default.</div>
<div class="scroll"><table><thead><tr><th>Passage</th><th>Probe z (per-length recalibrated)</th><th>Probe</th><th>TF-IDF</th><th>Native LR</th><th>Encoder</th></tr></thead><tbody>
{"".join(body)}</tbody></table></div>
<h2>How to read this</h2><ul>
<li><b>Threshold.</b> A passage is called emotional only if its largest recalibrated z exceeds the out-of-fold 95th percentile of max|z| on human neutral text of the same length (so about 5% of neutral human text is flagged by construction). Rule fixed before scoring these passages.</li>
<li><b>Short text.</b> A bucket is marked low confidence when the observed recalibrated false-alarm rate on human neutral text exceeds 5% (the shortest buckets). The paper's strict upper-bound criterion fails in every bucket, so even unmarked rows are approximate.</li>
<li><b>Baselines.</b> TF-IDF, native-state LR and a frozen sentence encoder are trained on the same data as the probe. Where they disagree with the probe, no one of them is ground truth.</li>
<li><b>Bundled neutral passages shown as emotional (small n, descriptive, by length bucket):</b> {rt}. This is not the acceptance test; the reserved human neutral set in <code>length_calibration.md</code> is.</li>
<li><b>Steering is a separate question.</b> Reading emotion text is not writing it: probe directions steered 0 of 7 emotions on Llama-3.2-3B and 4 of 7 (threshold pass) on Gemma-2-2B.</li></ul>
<p class="m">Numbers regenerate with <code>python probekit_viz_static.py</code>; source: docs/viz/static_scores.json.</p></main></html>"""


def main():
    ap = argparse.ArgumentParser()
    for k in ("model", "cache-dir", "eval-dir", "corpus"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--passages", default="data/viz/passages.json"); ap.add_argument("--out-dir", default="docs/viz")
    ap.add_argument("--pool", default="data/corpus/corpus_human_goemotions_pool.jsonl"); ap.add_argument("--testsets", default="data/testsets")
    ap.add_argument("--calib", default="data/testsets/goemotions_calibration.jsonl")
    a = ap.parse_args()
    c = T.setup(SimpleNamespace(model=a.model, cache_dir=a.cache_dir, eval_dir=a.eval_dir, corpus=a.corpus, pool=a.pool, testsets=a.testsets, calib=a.calib))
    lc = json.loads((Path(a.cache_dir) / "length_calibration.json").read_text())
    ps = json.loads(Path(a.passages).read_text())["passages"]
    rows = score(ps, c, lc, Path("data/viz/states_viz.npz"), a.model)
    meta = dict(model=a.model, layer=c["layer"], method=c["method"], n_train=int(c["llm"].sum() + len(c["yp"])))
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "static_scores.json").write_text(json.dumps({"meta": meta, "rows": rows}, indent=1))
    (out / "static.html").write_text(page(rows, meta, neutral_rate(rows)))
    print("wrote", out, neutral_rate(rows))


if __name__ == "__main__":
    main()
