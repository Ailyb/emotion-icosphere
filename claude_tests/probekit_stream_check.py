#!/usr/bin/env python3
"""
probekit_stream_check.py -- do the probes behave when scored TOKEN BY TOKEN, the way the live visualizer uses them?

Everything so far was validated on mean-pooled passages. Here every hand-written passage is run through the model,
each token position is scored with the token-calibrated probes, and we measure:
  * agreement between pooled scoring and several token-aggregation rules (mean z, max z, late-half mean z)
  * 8-way accuracy of each rule on the emotional passages
  * emotional-vs-neutral separation (AUC, threshold-free) of each rule
  * ONSET: how far into the passage the true emotion first crosses z >= 2 and how often it is the top emotion
  * a self-contained HTML viewer (stream_view.html): tokens shaded by the top emotion's z, hover for scores

    python probekit_stream_check.py            # uses data/probes_v3/<model>/probes.npz
"""
import argparse
import html
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from probekit_probes import NEUTRAL, ProbeSet

PALETTE = {"joy": "#f5b700", "sadness": "#3a6ea5", "fear": "#7b3fa0", "anger": "#d62828", "surprise": "#ff7f11",
           "disgust": "#5a8f29", "calm": "#2ec4b6", "love": "#e56b9d"}


def aggregate(z_tok: np.ndarray) -> dict:
    """z_tok: (n_tokens, E) -> {rule: (E,) vector}"""
    half = z_tok[len(z_tok) // 2:]
    return {"mean_z": z_tok.mean(0), "max_z": z_tok.max(0), "late_mean_z": half.mean(0)}


def onset(z_true: np.ndarray, thr: float = 2.0):
    """First token index where the true emotion's z >= thr, as a fraction of passage length (None if never)."""
    hit = np.where(z_true >= thr)[0]
    return None if len(hit) == 0 else float(hit[0] / max(1, len(z_true) - 1))


def analyze_passages(items, emotions):
    """items: list of dicts with label, z_tok (n,E), pooled_z (E,), subtype. Returns metrics dict."""
    out = {}
    emo = [it for it in items if it["label"] != NEUTRAL]
    neu = [it for it in items if it["label"] == NEUTRAL]
    ei = {e: i for i, e in enumerate(emotions)}
    rules = {"pooled": lambda it: it["pooled_z"]}
    rules.update({r: (lambda it, r=r: aggregate(it["z_tok"])[r]) for r in ("mean_z", "max_z", "late_mean_z")})
    for name, fn in rules.items():
        acc = float(np.mean([emotions[int(np.argmax(fn(it)))] == it["label"] for it in emo])) if emo else float("nan")
        auc = None
        if emo and neu:
            y = [1] * len(emo) + [0] * len(neu)
            auc = float(roc_auc_score(y, [fn(it).max() for it in emo + neu]))
        out[name] = {"emotion_accuracy_8way": acc, "emotional_vs_neutral_auc": auc}
    ons, top_frac = [], []
    for it in emo:
        zt = it["z_tok"][:, ei[it["label"]]]
        o = onset(zt)
        if o is not None:
            ons.append(o)
        top_frac.append(float((it["z_tok"].argmax(1) == ei[it["label"]]).mean()))
    out["onset_fraction_median"] = float(np.median(ons)) if ons else None
    out["onset_reached_share"] = float(len(ons) / max(1, len(emo)))
    out["true_emotion_top_token_share_mean"] = float(np.mean(top_frac)) if top_frac else None
    return out


def build_html(items, emotions, z_full):
    css = ("body{font-family:system-ui,sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem;line-height:2}"
           ".p{border:1px solid #ccc;border-radius:8px;padding:.6rem 1rem;margin:.8rem 0}"
           ".h{font-size:.8rem;color:#555;line-height:1.4;margin-bottom:.3rem}"
           ".t{padding:1px 0;border-radius:3px}.k{display:inline-block;padding:0 .5rem;border-radius:4px;color:#fff;margin-right:.3rem;font-size:.8rem}")
    parts = [f"<!doctype html><meta charset='utf-8'><title>Stream view</title><style>{css}</style>",
             "<h2>Token-level emotion scores</h2><p>Shading = z of the top emotion at that token "
             f"(full strength at z={z_full:.1f}); hover for the top three.</p><p>"]
    parts += [f"<span class='k' style='background:{PALETTE.get(e, '#888')}'>{e}</span>" for e in emotions]
    parts.append("</p>")
    for it in items:
        parts.append(f"<div class='p'><div class='h'>[{html.escape(it.get('subtype', ''))}] true: <b>{html.escape(it['label'])}</b>"
                     f" | pooled prediction: <b>{html.escape(it['pooled_pred'])}</b></div>")
        for tok, z in zip(it["tokens"], it["z_all"]):
            j = int(np.argmax(z)); a = float(np.clip(z[j] / z_full, 0, 1)) * 0.85
            tip = ", ".join(f"{emotions[k]} {z[k]:+.1f}" for k in np.argsort(z)[::-1][:3])
            col = PALETTE.get(emotions[j], "#888").lstrip("#")
            rgb = tuple(int(col[i:i + 2], 16) for i in (0, 2, 4))
            bg = f"background:rgba({rgb[0]},{rgb[1]},{rgb[2]},{a:.2f})" if z[j] > 0.5 else ""
            parts.append(f"<span class='t' style='{bg}' title='{html.escape(tip)}'>{html.escape(tok)}</span>")
        parts.append("</div>")
    return "".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Llama-3.2-3B-Instruct")
    ap.add_argument("--probes", default=None)
    ap.add_argument("--hardset", default="data/hardset/hardset.jsonl")
    ap.add_argument("--own-voice", default="data/hardset/own_voice.jsonl")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from build_emotion_vectors import slug
    from probekit_capture import Extractor
    from probekit_eval_v2 import load_jsonl, probe_predict
    pdir = Path(a.probes or f"data/probes_v3/{slug(a.model)}")
    out = Path(a.out or pdir); out.mkdir(parents=True, exist_ok=True)
    ps = ProbeSet.load(pdir / "probes.npz"); skip = int(ps.meta.get("skip", 4)); tau = float(ps.meta.get("tau", 2.0))
    z_full = float(ps.meta.get("z_full", 4.0))
    hrows = load_jsonl(a.hardset)
    if Path(a.own_voice).exists():
        hrows += [r for r in load_jsonl(a.own_voice) if r.get("text", "").strip()]
    ex = Extractor(a.model)
    items = []
    for r in hrows:
        ids = ex.encode(r["text"])
        st = ex.states_from_ids(ids, [ps.layer])[ps.layer]
        s = min(skip, len(st) - 1)
        z_all = ps.score(st, "token"); z_all[:s] = 0.0
        pooled = ps.score(st[s:].mean(0), "pooled")
        pred = probe_predict(ps, st[s:].mean(0)[None], tau)[0]
        items.append({"label": r["category"], "subtype": r.get("subtype", ""), "text": r["text"],
                      "tokens": [ex.tok.decode([i]) for i in ids], "z_all": z_all, "z_tok": z_all[s:],
                      "pooled_z": pooled, "pooled_pred": str(pred)})
    m = analyze_passages(items, ps.emotions)
    (out / "stream_report.json").write_text(json.dumps(m, indent=2))
    (out / "stream_view.html").write_text(build_html(items, ps.emotions, z_full))
    lines = [f"# Stream check ({len(items)} hand-written passages, layer {ps.layer}, method {ps.meta.get('method')})", "",
             "| rule | 8-way accuracy (emotional passages) | emotional-vs-neutral AUC |", "|---|---|---|"]
    for k in ("pooled", "mean_z", "max_z", "late_mean_z"):
        auc = m[k]["emotional_vs_neutral_auc"]
        lines.append(f"| {k} | {m[k]['emotion_accuracy_8way']:.3f} | {('%.3f' % auc) if auc is not None else 'n/a'} |")
    lines += ["", f"- true emotion reaches z>=2 in {m['onset_reached_share']:.0%} of emotional passages; "
                  f"median onset at {m['onset_fraction_median']} of the passage (0 = first token, 1 = last)",
              f"- share of tokens where the true emotion is the top emotion: {m['true_emotion_top_token_share_mean']:.2f}",
              "", "Open stream_view.html and read ~10 passages: do the highlighted tokens make sense, or does it light up on "
                  "stray words (names, punctuation)?"]
    (out / "stream_report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines)); print(f"\nviewer: {out/'stream_view.html'}")


if __name__ == "__main__":
    main()
