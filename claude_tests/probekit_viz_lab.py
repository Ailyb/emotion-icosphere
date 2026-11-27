#!/usr/bin/env python3
"""
probekit_viz_lab.py -- viz_plan Option 3 (static steering lab): every number comes from probekit_steer.analyze on the committed-method generation/judge files.

    python probekit_viz_lab.py [--out-dir docs/viz]   ->  docs/viz/lab_data.json, docs/viz/steering_lab.html

Direction types shown (never pooled across models): LDA probe, mean-difference fitted on the v1 corpus, and (Gemma only) mean-difference fitted on the same data as the LDA probe (T14).
Texts are model-written continuations of neutral prompts; no third-party data. Strength is in units of the mean token-state norm at the steered layer.
"""
import argparse
import html
import json
from pathlib import Path

import numpy as np

import probekit_steer as S

CONFIGS = [  # key, label, model label, dir, cond, note
    ("llama_probe", "Llama-3.2-3B, layer 16: LDA probe direction", "Llama-3.2-3B", "data/steering", "probe"),
    ("llama_md_v1", "Llama-3.2-3B, layer 16: mean-difference (v1 corpus)", "Llama-3.2-3B", "data/steering", "v1"),
    ("gemma_probe", "Gemma-2-2B, layer 14: LDA probe direction", "Gemma-2-2B", "data/steering_gemma", "probe"),
    ("gemma_md_v1", "Gemma-2-2B, layer 14: mean-difference (v1 corpus)", "Gemma-2-2B", "data/steering_gemma", "v1"),
    ("gemma_md_same", "Gemma-2-2B, layer 14: mean-difference (same data as the LDA probe, T14)", "Gemma-2-2B", "data/steering_gemma_md", "v1"),
]
N_SAMPLES = 3


def load(d):
    d = Path(d); gens = {}
    for l in (d / "generations.jsonl").open():
        g = json.loads(l); gens[g["gid"]] = g
    lab = {json.loads(l)["gid"]: json.loads(l)["label"] for l in (d / "judged.jsonl").open()}
    base_nll = {g["pid"]: g["nll"] for g in gens.values() if g["cond"] == "base"}
    rows = [{**{k: g[k] for k in ("cond", "emotion", "rid", "strength", "pid")}, "label": lab[gid],
             "ppl_ratio": float(np.exp(g["nll"] - base_nll[g["pid"]])) if g["pid"] in base_nll and np.isfinite(g["nll"]) else None}
            for gid, g in gens.items() if gid in lab]
    return gens, lab, rows


def samples(gens, cond, emotion, strength, n=N_SAMPLES):
    """First n prompts (by pid) where both the unsteered and the steered text exist."""
    base = {g["pid"]: g for g in gens.values() if g["cond"] == "base"}
    st = sorted((g for g in gens.values() if g["cond"] == cond and g["emotion"] == emotion and g["strength"] == strength), key=lambda g: g["pid"])
    return [dict(topic=g["topic"], base=base[g["pid"]]["text"], steered=g["text"]) for g in st if g["pid"] in base][:n]


def build():
    out = {}; cache = {}
    for key, label, model, d, cond in CONFIGS:
        if d not in cache:
            cache[d] = load(d)
        gens, lab, rows = cache[d]
        res = S.analyze(rows, cond=cond)
        base_lab = [r["label"] for r in rows if r["cond"] == "base"]; nb = len(base_lab)
        cells = {}
        for e, m in res["emotions"].items():
            for s, c in m.items():
                cells[f"{e}|{s:g}"] = dict(emotion=e, strength=s, delta=c["delta"], ci=list(c["ci"]), ppl=c["median_ppl_ratio"], fluent=c["fluent"], specific=c["specific"],
                                          random_max=c["random_max"], random=c["random_deltas"], base_rate=base_lab.count(e) / nb, samples=samples(gens, cond, e, s))
        out[key] = dict(label=label, model=model, strengths=[f"{s:g}" for s in res["strengths"]], emotions=list(res["emotions"]), n_prompts=res["n_prompts"], n_random=res["n_random"],
                        steerable=res["steerable"], cells=cells)
    return out


PAGE = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Steering lab (static)</title>
<style>:root{--bg:#fff;--ink:#222;--mute:#666;--line:#ddd;--a:#0072B2;--w:#D55E00;--ok:#009E73}@media(prefers-color-scheme:dark){:root{--bg:#16181c;--ink:#e6e6e6;--mute:#9aa;--line:#333;--a:#56B4E9;--w:#F08A4B;--ok:#4cc9a0}}
body{background:var(--bg);color:var(--ink);font:15px/1.45 system-ui,sans-serif;margin:0;padding:1rem 1rem 3rem}main{max-width:980px;margin:auto}.note{border-left:4px solid #E69F00;padding:.5rem .8rem;margin:1rem 0}
select{font:inherit;padding:.25rem;background:var(--bg);color:var(--ink);border:1px solid var(--line);margin:0 .5rem .5rem 0}.m{color:var(--mute);font-size:12px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:1rem}
.box{border:1px solid var(--line);border-radius:8px;padding:.6rem .8rem}.bad{color:var(--w);font-weight:700}.good{color:var(--ok);font-weight:700}svg{max-width:100%}table{border-collapse:collapse}td,th{border-bottom:1px solid var(--line);padding:.25rem .6rem;text-align:left}.scroll{overflow-x:auto}</style>
<main><h1>Steering lab <span class="m">(static; local models only)</span></h1>
<div class="note"><b>__NOTE__</b></div>
<p class="m">Steering = adding a direction (scaled by the mean token-state norm) to the residual stream while a model continues a neutral prompt. These numbers are from pre-registered runs (40 prompts, 20 random directions each); the lab only displays them. It is not a way to manipulate people, and it is not offered as a service.</p>
<div><select id="cfg"></select><select id="emo"></select><select id="str"></select></div>
<div id="verdict"></div><div class="grid"><div class="box"><b>Effect on the judge's label</b><div id="chart"></div><div id="eff" class="m"></div></div><div class="box"><b>Fluency</b><div id="flu"></div></div></div>
<h2>Example continuations</h2><p class="m">Same prompt, unsteered vs steered. Model-written text; read it yourself instead of trusting the label.</p><div id="samp" class="grid"></div>
<h2>Table view</h2><div class="scroll"><table id="tab"></table></div>
<p class="m">__JUDGE__</p></main>
<script>
const D=__DATA__,$=id=>document.getElementById(id);
const cfg=$('cfg'),emo=$('emo'),str=$('str');Object.entries(D).forEach(([k,v])=>cfg.add(new Option(v.label,k)));
function fill(){const c=D[cfg.value];emo.innerHTML='';c.emotions.forEach(e=>emo.add(new Option(e,e)));str.innerHTML='';c.strengths.forEach(s=>str.add(new Option('strength '+s+' x token norm',s)));show()}
function esc(t){const d=document.createElement('div');d.textContent=t;return d.innerHTML}
function chart(c){const W=420,H=70,lo=-0.3,hi=0.7,x=v=>20+(v-lo)/(hi-lo)*(W-40);let s=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="effect versus random directions"><line x1="20" x2="${W-20}" y1="40" y2="40" stroke="#888"/>`;
 [-0.2,0,0.2,0.4,0.6].forEach(t=>s+=`<line x1="${x(t)}" x2="${x(t)}" y1="36" y2="44" stroke="#888"/><text x="${x(t)}" y="58" font-size="9" text-anchor="middle" fill="currentColor">${t}</text>`);
 s+=`<line x1="${x(0)}" x2="${x(0)}" y1="12" y2="40" stroke="#888" stroke-dasharray="2"/>`;
 c.random.forEach(r=>s+=`<circle cx="${x(r)}" cy="40" r="3" fill="none" stroke="#777" opacity=".7"/>`);
 s+=`<line x1="${x(c.ci[0])}" x2="${x(c.ci[1])}" y1="24" y2="24" stroke="var(--a)" stroke-width="2"/><rect x="${x(c.delta)-5}" y="19" width="10" height="10" fill="var(--a)" transform="rotate(45 ${x(c.delta)} 24)"/>`;
 return s+`<text x="20" y="10" font-size="10" fill="currentColor">diamond: this direction (95% CI); circles: ${c.random.length} random directions; change in rate of the target label</text></svg>`}
function show(){const m=D[cfg.value],c=m.cells[emo.value+'|'+str.value];if(!c)return;
 $('chart').innerHTML=chart(c);const pct=v=>(v*100).toFixed(1)+'%';
 $('eff').textContent=`Unsteered rate of '${c.emotion}' labels ${pct(c.base_rate)}; change ${c.delta>=0?'+':''}${(c.delta*100).toFixed(1)} points [${(c.ci[0]*100).toFixed(1)}, ${(c.ci[1]*100).toFixed(1)}]; largest random-direction change ${(c.random_max*100).toFixed(1)} points.`;
 $('flu').innerHTML=`Median perplexity ratio ${c.ppl.toFixed(2)} (limit 1.5): <span class="${c.fluent?'good':'bad'}">${c.fluent?'fluent':'NOT FLUENT'}</span>`;
 $('verdict').innerHTML=`<p>Under the pre-registered rule (change above all ${c.random.length} random directions, CI above 0, and fluent): <span class="${c.specific?'good':'bad'}">${c.specific?'specifically steerable here':'not shown to be specifically steerable here'}</span>. Across all strengths in this setting, ${m.steerable.length} of 7 emotions pass: ${m.steerable.join(', ')||'none'}.</p>`;
 $('samp').innerHTML=c.samples.map(x=>`<div class="box"><div class="m">${esc(x.topic)}</div><b>unsteered</b><p>${esc(x.base)}</p><b>steered</b><p>${esc(x.steered)}</p></div>`).join('')||'<p class="m">no samples</p>';
 $('tab').innerHTML='<tr><th>emotion</th><th>strength</th><th>change</th><th>95% CI</th><th>random max</th><th>ppl ratio</th><th>specific</th></tr>'+Object.values(m.cells).map(k=>`<tr><td>${k.emotion}</td><td>${k.strength}</td><td>${(k.delta*100).toFixed(1)}</td><td>${(k.ci[0]*100).toFixed(1)} to ${(k.ci[1]*100).toFixed(1)}</td><td>${(k.random_max*100).toFixed(1)}</td><td>${k.ppl.toFixed(2)}</td><td>${k.specific?'yes':'no'}</td></tr>`).join('')}
cfg.onchange=fill;emo.onchange=show;str.onchange=show;fill();
</script></html>"""


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out-dir", default="docs/viz"); ap.add_argument("--judge-report", default="data/eval_merged2/judge_gen/judge_gen.json"); a = ap.parse_args()
    data = build(); out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "lab_data.json").write_text(json.dumps(data, default=float))
    jr = Path(a.judge_report)
    judge = ("Judge: granite4.2:8b. Its gate used human comments (macro-F1 >= 0.50), not steered text. " +
             ("Two-judge consistency on generated text (T16): " + ("rule MET (kappa >= 0.60)." if json.loads(jr.read_text())["rule_met"] else "rule NOT met: labels on steered text are not validated.") + " Agreement between two models is not accuracy; no human labels yet."
              if jr.exists() else "Consistency of this judge on generated text has not been checked (T16 pending); labels on steered text are unvalidated."))
    note = ("Results are model-dependent: LDA probe directions steered 0 of 7 emotions on Llama-3.2-3B and 4 of 7 on Gemma-2-2B (a threshold pass at the largest fluent strength). Mean-difference directions were not shown to beat them at matched fluency "
            "(a same-data comparison on Gemma found no advantage: 0 of 7 paired intervals above 0). Random directions are drawn so you can judge for yourself. This shows changes in expressed emotion in short neutral texts, not an isolated emotion mechanism and not a change in decisions.")
    (out / "steering_lab.html").write_text(PAGE.replace("__DATA__", json.dumps(data, default=float).replace("</", "<\\/")).replace("__NOTE__", html.escape(note)).replace("__JUDGE__", html.escape(judge)))
    print("wrote", out, {k: (v["steerable"]) for k, v in data.items()})


if __name__ == "__main__":
    main()
