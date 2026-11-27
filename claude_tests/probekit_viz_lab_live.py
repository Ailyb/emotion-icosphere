#!/usr/bin/env python3
"""
probekit_viz_lab_live.py -- viz_plan Option 3 (live, local): generate steered vs unsteered text for ONE topic with the directions used in the pre-registered runs.

    python probekit_viz_lab_live.py --model mlx-community/Llama-3.2-3B-Instruct --layer 16 --dir data/steering          # then http://127.0.0.1:8766
    python probekit_viz_lab_live.py --model mlx-community/gemma-2-2b-it-4bit --layer 14 --dir data/steering_gemma      # (or data/steering_gemma_md for same-data mean-difference)

Local only (127.0.0.1), one request at a time (MLX), no hosting. A run is ONE topic and one sample: an illustration, not evidence. The evidence is the pre-registered
40-prompt x 20-random-direction runs shown in docs/viz/steering_lab.html. Label = granite4.2:8b via local Ollama (labels on steered text are not human-validated; T16).
Steering here changes expressed emotion in short neutral text; the tool is not for manipulating people and is not offered as a service.
"""
import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np

import probekit_steer as S

TYPES = {"probe": "LDA probe direction", "v1": "mean-difference direction", "random": "random direction"}
MAX_STRENGTH, MAX_TOPIC, MAX_RANDOM = 1.0, 80, 8
FLUENT_LIMIT = S.PPL_LIMIT


def plan_request(p, n_dirs):
    """Validate a request -> (ok, dict | error string). Strength is in units of the mean token-state norm."""
    if not isinstance(p, dict):
        return False, "bad request"
    topic = p.get("topic"); emo = p.get("emotion"); typ = p.get("dtype", "probe")
    try:
        strength = float(p.get("strength")); n_rand = int(p.get("n_random", 5))
    except (TypeError, ValueError):
        return False, "strength and n_random must be numbers"
    if not isinstance(topic, str) or not topic.strip() or len(topic) > MAX_TOPIC:
        return False, f"topic must be 1-{MAX_TOPIC} characters"
    if emo not in S.EMOTIONS:
        return False, "unknown emotion"
    if typ not in TYPES:
        return False, "unknown direction type"
    if not (0 < strength <= MAX_STRENGTH):
        return False, f"strength must be in (0, {MAX_STRENGTH}]"
    if not (0 <= n_rand <= MAX_RANDOM):
        return False, f"n_random must be 0-{MAX_RANDOM}"
    return True, dict(topic=topic.strip(), emotion=emo, dtype=typ, strength=strength, n_random=min(n_rand, n_dirs), ei=S.EMOTIONS.index(emo))


def summarize(base_label, steered_label, rand_labels, emotion, ppl):
    """Plain-language summary of ONE comparison (never a significance claim)."""
    rl = [x for x in rand_labels if x is not None]
    return dict(target_in_steered=steered_label == emotion, target_in_base=base_label == emotion, random_with_target=sum(x == emotion for x in rl), n_random_labelled=len(rl),
                fluent=bool(ppl <= FLUENT_LIMIT))


def label(text):
    try:
        import probekit_judge as J
        return J.ask(text)
    except Exception:
        return None


PAGE = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Steering lab (live, local)</title>
<style>:root{--bg:#fff;--ink:#222;--mute:#666;--line:#ddd}@media(prefers-color-scheme:dark){:root{--bg:#16181c;--ink:#e6e6e6;--mute:#9aa;--line:#333}}
body{background:var(--bg);color:var(--ink);font:15px/1.45 system-ui,sans-serif;margin:0;padding:1rem 1rem 3rem}main{max-width:900px;margin:auto}.note{border-left:4px solid #E69F00;padding:.5rem .8rem;margin:1rem 0}
input,select,button{font:inherit;padding:.3rem;background:var(--bg);color:var(--ink);border:1px solid var(--line);margin:0 .5rem .5rem 0}.m{color:var(--mute);font-size:12px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:1rem}.box{border:1px solid var(--line);border-radius:8px;padding:.6rem .8rem}.bad{color:#D55E00;font-weight:700}</style>
<main><h1>Steering lab <span class="m">(live, local: __MODEL__)</span></h1><div class="note"><b>__NOTE__</b></div>
<div><input id="topic" maxlength="80" value="a rainy afternoon" aria-label="topic"><select id="emo"></select><select id="dtype"></select>
<label class="m">strength <input id="str" type="number" min="0.05" max="1" step="0.05" value="0.2" style="width:5em"> x token norm</label>
<label class="m">random directions <input id="nr" type="number" min="0" max="8" value="5" style="width:4em"></label><button id="go">Generate</button> <span id="st" class="m"></span></div>
<div id="out" hidden><div class="grid"><div class="box"><b>Unsteered</b><p id="b"></p><div class="m" id="bl"></div></div><div class="box"><b id="sh">Steered</b><p id="s"></p><div class="m" id="sl"></div></div></div>
<p id="sum"></p></div></main>
<script>
const $=id=>document.getElementById(id);["joy","sadness","fear","anger","surprise","disgust","love"].forEach(e=>$('emo').add(new Option(e,e)));
__TYPES__.forEach(([k,v])=>$('dtype').add(new Option(v,k)));
$('go').onclick=async()=>{$('st').textContent='generating (this takes a while)...';$('go').disabled=true;
 try{const r=await fetch('/generate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({topic:$('topic').value,emotion:$('emo').value,dtype:$('dtype').value,strength:+$('str').value,n_random:+$('nr').value})});
  const d=await r.json();if(!r.ok){$('st').textContent=d.error;return}$('st').textContent='';$('out').hidden=false;
  $('b').textContent=d.base;$('s').textContent=d.steered;$('bl').textContent='judge label: '+d.base_label;
  $('sl').innerHTML='judge label: '+d.steered_label+' · perplexity ratio '+d.ppl_ratio.toFixed(2)+(d.summary.fluent?' (fluent)':' <span class="bad">NOT FLUENT (over 1.5)</span>');
  const sm=d.summary;$('sum').textContent=`One topic, one sample. Target '${d.emotion}': unsteered ${sm.target_in_base?'yes':'no'}, steered ${sm.target_in_steered?'yes':'no'}; `+(sm.n_random_labelled?`${sm.random_with_target} of ${sm.n_random_labelled} random directions at the same strength also produced '${d.emotion}'. `:'no random directions run. ')+'This is an illustration; it is not a test. Judge labels on steered text are not human-validated.'}
 catch(e){$('st').textContent='request failed'}finally{$('go').disabled=false}};
</script></html>"""


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--layer", type=int, required=True); ap.add_argument("--dir", required=True); ap.add_argument("--port", type=int, default=8766)
    a = ap.parse_args()
    from mlx_lm import load
    S.LAYER = a.layer; S.DIR = Path(a.dir)
    model, tok = load(a.model); sb = S.install(model, a.layer); dirs = S.load_dirs(); norm = dirs["mean_norm"]
    note = ("Local only. One topic, one sample: an illustration of what the pre-registered runs measured, not evidence. Results are model-dependent (probe directions: 0 of 7 emotions on Llama-3.2-3B, "
            "4 of 7 on Gemma-2-2B). Steering changes expressed emotion in short neutral text; it is not an isolated emotion mechanism, and the tool must not be used to manipulate people.")

    def generate(p):
        ok, r = plan_request(p, len(dirs["random"]))
        if not ok:
            return 400, {"error": r}
        prompt = S.prompt_text(r["topic"]); unit = dirs[r["dtype"]][r["ei"]] if r["dtype"] != "random" else dirs["random"][0]
        vec = r["strength"] * norm * unit / (np.linalg.norm(unit) + 1e-9)
        base = S.gen(model, tok, sb, None, prompt); st = S.gen(model, tok, sb, vec, prompt)
        nb, ns = S.unsteered_nll(model, tok, sb, prompt, base), S.unsteered_nll(model, tok, sb, prompt, st)
        ppl = float(np.exp(ns - nb)) if np.isfinite(ns) and np.isfinite(nb) else float("inf")
        rl = []
        for k in range(r["n_random"]):
            u = dirs["random"][k]; rl.append(label(S.gen(model, tok, sb, r["strength"] * norm * u / (np.linalg.norm(u) + 1e-9), prompt)))
        bl, sl = label(base), label(st)
        return 200, dict(base=base, steered=st, base_label=bl, steered_label=sl, ppl_ratio=ppl, emotion=r["emotion"], summary=summarize(bl, sl, rl, r["emotion"], ppl))

    class H(BaseHTTPRequestHandler):
        def log_message(self, *x): pass
        def _send(self, code, body, ct="application/json"):
            b = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(code); self.send_header("Content-Type", ct); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
        def do_GET(self):
            page = PAGE.replace("__MODEL__", a.model).replace("__NOTE__", note).replace("__TYPES__", json.dumps(list(TYPES.items())))
            self._send(200, page.encode(), "text/html; charset=utf-8") if self.path == "/" else self._send(404, {"error": "not found"})
        def do_POST(self):
            if self.path != "/generate":
                return self._send(404, {"error": "not found"})
            try:
                p = json.loads(self.rfile.read(min(int(self.headers.get("Content-Length", 0)), 5000)) or b"{}")
            except Exception:
                return self._send(400, {"error": "bad request"})
            code, body = generate(p); self._send(code, body)
    srv = HTTPServer(("127.0.0.1", a.port), H); print(f"ready: http://127.0.0.1:{a.port}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
