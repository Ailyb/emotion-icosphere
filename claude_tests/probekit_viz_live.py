#!/usr/bin/env python3
"""
probekit_viz_live.py -- viz_plan Option 1 (live, local): the static page's scoring on text you type, served on 127.0.0.1 only.

    python probekit_viz_live.py --model M --cache-dir D --eval-dir E --corpus C [--port 8765]   ->  open http://127.0.0.1:8765

Same scorer as probekit_viz_static (HUM-trained probe, per-length recalibration, 95th-percentile threshold, TF-IDF / native LR / encoder trained on the same data).
Single-threaded on purpose (MLX streams are thread-bound). Typed text is scored in memory and never written to disk or logged; the server binds to localhost.
"""
import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import SimpleNamespace

MAX_CHARS = 2000
NOT = "What this is not: a reading of what the model feels. These are linear read-outs of emotion-concept text patterns; a TF-IDF classifier often does as well in-domain. Grey (neutral) is the default."


def handle_score(payload, extract, scorer):
    """Pure request logic: payload dict -> (status, body dict). `extract(text) -> 1-d state vector`, `scorer(X, passages) -> rows`."""
    text = payload.get("text") if isinstance(payload, dict) else None
    if not isinstance(text, str) or not text.strip():
        return 400, {"error": "empty text"}
    if len(text) > MAX_CHARS:
        return 400, {"error": f"text longer than {MAX_CHARS} characters"}
    import numpy as np
    row = scorer(np.asarray(extract(text))[None, :], [{"id": "live", "text": text, "intended": "unknown"}])[0]
    return 200, {k: row[k] for k in ("n_words", "bucket", "z_recal", "threshold", "low_confidence", "probe", "tfidf", "native_lr", "encoder")}


PAGE = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Reading the probe honestly (live, local)</title>
<style>:root{--bg:#fff;--ink:#222;--mute:#666;--line:#ddd;--e:#0072B2;--hi:#E69F00}@media(prefers-color-scheme:dark){:root{--bg:#16181c;--ink:#e6e6e6;--mute:#9aa;--line:#333;--e:#56B4E9;--hi:#F0C060}}
body{background:var(--bg);color:var(--ink);font:15px/1.45 system-ui,sans-serif;margin:0;padding:1rem 1rem 3rem}main{max-width:820px;margin:auto}
textarea{width:100%;box-sizing:border-box;min-height:7rem;font:inherit;padding:.5rem;background:var(--bg);color:var(--ink);border:1px solid var(--line)}
button{font:inherit;padding:.4rem 1rem;margin:.5rem 0}.note{border-left:4px solid var(--hi);padding:.5rem .8rem;margin:1rem 0}.m{color:var(--mute);font-size:12px}
.row{display:flex;align-items:center;gap:.5rem;font-variant-numeric:tabular-nums}.lab{width:5.5rem}.bar{height:10px;background:var(--mute);opacity:.5}.bar.hi{background:var(--e);opacity:1}
.cmp span{margin-right:1rem}.n{color:var(--mute)}.e{color:var(--e);font-weight:600}.lc{color:var(--hi)}</style>
<main><h1>Reading the probe honestly <span class="m">(live, local)</span></h1>
<div class="note"><b>__NOT__</b> Your text is scored on this computer and is not stored or logged.</div>
<textarea id="t" maxlength="2000" placeholder="Type or paste a passage (up to 2000 characters)"></textarea><br><button id="go">Score</button> <span id="st" class="m"></span>
<div id="out" hidden><h2 id="verdict"></h2><p class="m" id="meta"></p><div id="bars"></div><h3>Other labels, trained on the same data</h3><p class="cmp" id="cmp"></p>
<p class="m">Bars show recalibrated z per emotion; the vertical position of the threshold is the label cut-off for this length. The probe says "neutral" unless the largest z exceeds the threshold. Reading is not writing: probe directions did not reliably steer a model's output (paper section 2.8).</p></div></main>
<script>
const $=id=>document.getElementById(id), cls=l=>l==="neutral"?"n":"e";
$("go").onclick=async()=>{const t=$("t").value;$("st").textContent="scoring...";
 try{const r=await fetch("/score",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({text:t})});const d=await r.json();
  if(!r.ok){$("st").textContent=d.error;return}$("st").textContent="";$("out").hidden=false;
  $("verdict").innerHTML=`Probe: <span class="${cls(d.probe)}">${d.probe}</span>`;
  $("meta").innerHTML=`${d.n_words} words (length bucket ${d.bucket}); threshold: largest z above ${d.threshold.toFixed(1)}`+(d.low_confidence?' <span class="lc">low confidence: very short text, false alarms are above 5% here</span>':"");
  const mx=Math.max(8,...Object.values(d.z_recal));$("bars").innerHTML=Object.entries(d.z_recal).map(([e,v])=>`<div class="row"><span class="lab">${e}</span><span class="bar ${v>d.threshold?"hi":""}" style="width:${Math.max(0,v)/mx*60}%"></span><span class="m">${v>=0?"+":""}${v.toFixed(1)}</span></div>`).join("");
  $("cmp").innerHTML=["tfidf","native_lr","encoder"].map(k=>`<span>${k}: <span class="${cls(d[k])}">${d[k]}</span></span>`).join("")}
 catch(e){$("st").textContent="request failed"}};
</script></html>""".replace("__NOT__", NOT)


def make_handler(extract, scorer):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):                             # never log requests
            pass

        def _send(self, code, body, ctype="application/json"):
            b = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

        def do_GET(self):
            self._send(200, PAGE.encode(), "text/html; charset=utf-8") if self.path in ("/", "/index.html") else self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/score":
                return self._send(404, {"error": "not found"})
            try:
                n = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(min(n, 20000)) or b"{}")
            except Exception:
                return self._send(400, {"error": "bad request"})
            code, body = handle_score(payload, extract, scorer); self._send(code, body)
    return H


def main():
    ap = argparse.ArgumentParser()
    for k in ("model", "cache-dir", "eval-dir", "corpus"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--pool", default="data/corpus/corpus_human_goemotions_pool.jsonl"); ap.add_argument("--testsets", default="data/testsets")
    ap.add_argument("--calib", default="data/testsets/goemotions_calibration.jsonl")
    a = ap.parse_args()
    import probekit_t11 as T
    from probekit_capture import Extractor
    from probekit_viz_static import make_scorer
    c = T.setup(SimpleNamespace(model=a.model, cache_dir=a.cache_dir, eval_dir=a.eval_dir, corpus=a.corpus, pool=a.pool, testsets=a.testsets, calib=a.calib))
    lc = json.loads((Path(a.cache_dir) / "length_calibration.json").read_text())
    scorer = make_scorer(c, lc); ex = Extractor(a.model); layer, skip = c["layer"], c["skip"]
    extract = lambda text: ex.pooled(text, [layer], skip)[layer]
    srv = HTTPServer(("127.0.0.1", a.port), make_handler(extract, scorer))
    print(f"ready: http://127.0.0.1:{a.port}  (Ctrl-C to stop)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
