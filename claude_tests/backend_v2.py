#!/usr/bin/env python3
"""
backend_v2.py -- viz_plan Option 2: icosphere v2 on the audited probes (replaces the old contrastive-vector demo; backend.py is left untouched).

    python backend_v2.py --model M --cache-dir D --eval-dir E --corpus C [--port 8000]   -> http://127.0.0.1:8000  (serves ../frontend/v2.html + /ws)

Scores TEXT YOU SEND (not the model's own generation). Per token position: HUM-trained probe applied to the mean of the last 8 layer states (T15 window),
recalibrated against neutral streams, flagged above the 99th percentile of neutral positions. Per sentence: the audited pooled probe with per-length recalibration
plus the TF-IDF / native-LR / encoder labels. Localhost only; text is never stored or logged. MLX work runs on the main thread (streams are thread-bound).
"""
import argparse
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import probekit_stream_display as D
from probekit_viz_live import MAX_CHARS
from probekit_viz_static import NOT

PAGE = Path(__file__).resolve().parent.parent / "frontend" / "v2.html"


def agree_count(row):
    """How many of the three baselines name the same label as the probe (probe 'neutral' counts as agreeing only with 'neutral')."""
    return sum(row[k] == row["probe"] for k in ("tfidf", "native_lr", "encoder"))


def make_app(engine, ex, extract_pooled, sentence_scorer, anchors, meta, mode="token"):
    """engine: StreamEngine; ex: Extractor-like (encode, states_from_ids, tok.decode); extract_pooled(text)->(H,) state vector;
    sentence_scorer(X, passages)->rows (static scorer). Returns a FastAPI app."""
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse
    app = FastAPI(title="Emotion Icosphere v2 (local)")
    hello = dict(event="hello", emotions=list(D.EMO), anchors=np.asarray(anchors).tolist(), z_full=D.Z_FULL, window=engine.window, mode=mode,
                 note=NOT, meta=meta)

    @app.get("/", response_class=HTMLResponse)
    def index():
        return PAGE.read_text() if PAGE.exists() else "<p>frontend/v2.html not found</p>"

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        await sock.accept(); await sock.send_text(json.dumps(hello))
        try:
            while True:
                msg = json.loads(await sock.receive_text()); text = msg.get("text"); delay = float(msg.get("delay_ms", 60)) / 1000
                if not isinstance(text, str) or not text.strip() or len(text) > MAX_CHARS:
                    await sock.send_text(json.dumps(dict(event="error", message=f"send non-empty text up to {MAX_CHARS} characters"))); continue
                ids = ex.encode(text); st = ex.states_from_ids(ids, [meta["layer"]])[meta["layer"]]
                toks = [ex.tok.decode([i]) for i in ids]; sents = D.split_sentences(text)
                ends = [len(ex.encode(text[:b])) - 1 for _, b, _ in sents]; frames = engine.positions(st); si = 0
                await sock.send_text(json.dumps(dict(event="start", n_tokens=len(ids), n_sentences=len(sents))))
                for fr in frames:
                    out = dict(event="frame", token=toks[fr["pos"]], **fr, light=D.face_intensity(fr["z"], fr["flag"], fr["top"]))
                    if mode == "sentence":                                             # sentence mode lights nothing per token
                        out["flag"] = False; out["top"] = None; out["light"] = {e: 0.0 for e in D.EMO}
                    await sock.send_text(json.dumps(out))
                    while si < len(sents) and fr["pos"] >= ends[si]:
                        a, b, s = sents[si]
                        row = sentence_scorer(np.asarray(extract_pooled(s))[None, :], [{"id": f"s{si}", "text": s, "intended": "unknown"}])[0]
                        await sock.send_text(json.dumps(dict(event="sentence", index=si, text=s, probe=row["probe"], tfidf=row["tfidf"], native_lr=row["native_lr"],
                                                             encoder=row["encoder"], agree=agree_count(row), low_confidence=row["low_confidence"], bucket=row["bucket"],
                                                             threshold=row["threshold"], z_recal=row["z_recal"])))
                        si += 1
                    await asyncio.sleep(delay)
                await sock.send_text(json.dumps(dict(event="done")))
        except WebSocketDisconnect:
            pass
    return app


def build(a):
    import probekit_t11 as T
    from probekit_capture import Extractor
    from probekit_estimators import calibrate_pooled, fit_directions
    from probekit_viz_static import make_scorer
    from probekit_viz_stream import WINDOWS, build_streams, calib, run_part
    c = T.setup(SimpleNamespace(model=a.model, cache_dir=a.cache_dir, eval_dir=a.eval_dir, corpus=a.corpus, pool=a.pool, testsets=a.testsets, calib=a.calib))
    layer, skip = c["layer"], c["skip"]
    X_h, _, y_h = T.hum_training(c)
    ps = fit_directions(X_h, y_h, T.EMOTIONS, layer, method=c["method"], **c["cfg"]); calibrate_pooled(ps, c["Xc"][c["cal"]])
    ex = Extractor(a.model); cache = Path("data/viz/stream_calib.json")
    if cache.exists() and json.loads(cache.read_text()).get("model") == a.model:
        d = json.loads(cache.read_text()); norm, thr = (np.array(d["mu"]), np.array(d["sd"])), d["thr"]
    else:
        synth = [json.loads(l) for l in open(a.synth)]
        Z, meta = run_part([s for s in build_streams(synth, "val") if s["kind"] == "neutral"], ex, ps, layer, skip)
        norm, thr = calib(Z, meta, D.WINDOW)
        cache.write_text(json.dumps(dict(model=a.model, window=D.WINDOW, mu=norm[0].tolist(), sd=norm[1].tolist(), thr=thr)))
    lc = json.loads((Path(a.cache_dir) / "length_calibration.json").read_text())
    scorer = make_scorer(c, lc); engine = D.StreamEngine(ps, norm, thr, skip)
    t15 = Path(a.cache_dir) / "t15_stream.json"
    mode = "token" if t15.exists() and json.loads(t15.read_text()).get("usable") else "sentence"
    meta = dict(model=a.model, layer=layer, method=c["method"], threshold=float(thr), t15_usable=mode == "token")
    return make_app(engine, ex, lambda t: ex.pooled(t, [layer], skip)[layer], scorer, D.anchors_from_directions(ps.V), meta, mode)


def main():
    ap = argparse.ArgumentParser()
    for k in ("model", "cache-dir", "eval-dir", "corpus"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--port", type=int, default=8000); ap.add_argument("--synth", default="data/viz/synth.jsonl")
    ap.add_argument("--pool", default="data/corpus/corpus_human_goemotions_pool.jsonl"); ap.add_argument("--testsets", default="data/testsets")
    ap.add_argument("--calib", default="data/testsets/goemotions_calibration.jsonl")
    a = ap.parse_args(); import uvicorn
    uvicorn.run(build(a), host="127.0.0.1", port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
