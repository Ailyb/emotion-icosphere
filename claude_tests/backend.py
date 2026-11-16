#!/usr/bin/env python3
"""
backend.py (v2) -- FastAPI + WebSocket streaming of probe scores during generation.

Changes from v1
  * ONE forward pass per generated token, using a KV cache (v1: two full re-encodes per token, O(n^2))
  * causal mask applied (v1 captured bidirectional-attention activations)
  * scores are z-scores vs. neutral text from probes.npz (v1: raw cosine, needed 'auto contrast')
  * capture layer / skip / z_full come from the probes' own metadata, not hardcoded
  * sampler gets log-probs and applies temperature once (v1 divided by temperature twice)
  * stops at EOS; the model's last-position state is scored for the token it produces
  * decay state is per connection (v1 used a module-global shared by every client)
  * the PROMPT phase is streamed separately, so the UI can show how the model "reads" the prompt

Still on the main thread (no asyncio.to_thread): MLX streams are thread-bound. Each step yields
to the event loop with await asyncio.sleep(0).

Env: MODEL_NAME, PROBE_DIR (default data/probes/<model-slug>), FACET_MAPPING (default data/emotion_vectors/facet_mapping.json)

WebSocket protocol  (client -> server): {"prompt": str, "max_tokens": 60, "temperature": 0.7, "chat": false}
  server -> client:
    {"event":"start", "model", "layer", "emotions", "z_full"}
    {"event":"prompt", "tokens":[str], "scores":[[z per emotion] per token]}      # zeros for the first `skip` tokens
    {"event":"token", "i", "token", "emotion_scores":{e:z}, "emotion_intensity":{e:0..1},
                      "facet_intensities":[...], "layer", "timestamp"}               # state that PRODUCED this token
    {"event":"complete", "tokens", "tokens_per_sec"}
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List

import mlx.core as mx
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler
from pydantic import BaseModel

from probekit_capture import causal_mask
from probekit_probes import ProbeSet

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("emotion-backend")

MODEL_NAME = os.environ.get("MODEL_NAME", "mlx-community/Llama-3.2-3B-Instruct")
PROBE_DIR = Path(os.environ.get("PROBE_DIR", f"data/probes/{MODEL_NAME.replace('/', '__')}"))
FACET_PATH = Path(os.environ.get("FACET_MAPPING", "data/emotion_vectors/facet_mapping.json"))
NUM_FACETS = 1280
DECAY_RATE = 0.8

S: Dict[str, object] = {"model": None, "tok": None, "probes": None, "report": None, "facets": {}, "lock": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    S["model"], S["tok"] = load(MODEL_NAME)
    S["lock"] = asyncio.Lock()                      # one generation at a time on one GPU
    S["probes"] = ProbeSet.load(PROBE_DIR / "probes.npz")
    rp = PROBE_DIR / "report.json"
    S["report"] = json.loads(rp.read_text()) if rp.exists() else None
    ps: ProbeSet = S["probes"]
    log.info("probes: layer %d, %d emotions, skip=%s, z_full=%.2f", ps.layer, len(ps.emotions),
             ps.meta.get("skip"), ps.meta.get("z_full", 4.0))
    if FACET_PATH.exists():
        S["facets"] = json.loads(FACET_PATH.read_text())["facet_mapping"]
    else:
        log.warning("no facet mapping at %s; using equal split", FACET_PATH)
        per = NUM_FACETS // len(ps.emotions)
        S["facets"] = {e: list(range(i * per, (i + 1) * per if i < len(ps.emotions) - 1 else NUM_FACETS))
                       for i, e in enumerate(ps.emotions)}
    yield


app = FastAPI(title="Emotion Icosphere Backend", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


# ------------------------------------------------------------------ model plumbing
def forward(ids: mx.array, cache, layer: int):
    """Manual forward pass with KV cache; returns (logits, residual stream after block `layer`)."""
    m = S["model"].model
    h = m.embed_tokens(ids)
    mask = causal_mask(h.shape[1], h.dtype)         # we always prefill from an empty cache
    captured = None
    for i, (block, c) in enumerate(zip(m.layers, cache)):
        h = block(h, mask, c)
        if i == layer:
            captured = h
    h = m.norm(h)
    model = S["model"]
    logits = model.lm_head(h) if hasattr(model, "lm_head") else m.embed_tokens.as_linear(h)
    return logits, captured


def eos_ids(tok) -> set:
    ids = getattr(tok, "eos_token_ids", None)
    return set(ids) if ids else {tok.eos_token_id}


def to_intensity(z: np.ndarray, z_full: float) -> np.ndarray:
    """Map z-scores to 0..1 glow. 0 sigma = off; z_full sigma = full brightness (z_full is calibrated
    from emotional text in the build step, replacing the old rolling 'auto contrast' hack)."""
    return np.clip(z / z_full, 0.0, 1.0)


def facet_array(intensity: Dict[str, float]) -> np.ndarray:
    arr = np.zeros(NUM_FACETS)
    for e, idx in S["facets"].items():
        v = intensity.get(e, 0.0)
        for i in idx:
            arr[i] = max(arr[i], v)
    return arr


async def stream(prompt: str, max_tokens: int, temperature: float, chat: bool):
    ps: ProbeSet = S["probes"]; tok = S["tok"]
    layer, skip, z_full = ps.layer, int(ps.meta.get("skip", 4)), float(ps.meta.get("z_full", 4.0))
    if chat:
        prompt = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)
    ids = list(tok.encode(prompt))
    cache = make_prompt_cache(S["model"])

    yield {"event": "start", "model": MODEL_NAME, "layer": layer, "emotions": ps.emotions, "z_full": z_full}

    logits, cap = forward(mx.array([ids]), cache, layer)
    mx.eval(logits, cap)
    states = np.array(cap[0].astype(mx.float32))                  # (T, H)
    z_prompt = ps.score(states, "token")
    z_prompt[:skip] = 0.0                                         # BOS / early-token artifacts
    detok = tok.detokenizer
    pieces = [tok.decode([t]) for t in ids]
    yield {"event": "prompt", "tokens": pieces, "scores": np.round(z_prompt, 3).tolist()}

    sampler = make_sampler(temp=temperature, top_p=0.95)
    stops = eos_ids(tok)
    decay_state = np.zeros(NUM_FACETS)                            # per connection (this generator instance)
    last_state, next_logits = states[-1], logits[:, -1, :]
    detok.reset()
    t0, n = time.time(), 0
    for k in range(max_tokens):
        logprobs = next_logits - mx.logsumexp(next_logits, axis=-1, keepdims=True)
        y = sampler(logprobs)
        mx.eval(y)
        tid = int(y.item())

        z = ps.score(last_state, "token")                         # state at the position that produced `tid`
        zd = {e: float(v) for e, v in zip(ps.emotions, z)}
        inten = {e: float(v) for e, v in zip(ps.emotions, to_intensity(z, z_full))}
        decay_state = np.maximum(decay_state * DECAY_RATE, facet_array(inten))
        decay_state[decay_state < 0.01] = 0.0

        if tid in stops:
            break
        detok.add_token(tid)
        n += 1
        yield {"event": "token", "i": k, "token": detok.last_segment, "emotion_scores": zd,
               "emotion_intensity": inten, "facet_intensities": np.round(decay_state, 4).tolist(),
               "layer": layer, "timestamp": time.time()}

        logits, cap = forward(mx.array([[tid]]), cache, layer)    # KV-cached: one new token
        mx.eval(logits, cap)
        last_state, next_logits = np.array(cap[0, -1].astype(mx.float32)), logits[:, -1, :]
        await asyncio.sleep(0)                                    # let the event loop breathe

    detok.finalize()
    dt = max(time.time() - t0, 1e-6)
    yield {"event": "complete", "tokens": n, "tokens_per_sec": round(n / dt, 1)}


# ------------------------------------------------------------------ endpoints
@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            msg = json.loads(await websocket.receive_text())
            async with S["lock"]:
                try:
                    async for ev in stream(msg.get("prompt", ""), int(msg.get("max_tokens", 60)),
                                           float(msg.get("temperature", 0.7)), bool(msg.get("chat", False))):
                        await websocket.send_text(json.dumps(ev))
                except WebSocketDisconnect:
                    raise
                except Exception as e:
                    log.exception("generation error")
                    await websocket.send_text(json.dumps({"event": "error", "message": str(e)}))
    except WebSocketDisconnect:
        log.info("client disconnected")


class ScoreRequest(BaseModel):
    prompt: str


@app.post("/score")
async def score(req: ScoreRequest):
    """Non-streaming: z-scores for the whole passage (mean-pooled, same recipe the probes were validated on)."""
    ps: ProbeSet = S["probes"]; skip = int(ps.meta.get("skip", 4))
    ids = list(S["tok"].encode(req.prompt))
    async with S["lock"]:
        _, cap = forward(mx.array([ids]), make_prompt_cache(S["model"]), ps.layer)
        mx.eval(cap)
    st = np.array(cap[0].astype(mx.float32))
    pooled = st[min(skip, len(st) - 1):].mean(0)
    z = ps.score(pooled, "pooled")
    return {"emotion_scores": {e: float(v) for e, v in zip(ps.emotions, z)}, "layer": ps.layer, "mode": "pooled"}


@app.get("/")
async def root():
    ps: ProbeSet = S["probes"]
    return {"status": "ok", "model": MODEL_NAME, "layer": ps.layer, "emotions": ps.emotions}


@app.get("/probe_report")
async def probe_report():
    """Held-out validation numbers, so the UI can show a quality badge next to the sphere."""
    return S["report"] or {"error": "no report.json found"}


@app.get("/emotions")
async def emotions():
    return {"emotions": S["probes"].emotions}


@app.get("/facet_mapping")
async def facet_mapping():
    f = S["facets"]
    return {"facet_mapping": {e: len(v) for e, v in f.items()}, "facet_assignments": f,
            "total_facets": sum(len(v) for v in f.values())}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
