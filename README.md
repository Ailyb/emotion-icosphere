# Emotion Icosphere

Real-time visualization of emotion-concept probe activations from a local open-weight LLM, shown as glowing regions on a geodesic sphere.

> **Read this first.** The probes are linear read-outs of residual-stream activations that correlate with emotion-concept text. They are not the model's feelings or "internal emotional state". Measured accuracy, limits and negative results are in [`claude_tests/docs/FINDINGS.md`](claude_tests/docs/FINDINGS.md). Short version (three small models from three families, Llama-3.2-3B, Gemma-2-2B and Mistral-7B, all 4-bit; draft writeup: `claude_tests/docs/WRITEUP.md`; details, CIs and caveats in FINDINGS.md): on a frozen, hash-verified test set of 840 human reddit comments (GoEmotions, 8 classes, chance ~0.125) with every system trained on the same data (LLM-written rows plus a human GoEmotions pool, n=4,192) the probe scores macro-F1 0.549-0.587 (CIs about +/-0.03) -- better than a bag-of-embeddings baseline (+0.08 to +0.10) but **worse than a TF-IDF classifier (0.674; difference -0.086, -0.106 and -0.126 for Llama, Gemma and Mistral; every CI excludes 0)**. (The first pre-registered run, on a different training set, gave probe 0.51 vs TF-IDF 0.60: same conclusion; Appendix A of the writeup.) On three *other* human-labelled sources (tweets, event descriptions, dialogue; pre-registered test) the picture reverses for all three models: trained on the same data, the probe beats TF-IDF by +0.05 to +0.19 macro-F1 (CIs exclude 0), and by +0.13 to +0.26 when trained only on LLM-written text. So lexical features win in-domain and the probe transfers better across domains; a frozen 110M-parameter sentence encoder (all-mpnet-base-v2) trained on the same data does as well or better (the probe is no better than it on any external source and worse on ISEAR and DailyDialog), so the cross-domain advantage is not specific to LLM probes; a trained classifier on the same hidden states is above the probe in-domain (post hoc) and above the encoder there, so one direction per emotion leaves information on the table, but an MLP or a valence hierarchy does not help, so the information is linear at the chosen layer. The probe directions' geometry correlates with human valence-arousal-dominance distances between emotion words on two of the three models. A 7B model does not close the in-domain gap. Adding these probe directions to the model's activations did not change the emotion it writes any more than random directions did (a pre-registered steering test, 0 of 7 emotions); a plain mean-difference direction did shift joy and anger, so reading and writing directions differ. Human training data is the main lever (data-limited). **The raw z-scores are not calibrated across text length**: on neutral human text 15-83% of passages raise some |z| > 3 depending on length, and per-length recalibration is needed; simply re-fitting the neutral threshold on human neutral text restores neutral recall but costs 0.12-0.17 macro-F1 (pre-registered, not met). Earlier, more favourable numbers on a hand-written set are non-blind (that set was accidentally committed to this public repo). Plan: `claude_tests/PLAN_v2.md`; claim checklist: `claude_tests/docs/CLAIMS.md`.

## Audit your own probes (one command)

```bash
pip install "git+https://github.com/Ailyb/emotion-icosphere.git#subdirectory=claude_tests" && probekit-audit --demo
```

Runs a small bundled synthetic example (no model, no downloads) and writes `audit_demo.md`: probe vs TF-IDF, bag of embeddings and a hidden-state classifier, with bootstrap CIs, in-domain vs shifted. For your own data and caches see [`claude_tests/docs/AUDIT_QUICKSTART.md`](claude_tests/docs/AUDIT_QUICKSTART.md).

## Overview

This project captures residual-stream activations from a local open-weight LLM (via mlx-lm / Hugging Face), scores them against emotion direction vectors, maps those scores to facets on an icosphere in a similarity-aware way, and animates glow intensity with decay to produce a "firing" effect.

**Two code paths exist.** The top-level `backend.py` (this demo) still uses the original contrastive vectors (layer 18, last-token pooling, cosine similarity), which have only been checked for self-consistency, not against a baseline. The evaluated probes (mean-pooled, z-scored against neutral text, layer 14) live in `claude_tests/` (`probekit_*.py`); `claude_tests/backend.py` loads them. Only the latter is covered by the results above.

Based on:
- **Anthropic (April 2026):** "Emotion Concepts and their Function in a Large Language Model" — linear emotion vectors in the residual stream
- **EmotionScope** — open-weight replication + live probing + fluid orb visualization
- Open-model replications on Gemma-2 / Apertus family

## Current Status (2026-11-06)

| Component | Status |
|-----------|--------|
| 8 emotion vectors (demo) | ⚠️ Contrastive construction; diagonal dominance only, no baseline comparison |
| Evaluated probes (`claude_tests/`) | ✅ Tested vs lexical baseline and native LR; see FINDINGS.md |
| Similarity-Aware Facet Mapping | ✅ PCA-based, 1280 facets distributed by emotion similarity |
| REST API (`/score`) | ✅ Working |
| WebSocket Streaming | ✅ **Working** — direct model forward pass + sampler |
| Three.js Frontend | ✅ **Fully Functional** — WebSocket message parsing fixed |
| Static Visualization | ✅ PyVista render |

### Recent Fix (2026-11-06)
**Bug Fixed:** WebSocket message parsing in `frontend/index.html`
- **Issue:** `JSON.parse(event)` was used instead of `JSON.parse(event.data)`
- **Result:** Frontend received WebSocket messages but couldn't parse them, so emotion bars and icosphere didn't update
- **Fix:** Changed line 326 in `frontend/index.html` from `JSON.parse(event)` to `JSON.parse(event.data)`
- **Status:** ✅ **Demo now fully functional** — type a prompt, press Enter, watch real-time emotion visualization

## Project Structure

```
emotion-icosphere/
├── README.md
├── PROJECT_STATUS.md          # Detailed handoff document
├── requirements.txt
├── build_emotion_vectors.py
├── facet_mapping.py
├── static_visualization.py
├── backend.py                 # FastAPI + WebSocket (FIXED)
├── frontend/
│   └── index.html
├── data/
│   ├── hidden_states/
│   └── emotion_vectors/       # 8 contrastive vectors + self-consistency artifacts
├── notes/
│   ├── research_baseline.md
│   └── geometry_decision.md
└── scripts/
```

## Quick Start (macOS / M4 MacBook Pro)

### Prerequisites
- **macOS** with Apple Silicon (M1/M2/M3/M4)
- **Python 3.10+** (tested on 3.12+)
- **~8 GB free RAM** for model (4-bit quantized Llama-3.2-3B)

### 1. Install Dependencies

```bash
cd emotion-icosphere
pip install -r requirements.txt
```

Key dependencies:
- `mlx-lm` — Apple Silicon optimized LLM inference
- `fastapi`, `uvicorn` — Web server
- `numpy`, `scikit-learn` — Vector operations & PCA
- `pyvista` — Static 3D visualization (optional)

### 2. Start the Backend

```bash
python backend.py
```

Loads `mlx-community/Llama-3.2-3B-Instruct` (4-bit) and starts FastAPI on `http://127.0.0.0:8000`.
First run downloads ~2 GB model weights from Hugging Face.

**Expected startup logs:**
```
INFO:__main__:Model loaded. Hidden size: 3072
INFO:__main__:Number of layers: 28
INFO:__main__:Loaded 8/8 emotion vectors
INFO:__main__:Loaded similarity-aware facet mapping: 1280 facets
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 3. Start Frontend Server

**Must run from project directory** (so it finds `frontend/index.html`):

```bash
python -m http.server 8080
```

### 4. Open Demo in Browser

```
http://127.0.0.1:8080/frontend/index.html
```

**Important:** Use `127.0.0.1` (not `localhost`) to avoid Safari WebSocket cross-origin issues.

### 5. Test the Demo

1. Type a prompt: `"I am so happy and excited today!"`
2. Press **Enter**
3. Watch the icosphere glow with similarity-aware emotion regions
4. Read the generated continuation in the compact strip above the prompt field
5. Toggle **Auto contrast** to adapt the emotion bars and sphere brightness to a rolling baseline; this changes display contrast only, not model scores

### 6. Test REST Endpoint (Alternative)

```bash
curl -X POST http://127.0.0.1:8000/score \
  -H "Content-Type: application/json" \
  -d '{"prompt": "I am so happy today!", "max_tokens": 5}'
```

## Configuration (constants in `backend.py`)

| Setting | Value | Description |
|---------|-------|-------------|
| `MODEL_NAME` | `mlx-community/Llama-3.2-3B-Instruct` | 4-bit quantized, Apple Silicon optimized |
| `CAPTURE_LAYER_INDEX` | `18` | Mid-to-late layer for emotion signal |
| `EMOTION_POOLING` | `last_token` | Last-token residual stream pooling |
| `NUM_FACETS` | `1280` | nsub=3 icosphere cell count |

## Key Design Decisions

1. **Model:** Llama-3.2-3B-Instruct at 4-bit — fits in unified memory on M4
2. **Capture Layer:** Layer 18 (mid-to-late decoder block)
3. **Pooling:** Last-token residual stream
4. **Icosphere:** nsub=3 → 1280 triangular facets
5. **Mapping:** PCA on 8 emotion vectors → 2D → spherical projection → nearest-facet assignment
6. **Decay:** Exponential (0.8/frame) for "firing" effect

## Emotion Vector Self-Consistency Checks (demo vectors)

These checks show the vectors are internally consistent. They do not show the vectors track emotion beyond word choice; that test (against a lexical baseline) is in FINDINGS.md and applies to the `claude_tests/` probes.
- ✅ **Diagonal dominance** — each emotion most similar to itself
- ✅ **Clustering** — 3 clusters: {anger, disgust, love}, {surprise, calm}, {joy, sadness, fear}
- ✅ **PCA** — 40.8% variance explained in 2D (valence-arousal geometry)

Facet distribution (PCA-based):
| Emotion | Facets |
|---------|--------|
| fear | 282 |
| calm | 218 |
| disgust | 190 |
| love | 155 |
| surprise | 159 |
| sadness | 114 |
| joy | 92 |
| anger | 70 |

## Troubleshooting (macOS / M4)

| Issue | Fix |
|-------|-----|
| `Address already in use` | `lsof -ti:8000 -ti:8080 \| xargs kill -9` |
| Safari "dropped connection" | Use `127.0.0.1:8080` (not `localhost:8080`) |
| WebSocket fails | Check browser Console (F12) for errors |
| Frontend 404 | Run `python -m http.server 8080` from project directory |
| Model download fails | Check internet, retry (resumes automatically) |
| **Emotion bars/icosphere don't update** | **Fixed in v1.1** — was `JSON.parse(event)` bug, now `JSON.parse(event.data)` |

## Browser Compatibility

| Browser | Status |
|---------|--------|
| Chrome | ✅ Best support |
| Firefox | ✅ Works |
| Safari | ⚠️ Use `127.0.0.1` not `localhost` |

## Development

### Run Tests
```bash
# Test REST endpoint
curl -X POST http://127.0.0.1:8000/score \
  -H "Content-Type: application/json" \
  -d '{"prompt": "I am happy", "max_tokens": 5}'

# Test WebSocket
python -c "
import asyncio, json, websockets
async def t():
    async with websockets.connect('ws://127.0.0.1:8000/ws') as ws:
        await ws.send(json.dumps({'prompt': 'test', 'max_tokens': 2}))
        for _ in range(2): print(await ws.recv())
asyncio.run(t())
"
```

### Static Visualization
```bash
python static_visualization.py
# Output: static_visualization_example.png
```

### Build Emotion Vectors (Already Done)
```bash
python build_emotion_vectors.py
# Output: data/emotion_vectors/*.json + validation artifacts
```

### Facet Mapping (Already Done)
```bash
python facet_mapping.py
# Output: data/emotion_vectors/facet_mapping.json
```

## Next Steps (Roadmap)

1. **Steering Demo** — Add `α × emotion_vector` to residual stream during generation
2. **Performance** — KV caching for faster token generation (<50ms target)
3. **Validation Plots** — Clustering dendrogram, PCA trajectories, score heatmaps
4. **Scientific Caveats** — Add decoupling warnings, model-specificity notices to UI
5. **Packaging** — `pyproject.toml`, CLI entry points, GitHub Actions CI

## Scientific Caveats

- The sphere shows emotion-concept probe activations, not feelings or an internal emotional state; nothing here shows the model has either
- Probes are coarse and style-sensitive: informal text outside the training style is much worse, and very short text is unreliable (e.g. a neutral short sentence scored sadness z +4.9)
- Love is the weakest class in own-voice tests (0-2 of 5 depending on estimator); per-class n=5, so wide uncertainty
- The sphere is a similarity-preserving projection, not a literal brain map
- Results are from one model (Llama-3.2-3B-Instruct, 4-bit) and one layer plateau; vectors are model-specific
- Test sets are small (n=92 hand-written); see FINDINGS.md for CIs

## License

[The Unlicense](LICENSE): our code, tests, plans, reports and documentation are in the public domain; use them for anything. Third-party data, model outputs and model weights keep their own terms: see [NOTICE.md](NOTICE.md).
