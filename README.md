# Emotion Icosphere

Real-time emotion visualization on an icosphere brain — streaming a local open-weight LLM's internal emotional state as glowing regions firing across a geodesic sphere.

## Overview

This project captures residual-stream activations from a local open-weight LLM (via mlx-lm / Hugging Face), scores them against validated emotion direction vectors using cosine similarity, maps those scores to facets on an icosphere in a similarity-aware way, and animates glow intensity with decay to produce a "firing" effect.

Based on:
- **Anthropic (April 2026):** "Emotion Concepts and their Function in a Large Language Model" — linear emotion vectors in the residual stream
- **EmotionScope** — open-weight replication + live probing + fluid orb visualization
- Open-model replications on Gemma-2 / Apertus family

## Current Status (2026-11-06)

| Component | Status |
|-----------|--------|
| 8 Validated Emotion Vectors | ✅ Contrastive construction, diagonal dominance confirmed |
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
├── config.py
├── geometry_prototype.py
├── capture_hidden_state.py
├── emotion_vectors.py
├── build_emotion_vectors.py
├── score_emotions.py
├── facet_mapping.py
├── static_visualization.py
├── backend.py                 # FastAPI + WebSocket (FIXED)
├── frontend/
│   └── index.html
├── data/
│   ├── hidden_states/
│   └── emotion_vectors/       # 8 validated vectors + validation artifacts
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
4. Emotion bars on left update in real-time

### 6. Test REST Endpoint (Alternative)

```bash
curl -X POST http://127.0.0.1:8000/score \
  -H "Content-Type: application/json" \
  -d '{"prompt": "I am so happy today!", "max_tokens": 5}'
```

## Configuration (`config.py`)

| Setting | Value | Description |
|---------|-------|-------------|
| `MODEL_NAME` | `mlx-community/Llama-3.2-3B-Instruct` | 4-bit quantized, Apple Silicon optimized |
| `CAPTURE_LAYER_INDEX` | `18` | Mid-to-late layer for emotion signal |
| `EMOTION_POOLING` | `last_token` | Last-token residual stream pooling |
| `ICOSPHERE_NSUB` | `3` | Icosphere subdivision (1280 cells) |
| `EMOTION_NAMES` | 8 core emotions | joy, sadness, fear, anger, surprise, disgust, calm, love |

## Key Design Decisions

1. **Model:** Llama-3.2-3B-Instruct at 4-bit — fits in unified memory on M4
2. **Capture Layer:** Layer 18 (mid-to-late decoder block)
3. **Pooling:** Last-token residual stream
4. **Icosphere:** nsub=3 → 1280 triangular facets
5. **Mapping:** PCA on 8 emotion vectors → 2D → spherical projection → nearest-facet assignment
6. **Decay:** Exponential (0.8/frame) for "firing" effect

## Emotion Vector Validation

All 8 vectors pass validation:
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

- Activations can decouple from surface text — sphere shows internal state, not necessarily what the model says
- The sphere is a similarity-preserving projection, not a literal brain map
- Results are model- and layer-specific
- Emotion vectors are model-specific; validate before trusting visualization

## License

MIT
