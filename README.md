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
| Similarity-Aware Facet Mapping | ✅ PCA-based, 1280 facets |
| REST API (`/score`) | ✅ Working |
| WebSocket Streaming | ⚠️ Known bug in `stream_emotion_tokens()` |
| Three.js Frontend | ✅ Ready |
| Static Visualization | ✅ PyVista render |

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
├── backend.py                 # FastAPI + WebSocket (needs WebSocket fix)
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

## Quick Start

### 1. Install Dependencies

```bash
cd emotion-icosphere
pip install -r requirements.txt
```

### 2. Start the Backend

```bash
python backend.py
```

Loads `mlx-community/Llama-3.2-3B-Instruct` (4-bit) and starts FastAPI on `http://localhost:8000`.

### 3. Start Frontend Server

```bash
python -m http.server 8080
```

Open `http://127.0.0.1:8080/frontend/index.html` in browser.

### 4. Test REST Endpoint (Works Now)

```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{"prompt": "I am so happy today!", "max_tokens": 5}'
```

## Configuration (`config.py`)

| Setting | Value |
|---------|-------|
| `MODEL_NAME` | `mlx-community/Llama-3.2-3B-Instruct` |
| `CAPTURE_LAYER_INDEX` | `18` |
| `EMOTION_POOLING` | `last_token` |
| `ICOSPHERE_NSUB` | `3` (1280 cells) |
| `EMOTION_NAMES` | joy, sadness, fear, anger, surprise, disgust, calm, love |

## Key Design Decisions

1. **Model:** Llama-3.2-3B-Instruct at 4-bit — fits in 64 GB unified memory
2. **Capture Layer:** Layer 18 (mid-to-late decoder block)
3. **Pooling:** Last-token residual stream
4. **Icosphere:** nsub=3 → 1280 cells
5. **Mapping:** PCA on emotion vectors → 2D → spherical projection → nearest-facet assignment
6. **Decay:** Exponential (0.8/frame) for "firing" effect

## Next Steps

1. **Fix WebSocket streaming** in `backend.py` — `stream_generate` passes unsupported kwargs to `generate_step`
2. **Live demo** — test with frontend once WebSocket works
3. **Steering demo** — add `α × emotion_vector` to residual stream during generation
4. **Polish** — validation plots, performance metrics, scientific caveats in UI

## Scientific Caveats

- Activations can decouple from surface text — sphere shows internal state, not necessarily what the model says
- The sphere is a similarity-preserving projection, not a literal brain map
- Results are model- and layer-specific
- Emotion vectors are model-specific; validate before trusting visualization

## License

MIT
