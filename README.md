# Emotion Icosphere

Real-time emotion visualization on an icosphere brain — streaming a local open-weight LLM's internal emotional state as glowing regions firing across a geodesic sphere.

## Overview

This project captures residual-stream activations from a local open-weight LLM (via mlx-lm / Hugging Face), scores them against validated emotion direction vectors using cosine similarity, maps those scores to facets on an icosphere in a similarity-aware way, and animates glow intensity with decay to produce a "firing" effect.

Based on:
- **Anthropic (April 2026):** "Emotion Concepts and their Function in a Large Language Model" — linear emotion vectors in the residual stream
- **EmotionScope** — open-weight replication + live probing + fluid orb visualization
- Open-model replications on Gemma-2 / Apertus family

## Project Structure

```
emotion-icosphere/
├── README.md
├── requirements.txt
├── config.py                    # model, layer(s), nsub, emotion→facet map, decay, etc.
├── geometry_prototype.py        # Step 1: icosphere generation & inspection
├── capture_hidden_state.py      # Step 2: model inference + residual-stream capture
├── emotion_vectors.py           # Step 3: emotion vector construction + validation
├── score_emotions.py            # Step 3/4: cosine similarity scoring
├── static_visualization.py      # Step 4: static PyVista render
├── backend.py                   # Step 5: FastAPI + WebSocket server
├── frontend/
│   └── index.html               # Step 5: Three.js frontend
├── data/
│   ├── hidden_states/           # captured residual-stream vectors
│   └── emotion_vectors/         # validated emotion vectors + metadata
├── notes/
│   ├── research_baseline.md     # Step 0: literature summaries & model choice
│   ├── geometry_decision.md     # Step 1: nsub & facet mapping decision
│   ├── validation_results.md    # Step 3: emotion vector validation
│   └── performance.md           # Step 6: latency, FPS, memory notes
└── scripts/                     # helpers for steering, batch evaluation, etc.
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

This loads `mlx-community/Llama-3.2-3B-Instruct` at 4-bit quantization and starts a FastAPI server on `http://localhost:8000`.

### 3. Open the Frontend

Open `frontend/index.html` in your browser. The page will connect to the WebSocket at `ws://localhost:8000/ws`.

### 4. Use the Demo

Type a prompt in the input box and press Enter. Watch the icosphere light up as emotion scores stream in real time.

## Implementation Steps

| Step | File | Status | Description |
|------|------|--------|-------------|
| 0 | `notes/research_baseline.md` | ✅ | Literature review, model selection |
| 1 | `geometry_prototype.py` | ✅ | Icosphere generation (nsub=3, 1280 cells) |
| 1 | `notes/geometry_decision.md` | ✅ | nsub=3 chosen, facet mapping scheme |
| 2 | `capture_hidden_state.py` | ✅ | mlx-lm residual stream capture (layer 18) |
| 3 | `emotion_vectors.py` | ✅ | Emotion vector construction + validation |
| 3 | `score_emotions.py` | ✅ | Cosine similarity scoring |
| 4 | `static_visualization.py` | ✅ | PyVista static render with cell intensities |
| 5 | `backend.py` | ✅ | FastAPI + WebSocket streaming backend |
| 5 | `frontend/index.html` | ✅ | Three.js frontend with decay animation |
| 6 | — | ⬜ | Optional steering demo |
| 7 | — | ⬜ | Polish, packaging, README write-up |

## Configuration (`config.py`)

| Setting | Value | Description |
|---------|-------|-------------|
| `MODEL_NAME` | `mlx-community/Llama-3.2-3B-Instruct` | Small instruction-tuned open model |
| `CAPTURE_LAYER_INDEX` | `18` | Mid-to-late layer for emotion capture |
| `EMOTION_POOLING` | `last_token` | Last-token residual stream pooling |
| `ICOSPHERE_NSUB` | `3` | Icosphere subdivision (1280 cells) |
| `EMOTION_NAMES` | 8 core emotions | joy, sadness, fear, anger, surprise, disgust, calm, love |

## Key Design Decisions

1. **Model:** Llama-3.2-3B-Instruct at 4-bit — fits comfortably in 64 GB unified memory on Mac Pro M4
2. **Capture Layer:** Layer 18 (mid-to-late decoder block) for emotion signal
3. **Pooling:** Last-token residual stream — matches literature standard for emotion vectors
4. **Icosphere:** nsub=3 → 1280 cells — good balance between granularity and visual distinctness
5. **Mapping:** Cells distributed across emotions equally (simplified); similarity-aware mapping planned
6. **Decay:** Exponential decay (~200ms half-life) for the "firing" effect

## Future Work

- **Step 3:** Construct validated emotion vectors using contrastive construction (emotion vs neutral stories)
- **Step 6:** Implement activation steering (add α × emotion_vector to residual stream)
- **Step 7:** Package as installable Python package + browser demo; add scientific caveats in UI

## Scientific Caveats

- Activations can decouple from surface text — the sphere shows internal state, not necessarily what the model says
- The sphere is a similarity-preserving projection, not a literal brain map
- Results are model- and layer-specific
- Emotion vectors are model-specific; validate before trusting visualization

## License

MIT