# Emotion Icosphere — Project Status & Handoff Document

**Last Updated:** 2026-11-06  
**Working Directory:** `/Users/justinvanwart/Projects/emotion-icosphere`  
**Backend Process:** May need restart (see below)

---

## 🎯 Project Goal
Real-time emotion visualization on an icosphere brain — streaming a local open-weight LLM's internal emotional state as glowing regions firing across a geodesic sphere.

**Pipeline:** Prompt → mlx-lm (Llama-3.2-3B-Instruct) → Residual stream capture (layer 18) → Cosine similarity vs 8 validated emotion vectors → Similarity-aware facet mapping (PCA) → WebSocket stream → Three.js icosphere with decay animation.

---

## ✅ Completed Steps

| Step | Component | Status | Key Files |
|------|-----------|--------|-----------|
| 0 | Research Baseline | ✅ | `notes/research_baseline.md` |
| 1 | Geometry Prototype (nsub=3 → 1280 cells) | ✅ | `geometry_prototype.py`, `notes/geometry_decision.md` |
| 2 | Residual Capture (mlx-lm, layer 18, 3072-dim) | ✅ | `capture_hidden_state.py` |
| 3 | Emotion Vectors (contrastive construction) | ✅ | `build_emotion_vectors.py`, `emotion_vectors.py` |
| 3 | Validation (diagonal dominance ✅, clustering, PCA) | ✅ | `data/emotion_vectors/validation.json`, `clusters.json`, `pca_plot.png` |
| 4 | Scoring (cosine similarity) | ✅ | `score_emotions.py` |
| 4 | Static Visualization (PyVista) | ✅ | `static_visualization.py` |
| 4★ | **Similarity-Aware Facet Mapping (PCA-based)** | ✅ | `facet_mapping.py`, `data/emotion_vectors/facet_mapping.json` |
| 5 | Backend (FastAPI + WebSocket) | ✅ | `backend.py` |
| 5 | Frontend (Three.js) | ✅ | `frontend/index.html` |

### Key Artifacts Created
```
data/emotion_vectors/
├── joy.json, sadness.json, fear.json, anger.json,
│   surprise.json, disgust.json, calm.json, love.json   # 8 validated 3072-dim vectors
├── validation.json          # diagonal_dominant: true, pairwise similarities
├── clusters.json            # Hierarchical clustering (3 clusters)
├── pca_plot.png             # 2D PCA (40.8% variance explained)
└── facet_mapping.json       # Similarity-aware mapping (1280 facets)
```

### Facet Distribution (PCA-based, not equal blocks)
| Emotion | Facets |
|---------|--------|
| joy | 92 |
| sadness | 114 |
| fear | 282 |
| anger | 70 |
| surprise | 159 |
| disgust | 190 |
| calm | 218 |
| love | 155 |
| **Total** | **1280** |

---

## 🔧 Current Backend State

**File:** `backend.py` — FastAPI server on port 8000

**Key Features Working:**
- REST endpoint `/score` → returns emotion scores + 1280 facet intensities
- REST endpoint `/facet_mapping` → returns facet counts per emotion
- WebSocket `/ws` protocol defined (client sends prompt, server streams tokens + scores)

**Known Issue:** WebSocket streaming has a bug in `stream_emotion_tokens()` generator.
- `stream_generate()` with `make_sampler()` causes `generate_step()` argument errors
- The generator blocks or errors before yielding first token
- **Workaround:** REST `/score` endpoint works perfectly for testing

**To restart backend:**
```bash
cd /Users/justinvanwart/Projects/emotion-icosphere
pkill -f "python backend.py"
python backend.py &
```

**To test REST endpoint:**
```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{"prompt": "I am so happy today!", "max_tokens": 5}'
```

---

## 🌐 Frontend Access

**File:** `frontend/index.html` — Three.js visualization

**Must serve via HTTP** (not `file://`) due to WebSocket CORS:
```bash
cd /Users/justinvanwart/Projects/emotion-icosphere
python -m http.server 8080
# Open http://127.0.0.1:8080/frontend/index.html
```

Frontend connects to `ws://localhost:8000/ws` automatically.

---

## 📋 Next Steps (Priority Order)

### 1. Fix WebSocket Streaming (Critical)
The `stream_emotion_tokens()` function in `backend.py` needs debugging:
- `stream_generate(model, tokenizer, prompt, max_tokens=1, sampler=sampler)` passes `verbose` internally to `generate_step` which doesn't accept it
- Options:
  - Use `generate()` instead of `stream_generate()` for single-token generation
  - Patch `generate_step` call or use lower-level API
  - Run generation in separate thread with proper error handling

### 2. Test Live Demo
Once WebSocket works:
- Open `http://127.0.0.1:8080/frontend/index.html`
- Type prompt → press Enter → watch icosphere glow with similarity-aware regions

### 3. Optional: Steering Demo (Step 6)
Add `α × emotion_vector` to residual stream during generation:
- Modify `capture_residual_stream` or add hook to inject steering vector
- Show side-by-side: steered vs unsteered text + sphere changes

### 4. Polish & Write-up (Step 7)
- Validation plots: clustering dendrogram, PCA trajectories, score trajectories
- Performance metrics: latency per token, FPS, memory usage
- Scientific caveats in UI (decoupling warning, model-specificity notice)

---

## 🛠️ Key Files Reference

| File | Purpose |
|------|---------|
| `backend.py` | FastAPI + WebSocket server (needs WebSocket fix) |
| `frontend/index.html` | Three.js icosphere with decay animation |
| `facet_mapping.py` | PCA-based similarity-aware facet assignment |
| `build_emotion_vectors.py` | Contrastive vector construction + validation |
| `score_emotions.py` | Cosine similarity scoring logic |
| `capture_hidden_state.py` | Residual stream capture from mlx-lm |
| `static_visualization.py` | PyVista static render |
| `config.py` | Central configuration |
| `data/emotion_vectors/facet_mapping.json` | Precomputed facet assignment |

---

## 💡 Architecture Notes

**Model:** `mlx-community/Llama-3.2-3B-Instruct` (4-bit, 3072 hidden, 28 layers)  
**Capture Layer:** 18 (mid-to-late)  
**Pooling:** Last-token residual stream  
**Emotion Set:** 8 core (joy, sadness, fear, anger, surprise, disgust, calm, love)  
**Geometry:** Icosphere nsub=3 → 1280 triangular facets  
**Mapping:** PCA on 8 emotion vectors → 2D → spherical projection → nearest-facet assignment  
**Decay:** Exponential (0.8/frame) for "firing" effect  

---

## 🚀 Quick Start for Next Session

```bash
cd /Users/justinvanwart/Projects/emotion-icosphere

# 1. Kill any stale processes
pkill -f "python backend.py"
pkill -f "http.server 8080"

# 2. Start backend
python backend.py &

# 3. Start frontend server
python -m http.server 8080 &

# 4. Open browser
open http://127.0.0.1:8080/frontend/index.html

# 5. Test REST endpoint (works now)
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{"prompt": "I am terrified of the dark", "max_tokens": 5}'
```

---

## 📝 Context for Next LLM Session

> **Project:** Emotion Icosphere — real-time LLM emotion visualization  
> **Status:** 8 validated emotion vectors built, similarity-aware facet mapping complete, REST API working, WebSocket streaming needs fix.  
> **Immediate Task:** Debug `stream_emotion_tokens()` in `backend.py` to make WebSocket token streaming work with `mlx_lm.generate.stream_generate` + `make_sampler`.  
> **Key Insight:** The `stream_generate` function passes kwargs to `generate_step` which doesn't accept `verbose` or `temperature` — use `sampler=make_sampler(temp=temperature)` and avoid `verbose`.  
> **Fallback:** REST `/score` endpoint works perfectly for testing emotion scoring + facet intensities.