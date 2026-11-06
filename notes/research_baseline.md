# Research Baseline — Emotion Vectors & Icosphere Mapping

## 1. Core Literature

### Anthropic (April 2026): "Emotion Concepts and their Function in a Large Language Model"

**Key findings:**
- Emotion concepts exist as *linear direction vectors* in the residual stream of Claude.
- These vectors activate appropriately on emotion-eliciting inputs (high cosine similarity to the emotion's vector vs neutral inputs).
- Emotion vectors exhibit *valence–arousal geometry* — they map onto a meaningful 2D space where valence (positive/negative) and arousal (high/low activation) are recoverable axes.
- *Steerability*: Adding scaled emotion vectors to the residual stream changes generated text in emotion-consistent ways.
- Vectors can be constructed contrastively: activations on emotion-eliciting stories minus activations on neutral control stories, normalized to unit length.

**Implication for this project:** The same pipeline can be applied to open-weight models with mlx-lm: capture residual-stream activations at a mid-to-late layer, build contrastive emotion vectors, and score live activations via cosine similarity.

---

### EmotionScope (AidanZach/EmotionScope)

**Architecture:**
- Python backend (mlx-lm / Transformers) captures hidden states.
- FastAPI WebSocket server streams scores per token.
- React Three Fiber frontend renders a fluid 3D orb with per-emotion coloring.

**Why reference it:**
- Proven pipeline for extracting + probing + streaming emotion vectors.
- WebSocket + React Three Fiber stack is directly transferable.
- The orb aesthetic differs from our discrete-facet icosphere design.

---

### Open-Model Replications

**Gemma-2 / Apertus family:**
- Independent replications confirm emotion vectors transfer to open models (Gemma-2, Apertus-8B).
- Valence–arousal geometry is reproducible in models at 2B–9B scale.
- Small models at 4-bit quantization retain usable emotion-direction signal.

**Decision:** Use a small instruction-tuned open model (~2–8B parameters) at 4-bit so it fits comfortably on the Mac Pro M4 (64 GB unified memory).

---

## 2. Chosen Model

| Attribute | Choice |
|-----------|--------|
| Model | `Llama-3.2-3B-Instruct` (via mlx-lm) |
| Quantization | 4-bit |
| Backend | `mlx-lm` |
| Hidden size | ~3072 (Llama-3.2-3B) |
| Number of layers | 32 decoder blocks |
| Primary capture layer | 18 (mid-to-late) |
| Pooling | Last-token residual stream |

**Rationale:** Llama-3.2-3B-Instruct is small enough to run quickly on Apple Silicon, fits in 64 GB memory with headroom, and is instruction-tuned so it can follow chat prompts well for the live demo.

If mlx-lm residual-stream access proves insufficient, fallback to Hugging Face Transformers + forward hooks on the same model weights.

---

## 3. Planned Emotion Set (Phase 1)

8 core emotions:
1. joy
2. sadness
3. fear
4. anger
5. surprise
6. disgust
7. calm
8. love

These cover both high-arousal and low-arousal states across positive/negative valence, enabling valence–arousal geometry checks.

---

## 4. Validation Plan

1. **Contrastive construction:** Collect short emotion-eliciting vs neutral scenarios per emotion.
2. **Cosine similarity diagonal dominance:** Held-out emotion scenarios should score highest against their own vector.
3. **Pairwise similarity clustering:** Similar emotions (e.g., joy & love, fear & anxiety) should cluster together.
4. **PCA / MDS valence recovery:** A clear valence axis should emerge.
5. **Optional steering test:** Add scaled emotion vector to residual stream; observe text shift.

---

## 5. Geometry Plan

- Generate `pv.Icosphere(radius=1.0, nsub=N)` for N = 2, 3, 4.
- Choose nsub where 8 emotion regions remain perceptually distinct (start with nsub = 3 → ~512 facets).
- Map emotion vectors into 2D/3D (PCA or UMAP) and place similar emotions on neighboring facets.
- Facets are display addresses only; mapping preserves similarity structure.
