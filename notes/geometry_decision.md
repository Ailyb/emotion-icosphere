# Geometry Decision — Icosphere Subdivision and Facet Mapping

## 1. Icosphere Subdivision (nsub)

Icosphere facets count follows: **(nsub + 2)^2** (approximately, for geodesic sphere approximations).

| nsub | Facet count (approx) | Visual characteristics |
|------|----------------------|------------------------|
| 2    | 16                   | Too coarse - each emotion would cover >10% of sphere |
| 3    | 81                   | Good balance - 8 emotions can occupy ~10 facets each |
| 4    | 324                  | Too fine-grained - regions become too small for perceptible differences |

### Recommended: nsub = 3

- **81 facets** provides enough granularity for 8–20 emotions while keeping regions visually distinct.
- Each emotion can be assigned a contiguous cluster of 3–10 facets, preserving similarity structure.
- Facet size is large enough for clear glow effects in rendering.

---

## 2. Facet Mapping Strategy

### 2.1 Emotion Grouping
- **Core emotions (8):** joy, sadness, fear, anger, surprise, disgust, calm, love
- **Clusters (optional):** 
  - Positive emotions: joy, love, calm
  - Negative emotions: sadness, fear, anger, surprise, disgust
  - High-arousal: fear, anger, surprise
  - Low-arousal: calm, sadness

### 2.2 Projection Approach
1. Extract emotion vectors (8–20 dimensions)
2. Reduce to 2D/3D using PCA or UMAP
3. Place emotion vectors on 2D/3D plane
4. Map 2D/3D positions to nearby facets on the icosphere

### 2.3 Facet Assignment Algorithm
1. Compute convex hull of emotion vectors in 2D/3D space
2. For each emotion vector:
   - Find nearest facet center on icosphere
   - Assign to facet or cluster of facets
   - Ensure spatial contiguity (neighboring facets)
3. Normalize scores to [0, 1] for glow intensity

---

## 3. Implementation Plan

1. **Prototype geometry** (nsub=3) → visualize and decide facet assignment
2. **Map emotion vectors** to facet clusters
3. **Render static visualization** with glow intensities
4. **Implement decay animation** for real-time effect

## 4. Next Steps

- Run `geometry_prototype.py` to visualize nsub=3
- Document facet count and decide exact region assignment scheme
- Update `config.py` with chosen nsub and facet mapping strategy