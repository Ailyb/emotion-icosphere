#!/usr/bin/env python3
"""
Similarity-Aware Facet Mapping for Icosphere

Projects emotion vectors to 2D/3D using PCA, then maps them to icosphere facets
so that similar emotions occupy neighboring regions on the sphere.
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


def load_emotion_vectors(vectors_dir: str = "data/emotion_vectors") -> Dict[str, np.ndarray]:
    """Load validated emotion vectors from disk."""
    vectors = {}
    vectors_dir = Path(vectors_dir)
    
    EMOTION_NAMES = [
        "joy", "sadness", "fear", "anger",
        "surprise", "disgust", "calm", "love"
    ]
    
    for emotion in EMOTION_NAMES:
        filepath = vectors_dir / f"{emotion}.json"
        if filepath.exists():
            with open(filepath, 'r') as f:
                data = json.load(f)
                vector = np.array(data['vector'])
                # Normalize to unit vector
                vector = vector / (np.linalg.norm(vector) + 1e-8)
                vectors[emotion] = vector
    
    return vectors


def generate_icosphere_facet_centers(nsub: int = 3, radius: float = 1.0) -> np.ndarray:
    """
    Generate icosphere facet centers using icosphere geometry.
    Returns array of shape (num_facets, 3) with x, y, z coordinates.
    
    For nsub=3, this gives 1280 facets (matching Three.js IcosahedronGeometry).
    """
    # Use PyVista if available, otherwise generate mathematically
    try:
        import pyvista as pv
        mesh = pv.Icosphere(radius=radius, nsub=nsub)
        # Get face centers
        centers = mesh.cell_centers().points
        return centers
    except ImportError:
        # Fallback: generate mathematically using icosphere subdivision
        return _generate_icosphere_centers_math(nsub, radius)


def _generate_icosphere_centers_math(nsub: int, radius: float) -> np.ndarray:
    """Generate icosphere facet centers mathematically."""
    # Golden ratio
    phi = (1 + np.sqrt(5)) / 2
    
    # Initial icosahedron vertices (12 vertices)
    vertices = np.array([
        [-1,  phi,  0], [ 1,  phi,  0], [-1, -phi,  0], [ 1, -phi,  0],
        [ 0, -1,  phi], [ 0,  1,  phi], [ 0, -1, -phi], [ 0,  1, -phi],
        [ phi,  0, -1], [ phi,  0,  1], [-phi,  0, -1], [-phi,  0,  1]
    ], dtype=float)
    
    # Normalize to radius
    vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True) * radius
    
    # Initial icosahedron faces (20 triangles)
    faces = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ])
    
    def subdivide(vertices, faces):
        """Subdivide each triangle into 4 smaller triangles."""
        new_vertices = list(vertices)
        vertex_map = {}  # (v1, v2) -> new_vertex_index
        
        def get_midpoint(v1, v2):
            key = tuple(sorted([v1, v2]))
            if key not in vertex_map:
                mid = (new_vertices[v1] + new_vertices[v2]) / 2
                mid = mid / np.linalg.norm(mid) * radius
                vertex_map[key] = len(new_vertices)
                new_vertices.append(mid)
            return vertex_map[key]
        
        new_faces = []
        for face in faces:
            a, b, c = face
            ab = get_midpoint(a, b)
            bc = get_midpoint(b, c)
            ca = get_midpoint(c, a)
            
            new_faces.append([a, ab, ca])
            new_faces.append([b, bc, ab])
            new_faces.append([c, ca, bc])
            new_faces.append([ab, bc, ca])
        
        return np.array(new_vertices), np.array(new_faces)
    
    # Subdivide nsub times
    current_vertices = vertices
    current_faces = faces
    for _ in range(nsub):
        current_vertices, current_faces = subdivide(current_vertices, current_faces)
    
    # Compute face centers
    centers = []
    for face in current_faces:
        center = (current_vertices[face[0]] + current_vertices[face[1]] + current_vertices[face[2]]) / 3
        center = center / np.linalg.norm(center) * radius
        centers.append(center)
    
    return np.array(centers)


def compute_similarity_aware_mapping(
    emotion_vectors: Dict[str, np.ndarray],
    nsub: int = 3,
    num_facets: int = 1280,
    radius: float = 1.0
) -> Tuple[Dict[str, List[int]], np.ndarray]:
    """
    Compute similarity-aware facet mapping.
    
    Returns:
        - facet_mapping: Dict[emotion_name -> list of facet indices]
        - emotion_positions_2d: 2D PCA positions for each emotion
    """
    emotions = list(emotion_vectors.keys())
    vectors = np.array([emotion_vectors[e] for e in emotions])
    
    # Project to 2D using PCA
    pca = PCA(n_components=2)
    positions_2d = pca.fit_transform(vectors)
    
    print(f"PCA explained variance: {pca.explained_variance_ratio_}")
    print(f"Total variance explained: {pca.explained_variance_ratio_.sum():.3f}")
    
    # Generate icosphere facet centers
    facet_centers = generate_icosphere_facet_centers(nsub, radius)
    print(f"Generated {len(facet_centers)} facet centers")
    
    # Project facet centers to 2D using the same PCA (project onto PC1-PC2 plane)
    # We need to project 3D sphere points onto the PCA plane
    # The PCA was fit on emotion vectors (3072D), so we can't directly use it for 3D points.
    # Instead, we'll map the 2D PCA positions to the sphere surface.
    
    # Normalize 2D positions to unit circle, then map to sphere
    # Scale positions to reasonable range
    pos_min = positions_2d.min(axis=0)
    pos_max = positions_2d.max(axis=0)
    pos_range = pos_max - pos_min
    if pos_range[0] == 0: pos_range[0] = 1
    if pos_range[1] == 0: pos_range[1] = 1
    
    # Normalize to [-1, 1] range
    positions_2d_norm = (positions_2d - pos_min) / pos_range * 2 - 1
    
    # Map 2D positions to 3D sphere using inverse azimuthal equidistant projection
    # or simple spherical mapping
    emotion_positions_3d = []
    for pos in positions_2d_norm:
        x, y = pos
        # Map to sphere: use x as longitude, y as latitude
        # Scale to reasonable range
        lon = x * np.pi * 0.8  # -144° to 144°
        lat = y * np.pi * 0.4  # -72° to 72°
        
        px = np.cos(lat) * np.sin(lon)
        py = np.sin(lat)
        pz = np.cos(lat) * np.cos(lon)
        emotion_positions_3d.append([px, py, pz])
    
    emotion_positions_3d = np.array(emotion_positions_3d)
    emotion_positions_3d = emotion_positions_3d / np.linalg.norm(emotion_positions_3d, axis=1, keepdims=True) * radius
    
    # Assign each facet to the nearest emotion in 3D space
    facet_mapping = {e: [] for e in emotions}
    
    for i, center in enumerate(facet_centers):
        # Find nearest emotion
        distances = np.linalg.norm(emotion_positions_3d - center, axis=1)
        nearest_idx = np.argmin(distances)
        nearest_emotion = emotions[nearest_idx]
        facet_mapping[nearest_emotion].append(i)
    
    # Print stats
    for emotion in emotions:
        print(f"  {emotion}: {len(facet_mapping[emotion])} facets")
    
    return facet_mapping, positions_2d


def map_scores_to_facets_similarity(
    scores: Dict[str, float],
    facet_mapping: Dict[str, List[int]],
    num_facets: int = 1280,
    spread: float = 0.3
) -> List[float]:
    """
    Map emotion scores to facet intensities using similarity-aware mapping.
    
    Args:
        scores: Dict of emotion -> cosine similarity score
        facet_mapping: Dict of emotion -> list of facet indices (from compute_similarity_aware_mapping)
        num_facets: Total number of facets
        spread: How much to spread intensity to neighboring facets (0-1)
    
    Returns:
        List of facet intensities
    """
    intensities = np.zeros(num_facets)
    
    # Apply ReLU to scores (only positive similarities)
    relu_scores = {e: max(s, 0.0) for e, s in scores.items()}
    
    # Assign scores to primary facets
    for emotion, facet_indices in facet_mapping.items():
        score = relu_scores.get(emotion, 0.0)
        for facet_idx in facet_indices:
            intensities[facet_idx] = max(intensities[facet_idx], score)
    
    # Optional: spread to neighbors for smoother transitions
    if spread > 0:
        # Simple diffusion: each facet gets a fraction of its neighbors' intensity
        # This requires adjacency info - for now we'll skip and just use primary assignment
        pass
    
    return intensities.tolist()


def save_mapping(facet_mapping: Dict[str, List[int]], positions_2d: np.ndarray, 
                 output_path: str = "data/emotion_vectors/facet_mapping.json"):
    """Save facet mapping to disk."""
    emotions = list(facet_mapping.keys())
    
    data = {
        "facet_mapping": facet_mapping,
        "emotion_positions_2d": {
            emotions[i]: positions_2d[i].tolist() 
            for i in range(len(emotions))
        },
        "num_facets": sum(len(v) for v in facet_mapping.values()),
        "emotions": emotions
    }
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Saved facet mapping to {output_path}")


def load_mapping(input_path: str = "data/emotion_vectors/facet_mapping.json") -> Tuple[Dict[str, List[int]], np.ndarray]:
    """Load facet mapping from disk."""
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    facet_mapping = data["facet_mapping"]
    emotions = data["emotions"]
    positions_2d = np.array([data["emotion_positions_2d"][e] for e in emotions])
    
    return facet_mapping, positions_2d


def main():
    """Build and save similarity-aware facet mapping."""
    print("=" * 60)
    print("Building Similarity-Aware Facet Mapping")
    print("=" * 60)
    
    # Load emotion vectors
    emotion_vectors = load_emotion_vectors()
    print(f"Loaded {len(emotion_vectors)} emotion vectors")
    
    # Compute mapping
    facet_mapping, positions_2d = compute_similarity_aware_mapping(
        emotion_vectors, nsub=3, num_facets=1280, radius=1.0
    )
    
    # Save mapping
    save_mapping(facet_mapping, positions_2d)
    
    print("\nDone!")
    
    # Test with sample scores
    print("\n--- Test Mapping ---")
    test_scores = {
        "joy": 0.8, "sadness": -0.1, "fear": 0.2, "anger": -0.2,
        "surprise": 0.3, "disgust": -0.3, "calm": 0.4, "love": 0.5
    }
    intensities = map_scores_to_facets_similarity(test_scores, facet_mapping)
    print(f"Test intensities: min={min(intensities):.3f}, max={max(intensities):.3f}, "
          f"nonzero={sum(1 for x in intensities if x > 0)}")


if __name__ == "__main__":
    main()
