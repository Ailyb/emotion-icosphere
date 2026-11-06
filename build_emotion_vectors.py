#!/usr/bin/env python3
"""
Build validated emotion vectors via contrastive construction.

For each emotion:
1. Capture residual-stream activation on emotion-eliciting story
2. Capture residual-stream activation on matched neutral story
3. Construct vector = normalize(emotion_vec - neutral_vec)
4. Save to data/emotion_vectors/

Then run validation: diagonal dominance, clustering, PCA valence recovery.
"""

import mlx.core as mx
from mlx_lm import load
import json
import os
import numpy as np
from typing import Dict, List, Tuple
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize as sk_normalize
import matplotlib.pyplot as plt

# Configuration
MODEL_NAME = "mlx-community/Llama-3.2-3B-Instruct"
CAPTURE_LAYER_INDEX = 18
EMOTION_VECTORS_DIR = "data/emotion_vectors"

# Emotion-eliciting vs neutral story pairs
# Each pair: (emotion prompt, neutral prompt with similar structure but no strong emotion)
EMOTION_PAIRS = {
    "joy": (
        "I just found out I got accepted to my dream college and my whole family is celebrating together.",
        "I received a letter from the admissions office and my family gathered to read it together."
    ),
    "sadness": (
        "My beloved dog of fifteen years passed away peacefully, and I miss him terribly every day.",
        "My dog lived a long life of fifteen years and spent time with our family every day."
    ),
    "fear": (
        "I hear footsteps behind me in the dark alley and my heart starts racing as I realize I'm not alone.",
        "I hear footsteps behind me on the street and notice there is another person walking nearby."
    ),
    "anger": (
        "Someone stole my wallet and used my credit card to buy expensive items without my permission.",
        "Someone found my wallet and returned it to the lost and found at the store."
    ),
    "surprise": (
        "When I opened the door, there was a marching band playing my favorite song right in front of my house.",
        "When I opened the door, there was a delivery person with a package addressed to me."
    ),
    "disgust": (
        "The milk had been left out for days and smelled so rotten that I had to throw it away immediately.",
        "The milk had been in the refrigerator for several days and I poured myself a glass."
    ),
    "calm": (
        "Sitting by the quiet lake at sunset, the water is still and I feel completely at peace.",
        "Sitting by the lake at sunset, the water is calm and I look at the horizon."
    ),
    "love": (
        "Holding my partner's hand as we walk through the park, I feel a deep warmth and connection.",
        "Walking through the park with my partner, we hold hands and enjoy the afternoon together."
    ),
}


def capture_residual_stream(model, tokenizer, prompt: str, layer_index: int) -> np.ndarray:
    """Capture residual-stream activation at the specified layer."""
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])

    h = model.model.embed_tokens(input_ids)

    for i, layer in enumerate(model.model.layers):
        if i == layer_index:
            captured = h
            h = layer(captured)
            break
        else:
            h = layer(h)

    captured_np = np.array(captured)
    # Last-token pooling
    vector = captured_np[0, -1]
    return vector


def build_emotion_vector(model, tokenizer, emotion: str, emotion_prompt: str, neutral_prompt: str) -> Tuple[np.ndarray, Dict]:
    """Build a single emotion vector via contrastive construction."""
    print(f"\nBuilding vector for '{emotion}'...")
    
    # Capture both activations
    emotion_vec = capture_residual_stream(model, tokenizer, emotion_prompt, CAPTURE_LAYER_INDEX)
    neutral_vec = capture_residual_stream(model, tokenizer, neutral_prompt, CAPTURE_LAYER_INDEX)
    
    # Contrastive: emotion - neutral
    contrastive = emotion_vec - neutral_vec
    
    # Normalize to unit vector
    norm = np.linalg.norm(contrastive)
    normalized = contrastive / norm if norm > 0 else contrastive
    
    metadata = {
        "emotion": emotion,
        "emotion_prompt": emotion_prompt,
        "neutral_prompt": neutral_prompt,
        "layer": CAPTURE_LAYER_INDEX,
        "contrastive_norm_before": float(norm),
        "shape": normalized.shape
    }
    
    print(f"  Contrastive norm: {norm:.4f}")
    print(f"  Vector shape: {normalized.shape}")
    
    return normalized, metadata


def save_vector(emotion: str, vector: np.ndarray, metadata: Dict):
    """Save emotion vector to disk."""
    os.makedirs(EMOTION_VECTORS_DIR, exist_ok=True)
    
    filepath = os.path.join(EMOTION_VECTORS_DIR, f"{emotion}.json")
    data = {
        "emotion": emotion,
        "vector": vector.tolist(),
        "shape": vector.shape,
        "norm": float(np.linalg.norm(vector)),
        "metadata": metadata
    }
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"  Saved to {filepath}")


def load_vectors() -> Dict[str, np.ndarray]:
    """Load all emotion vectors from disk."""
    vectors = {}
    if not os.path.exists(EMOTION_VECTORS_DIR):
        return vectors
    
    for filename in os.listdir(EMOTION_VECTORS_DIR):
        if filename.endswith('.json'):
            filepath = os.path.join(EMOTION_VECTORS_DIR, filename)
            with open(filepath, 'r') as f:
                data = json.load(f)
                vectors[data['emotion']] = np.array(data['vector'])
    
    return vectors


def validate_vectors(vectors: Dict[str, np.ndarray]):
    """Run validation suite on emotion vectors."""
    print("\n" + "=" * 60)
    print("EMOTION VECTOR VALIDATION")
    print("=" * 60)
    
    emotions = list(vectors.keys())
    print(f"\nValidating {len(emotions)} emotion vectors...")
    
    # Stack all vectors
    X = np.array([vectors[e] for e in emotions])
    
    # 1. PCA - check valence-arousal geometry
    print("\n--- PCA Analysis ---")
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(X)
    print(f"PC1 explained variance: {pca.explained_variance_ratio_[0]:.3f}")
    print(f"PC2 explained variance: {pca.explained_variance_ratio_[1]:.3f}")
    print(f"Total explained variance: {pca.explained_variance_ratio_.sum():.3f}")
    
    # Save PCA plot
    plt.figure(figsize=(10, 8))
    for i, emotion in enumerate(emotions):
        plt.scatter(reduced[i, 0], reduced[i, 1], s=100, label=emotion, alpha=0.7)
        plt.annotate(emotion, (reduced[i, 0], reduced[i, 1]), xytext=(5, 5), 
                    textcoords='offset points', fontsize=10)
    plt.title("Emotion Vectors in 2D (PCA)")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(EMOTION_VECTORS_DIR, "pca_plot.png"), dpi=150)
    print("  PCA plot saved to data/emotion_vectors/pca_plot.png")
    
    # 2. Hierarchical clustering
    print("\n--- Hierarchical Clustering ---")
    clustering = AgglomerativeClustering(n_clusters=min(3, len(emotions) - 1))
    labels = clustering.fit_predict(X)
    
    for emotion, label in zip(emotions, labels):
        print(f"  {emotion}: cluster {label}")
    
    # Save clustering info
    cluster_data = {emotion: int(label) for emotion, label in zip(emotions, labels)}
    with open(os.path.join(EMOTION_VECTORS_DIR, "clusters.json"), 'w') as f:
        json.dump(cluster_data, f, indent=2)
    
    # 3. Pairwise cosine similarity matrix
    print("\n--- Pairwise Cosine Similarity ---")
    X_norm = sk_normalize(X)
    sim_matrix = X_norm @ X_norm.T
    
    # Print matrix
    header = " " * 12 + " ".join(f"{e[:8]:>8}" for e in emotions)
    print(header)
    for i, emotion in enumerate(emotions):
        row = f"{emotion[:12]:12} " + " ".join(f"{sim_matrix[i,j]:8.3f}" for j in range(len(emotions)))
        print(row)
    
    # Check diagonal dominance (each emotion should be most similar to itself)
    print("\n--- Diagonal Dominance Check ---")
    all_dominant = True
    for i, emotion in enumerate(emotions):
        row = sim_matrix[i].copy()
        row[i] = -1  # Exclude self
        max_other = row.max()
        if sim_matrix[i,i] <= max_other:
            print(f"  {emotion}: NOT dominant (self={sim_matrix[i,i]:.3f}, max_other={max_other:.3f})")
            all_dominant = False
        else:
            print(f"  {emotion}: dominant (self={sim_matrix[i,i]:.3f})")
    
    if all_dominant:
        print("  All emotion vectors show diagonal dominance!")
    
    # Save validation results
    validation = {
        "diagonal_dominant": all_dominant,
        "pca_explained_variance": pca.explained_variance_ratio_.tolist(),
        "clusters": cluster_data,
        "pairwise_similarities": sim_matrix.tolist(),
        "emotions": emotions
    }
    with open(os.path.join(EMOTION_VECTORS_DIR, "validation.json"), 'w') as f:
        json.dump(validation, f, indent=2)
    
    print("\nValidation results saved to data/emotion_vectors/validation.json")
    return validation


def main():
    print("=" * 60)
    print("Building Emotion Vectors via Contrastive Construction")
    print("=" * 60)
    
    # Load model
    print(f"\nLoading model: {MODEL_NAME}")
    model, tokenizer = load(MODEL_NAME)
    print(f"Model loaded. Hidden size: {model.model.args.hidden_size}")
    print(f"Number of layers: {model.model.num_hidden_layers}")
    
    # Build vectors for each emotion
    all_vectors = {}
    
    for emotion, (emotion_prompt, neutral_prompt) in EMOTION_PAIRS.items():
        vector, metadata = build_emotion_vector(model, tokenizer, emotion, emotion_prompt, neutral_prompt)
        save_vector(emotion, vector, metadata)
        all_vectors[emotion] = vector
    
    print(f"\n{'=' * 60}")
    print(f"Built {len(all_vectors)} emotion vectors")
    print(f"{'=' * 60}")
    
    # Run validation
    validation = validate_vectors(all_vectors)
    
    print("\nDone! Emotion vectors saved to data/emotion_vectors/")


if __name__ == "__main__":
    main()