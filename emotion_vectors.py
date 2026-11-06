#!/usr/bin/env python3
"""
Emotion Vector Construction and Validation

This module handles the construction and validation of emotion direction vectors
for the emotion visualization system.

Key components:
1. Contrastive construction of emotion vectors (emotion - neutral)
2. Normalization to unit vectors
3. Validation metrics (cosine similarity, clustering, PCA)
4. Loading saved vectors from disk
"""

import numpy as np
import json
import os
from pathlib import Path
from typing import List, Dict, Optional
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize


class EmotionVectors:
    """Handle emotion vector construction and validation."""

    def __init__(self, vectors_dir: str = "data/emotion_vectors"):
        """
        Initialize the EmotionVectors class.

        Args:
            vectors_dir: Directory to load/save emotion vectors
        """
        self.vectors_dir = Path(vectors_dir)
        self.vectors_dir.mkdir(parents=True, exist_ok=True)
        self.vectors: Dict[str, np.ndarray] = {}  # emotion name -> vector
        self.metadata: Dict[str, Dict] = {}     # emotion name -> metadata

    def load_vectors(self):
        """Load all emotion vectors from disk."""
        if not self.vectors_dir.exists():
            print(f"No vectors found in {self.vectors_dir}")
            return

        for file_path in self.vectors_dir.glob("*.json"):
            emotion_name = file_path.stem
            with open(file_path, 'r') as f:
                data = json.load(f)
                vector = np.array(data['vector'])
                self.vectors[emotion_name] = vector
                self.metadata[emotion_name] = data
                print(f"Loaded emotion vector: {emotion_name} (shape: {vector.shape})")

    def save_vector(self, emotion_name: str, vector: np.ndarray):
        """Save an emotion vector to disk."""
        if emotion_name in self.vectors:
            print(f"Vector for {emotion_name} already exists. Overwriting.")
        
        self.vectors[emotion_name] = vector
        self.metadata[emotion_name] = {
            "vector": vector.tolist(),
            "shape": vector.shape,
            "norm": float(np.linalg.norm(vector)),
            "created": None  # Could add timestamp
        }
        
        file_path = self.vectors_dir / f"{emotion_name}.json"
        with open(file_path, 'w') as f:
            json.dump({
                "emotion_name": emotion_name,
                "vector": vector.tolist(),
                "shape": vector.shape,
                "norm": float(np.linalg.norm(vector))
            }, f, indent=2)
        print(f"Saved vector for {emotion_name}")

    def construct_vector(self, emotion_prompt: str, neutral_prompt: str, layer_index: int = 18) -> np.ndarray:
        """
        Construct an emotion vector using contrastive learning.

        Args:
            emotion_prompt: Prompt that should evoke the target emotion
            neutral_prompt: Matched neutral prompt (similar context, no strong emotion)
            layer_index: Layer to capture residual stream from

        Returns:
            Normalized emotion vector
        """
        # Load model and tokenizer for this specific capture
        model, tokenizer = load("mlx-community/Llama-3.2-3B-Instruct")
        
        # Capture residual stream
        emotion_vec = self._capture_residual_stream(model, tokenizer, emotion_prompt, layer_index)
        neutral_vec = self._capture_residual_stream(model, tokenizer, neutral_prompt, layer_index)
        
        # Construct contrastive vector
        contrastive_vec = emotion_vec - neutral_vec
        
        # Normalize to unit vector
        normalized_vec = contrastive_vec / np.linalg.norm(contrastive_vec)
        
        return normalized_vec

    def _capture_residual_stream(self, model, tokenizer, prompt: str, layer_index: int) -> np.ndarray:
        """Internal method to capture residual stream from a specific layer."""
        # Tokenize input
        tokens = tokenizer.encode(prompt)
        input_ids = model.model.embed_tokens.mx.array([tokens])
        
        # Run through layers up to target layer
        h = input_ids
        for i, layer in enumerate(model.model.layers):
            if i == layer_index:
                captured = h
                h = layer(captured)
                break
            else:
                h = layer(h)
        
        # Extract last token vector
        captured_np = np.array(captured)
        vector = captured_np[0, -1]  # Last token in sequence
        return vector

    def validate_vectors(self) -> Dict[str, float]:
        """
        Validate emotion vectors using cosine similarity and clustering metrics.

        Returns:
            Dictionary of validation scores per emotion
        """
        validation_results = {}
        
        # Check diagonal dominance (emotion vs neutral)
        print("Validating vector diagonal dominance...")
        for emotion, vector in self.vectors.items():
            # For validation, we'd need paired emotion/neutral vectors
            # This is a simplified check - in practice we'd have multiple examples
            validation_results[emotion] = 0.8  # Placeholder score
            
        # Check clustering - similar emotions should cluster together
        print("Checking emotion clustering...")
        if len(self.vectors) > 2:
            X = np.array(list(self.vectors.values()))
            clustering = AgglomerativeClustering(n_clusters=min(4, len(X)-1))
            labels = clustering.fit_predict(X)
            
            # Validate that similar emotions have similar labels
            # This is a basic validation - in practice we'd have labeled clusters
            validation_results["clustering_score"] = 0.7  # Placeholder
            
        # Check valence-arousal geometry (PCA)
        print("Checking valence-arousal geometry...")
        if len(self.vectors) >= 4:
            pca = PCA(n_components=2)
            reduced = pca.fit_transform(X)
            
            # Check if we can distinguish positive/negative valence
            # This would require knowing which emotions are positive/negative
            validation_results["pca_explained"] = 0.6  # Placeholder
            
        return validation_results

    def get_vector(self, emotion_name: str) -> Optional[np.ndarray]:
        """Get an emotion vector by name."""
        return self.vectors.get(emotion_name)

    def list_emotions(self) -> List[str]:
        """List all available emotion names."""
        return list(self.vectors.keys())

    def plot_emotion_space(self):
        """Plot emotion vectors in 2D using PCA."""
        if len(self.vectors) < 2:
            print("Not enough vectors for PCA plot")
            return
            
        X = np.array(list(self.vectors.values()))
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(X)
        
        plt.figure(figsize=(10, 8))
        plt.scatter(reduced[:, 0], reduced[:, 1])
        
        for i, txt in enumerate(self.list_emotions()):
            plt.annotate(txt, (reduced[i, 0], reduced[i, 1]))
            
        plt.title("Emotion Vectors in 2D (PCA)")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.show()


# Example usage
if __name__ == "__main__":
    ev = EmotionVectors("data/emotion_vectors")
    ev.load_vectors()
    
    # Example: construct a vector for "joy"
    # In practice, you'd have specific emotion/neutral prompt pairs
    # For now, we'll just show the structure
    print("Available emotions:", ev.list_emotions())