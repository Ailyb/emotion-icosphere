#!/usr/bin/env python3
"""
Emotion Scoring Module

Scores residual-stream activations against emotion direction vectors
using cosine similarity.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import json


class EmotionScorer:
    """Score hidden state activations against emotion vectors."""

    def __init__(self, vectors: Dict[str, np.ndarray]):
        """
        Initialize with emotion vectors.

        Args:
            vectors: Dict mapping emotion_name -> unit vector
        """
        self.vectors = {k: v / np.linalg.norm(v) for k, v in vectors.items()}
        self.emotion_names = list(self.vectors.keys())

    def score(self, activation: np.ndarray) -> Dict[str, float]:
        """
        Score an activation vector against all emotion vectors.

        Args:
            activation: Residual stream activation vector (hidden_size,)

        Returns:
            Dict mapping emotion_name -> cosine similarity score
        """
        # Normalize activation
        activation_norm = activation / (np.linalg.norm(activation) + 1e-8)
        
        scores = {}
        for emotion, vector in self.vectors.items():
            # Cosine similarity = dot product of normalized vectors
            scores[emotion] = float(np.dot(activation_norm, vector))
        
        return scores

    def score_batch(self, activations: np.ndarray) -> List[Dict[str, float]]:
        """
        Score multiple activations.

        Args:
            activations: Array of shape (batch, hidden_size)

        Returns:
            List of score dicts
        """
        results = []
        for i in range(activations.shape[0]):
            results.append(self.score(activations[i]))
        return results

    def get_top_emotions(self, scores: Dict[str, float], k: int = 3) -> List[Tuple[str, float]]:
        """
        Get top k emotions by score.

        Args:
            scores: Dict of emotion -> score
            k: Number of top emotions to return

        Returns:
            List of (emotion, score) tuples sorted by score descending
        """
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_scores[:k]

    def map_to_facets(self, scores: Dict[str, float], facet_mapping: Dict[str, List[int]]) -> np.ndarray:
        """
        Map emotion scores to icosphere facet intensities.

        Args:
            scores: Dict of emotion -> score
            facet_mapping: Dict mapping emotion_name -> list of facet indices

        Returns:
            Array of facet intensities (num_facets,)
        """
        # Determine number of facets
        all_facets = set()
        for facets in facet_mapping.values():
            all_facets.update(facets)
        num_facets = max(all_facets) + 1 if all_facets else 0
        
        # Initialize facet intensities
        facet_intensities = np.zeros(num_facets)
        
        # Assign scores to facets
        for emotion, score in scores.items():
            if emotion in facet_mapping:
                # Normalize score to [0, 1] using ReLU or softplus
                intensity = np.maximum(score, 0)  # ReLU - only positive similarities
                for facet_idx in facet_mapping[emotion]:
                    facet_intensities[facet_idx] = max(facet_intensities[facet_idx], intensity)
        
        return facet_intensities


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    a_norm = a / (np.linalg.norm(a) + 1e-8)
    b_norm = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a_norm, b_norm))


# Example usage
if __name__ == "__main__":
    # Example emotion vectors (random for testing)
    np.random.seed(42)
    hidden_size = 3072
    emotions = ["joy", "sadness", "fear", "anger", "surprise", "disgust", "calm", "love"]
    
    vectors = {}
    for emotion in emotions:
        v = np.random.randn(hidden_size)
        v = v / np.linalg.norm(v)
        vectors[emotion] = v
    
    scorer = EmotionScorer(vectors)
    
    # Test with random activation
    activation = np.random.randn(hidden_size)
    scores = scorer.score(activation)
    
    print("Emotion scores:")
    for emotion, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        print(f"  {emotion}: {score:.4f}")
    
    top = scorer.get_top_emotions(scores, k=3)
    print(f"\nTop 3: {top}")