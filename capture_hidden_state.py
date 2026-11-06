#!/usr/bin/env python3
"""
Capture residual-stream activations from a local open-weight LLM.

This script captures hidden states from specific layers of a Llama-3.2-3B-Instruct model
and saves them for emotion scoring and visualization.
"""

import mlx.core as mx
from mlx_lm import load
import json
import os
from typing import List, Dict
import numpy as np

# Configuration
MODEL_NAME = "mlx-community/Llama-3.2-3B-Instruct"

# Capture layer(s) - mid-to-late layers for emotion signal
CAPTURE_LAYER_INDEX = 18  # Primary layer
SECONDARY_LAYERS = [20, 24]  # Additional layers for comparison

# Pooling strategy: last-token residual stream
EMOTION_POOLING = "last_token"

# Output directory
OUTPUT_DIR = "data/hidden_states"


def capture_residual_stream(
    model,
    tokenizer,
    prompt: str,
    layer_index: int,
    pooling: str = "last_token",
) -> np.ndarray:
    """
    Capture residual-stream activations from a specific layer.
    
    Args:
        model: Loaded mlx-lm model
        tokenizer: Tokenizer
        prompt: Input prompt
        layer_index: Layer index to capture from
        pooling: Pooling strategy ("last_token" or "mean_pool")
    
    Returns:
        Residual stream vector(s) as numpy array
    """
    # Tokenize input
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])
    
    # Get embeddings
    h = model.model.embed_tokens(input_ids)
    
    # Run through layers up to target layer
    for i, layer in enumerate(model.model.layers):
        if i == layer_index:
            # Capture before applying the layer (residual stream input)
            captured = h
            # Apply the layer to continue forward pass
            h = layer(captured)  # TransformerBlock returns single mx.array
            break
        else:
            h = layer(h)  # TransformerBlock returns single mx.array
    
    # Convert to numpy
    captured_np = np.array(captured)
    
    # Apply pooling
    if pooling == "last_token":
        # Take last token in sequence (batch, seq, hidden) -> (seq, hidden) -> take last token
        vector = captured_np[0, -1]  # Shape: [hidden_size]
    elif pooling == "mean_pool":
        # Mean over sequence
        vector = captured_np.mean(axis=0)  # Shape: [hidden_size]
    else:
        vector = captured_np
    
    return vector


def capture_multiple_layers(
    model,
    tokenizer,
    prompt: str,
    layer_indices: List[int],
    pooling: str = "last_token",
) -> Dict[int, np.ndarray]:
    """
    Capture residual-stream activations from multiple layers.
    
    Returns:
        Dict mapping layer_index -> residual vector
    """
    results = {}
    for layer_idx in layer_indices:
        vec = capture_residual_stream(model, tokenizer, prompt, layer_idx, pooling)
        results[layer_idx] = vec
    return results


def save_vector(
    vector: np.ndarray,
    prompt: str,
    layer_index: int,
    filename: str,
):
    """Save captured vector with metadata."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    data = {
        "prompt": prompt,
        "layer_index": layer_index,
        "vector": vector.tolist(),
        "shape": vector.shape,
        "pooling": EMOTION_POOLING,
    }
    
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"Saved vector to {filename}")


def main():
    """Main entry point."""
    print("=" * 60)
    print("Model Inference + Residual-Stream Capture")
    print("=" * 60)
    
    # Load model
    print(f"\nLoading model: {MODEL_NAME}")
    model, tokenizer = load(MODEL_NAME)
    print(f"Model loaded. Hidden size: {model.model.args.hidden_size}")
    print(f"Number of layers: {model.model.num_hidden_layers}")
    
    # Test prompts
    test_prompts = [
        "The sun is shining brightly today.",
        "I feel very happy and joyful about this.",
        "This makes me feel sad and depressed.",
        "I am angry about what happened.",
    ]
    
    # Capture from primary layer for each prompt
    all_layer_indices = [CAPTURE_LAYER_INDEX] + SECONDARY_LAYERS
    
    print(f"\nCapturing from layers: {all_layer_indices}")
    print(f"Pooling strategy: {EMOTION_POOLING}")
    
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt[:50]}...")
        results = capture_multiple_layers(
            model, tokenizer, prompt, all_layer_indices, EMOTION_POOLING
        )
        
        for layer_idx, vec in results.items():
            filename = f"{OUTPUT_DIR}/prompt_{abs(hash(prompt))}_layer{layer_idx}.json"
            save_vector(vec, prompt, layer_idx, filename)
            print(f"  Layer {layer_idx}: shape {vec.shape}")
    
    print("\nCapture complete. Vectors saved to data/hidden_states/")


if __name__ == "__main__":
    main()