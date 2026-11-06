"""
Configuration for the Real-Time Emotion Visualization on an Icosphere.

Central configuration for:
- Model selection and inference backend
- Hidden layer capture
- Emotion scoring
- Geometry (icosphere subdivision)
- Streaming/decay settings
- Visualization parameters
"""

# ------------------------------------------------------------------
# Model Configuration
# ------------------------------------------------------------------
MODEL_NAME = "mlx-community/Llama-3.2-3B-Instruct"  # Small open model for prototyping
# Alternative small options:
#   - "mlx-community/Gemma-2-9B-it"
#   - "Qwen/Qwen2.5-3B-Instruct"
QUANTIZATION = "4bit"  # or "8bit", "fp16"
DEVICE = "mlx"  # "mlx" on Apple Silicon, "cuda" for NVIDIA

# Layers to capture activations from (0-indexed within decoder blocks)
# Mid-to-late layer is preferred for emotion signals.
CAPTURE_LAYER_INDICES = [18, 20, 24]  # mid, later, later-later (fallback set)
PRIMARY_LAYER = 18  # Primary layer used for emotion scoring

# Pooling strategy: "last_token" or "mean_pool"
EMOTION_POOLING = "last_token"

# ------------------------------------------------------------------
# Emotion Configuration
# ------------------------------------------------------------------
EMOTION_NAMES = [
    "joy",
    "sadness",
    "fear",
    "anger",
    "surprise",
    "disgust",
    "calm",
    "love",
]

EMOTION_VECTORS_DIR = "data/emotion_vectors"

# ------------------------------------------------------------------
# Geometry Configuration
# ------------------------------------------------------------------
ICOSPHERE_RADIUS = 1.0
ICOSPHERE_NSUB = 3  # Base subdivision; adjust per geometry_decision.md

# Facet mapping
FACET_MAX = None  # will be set after computing icosphere facet count
