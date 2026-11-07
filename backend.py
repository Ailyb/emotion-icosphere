#!/usr/bin/env python3
"""
FastAPI Backend with WebSocket streaming for real-time emotion visualization.

Accepts prompts, runs model inference, captures residual-stream activations,
scores them against validated emotion direction vectors via cosine similarity,
and streams results to frontend.
"""

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import mlx.core as mx
from mlx_lm import load
from mlx_lm.sample_utils import make_sampler
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Emotion Icosphere Backend")

# Add CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== Configuration ====================
MODEL_NAME = "mlx-community/Llama-3.2-3B-Instruct"
CAPTURE_LAYER_INDEX = 18
EMOTION_POOLING = "last_token"
DECAY_RATE = 0.8  # Exponential decay factor per frame
EMOTION_VECTORS_DIR = "data/emotion_vectors"

# Emotion names matching the visualization set
EMOTION_NAMES = [
    "joy", "sadness", "fear", "anger",
    "surprise", "disgust", "calm", "love"
]

NUM_FACETS = 1280  # nsub=3 icosphere cell count

# ==================== Model & State ====================
model = None
tokenizer = None

# Load validated emotion vectors
EMOTION_VECTORS: Dict[str, np.ndarray] = {}

# Similarity-aware facet mapping
FACET_MAPPING: Dict[str, List[int]] = {}

# Store active emotion intensities (with decay)
current_intensities: Dict[int, float] = {}  # cell_index -> intensity

# ==================== Pydantic Models ====================
class PromptRequest(BaseModel):
    prompt: str
    max_tokens: int = 50
    temperature: float = 0.7
    stream: bool = True


class ScoreResponse(BaseModel):
    token: str
    emotion_scores: Dict[str, float]
    facet_intensities: List[float]
    layer: int


# ==================== Helpers ====================

def load_emotion_vectors():
    """Load validated emotion vectors from disk."""
    global EMOTION_VECTORS
    vectors_dir = Path(EMOTION_VECTORS_DIR)
    
    if not vectors_dir.exists():
        logger.warning(f"No emotion vectors directory found at {vectors_dir}")
        return
    
    for emotion in EMOTION_NAMES:
        filepath = vectors_dir / f"{emotion}.json"
        if filepath.exists():
            with open(filepath, 'r') as f:
                data = json.load(f)
                vector = np.array(data['vector'])
                # Normalize to unit vector
                vector = vector / (np.linalg.norm(vector) + 1e-8)
                EMOTION_VECTORS[emotion] = vector
                logger.info(f"Loaded emotion vector: {emotion}")
        else:
            logger.warning(f"Missing emotion vector: {emotion}")
    
    logger.info(f"Loaded {len(EMOTION_VECTORS)}/{len(EMOTION_NAMES)} emotion vectors")


def load_facet_mapping():
    """Load similarity-aware facet mapping from disk."""
    global FACET_MAPPING
    mapping_path = Path(EMOTION_VECTORS_DIR) / "facet_mapping.json"
    
    if not mapping_path.exists():
        logger.warning(f"No facet mapping found at {mapping_path}, falling back to equal distribution")
        # Fallback: equal distribution
        cells_per_emotion = NUM_FACETS // len(EMOTION_NAMES)
        for i, emotion in enumerate(EMOTION_NAMES):
            start_idx = i * cells_per_emotion
            end_idx = start_idx + cells_per_emotion if i < len(EMOTION_NAMES) - 1 else NUM_FACETS
            FACET_MAPPING[emotion] = list(range(start_idx, end_idx))
        return
    
    with open(mapping_path, 'r') as f:
        data = json.load(f)
    
    FACET_MAPPING = data["facet_mapping"]
    logger.info(f"Loaded similarity-aware facet mapping: {sum(len(v) for v in FACET_MAPPING.values())} facets")
    for emotion, facets in FACET_MAPPING.items():
        logger.info(f"  {emotion}: {len(facets)} facets")


def load_model():
    """Load the mlx-lm model and tokenizer."""
    global model, tokenizer
    logger.info(f"Loading model: {MODEL_NAME}")
    model, tokenizer = load(MODEL_NAME)
    logger.info(f"Model loaded. Hidden size: {model.model.args.hidden_size}")
    logger.info(f"Number of layers: {model.model.num_hidden_layers}")


def capture_residual_stream(prompt: str, layer_index: int = CAPTURE_LAYER_INDEX) -> np.ndarray:
    """
    Capture residual-stream activation for a prompt at a specific layer.
    Returns last-token residual stream vector.
    """
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
    # Last token pooling
    vector = captured_np[0, -1]  # Shape: [hidden_size]
    return vector


def score_emotion_vector(activation: np.ndarray) -> Dict[str, float]:
    """
    Score an activation vector against all emotion vectors using cosine similarity.
    Returns dict of emotion_name -> score.
    """
    activation_norm = activation / (np.linalg.norm(activation) + 1e-8)
    scores = {}
    for emotion, vector in EMOTION_VECTORS.items():
        scores[emotion] = float(np.dot(activation_norm, vector))
    return scores


def map_scores_to_facets(scores: Dict[str, float], num_facets: int = NUM_FACETS) -> List[float]:
    """
    Map emotion scores to facet intensities on the icosphere.
    
    Uses similarity-aware mapping: facets assigned via PCA projection of emotion vectors,
    so similar emotions occupy neighboring regions on the sphere.
    """
    intensities = np.zeros(num_facets)
    
    # Apply ReLU - only positive similarities glow
    relu_scores = {e: max(s, 0.0) for e, s in scores.items()}
    
    # Assign scores to facets based on similarity-aware mapping
    for emotion, facet_indices in FACET_MAPPING.items():
        score = relu_scores.get(emotion, 0.0)
        for facet_idx in facet_indices:
            intensities[facet_idx] = max(intensities[facet_idx], score)
    
    return intensities.tolist()


def apply_decay(intensities: Dict[int, float], decay: float = DECAY_RATE) -> Dict[int, float]:
    """
    Apply exponential decay to current intensities to produce fading effect.
    """
    decayed = {}
    for cell_idx, intensity in intensities.items():
        new_intensity = intensity * decay
        if new_intensity > 0.01:  # Threshold to avoid near-zero values
            decayed[cell_idx] = new_intensity
    return decayed


# ==================== Startup Event ====================

@app.on_event("startup")
async def startup():
    load_model()
    load_emotion_vectors()
    load_facet_mapping()


# ==================== WebSocket Handler ====================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time emotion streaming.
    
    Protocol:
    - Client sends: {"prompt": "...", "max_tokens": 50, ...}
    - Server sends: {"token": "...", "emotion_scores": {...}, "facet_intensities": [...]}
    """
    await websocket.accept()
    logger.info("WebSocket client connected")
    
    try:
        while True:
            # Receive prompt from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            prompt = message.get("prompt", "")
            max_tokens = message.get("max_tokens", 30)
            temperature = message.get("temperature", 0.7)
            
            logger.info(f"Received prompt: {prompt[:50]}...")
            
            # Stream generation token by token
            try:
                async for token_data in stream_emotion_tokens(prompt, max_tokens, temperature):
                    await websocket.send_text(json.dumps(token_data))
            except Exception as e:
                logger.exception("Prompt processing error")
                await websocket.send_text(json.dumps({
                    "event": "error",
                    "message": str(e)
                }))
                
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")


async def stream_emotion_tokens(prompt: str, max_tokens: int, temperature: float):
    """
    Generator that streams tokens with emotion scores.
    Runs model forward pass on main thread (no asyncio.to_thread).
    """
    current_text = prompt
    
    for token_idx in range(max_tokens):
        # Capture residual stream at current state
        activation = capture_residual_stream(current_text, CAPTURE_LAYER_INDEX)
        
        # Score emotions
        emotion_scores = score_emotion_vector(activation)
        
        # Map to facets
        facet_intensities = map_scores_to_facets(emotion_scores, NUM_FACETS)
        
        # Apply decay to existing intensities for fading effect
        global current_intensities
        current_intensities = apply_decay(current_intensities)
        
        # Add new intensities
        for i, intensity in enumerate(facet_intensities):
            if intensity > current_intensities.get(i, 0):
                current_intensities[i] = intensity
        
        # Convert current intensities to list for sending
        current_intensities_list = [
            current_intensities.get(i, 0.0) for i in range(NUM_FACETS)
        ]
        
        response_data = {
            "emotion_scores": emotion_scores,
            "facet_intensities": current_intensities_list,
            "layer": CAPTURE_LAYER_INDEX,
            "timestamp": time.time()
        }
        
        # Generate next token using simple forward pass (on main thread)
        try:
            # Encode current text
            tokens = tokenizer.encode(current_text)
            input_ids = mx.array([tokens])
            
            # Run full forward pass
            logits = model(input_ids)
            
            # Get logits for last token
            last_logits = logits[0, -1, :]  # Shape: [vocab_size]
            
            # Apply temperature
            if temperature > 0:
                last_logits = last_logits / temperature
            
            # Sample
            sampler = make_sampler(temp=temperature)
            next_token = sampler(last_logits)
            
            # Detokenize
            detokenizer = tokenizer.detokenizer
            detokenizer.add_token(int(next_token.item()))
            next_token_text = detokenizer.last_segment
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
            yield {
                "event": "error",
                "message": "Text generation failed. Check the backend log for details."
            }
            return
        
        response_data["token"] = next_token_text
        current_text += next_token_text
        
        yield response_data
        
        # Small delay for visual effect (yields to event loop)
        await asyncio.sleep(0.1)

    yield {"event": "complete"}


# ==================== REST API Endpoints ====================

@app.get("/")
async def root():
    return {"status": "ok", "model": MODEL_NAME, "layer": CAPTURE_LAYER_INDEX}


@app.get("/emotions")
async def list_emotions():
    return {"emotions": EMOTION_NAMES}


@app.get("/facet_mapping")
async def get_facet_mapping():
    """Get facet assignments and counts for the visualization."""
    return {
        "facet_mapping": {e: len(v) for e, v in FACET_MAPPING.items()},
        "facet_assignments": FACET_MAPPING,
        "total_facets": sum(len(v) for v in FACET_MAPPING.values())
    }


@app.post("/score")
async def score_prompt(request: PromptRequest):
    """
    Single-shot scoring endpoint (non-streaming).
    Runs generation once and returns final emotion scores.
    """
    activation = capture_residual_stream(request.prompt, CAPTURE_LAYER_INDEX)
    emotion_scores = score_emotion_vector(activation)
    facet_intensities = map_scores_to_facets(emotion_scores, NUM_FACETS)
    
    return ScoreResponse(
        token="",
        emotion_scores=emotion_scores,
        facet_intensities=facet_intensities,
        layer=CAPTURE_LAYER_INDEX
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
