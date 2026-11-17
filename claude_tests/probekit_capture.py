#!/usr/bin/env python3
"""
probekit_capture.py -- residual-stream capture for mlx-lm models, done correctly.

Fixes vs. the old capture_residual_stream():
  * applies a CAUSAL attention mask (calling layer(h) with no mask => bidirectional attention,
    i.e. activations the model never produces in real use)
  * captures many layers in ONE forward pass and stops at the deepest one requested
  * returns per-token states so callers choose the pooling (and can skip BOS / early tokens,
    which carry huge outlier activations)

Layer convention (changed!): "layer L" = residual stream AFTER decoder block L (0-indexed),
i.e. the output of model.model.layers[L]. The old code captured the INPUT to block 18, which is
the output of block 17, so old "18" == new "17".

Verify on your install:  python probekit_capture.py --check
"""
from __future__ import annotations

import argparse
from typing import Dict, List, Sequence

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm import load


def causal_mask(T: int, dtype) -> "mx.array | None":
    """Additive causal mask for a fresh (empty-cache) sequence. Single tokens need no mask."""
    if T <= 1:
        return None
    return nn.MultiHeadAttention.create_additive_causal_mask(T).astype(dtype)


class Extractor:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model, self.tok = load(model_name)
        self.blocks = self.model.model.layers
        self.n_layers = len(self.blocks)
        # Gemma-family models multiply the embeddings by sqrt(hidden_size) inside their own forward pass; a manual layer loop
        # that skips it feeds the blocks activations at the wrong scale. Detected by model_type; verified by fidelity_check().
        mt = str(getattr(self.model, "model_type", "")).lower()
        self.embed_scale = float(self.model.model.args.hidden_size) ** 0.5 if mt.startswith("gemma") else 1.0

    def encode(self, text: str) -> List[int]:
        return list(self.tok.encode(text))

    def states_from_ids(self, ids: Sequence[int], layers: Sequence[int], use_mask: bool = True) -> Dict[int, np.ndarray]:
        """-> {layer: (T, H) float32}. use_mask=False exists only to demonstrate the old bug."""
        want, last = set(layers), max(layers)
        h = self.model.model.embed_tokens(mx.array([list(ids)])) * self.embed_scale
        mask = causal_mask(h.shape[1], h.dtype) if use_mask else None
        out = {}
        for i, block in enumerate(self.blocks):
            h = block(h, mask, None)
            if i in want:
                out[i] = np.array(h[0].astype(mx.float32))
            if i == last:
                break
        return out

    def states(self, text: str, layers: Sequence[int]) -> Dict[int, np.ndarray]:
        return self.states_from_ids(self.encode(text), layers)

    def pooled(self, text: str, layers: Sequence[int], skip: int = 4) -> Dict[int, np.ndarray]:
        """Mean over tokens [skip:], dropping BOS and the first few tokens (attention-sink outliers)."""
        st = self.states(text, layers)
        return {l: s[min(skip, len(s) - 1):].mean(0) for l, s in st.items()}

    def embedded_pooled(self, text: str, skip: int = 4) -> np.ndarray:
        """Bag-of-embeddings baseline: mean of the input token embeddings (before ANY decoder block) over tokens [skip:].
        No attention, no context: a purely lexical (token-identity) representation of the same passage."""
        e = np.array((self.model.model.embed_tokens(mx.array([self.encode(text)])) * self.embed_scale)[0].astype(mx.float32))
        return e[min(skip, len(e) - 1):].mean(0)

    PROMPT_FRAME = "\n\nThe emotion expressed in the text above is"

    def variant_states(self, text: str, layers: Sequence[int], skip: int = 4) -> Dict[int, Dict[str, np.ndarray]]:
        """Alternative poolings from ONE forward pass: last token, max-pool over tokens[skip:], mean of the last 4 tokens."""
        out = {}
        for l, s in self.states(text, layers).items():
            t = s[min(skip, len(s) - 1):]
            out[l] = {"last": s[-1], "max": t.max(0), "last4": s[-4:].mean(0)}
        return out

    def prompted_last(self, text: str, layers: Sequence[int], frame: str = None) -> Dict[int, np.ndarray]:
        """Prompted read-out: append a fixed framing sentence and return the last token's state per layer."""
        ids = self.encode(text + (self.PROMPT_FRAME if frame is None else frame))
        return {l: s[-1] for l, s in self.states_from_ids(ids, layers).items()}

    def fidelity_check(self, text: str) -> Dict[str, float]:
        """The manual layer loop (all blocks, then the model's final norm) must reproduce the model's own forward pass.
        Returns the max relative difference; large values mean the loop misses an architecture-specific step (e.g. embedding scale)."""
        ids = self.encode(text)
        mine = self.states_from_ids(ids, [self.n_layers - 1])[self.n_layers - 1]
        mine = np.array(self.model.model.norm(mx.array(mine[None])).astype(mx.float32))[0]
        ref = np.array(self.model.model(mx.array([ids]))[0].astype(mx.float32))
        return {"max_rel_diff": float(np.abs(mine - ref).max() / (np.abs(ref).max() + 1e-8)), "embed_scale": self.embed_scale}

    def token_states(self, text: str, layer: int, skip: int = 4) -> np.ndarray:
        s = self.states(text, [layer])[layer]
        return s[min(skip, len(s) - 1):]

    def causality_check(self, text: str, layer: int, k: int = 6) -> Dict[str, float]:
        """Earlier tokens' states must not change when later tokens are appended (causal model).
        Returns max relative difference with and without the mask."""
        ids = self.encode(text)
        k = min(k, len(ids) - 1)
        res = {}
        for use_mask in (True, False):
            full = self.states_from_ids(ids, [layer], use_mask)[layer][:k]
            pref = self.states_from_ids(ids[:k], [layer], use_mask)[layer]
            res["masked" if use_mask else "unmasked"] = float(
                np.abs(full - pref).max() / (np.abs(pref).max() + 1e-8))
        return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Llama-3.2-3B-Instruct")
    ap.add_argument("--layer", type=int, default=17)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--fidelity", action="store_true", help="manual layer loop must reproduce the model's own forward pass")
    a = ap.parse_args()
    ex = Extractor(a.model)
    print(f"{a.model}: {ex.n_layers} blocks")
    if a.check:
        r = ex.causality_check("I opened the letter slowly, and my hands began to shake as I read it.", a.layer)
        print(f"relative prefix mismatch @ layer {a.layer}: masked={r['masked']:.2e}  unmasked={r['unmasked']:.2e}")
        print("PASS" if r["masked"] < 5e-2 and r["unmasked"] > 5 * r["masked"] else
              "CHECK: masked should be tiny (fp16 noise ok) and unmasked clearly larger")
    if a.fidelity:
        f = ex.fidelity_check("I opened the letter slowly, and my hands began to shake as I read it.")
        print(f"fidelity: max relative diff vs the model's own forward = {f['max_rel_diff']:.2e} (embed scale {f['embed_scale']:.2f})")
        print("PASS" if f["max_rel_diff"] < 5e-2 else "FAIL: the manual loop does not match this architecture; do not use its states")
