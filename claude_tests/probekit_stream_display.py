#!/usr/bin/env python3
"""
probekit_stream_display.py -- viz_plan Option 2: model-free pieces of the icosphere v2 display (token-window scoring from T15, sentence splitting,
icosphere anchors from the probe directions). Used by backend_v2.py; tested by test_stream_display.py.
"""
import re

import numpy as np

from probekit_viz_stream import EMO, window_means

Z_FULL = 6.0            # fixed colour anchor (recalibrated z at which a face is fully lit); not tuned on any display, shown in the UI
WINDOW = 8              # T15-selected window (data/eval_merged2/<slug>/t15_stream.md)
_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text):
    """-> list of (start_char, end_char, sentence) for non-empty sentences."""
    out, pos = [], 0
    for part in _SPLIT.split(text.strip()):
        if not part:
            continue
        i = text.find(part, pos); out.append((i, i + len(part), part)); pos = i + len(part)
    return out


def anchors_from_directions(V):
    """(7,H) unit probe directions -> (7,3) unit vectors on the sphere: PCA of the directions, so emotions whose probe directions are similar
    sit near each other. Deterministic sign convention (first nonzero component of each PCA axis positive)."""
    X = V - V.mean(0); U, S, Vt = np.linalg.svd(X, full_matrices=False)
    P = X @ Vt[:3].T
    for j in range(P.shape[1]):
        if P[np.argmax(np.abs(P[:, j])), j] < 0:
            P[:, j] *= -1
    P = np.hstack([P, np.zeros((len(P), max(0, 3 - P.shape[1])))])
    P = P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-9)
    return P


class StreamEngine:
    """Token-window scorer: z' = (pooled-probe z over the last WINDOW tokens - mu) / sd, flagged when max z' > thr."""

    def __init__(self, ps, norm, thr, skip, window=WINDOW):
        self.ps, self.mu, self.sd, self.thr, self.skip, self.window = ps, norm[0], norm[1], float(thr), skip, window

    def positions(self, st):
        """st: (T,H) layer states of the whole text -> list of frame dicts (one per token position from `skip`)."""
        M, ts = window_means(st, self.skip, self.window)
        Z = (self.ps.score(M, "pooled") - self.mu) / self.sd
        frames = []
        for i, t in enumerate(ts):
            j = int(Z[i].argmax()); flag = bool(Z[i, j] > self.thr)
            frames.append(dict(pos=int(t), z=dict(zip(EMO, map(float, Z[i]))), flag=flag, top=EMO[j] if flag else None,
                               warmup=bool(t - self.skip + 1 < self.window)))
        return frames


def face_intensity(z, flag, top, z_full=Z_FULL):
    """Per-emotion light level in [0,1]: only the flagged top emotion lights up; everything else stays dark (neutral is the default look)."""
    return {e: (float(np.clip(z[e] / z_full, 0, 1)) if (flag and e == top) else 0.0) for e in z}
