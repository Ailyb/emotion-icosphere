#!/usr/bin/env python3
"""
probekit_probes.py -- fit, score, and evaluate concept probes (numpy/sklearn only).

No MLX imports here on purpose: this module is shared by the offline builder and
the live backend, and it can be unit-tested without a model.

Scoring pipeline for an activation x (residual stream, one layer):
    1. center:           x' = x - mu                 (mu = mean of neutral training texts)
    2. remove nuisance:  x'' = x' - B^T B x'         (B = top within-class PCs: topic/style/length)
    3. project:          p_e = x'' . v_e             (v_e unit direction for concept e)
    4. standardize:      z_e = (p_e - bias_e) / sigma_e   (sigma from NEUTRAL text, so z = "sigmas above neutral")

z-scores replace the old raw cosine + "auto contrast" display hack: a z of 3 means the same
thing for every concept, layer, and model, which is what you need for alert thresholds.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

NEUTRAL = "neutral"
DEFAULT_EMOTIONS = ["joy", "sadness", "fear", "anger", "surprise", "disgust", "calm", "love"]
EPS = 1e-8


@dataclass
class ProbeSet:
    emotions: List[str]
    layer: int
    mu: np.ndarray          # (H,)
    basis: np.ndarray       # (k, H) orthonormal nuisance directions (k may be 0)
    V: np.ndarray           # (E, H) unit concept directions (already nuisance-free)
    bias_pool: np.ndarray   # (E,) calibration for mean-pooled passages
    sigma_pool: np.ndarray  # (E,)
    bias_tok: np.ndarray    # (E,) calibration for single-token states (live streaming)
    sigma_tok: np.ndarray   # (E,)
    meta: dict = field(default_factory=dict)

    # ---- scoring -------------------------------------------------------
    def project(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        Xc = X - self.mu.astype(np.float32)
        if len(self.basis):
            B = self.basis.astype(np.float32)
            Xc = Xc - (Xc @ B.T) @ B
        return Xc @ self.V.astype(np.float32).T

    def score(self, X: np.ndarray, mode: str = "token") -> np.ndarray:
        """X: (..., H) -> z-scores (..., E). mode: 'token' | 'pooled' | 'raw'."""
        proj = self.project(X)
        if mode == "raw":
            return proj
        if mode == "pooled":
            b, s = self.bias_pool, self.sigma_pool
        elif mode == "token":
            b, s = self.bias_tok, self.sigma_tok
        else:
            raise ValueError(f"unknown mode {mode!r}")
        return (proj - b.astype(np.float32)) / s.astype(np.float32)

    # ---- token-level calibration (needs per-token states) ---------------
    def calibrate_tokens(self, neutral_tokens: np.ndarray, emotional_tokens: Optional[np.ndarray] = None,
                         pct: float = 90.0) -> None:
        """Single-token states are noisier than mean-pooled passages, so they need their own sigma.
        Also records z_full: the `pct` percentile of max-z on emotional tokens (UI full-scale anchor)."""
        proj = self.project(neutral_tokens)
        self.bias_tok = proj.mean(0).astype(np.float64)
        self.sigma_tok = proj.std(0).astype(np.float64) + EPS
        if emotional_tokens is not None and len(emotional_tokens):
            z = self.score(emotional_tokens, "token")
            self.meta["z_full"] = float(np.percentile(z.max(1), pct))

    # ---- persistence -----------------------------------------------------
    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path, emotions=np.array(self.emotions), layer=self.layer, mu=self.mu, basis=self.basis,
            V=self.V, bias_pool=self.bias_pool, sigma_pool=self.sigma_pool,
            bias_tok=self.bias_tok, sigma_tok=self.sigma_tok, meta=json.dumps(self.meta),
        )

    @staticmethod
    def load(path) -> "ProbeSet":
        d = np.load(path, allow_pickle=False)
        return ProbeSet(
            emotions=[str(e) for e in d["emotions"]], layer=int(d["layer"]), mu=d["mu"], basis=d["basis"],
            V=d["V"], bias_pool=d["bias_pool"], sigma_pool=d["sigma_pool"],
            bias_tok=d["bias_tok"], sigma_tok=d["sigma_tok"], meta=json.loads(str(d["meta"])),
        )


# ======================================================================
# Fitting
# ======================================================================
def _nuisance_basis(X: np.ndarray, cats: np.ndarray, var_frac: float, max_k: int) -> np.ndarray:
    """Top principal components of WITHIN-category variation (topic, style, length, outlier dims).
    Removing them can't remove between-category signal by construction of the residuals."""
    if var_frac <= 0 or max_k <= 0:
        return np.zeros((0, X.shape[1]))
    R = X.copy()
    for c in np.unique(cats):
        m = cats == c
        R[m] -= R[m].mean(0)
    _, S, Vt = np.linalg.svd(R, full_matrices=False)
    var = S ** 2
    frac = np.cumsum(var) / var.sum()
    k = int(min(max_k, np.searchsorted(frac, var_frac) + 1))
    return Vt[:k]


def fit_probes(X: np.ndarray, cats, emotions: List[str], layer: int, var_frac: float = 0.9,
               max_k: int = 24, contrast: str = "mean_emotion") -> ProbeSet:
    """
    X: (N, H) mean-pooled passage states at one layer. cats: (N,) category per row (emotion or 'neutral').

    contrast='mean_emotion' (default): v_e = mean(e) - mean over emotion class means. Directions are
        discriminative (joy-vs-the-others) and don't share a generic 'this text is emotional' component.
    contrast='neutral': v_e = mean(e) - mean(neutral). Detects 'emotional at all' but concepts are correlated.
    """
    X = np.asarray(X, dtype=np.float64)
    cats = np.asarray(cats)
    missing = [e for e in emotions if not (cats == e).any()]
    if missing:
        raise ValueError(f"no training examples for emotion(s) {missing}; labels present: {sorted(set(cats.tolist()))}")
    neutral = X[cats == NEUTRAL]
    if len(neutral) < 5:
        raise ValueError("need >=5 neutral training texts for baseline/calibration")
    mu = neutral.mean(0)
    B = _nuisance_basis(X, cats, var_frac, max_k)

    def clean(A):
        A = A - mu
        return A - (A @ B.T) @ B if len(B) else A

    C = np.stack([clean(X[cats == e]).mean(0) for e in emotions])
    D = C - C.mean(0, keepdims=True) if contrast == "mean_emotion" else C
    V = D / (np.linalg.norm(D, axis=1, keepdims=True) + EPS)

    proj = clean(neutral) @ V.T
    bias, sigma = proj.mean(0), proj.std(0) + EPS
    return ProbeSet(
        emotions=list(emotions), layer=layer, mu=mu, basis=B, V=V,
        bias_pool=bias, sigma_pool=sigma, bias_tok=bias.copy(), sigma_tok=sigma.copy(),
        meta={"contrast": contrast, "nuisance_k": int(len(B)), "nuisance_var_frac": var_frac},
    )


# ======================================================================
# Evaluation
# ======================================================================
def evaluate(ps: ProbeSet, X: np.ndarray, cats, mode: str = "pooled") -> dict:
    cats = np.asarray(cats)
    Z = ps.score(X, mode)
    emo = cats != NEUTRAL
    pred = np.array(ps.emotions)[Z[emo].argmax(1)]
    y = cats[emo]
    out = {
        "n_eval": int(len(cats)),
        "n_per_emotion": {e: int((cats == e).sum()) for e in ps.emotions},
        "chance_accuracy": 1.0 / len(ps.emotions),
        "accuracy": float((pred == y).mean()),
        "macro_f1": float(f1_score(y, pred, labels=ps.emotions, average="macro", zero_division=0)),
        "confusion": confusion_matrix(y, pred, labels=ps.emotions).tolist(),
        "auc_one_vs_rest": {},
        "auc_emotional_vs_neutral": None,
    }
    for j, e in enumerate(ps.emotions):
        pos = cats == e
        if pos.any() and (~pos).any():
            out["auc_one_vs_rest"][e] = float(roc_auc_score(pos, Z[:, j]))
    if emo.any() and (~emo).any():
        out["auc_emotional_vs_neutral"] = float(roc_auc_score(emo, Z.max(1)))
    return out


def null_distribution(X_tr, c_tr, X_ev, c_ev, emotions, layer, runs: int = 20, seed: int = 0, **fit_kw) -> dict:
    """Shuffle emotion labels among training texts, refit, evaluate on the same held-out set.
    Real probes must beat the upper tail of this, otherwise the 'signal' is fitting noise."""
    rng = np.random.default_rng(seed)
    c_tr = np.asarray(c_tr)
    idx = np.where(c_tr != NEUTRAL)[0]
    f1s = []
    for _ in range(runs):
        c = c_tr.copy()
        c[idx] = rng.permutation(c[idx])
        f1s.append(evaluate(fit_probes(X_tr, c, emotions, layer, **fit_kw), X_ev, c_ev)["macro_f1"])
    f1s = np.array(f1s)
    return {"runs": runs, "mean": float(f1s.mean()), "std": float(f1s.std()),
            "p95": float(np.percentile(f1s, 95)), "max": float(f1s.max())}


def split_half_reliability(X, cats, emotions, layer, reps: int = 20, seed: int = 0, **fit_kw) -> Dict[str, float]:
    """Cosine between directions built from two random halves of the training data (stratified).
    Low values mean the vector is mostly noise from too few examples."""
    rng = np.random.default_rng(seed)
    cats = np.asarray(cats)
    sims = {e: [] for e in emotions}
    for _ in range(reps):
        a, b = [], []
        for c in np.unique(cats):
            idx = rng.permutation(np.where(cats == c)[0])
            h = len(idx) // 2
            a += list(idx[:h]); b += list(idx[h:2 * h])
        pa = fit_probes(X[a], cats[a], emotions, layer, **fit_kw)
        pb = fit_probes(X[b], cats[b], emotions, layer, **fit_kw)
        for i, e in enumerate(emotions):
            sims[e].append(float(pa.V[i] @ pb.V[i]))
    return {e: float(np.mean(v)) for e, v in sims.items()}
