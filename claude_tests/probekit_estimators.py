#!/usr/bin/env python3
"""
probekit_estimators.py -- better ways to estimate one direction per concept (numpy/sklearn only).

All estimators return a ProbeSet (same class the backend loads) so they are drop-in replacements:
    meandiff  mean(e) - mean over emotions            (v1/v2 behavior; cheap, ignores noise structure)
    lda       shrinkage-LDA direction  Sigma^-1 (mu_e - mu)   (whitens by within-class noise: suppresses
              directions that vary a lot for reasons unrelated to emotion)
    logreg    multinomial logistic-regression weights, folded back to raw activation space
Pipeline for every method: center on neutral mean -> remove top within-class nuisance PCs -> fit -> unit-normalize.

IMPORTANT: calibrate with HELD-OUT neutral text (calibrate_pooled). For supervised estimators the
training neutral passages are fit too well, so their projections under-estimate neutral spread.
"""
from __future__ import annotations

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

from probekit_probes import EPS, NEUTRAL, ProbeSet, _nuisance_basis

METHODS = ("meandiff", "lda", "logreg")


def fit_directions(X, cats, emotions, layer, method="meandiff", var_frac=0.8, max_k=24,
                   C=0.03, shrinkage="auto") -> ProbeSet:
    X = np.asarray(X, dtype=np.float64)
    cats = np.asarray(cats)
    neutral = X[cats == NEUTRAL]
    if len(neutral) < 5:
        raise ValueError("need >=5 neutral training passages")
    mu = neutral.mean(0)
    B = _nuisance_basis(X, cats, var_frac, max_k)
    Xc = X - mu
    if len(B):
        Xc = Xc - (Xc @ B.T) @ B

    if method == "meandiff":
        M = np.stack([Xc[cats == e].mean(0) for e in emotions])
        D = M - M.mean(0, keepdims=True)
    elif method == "lda":
        lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=shrinkage).fit(Xc, cats)
        classes = list(lda.classes_)
        W = lda.coef_
        D = W[[classes.index(e) for e in emotions]] - W.mean(0, keepdims=True)
    elif method == "logreg":
        s = Xc.std(0)
        s = np.maximum(s, 1e-3 * s.mean() + 1e-12)
        clf = LogisticRegression(C=C, max_iter=2000).fit(Xc / s, cats)
        classes = list(clf.classes_)
        W = clf.coef_ / s                                  # fold the feature scaling back into raw space
        D = W[[classes.index(e) for e in emotions]] - W.mean(0, keepdims=True)
    else:
        raise ValueError(f"unknown method {method!r}")

    V = D / (np.linalg.norm(D, axis=1, keepdims=True) + EPS)
    proj = (neutral - mu) @ V.T                            # placeholder calibration; replace via calibrate_pooled
    if len(B):
        proj = ((neutral - mu) - ((neutral - mu) @ B.T) @ B) @ V.T
    ps = ProbeSet(emotions=list(emotions), layer=layer, mu=mu, basis=B, V=V,
                  bias_pool=proj.mean(0), sigma_pool=proj.std(0) + EPS,
                  bias_tok=proj.mean(0), sigma_tok=proj.std(0) + EPS,
                  meta={"method": method, "var_frac": var_frac, "nuisance_k": int(len(B)), "C": C,
                        "shrinkage": shrinkage})
    return ps


def fit_from_config(X, cats, emotions, layer, cfg: dict) -> ProbeSet:
    """cfg: {"method":..., "var_frac":..., ["C":..., "shrinkage":...]}"""
    kw = {k: v for k, v in cfg.items() if k in ("var_frac", "max_k", "C", "shrinkage")}
    return fit_directions(X, cats, emotions, layer, method=cfg["method"], **kw)


def calibrate_pooled(ps: ProbeSet, X_neutral: np.ndarray) -> ProbeSet:
    """Set bias/sigma from HELD-OUT neutral pooled passages (z = sigmas above neutral)."""
    proj = ps.project(X_neutral)
    ps.bias_pool = proj.mean(0).astype(np.float64)
    ps.sigma_pool = proj.std(0).astype(np.float64) + EPS
    ps.bias_tok, ps.sigma_tok = ps.bias_pool.copy(), ps.sigma_pool.copy()
    return ps
