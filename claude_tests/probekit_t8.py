#!/usr/bin/env python3
"""
probekit_t8.py -- PLAN_v2 T8 (pre-registered), cached-state experiments:
  (a) hierarchical valence-group -> class LR vs flat LR     (e) MLP vs flat LR (nonlinear ceiling)
  (b) similarity of the probe directions to NRC VAD valence/arousal/dominance distances (permutation null)

    python probekit_t8.py --cache-dir data/eval_merged2/<slug> --eval-dir data/eval_merged2/<slug>/v3 \
        --corpus data/corpus/corpus_merged2_scored.jsonl --vad data/external/nrc/vad/NRC-VAD-Lexicon-v2.1/NRC-VAD-Lexicon-v2.1.txt --out <md>
Evaluation sets: GoEmotions frozen (primary) and the three external sources (predictions restricted to each source's classes).
The VAD lexicon is third-party (non-commercial research use, no redistribution): never committed.
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from probekit_eval_v2 import LR_CS, f1_9, load_jsonl
from probekit_eval_v3 import paired_bootstrap
from probekit_external_eval import argmax_classes, columns, fit_lr
from probekit_frozen_eval import EMOTIONS, boot_ci

GROUPS = {"positive": ["joy", "love"], "negative": ["sadness", "fear", "anger", "disgust"], "ambiguous": ["surprise"], "neutral": ["neutral"]}
GROUP_OF = {c: g for g, cs in GROUPS.items() for c in cs}


# ---------------------------------------------------------------- (a) hierarchical
class Hierarchical:
    """Two-stage LR with soft composition P(class) = P(group) * P(class | group); restrictable to a class subset."""
    def __init__(self, C1, C2):
        self.C1, self.C2 = C1, C2

    def fit(self, X, y):
        self.sc = StandardScaler().fit(X); Xs = self.sc.transform(X)
        self.g1 = LogisticRegression(C=self.C1, max_iter=3000).fit(Xs, [GROUP_OF[c] for c in y])
        self.g2 = {}
        for g, cs in GROUPS.items():
            m = np.isin(y, cs)
            if len(cs) > 1 and m.sum() > 0 and len(set(y[m])) > 1:
                self.g2[g] = LogisticRegression(C=self.C2, max_iter=3000).fit(Xs[m], y[m])
        return self

    def proba(self, X, classes):
        Xs = self.sc.transform(X); P1 = self.g1.predict_proba(Xs); gi = {g: i for i, g in enumerate(self.g1.classes_)}
        out = np.zeros((len(X), len(classes)))
        for j, c in enumerate(classes):
            g = GROUP_OF[c]
            if g not in gi:
                continue
            if g in self.g2:
                P2 = self.g2[g].predict_proba(Xs); out[:, j] = P1[:, gi[g]] * P2[:, list(self.g2[g].classes_).index(c)] if c in self.g2[g].classes_ else 0
            else:
                out[:, j] = P1[:, gi[g]]
        return out

    def predict(self, X, classes):
        return argmax_classes(self.proba(X, classes), classes)


def fit_hierarchical(Xtr, ytr, Xva, yva):
    best = None
    for C1 in LR_CS:
        for C2 in LR_CS:
            h = Hierarchical(C1, C2).fit(Xtr, ytr)
            f = f1_9(yva, h.predict(Xva, sorted(set(ytr))))
            if best is None or f > best[0]:
                best = (f, h)
    return best[1]


# ---------------------------------------------------------------- (e) MLP
def fit_mlp(Xtr, ytr, Xva, yva, seed=0):
    sc = StandardScaler().fit(Xtr); best = None
    for alpha in (1e-3, 1e-2, 1e-1):
        m = MLPClassifier(hidden_layer_sizes=(256,), alpha=alpha, early_stopping=True, max_iter=300, random_state=seed).fit(sc.transform(Xtr), ytr)
        f = f1_9(yva, m.predict(sc.transform(Xva)))
        if best is None or f > best[0]:
            best = (f, m, alpha)
    f, m, alpha = best
    return sc, m, alpha


# ---------------------------------------------------------------- (b) similarity to VAD
def load_vad(path, words):
    got = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines()[1:]:
        t = line.split("\t")
        if len(t) == 4 and t[0] in words:
            got[t[0]] = np.array([float(x) for x in t[1:]])
    return got


def pair_vectors(V, vad, emotions):
    """Pairwise cosine dissimilarity of directions and Euclidean distance of VAD vectors, in the same pair order."""
    idx = {e: i for i, e in enumerate(emotions)}
    A = V[[idx[e] for e in emotions]]; A = A / np.linalg.norm(A, axis=1, keepdims=True)
    D = 1 - A @ A.T; W = np.stack([vad[e] for e in emotions]); E = np.linalg.norm(W[:, None] - W[None], axis=2)
    pairs = list(itertools.combinations(range(len(emotions)), 2))
    return np.array([D[i, j] for i, j in pairs]), np.array([E[i, j] for i, j in pairs]), pairs


def rsa_permutation(dirs_dissim_matrix, vad_dist_matrix):
    """Spearman rho between the two distance matrices (upper triangles) and the exact one-sided permutation p-value over relabelings."""
    n = dirs_dissim_matrix.shape[0]; iu = np.triu_indices(n, 1)
    rho = spearmanr(dirs_dissim_matrix[iu], vad_dist_matrix[iu])[0]
    count = 0; total = 0
    for perm in itertools.permutations(range(n)):
        p = np.array(perm); r = spearmanr(dirs_dissim_matrix[iu], vad_dist_matrix[np.ix_(p, p)][iu])[0]
        count += r >= rho - 1e-12; total += 1
    return float(rho), count / total


def dist_matrices(V, vad, emotions):
    A = V / np.linalg.norm(V, axis=1, keepdims=True); D = 1 - A @ A.T
    W = np.stack([vad[e] for e in emotions]); E = np.linalg.norm(W[:, None] - W[None], axis=2)
    return D, E


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Llama-3.2-3B-Instruct")
    ap.add_argument("--cache-dir", required=True); ap.add_argument("--eval-dir", required=True); ap.add_argument("--corpus", required=True)
    ap.add_argument("--pool", default="data/corpus/corpus_human_goemotions_pool.jsonl"); ap.add_argument("--testsets", default="data/testsets")
    ap.add_argument("--vad", default=None); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from probekit_estimators import calibrate_pooled, fit_directions
    from probekit_eval_v3 import load_states
    from probekit_external import load_source
    from probekit_frozen_eval import cfg_for, extract_cached, pool_new_rows, EXTRACT_LAYERS
    from probekit_l0_baseline import load_embed
    from probekit_testset import load_frozen
    eval_dir, cache = Path(a.eval_dir), Path(a.cache_dir)
    best = json.loads((eval_dir / "best_config.json").read_text()); layer, skip, sel = best["layer"], best["skip"], best["method"]
    rows = load_jsonl(a.corpus); pool = load_jsonl(a.pool)
    Xc = load_states(rows, cache / "states_corpus.npz", [layer], skip, a.model)[layer]
    new = pool_new_rows(pool, rows); Xn = extract_cached(new, cache / "states_pool_new.npz", EXTRACT_LAYERS, skip, a.model)
    cat = np.array([r["category"] for r in rows]); spl = np.array([r["split"] for r in rows])
    wr = np.array([r.get("writer", "llama-3.2-3b") for r in rows]); hard = np.array([bool(r.get("hard", True)) for r in rows])
    keep = cat != "calm"; llm = (spl == "train") & keep & (wr != "reddit-goemotions"); va = (spl == "val") & hard & keep; cal = (spl == "val") & (cat == "neutral")
    byid = {r["id"]: i for i, r in enumerate(rows)}; nid = {r["id"]: i for i, r in enumerate(new)}
    pr = [r for r in pool if r["split"] == "train"]
    Xp = np.stack([Xc[byid[r["id"]]] if r["id"] in byid else Xn[layer][nid[r["id"]]] for r in pr]); yp = np.array([r["category"] for r in pr], dtype=object)
    Xtr = np.vstack([Xc[llm], Xp]); ytr = np.concatenate([cat[llm], yp])
    lr_sc, lr = fit_lr(Xtr, ytr, Xc[va], cat[va]); hier = fit_hierarchical(Xtr, ytr, Xc[va], cat[va]); mlp_sc, mlp, alpha = fit_mlp(Xtr, ytr, Xc[va], cat[va])
    L = [f"# T8 cached-state experiments ({a.model}, layer {layer}; pre-registered in PLAN_v2)", "",
         f"Trained on HUM (n={len(ytr)}); hierarchy C1={hier.C1}, C2={hier.C2}; MLP alpha={alpha}. Paired macro-F1 differences [95% CI]; * = excludes 0.", "",
         "| set (n, classes) | flat LR | hierarchical | MLP | hierarchical - flat (a) | MLP - flat (e) |", "|---|---|---|---|---|---|"]
    wins = {"a": 0, "e": 0}; sets = []
    frozen, _ = load_frozen(f"{a.testsets}/goemotions_test_frozen.manifest.json"); sets.append(("goemotions (frozen)", frozen, cache / "states_frozen.npz"))
    for s in ("dair", "isear", "dailydialog"):
        items, _ = load_source(s, a.testsets); sets.append((s, items, cache / f"states_ext_{s}.npz"))
    for name, items, path in sets:
        Xe = extract_cached(items, path, EXTRACT_LAYERS, skip, a.model)[layer]
        y = np.array([r["category"] for r in items], dtype=object); cl = sorted(set(y))
        p_lr = argmax_classes(columns(lr, lr.predict_proba(lr_sc.transform(Xe)), cl), cl)
        p_h = hier.predict(Xe, cl)
        p_m = argmax_classes(columns(mlp, mlp.predict_proba(mlp_sc.transform(Xe)), cl), cl)
        da, de = paired_bootstrap(y, p_h, p_lr), paired_bootstrap(y, p_m, p_lr)
        wins["a"] += da[1] > 0; wins["e"] += de[1] > 0
        fm = lambda d: f"{d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}]{'*' if d[1] > 0 or d[2] < 0 else ''}"
        L.append(f"| {name} ({len(y)}, {len(cl)}) | {f1_9(y, p_lr):.3f} | {f1_9(y, p_h):.3f} | {f1_9(y, p_m):.3f} | {fm(da)} | {fm(de)} |")
    L += ["", f"Pre-registered tally: hierarchical > flat on {wins['a']}/4 sets (claim needs >= 3); MLP > flat on {wins['e']}/4 sets (claim needs >= 3).",
          "Reading: " + ("hierarchy helps; " if wins["a"] >= 3 else "no evidence that hierarchy helps; ") +
          ("the MLP finds nonlinear information the linear read-out misses." if wins["e"] >= 3 else "the linear read-out is sufficient at this layer.")]
    if a.vad:
        vad = load_vad(a.vad, set(EMOTIONS) - {"neutral"}); emos = [e for e in EMOTIONS if e in vad]
        rows_b = []
        for method in (sel, "meandiff"):
            train_mask = (spl == "train") & keep
            ps = fit_directions(Xc[train_mask], cat[train_mask], EMOTIONS, layer, method=method, **cfg_for(eval_dir, method, layer)); calibrate_pooled(ps, Xc[cal])
            D, E = dist_matrices(ps.V[[ps.emotions.index(e) for e in emos]], vad, emos)
            rho, p = rsa_permutation(D, E)
            iu = np.triu_indices(len(emos), 1)
            rows_b.append((method, rho, p))
        L += ["", f"## (b) Probe-direction dissimilarity vs NRC VAD distance (7 emotions, {len(emos)*(len(emos)-1)//2} pairs; exact permutation over {int(np.prod(range(1, len(emos)+1)))} labelings)", "",
              "| estimator | Spearman rho | one-sided permutation p |", "|---|---|---|"] + [f"| {m} | {r:+.2f} | {p:.3f} |" for m, r, p in rows_b]
        L += ["", "Claim rule: only if p < 0.05 for both estimators. Seven points have low power; a non-significant result does not show the geometries differ."]
    Path(a.out).write_text("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()
