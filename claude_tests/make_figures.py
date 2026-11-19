#!/usr/bin/env python3
"""make_figures.py -- writeup Figures 1-3 from committed reports only (SVG, no data beyond report numbers).
    python make_figures.py   ->  docs/figures/fig1_forest.svg fig2_learning_curve.svg fig3_neutral_false_alarm.svg"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta

R = Path("data/eval_merged2"); OUT = Path("docs/figures"); OUT.mkdir(parents=True, exist_ok=True)
MODELS = {"Llama-3.2-3B": "mlx-community__Llama-3.2-3B-Instruct", "Gemma-2-2B": "mlx-community__gemma-2-2b-it-4bit", "Mistral-7B": "mlx-community__Mistral-7B-Instruct-v0.3-4bit"}
C = {"a": "#0072B2", "b": "#E69F00", "c": "#009E73", "ink": "#222222", "mute": "#777777"}      # Okabe-Ito subset (colour-vision safe)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"})


def fig1():
    sets = ["frozen", "dair", "isear", "dailydialog"]; lab = ["GoEmotions\n(in-domain, shaded)", "dair\n(tweets)", "ISEAR", "DailyDialog"]
    comps = [("tfidf", "vs TF-IDF", C["a"], "o"), ("encoder", "vs sentence encoder", C["b"], "s"), ("embed", "vs bag of embeddings", C["c"], "^")]
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.6), sharey=True)
    for ax, (mn, slug) in zip(axs, MODELS.items()):
        t = json.loads((R / slug / "t11.json").read_text()); e = json.loads((R / slug / "external_eval.json").read_text()); ph = json.loads((R / slug / "t11_posthoc.json").read_text())
        ax.axhspan(-0.5, 0.5, color="#f0f0f0", zorder=0)
        for ci, (k, name, col, mk) in enumerate(comps):
            for si, s in enumerate(sets):
                if k == "embed":
                    m, lo, hi = ph["sets"][s]["probe_minus_embed"] if s == "frozen" else e[s]["HUM|probe-embed"]
                else:
                    m, lo, hi = t["sets"][s]["probe_minus"][k]
                y = si + (ci - 1) * 0.22
                ax.plot([lo, hi], [y, y], color=col, lw=1.4); ax.plot(m, y, mk, color=col, ms=5, label=name if si == 0 else None)
        ax.axvline(0, color=C["ink"], lw=0.8); ax.set_ylim(3.6, -0.5); ax.set_title(mn); ax.set_yticks(range(4)); ax.set_yticklabels(lab); 
        ax.set_xlabel("probe − baseline, macro-F1 (95% CI)"); ax.grid(axis="x", color="#e5e5e5", lw=0.6)
    h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(OUT / "fig1_forest.svg"); plt.close(fig)


def fig2():
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.4), sharey=True)
    for ax, (mn, slug) in zip(axs, MODELS.items()):
        cur = json.loads((R / slug / "frozen_eval.json").read_text())["curve"]
        for key, col in (("probe", C["a"]), ("native_lr", C["b"]), ("tfidf", C["c"])):
            for llm, ls in ((True, "-"), (False, "--")):
                ks = sorted({r["k"] for r in cur if r["llm"] == llm})
                if not ks: continue
                mu = [np.mean([r[key] for r in cur if r["llm"] == llm and r["k"] == k]) for k in ks]
                sd = [np.std([r[key] for r in cur if r["llm"] == llm and r["k"] == k]) for k in ks]
                ax.errorbar(ks, mu, yerr=sd, color=col, ls=ls, lw=1.4, marker="o", ms=3, capsize=2,
                            label=f"{ {'probe': 'probe', 'native_lr': 'native LR', 'tfidf': 'TF-IDF'}[key] }, {'LLM + human' if llm else 'human only'}")
        ax.set_title(mn); ax.set_xlabel("human rows per class"); ax.grid(color="#e5e5e5", lw=0.6)
    axs[0].set_ylabel("frozen GoEmotions macro-F1"); h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.06)); fig.tight_layout(rect=(0, 0.16, 1, 1))
    fig.savefig(OUT / "fig2_learning_curve.svg"); plt.close(fig)


def cp(k, n):
    return (0.0 if k == 0 else beta.ppf(0.025, k, n - k + 1)), (1.0 if k == n else beta.ppf(0.975, k + 1, n - k))


def fig3():
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.3), sharey=True)
    for ax, (mn, slug) in zip(axs, MODELS.items()):
        d = json.loads((R / slug / "length_calibration.json").read_text())
        for key, col, name, off in (("raw", C["a"], "raw z-scores", -0.08), ("recal", C["b"], "per-length recalibrated", 0.08)):
            rows = [r for r in d[key] if r["n"] > 0]; x = np.arange(len(rows)) + off
            p = np.array([r["passage_rate"] for r in rows]); ci = np.array([cp(int(round(r["passage_rate"] * r["n"])), r["n"]) for r in rows])
            ax.errorbar(x, p, yerr=[p - ci[:, 0], ci[:, 1] - p], color=col, marker="o", ms=4, lw=1.4, capsize=2, label=name)
        ax.axhline(0.05, color=C["mute"], ls=":", lw=1); ax.set_xticks(range(len(rows))); ax.set_xticklabels([r["bucket"] for r in rows])
        ax.set_title(mn); ax.set_xlabel("words per passage"); ax.grid(axis="y", color="#e5e5e5", lw=0.6)
    axs[0].set_ylabel("neutral passages with any |z| > 3"); h, l = axs[0].get_legend_handles_labels()
    axs[0].annotate("5% bound", (2.6, 0.095), color=C["mute"], fontsize=8)
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.03)); fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(OUT / "fig3_neutral_false_alarm.svg"); plt.close(fig)


if __name__ == "__main__":
    fig1(); fig2(); fig3(); print("wrote", sorted(p.name for p in OUT.iterdir()))
