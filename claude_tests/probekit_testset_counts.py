#!/usr/bin/env python3
"""
probekit_testset_counts.py -- PLAN_v2 T2 step 1: COUNTS ONLY for a frozen human test set from GoEmotions.

Reads the original GoEmotions release (per-rater CSVs goemotions_{1,2,3}.csv plus the official train/dev/test .tsv
split files from github.com/google-research/google-research/tree/master/goemotions/data, Apache-2.0) and reports, for the
official TEST split, how many items each of our 8 classes (joy, sadness, fear, anger, surprise, disgust, love, neutral)
gets under three label-agreement rules:

  strict   every rater gave exactly one label and all raters gave the same one (>= 2 raters), none flagged very unclear
  shared   the intersection of all raters' label sets is exactly one label (raters may add extra labels), >= 2 raters
  simple   the dataset's own "simplified" label: a single label chosen by rater agreement (what the current loader uses)

each with and without the training corpus's length filter (10-80 words). It writes nothing but this report; it never
builds a test set. Run with an isolated interpreter and the data in its own directory:

    python -I probekit_testset_counts.py --data-dir <dir with the downloaded files> [--out report.md]
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

OURS = ["joy", "sadness", "fear", "anger", "surprise", "disgust", "love", "neutral"]


def rater_label_sets(raw: pd.DataFrame, label_cols):
    """id -> list of frozenset(labels) per rater row, plus id -> any 'very unclear' flag and text."""
    g = {}
    arr = raw[label_cols].to_numpy()
    for i, (rid, unclear, text) in enumerate(zip(raw["id"], raw["example_very_unclear"], raw["text"])):
        sets, flag, _ = g.setdefault(rid, [[], False, text])
        sets.append(frozenset(c for c, v in zip(label_cols, arr[i]) if v == 1))
        g[rid][1] = flag or bool(unclear)
    return g


def agreed_label(sets, flag, rule):
    """Return the single agreed label under `rule` ('strict' | 'shared') or None."""
    if len(sets) < 2:
        return None
    if rule == "strict":
        if flag or any(len(s) != 1 for s in sets):
            return None
        return next(iter(sets[0])) if all(s == sets[0] for s in sets) else None
    if rule == "shared":
        common = frozenset.intersection(*sets)
        return next(iter(common)) if len(common) == 1 else None
    raise ValueError(rule)


def count_table(items, split_ids, simple_labels, min_words=10, max_words=80):
    """items: id -> (sets, flag, text). Returns {(rule, length_filtered): {class: n, '_unmapped': n, '_none': n}}."""
    out = {}
    for rule in ("strict", "shared", "simple"):
        for lf in (False, True):
            cnt = {c: 0 for c in OURS}; cnt["_unmapped"] = 0; cnt["_no_agreement"] = 0
            for i in split_ids:
                if i not in items:
                    continue
                sets, flag, text = items[i]
                if lf and not (min_words <= len(str(text).split()) <= max_words):
                    continue
                lab = simple_labels.get(i) if rule == "simple" else agreed_label(sets, flag, rule)
                if lab is None:
                    cnt["_no_agreement"] += 1
                elif lab in OURS:
                    cnt[lab] += 1
                else:
                    cnt["_unmapped"] += 1
            out[(rule, lf)] = cnt
    return out


def render(tables, n_split, n_items):
    L = [f"# T2 step 1: GoEmotions official TEST split, label counts (n items in split: {n_split}; with rater data: {n_items})", "",
         "Counts only; nothing was built. `_unmapped` = agreed label is a GoEmotions class we do not use (e.g. admiration); "
         "`_no_agreement` = fails the rule. Criterion (PLAN_v2 T2): at least 6 of the 7 emotions, and neutral, reach n >= 50.", ""]
    for (rule, lf), cnt in tables.items():
        ok = [c for c in OURS if cnt[c] >= 50]
        L += [f"## rule={rule}, length filter 10-80 words: {'on' if lf else 'off'}", "",
              "| " + " | ".join(OURS) + " | unmapped | no agreement |", "|" + "---|" * (len(OURS) + 2),
              "| " + " | ".join(str(cnt[c]) for c in OURS) + f" | {cnt['_unmapped']} | {cnt['_no_agreement']} |",
              f"\nclasses with n >= 50: {len(ok)}/8 ({', '.join(ok) or 'none'}); emotions among them: "
              f"{sum(c in ok for c in OURS if c != 'neutral')}/7\n"]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    d = Path(a.data_dir)
    emotions = (d / "emotions.txt").read_text().split()           # 27 labels, order = tsv label ids; 'neutral' is the last
    raw = pd.concat([pd.read_csv(d / f"goemotions_{k}.csv") for k in (1, 2, 3)], ignore_index=True)
    label_cols = [c for c in raw.columns if c in set(emotions)]
    assert len(label_cols) == 28 or len(label_cols) == len(emotions), (len(label_cols), len(emotions))
    items = rater_label_sets(raw, label_cols)
    names = emotions if emotions[-1] == "neutral" else emotions + ["neutral"]
    test = pd.read_csv(d / "test.tsv", sep="\t", header=None, names=["text", "labels", "id"])
    simple = {r.id: (names[int(str(r.labels).split(",")[0])] if "," not in str(r.labels) else None) for r in test.itertuples()}
    tables = count_table(items, list(test["id"]), simple)
    md = render(tables, len(test), sum(i in items for i in test["id"]))
    print(md)
    if a.out:
        Path(a.out).write_text(md + "\n")


if __name__ == "__main__":
    sys.exit(main())
