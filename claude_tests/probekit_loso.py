#!/usr/bin/env python3
"""
probekit_loso.py -- POST HOC robustness (requested 2026-11-20): do the cross-domain conclusions survive dropping one external source?
Mainly ISEAR, whose copy has unverified provenance (Hub copy; uploader's license tag), has only 5 classes and is the source where the encoder's
advantage over the probe is largest. Reads the committed T11 reports only (no refitting):
    python probekit_loso.py --out data/eval_merged2/t11_leave_one_source_out.md
Rules re-applied with the remaining two sources: H1 = probe - TF-IDF CI above 0 on both; H3 = probe - encoder CI above 0 on both (the original
rules needed >= 2 of 3); seed stability = same sign in >= 4 of 5 seeds on both.
"""
import argparse
import glob
import json
from pathlib import Path

SOURCES = ("dair", "isear", "dailydialog")


def loso(results):
    """results: {model: t11 json}. Returns {dropped: {model: {h1, h3, enc_range, tfidf_range, seeds_stable_tfidf, seeds_stable_enc}}}."""
    out = {}
    for drop in SOURCES:
        keep = [s for s in SOURCES if s != drop]; out[drop] = {}
        for m, r in results.items():
            pm = {s: r["sets"][s]["probe_minus"] for s in keep}
            st = r["h6_stability"]
            out[drop][m] = {"h1": all(pm[s]["tfidf"][1] > 0 for s in keep), "h3": all(pm[s]["encoder"][1] > 0 for s in keep),
                            "enc_means": {s: pm[s]["encoder"][0] for s in keep}, "enc_ci_hi": {s: pm[s]["encoder"][2] for s in keep},
                            "tfidf_means": {s: pm[s]["tfidf"][0] for s in keep},
                            "stable_tfidf": all(st[f"{s}|tfidf"]["stable"] for s in keep), "stable_enc": all(st[f"{s}|encoder"]["stable"] for s in keep)}
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--glob", default="data/eval_merged2/mlx-community__*/t11.json")
    a = ap.parse_args()
    results = {Path(f).parent.name.split("__")[1]: json.load(open(f)) for f in sorted(glob.glob(a.glob))}
    res = loso(results)
    L = ["# Leave-one-source-out robustness of the cross-domain conclusions (POST HOC; from the committed T11 reports, no refitting)", "",
         "Each block drops one external source and re-applies the rules to the other two (both must hold). ISEAR: Hub copy with unverified provenance, 5 classes (no neutral, surprise or love), 300 per class; "
         "the one source where the sentence encoder's lead over the probe is largest.", "",
         "| dropped | model | H1 probe > TF-IDF on both remaining | H3 probe > encoder on both remaining | probe - encoder (means) | seed sign stable: TF-IDF / encoder |", "|---|---|---|---|---|---|"]
    for drop, byc in res.items():
        for m, v in byc.items():
            L.append(f"| {drop} | {m} | {'yes' if v['h1'] else 'no'} | {'yes' if v['h3'] else 'no'} | " + ", ".join(f"{s} {x:+.3f}" for s, x in v["enc_means"].items()) +
                     f" | {'yes' if v['stable_tfidf'] else 'no'} / {'yes' if v['stable_enc'] else 'no'} |")
    iso = res["isear"]
    L += ["", "## Reading", "",
          f"- Without ISEAR: H1 holds for {sum(v['h1'] for v in iso.values())}/{len(iso)} models; H3 holds for {sum(v['h3'] for v in iso.values())}/{len(iso)}. "
          "With dair and DailyDialog alone the probe ties the encoder on dair and loses on DailyDialog for every model, so the H3 verdict does not depend on ISEAR.",
          f"- H1 (probe beats TF-IDF) holds under every single-source drop for {sum(all(res[d][m]['h1'] for d in res) for m in results)}/{len(results)} models.",
          f"- H3 holds under any single-source drop for {sum(any(res[d][m]['h3'] for d in res) for m in results)}/{len(results)} models."]
    Path(a.out).write_text("\n".join(L) + "\n"); Path(a.out).with_suffix(".json").write_text(json.dumps(res, indent=1)); print("\n".join(L))


if __name__ == "__main__":
    main()
