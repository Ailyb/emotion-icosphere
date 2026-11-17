#!/usr/bin/env python3
"""
probekit_testset.py -- PLAN_v2 T2 steps 2-3: frozen GoEmotions test set, rebuilt human TRAINING corpus, calibration-id
reservation, leakage check, and manifest verification. Stdlib only (no pandas, no model).

Source: the original GoEmotions release (Demszky et al. 2020, Apache-2.0): per-rater CSVs goemotions_{1,2,3}.csv and the
official train/dev/test .tsv split files plus emotions.txt, kept under data/external/goemotions/ (gitignored, re-downloadable).

    python probekit_testset.py build   --data-dir data/external/goemotions --out-dir data/testsets --corpus-out data/corpus/corpus_human_goemotions_train.jsonl
    python probekit_testset.py leakage --out-dir data/testsets --data-dir data/external/goemotions --corpus data/corpus/corpus_v2.jsonl ...
    python probekit_testset.py verify  --out-dir data/testsets

Rules (all recorded in the manifest):
  * TEST: the official TEST split, items whose official label is a single label in our 8 classes (joy, sadness, fear, anger,
    surprise, disgust, love, neutral). No length filter. Neutral is capped at the size of the largest other class (seeded sample).
    Per item we keep n_raters and `agree` (share of raters whose label set contains the label) as a covariate, and mined
    subtypes: negation / emotion_word_neutral / no_emotion_lexicon_match (lexicon = probekit_corpus_v2.STEM_RE2).
  * CALIBRATION: a seeded reservation of single-label neutral items from the official TRAIN split, any length (for T3).
  * HUMAN TRAIN corpus: from the official TRAIN split only (never dev or test), excluding calibration ids and any text equal to a
    test text; 10-80 words; split into train/val by comment id; same per-class sizes as the previous corpus (80 train / 20 val).
"""
import argparse
import csv
import hashlib
import json
import random
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

from probekit_corpus_hf import clean_text
from probekit_corpus_v2 import STEM_RE2
from probekit_overlap import containments, norm, overlap

OURS = ["joy", "sadness", "fear", "anger", "surprise", "disgust", "love", "neutral"]
NEGATORS = {"not", "no", "never", "without", "hardly", "neither", "nor", "cannot", "cant", "dont", "wont"}
STYLE = "informal social-media comment (reddit)"
WRITER = "reddit-goemotions"
SEED = 0


# ---------------------------------------------------------------- loading
def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_tsv(path, names):
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        text, labels, rid = line.split("\t")
        lab = [int(x) for x in labels.split(",") if x != ""]
        out.append({"id": rid, "text": text, "label": names[lab[0]] if len(lab) == 1 else None, "n_labels": len(lab)})
    return out


def read_rater_stats(data_dir, names):
    """id -> (n_raters, {label: n_raters_with_label}) from the per-rater CSVs."""
    n, per = Counter(), defaultdict(Counter)
    for k in (1, 2, 3):
        with open(Path(data_dir) / f"goemotions_{k}.csv", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                n[row["id"]] += 1
                for c in names:
                    if row.get(c) == "1":
                        per[row["id"]][c] += 1
    return n, per


def tokens(text):
    return re.findall(r"[a-z0-9]+(?:['’][a-z]+)?", text.lower())


def mined_subtypes(text, label):
    """Rule-mined hard subtypes (no hand authoring). Lexicon = the emotion-word filter used for corpus generation."""
    low = text.lower()
    hits = [m.start() for m in STEM_RE2.finditer(low)]
    sub = []
    if hits:
        for pos in hits:                                              # negator within the 3 words before a lexicon match
            before = tokens(low[:pos])[-3:]
            if any(t in NEGATORS or t.endswith("n't") or t.endswith("n’t") for t in before):
                sub.append("negation"); break
        if label == "neutral":
            sub.append("emotion_word_neutral")
    elif label != "neutral":
        sub.append("no_emotion_lexicon_match")
    return sub


def n_words(t):
    return len(t.split())


# ---------------------------------------------------------------- builders
def build_test(items, names, rater_n, rater_per, train_texts, seed=SEED, near_dup_texts=()):
    rng = random.Random(seed)
    drops = Counter()
    keep, seen = [], set()
    for it in items:
        if it["label"] is None:
            drops["multi_label"] += 1; continue
        if it["label"] not in OURS:
            drops["unmapped_label"] += 1; continue
        text = clean_text(it["text"])
        key = " ".join(norm(text))
        if not key:
            drops["empty"] += 1; continue
        if key in seen:
            drops["duplicate_in_test"] += 1; continue
        if key in train_texts:
            drops["text_also_in_official_train_or_dev"] += 1; continue
        seen.add(key)
        n = rater_n.get(it["id"], 0)
        keep.append({"id": it["id"], "text": text, "category": it["label"], "n_raters": n,
                     "agree": round(rater_per[it["id"]][it["label"]] / n, 3) if n else None, "n_words": n_words(text),
                     "subtypes": mined_subtypes(text, it["label"]), "writer": "human-goemotions", "split": "test"})
    if near_dup_texts:                       # drop items whose 5-grams are >= 50% contained in an official train/dev passage
        nd = containments([r["text"] for r in keep], list(near_dup_texts))
        drops["near_duplicate_of_official_train_or_dev"] = sum(c >= 0.5 for c in nd)
        keep = [r for r, c in zip(keep, nd) if c < 0.5]
    emo = [r for r in keep if r["category"] != "neutral"]
    cap = max(Counter(r["category"] for r in emo).values())
    neu = [r for r in keep if r["category"] == "neutral"]
    rng.shuffle(neu)
    drops["neutral_over_cap"] = max(0, len(neu) - cap)
    final = sorted(emo + neu[:cap], key=lambda r: r["id"])
    return final, dict(drops), cap


def reserve_calibration(items, k, exclude_texts, seed=SEED):
    rng = random.Random(seed + 1)
    pool = [it for it in items if it["label"] == "neutral" and " ".join(norm(clean_text(it["text"]))) not in exclude_texts]
    rng.shuffle(pool)
    sel = pool[:k]
    return sorted(it["id"] for it in sel), Counter(min(n_words(clean_text(it["text"])) // 5 * 5, 30) for it in sel)


def build_human_train(items, calib_ids, test_keys, n_train=80, n_val=20, seed=SEED, min_words=10, max_words=80):
    rng = random.Random(seed + 2)
    calib = set(calib_ids)
    by = defaultdict(list)
    seen = set()
    for it in items:
        if it["label"] not in OURS or it["id"] in calib:
            continue
        text = clean_text(it["text"])
        key = " ".join(norm(text))
        if not (min_words <= n_words(text) <= max_words) or key in seen or key in test_keys:
            continue
        seen.add(key); by[it["label"]].append((it["id"], text))
    rows = []
    for cat in OURS:
        pool = by[cat]; rng.shuffle(pool)
        for j, (rid, text) in enumerate(pool[: n_train + n_val]):
            sp = "train" if j < n_train else "val"
            rows.append({"id": f"hf|goemotions|train|{rid}", "category": cat, "topic": f"goemotions|train|{rid}",
                         "style": STYLE, "split": sp, "writer": WRITER, "source": "goemotions", "text": text})
    return rows


# ---------------------------------------------------------------- manifest / verification
def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def git_rev():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5).stdout.strip() or None
    except Exception:
        return None


def verify(manifest_path, testset_path=None):
    """Raise RuntimeError if the frozen test set no longer matches its manifest hash. Returns the manifest."""
    m = json.loads(Path(manifest_path).read_text())
    p = Path(testset_path or Path(manifest_path).with_name(m["testset_file"]))
    got = sha256_file(p)
    if got != m["testset_sha256"]:
        raise RuntimeError(f"frozen test set {p} changed: sha256 {got[:12]}... != manifest {m['testset_sha256'][:12]}...")
    return m


def load_frozen(manifest_path):
    """Evals call this: verifies the hash, then returns the rows. Refuses to run on a modified file."""
    m = verify(manifest_path)
    p = Path(manifest_path).with_name(m["testset_file"])
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()], m


# ---------------------------------------------------------------- commands
def cmd_build(a):
    d = Path(a.data_dir); out = Path(a.out_dir)
    names = (d / "emotions.txt").read_text().split()
    train, dev, test = (read_tsv(d / f"{s}.tsv", names) for s in ("train", "dev", "test"))
    rater_n, rater_per = read_rater_stats(d, names)
    other_keys = {" ".join(norm(clean_text(it["text"]))) for it in train + dev}
    test_rows, drops, cap = build_test(test, names, rater_n, rater_per, other_keys,
                                       near_dup_texts=[clean_text(it['text']) for it in train + dev])
    test_keys = {" ".join(norm(r["text"])) for r in test_rows}
    calib_ids, calib_len = reserve_calibration(train, a.calibration_n, test_keys)
    corpus = build_human_train(train, calib_ids, test_keys)
    tpath = out / "goemotions_test_frozen.jsonl"; write_jsonl(tpath, test_rows)
    cpath = out / "goemotions_calibration_ids.json"
    cpath.write_text(json.dumps({"ids": calib_ids, "n": len(calib_ids), "word_len_buckets": dict(sorted(calib_len.items()))}, indent=1))
    write_jsonl(a.corpus_out, corpus)
    sub = Counter(s for r in test_rows for s in r["subtypes"])
    manifest = {
        "dataset": "GoEmotions (Demszky et al. 2020), original release, Apache-2.0",
        "source_files_sha256": {p.name: sha256_file(p) for p in sorted(d.iterdir()) if p.is_file()},
        "split": "official TEST (test.tsv)", "n_source_items": len(test),
        "label_rule": "official single label (test.tsv has exactly one label id), mapped identity to 8 classes; multi-label and other classes dropped",
        "classes": OURS, "length_filter": "none", "text_cleaning": "probekit_corpus_hf.clean_text",
        "dedup": "normalized text duplicates within test dropped; texts also in official train/dev dropped; items with >= 50% of their word 5-grams inside one official train/dev passage dropped",
        "neutral_cap": cap, "neutral_cap_rule": "seeded sample, size = largest non-neutral class", "seed": SEED,
        "drops": drops, "n": len(test_rows), "class_counts": dict(Counter(r["category"] for r in test_rows)),
        "subtype_counts": dict(sub), "subtype_rules": {
            "negation": "emotion-lexicon match with a negator in the 3 words before it",
            "emotion_word_neutral": "neutral item containing an emotion-lexicon match",
            "no_emotion_lexicon_match": "emotion-labelled item with no lexicon match"},
        "lexicon": "probekit_corpus_v2.STEM_RE2", "agreement_covariate": "agree = share of raters whose label set contains the label",
        "testset_file": tpath.name, "testset_sha256": sha256_file(tpath),
        "calibration_ids_file": cpath.name, "calibration_ids_sha256": sha256_file(cpath), "calibration_n": len(calib_ids),
        "human_train_corpus": str(a.corpus_out), "human_train_corpus_sha256": sha256_file(a.corpus_out),
        "human_train_class_counts": {f"{sp}|{c}": n for (sp, c), n in sorted(Counter((r["split"], r["category"]) for r in corpus).items())},
        "code_version": git_rev(), "script": "probekit_testset.py"}
    (out / "goemotions_test_frozen.manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    print(f"frozen test: n={len(test_rows)} {manifest['class_counts']}\nsubtypes: {manifest['subtype_counts']}\n"
          f"drops: {drops}\ncalibration ids: {len(calib_ids)}; human train corpus: {len(corpus)} rows -> {a.corpus_out}\n"
          f"test sha256 {manifest['testset_sha256'][:16]}...")


def cmd_leakage(a):
    out = Path(a.out_dir); d = Path(a.data_dir)
    rows, m = load_frozen(out / "goemotions_test_frozen.manifest.json")
    texts = [r["text"] for r in rows]; ids = {r["id"] for r in rows}
    calib = set(json.loads((out / m["calibration_ids_file"]).read_text())["ids"])
    names = (d / "emotions.txt").read_text().split()
    ok = True
    retired = set(a.retired or [])
    print(f"frozen test n={len(rows)} (hash verified); calibration ids n={len(calib)}")
    print(f"test/calibration id overlap: {len(ids & calib)}")
    ok &= not (ids & calib)
    for c in list(a.corpus) + sorted(retired):
        crow = [json.loads(l) for l in open(c) if l.strip()]
        cids = {r["id"].split("|")[-1] for r in crow if r.get("source") == "goemotions"}
        r = overlap(texts, [x["text"] for x in crow])
        idov = len(cids & ids) + len(cids & calib)
        print(f"{c}: n={len(crow)} exact={r['exact']} containment>=0.5={r['containment>=0.5']} max={r['max_containment']} "
              f"goemotions-id overlap with test/calibration={idov}")
        if c in retired:
            print("   (retired corpus: reported, not counted)"); continue
        ok &= r["exact"] == 0 and r["containment>=0.5"] == 0 and idov == 0
    for s in ("train", "dev"):
        items = read_tsv(d / f"{s}.tsv", names)
        r = overlap(texts, [clean_text(it["text"]) for it in items])
        print(f"official {s}.tsv: n={len(items)} exact={r['exact']} containment>=0.5={r['containment>=0.5']} max={r['max_containment']}")
        ok &= r["exact"] == 0 and r["containment>=0.5"] == 0
    print("LEAKAGE CHECK:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def cmd_verify(a):
    m = verify(Path(a.out_dir) / "goemotions_test_frozen.manifest.json")
    print(f"OK: {m['testset_file']} n={m['n']} sha256 {m['testset_sha256'][:16]}...")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    b = sp.add_parser("build")
    b.add_argument("--data-dir", default="data/external/goemotions"); b.add_argument("--out-dir", default="data/testsets")
    b.add_argument("--corpus-out", default="data/corpus/corpus_human_goemotions_train.jsonl")
    b.add_argument("--calibration-n", type=int, default=3000)
    l = sp.add_parser("leakage")
    l.add_argument("--data-dir", default="data/external/goemotions"); l.add_argument("--out-dir", default="data/testsets")
    l.add_argument("--corpus", nargs="+", required=True)
    l.add_argument("--retired", nargs="*", default=[], help="corpora that are no longer used (e.g. the old human corpus): reported only")
    v = sp.add_parser("verify"); v.add_argument("--out-dir", default="data/testsets")
    a = ap.parse_args()
    sys.exit({"build": lambda: cmd_build(a), "leakage": lambda: cmd_leakage(a), "verify": lambda: cmd_verify(a)}[a.cmd]() or 0)


if __name__ == "__main__":
    main()
