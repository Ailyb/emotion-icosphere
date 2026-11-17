#!/usr/bin/env python3
"""
probekit_external.py -- second (and third, fourth) human-labelled test sources for PLAN_v2 T2/T0d: cross-domain checks that
use a DIFFERENT dataset from the one the probes were trained on (GoEmotions). Stdlib only; rows come from the Hugging Face
datasets-server JSON API, so no pyarrow/datasets install is needed. Raw data goes to data/external/ (gitignored, not
redistributed); only manifests (hashes, counts, rules, provenance) and aggregate results are committed.

    python probekit_external.py build --source dair      # tweets, hashtag-derived labels, 6 classes
    python probekit_external.py build --source isear     # first-person event descriptions, 5 mapped classes
    python probekit_external.py build --source dailydialog   # utterances, 7 classes incl. neutral and disgust
    python probekit_external.py leakage --corpus data/corpus/corpus_merged2.jsonl data/corpus/corpus_human_goemotions_pool.jsonl

Provenance and limits are written into each manifest. dair-ai/emotion: license "other" with no stated terms, labels are
machine-generated from hashtags (distant supervision), text is dominated by "i feel ..." patterns, so it is strongly lexical.
ISEAR: the original host (unige.ch) refused my request; the Hub copy `savalera/isear-from-original` declares apache-2.0 but
that is the uploader's tag, not the original authors' terms: used for local research evaluation only, raw text not committed.
DailyDialog (ConvLab copy): CC-BY-NC-SA-4.0 (non-commercial research use).
"""
import argparse
import hashlib
import json
import random
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

from probekit_overlap import containments, norm, overlap

OUR = ["joy", "sadness", "fear", "anger", "surprise", "disgust", "love", "neutral"]
SEED = 0
API = "https://datasets-server.huggingface.co"


def fetch_rows(dataset, config, split, where=None, page=100, pause=0.6, get=None, max_rows=None, start=0):
    """Page through the datasets-server rows (or /filter) endpoint. `get(url)->dict` is injectable for tests."""
    get = get or _http_json
    out, off = [], start
    while True:
        q = {"dataset": dataset, "config": config, "split": split, "offset": off, "length": page}
        if where:
            q["where"] = where
        data = get(f"{API}/{'filter' if where else 'rows'}?{urllib.parse.urlencode(q)}")
        rows = [r["row"] for r in data["rows"]]
        out += rows
        off += len(rows)
        total = data.get("num_rows_total", off)
        if not rows or off >= total or (max_rows and off >= max_rows):
            return out
        time.sleep(pause)


def _ssl_context():
    """Default verification; if the interpreter has no CA bundle (python.org builds), use certifi's when installed."""
    import ssl
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def _http_json(url, tries=8):
    """GET json with back-off; honours Retry-After on 429 (the datasets-server rate-limits bursts)."""
    ctx = _ssl_context()
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60, context=ctx) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if k == tries - 1 or e.code not in (429, 500, 502, 503, 504):
                raise
            wait = float(e.headers.get("Retry-After", 0) or 0) or min(60, 3 * 2 ** k)
            time.sleep(wait)
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(2 ** k)


def fetch_zip_json(url, member, get_bytes=None):
    """Download a zip and return the JSON list in `member` (used where the datasets-server cannot serve the rows)."""
    import io, zipfile
    data = (get_bytes or _http_bytes)(url)
    return json.loads(zipfile.ZipFile(io.BytesIO(data)).read(member))


def _http_bytes(url, tries=4):
    ctx = _ssl_context()
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=180, context=ctx) as r:
                return r.read()
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(2 ** k)


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(p):
    return sha256_bytes(Path(p).read_bytes())


# ---------------------------------------------------------------- source adapters: raw rows -> (id, text, label_in_our_scheme|None, raw_label)
def adapt_dair(rows, names=("sadness", "joy", "love", "anger", "fear", "surprise")):
    return [(f"dair|test|{i}", r["text"], names[int(r["label"])], names[int(r["label"])]) for i, r in enumerate(rows)]


ISEAR_MAP = {"joy": "joy", "fear": "fear", "anger": "anger", "sadness": "sadness", "disgust": "disgust"}   # shame, guilt: no counterpart


def adapt_isear(rows):
    out = []
    for i, r in enumerate(rows):
        text = (r.get("SIT") or "").replace("\\n", " ").replace("\n", " ").strip()
        if not text or text.lower().startswith("[ no response") or text.lower().startswith("[no response"):
            continue
        raw = str(r.get("Field1", "")).strip().lower()
        out.append((f"isear|{r.get('ID', i)}|{i}", text, ISEAR_MAP.get(raw), raw))
    return out


DD_MAP = {"no emotion": "neutral", "happiness": "joy", "anger": "anger", "disgust": "disgust", "fear": "fear", "sadness": "sadness", "surprise": "surprise"}


def adapt_dailydialog(rows):
    out = []
    for r in rows:
        turns = r["turns"] if isinstance(r["turns"], list) else json.loads(r["turns"])
        for t in turns:
            raw = str(t.get("emotion", "")).strip().lower()
            text = (t.get("utterance") or "").strip()
            if text:
                out.append((f"dd|{r['dialogue_id']}|{t.get('utt_idx')}", text, DD_MAP.get(raw), raw))
    return out


SOURCES = {
    "dair": {"fetch": [("dair-ai/emotion", "split", "test", None)], "adapt": adapt_dair, "cap_per_class": None,
             "license": "other (no terms stated on the card)", "labels": "machine-generated from hashtags (distant supervision)",
             "domain": "tweets", "classes": ["sadness", "joy", "love", "anger", "fear", "surprise"]},
    "isear": {"fetch": [("savalera/isear-from-original", "original", s, None) for s in ("train", "validation", "test")],
              "adapt": adapt_isear, "cap_per_class": 300, "license": "Hub tag apache-2.0 (uploader's claim; original terms unverified)",
              "labels": "self-reported emotion of the writer for their own event", "domain": "first-person event descriptions",
              "classes": ["joy", "fear", "anger", "sadness", "disgust"]},
    "dailydialog": {"zip": ("https://huggingface.co/datasets/ConvLab/dailydialog/resolve/main/data.zip", "data/dialogues.json"),
                    "fetch": [], "row_filter": ("data_split", "test"), "adapt": adapt_dailydialog,
                    "cap_per_class": None, "license": "CC-BY-NC-SA-4.0", "labels": "human annotated per utterance",
                    "domain": "everyday dialogue utterances (no context given to the model)",
                    "classes": ["joy", "sadness", "fear", "anger", "surprise", "disgust", "neutral"]},
}


def build_items(items, cap_per_class, seed=SEED, min_words=3):
    """Apply label mapping drops, dedup, a per-class cap, and the neutral cap (= size of the largest other class)."""
    rng = random.Random(seed)
    drops, seen, keep = Counter(), set(), []
    for rid, text, lab, raw in items:
        key = " ".join(norm(text))
        if lab is None:
            drops["unmapped_label"] += 1; continue
        if len(text.split()) < min_words:
            drops["too_short"] += 1; continue
        if key in seen:
            drops["duplicate"] += 1; continue
        seen.add(key); keep.append({"id": rid, "text": text, "category": lab, "raw_label": raw, "n_words": len(text.split())})
    by = {}
    for r in keep:
        by.setdefault(r["category"], []).append(r)
    for c, rs in by.items():
        rng.shuffle(rs)
    if cap_per_class:
        for c in by:
            drops[f"over_cap_{c}"] = max(0, len(by[c]) - cap_per_class); by[c] = by[c][:cap_per_class]
    if "neutral" in by:
        cap = max([len(v) for c, v in by.items() if c != "neutral"] or [0])
        drops["neutral_over_cap"] = max(0, len(by["neutral"]) - cap); by["neutral"] = by["neutral"][:cap]
    rows = sorted((r for v in by.values() for r in v), key=lambda r: r["id"])
    return rows, {k: v for k, v in drops.items() if v}


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def drop_near_duplicates(rows, corpus_texts, thr=0.5):
    c = containments([r["text"] for r in rows], corpus_texts)
    ex = {" ".join(norm(t)) for t in corpus_texts}
    keep = [r for r, x in zip(rows, c) if x < thr and " ".join(norm(r["text"])) not in ex]
    return keep, len(rows) - len(keep)


def build_source(name, out_dir="data/testsets", raw_dir="data/external/sources", train_corpora=(), get=None, pause=0.6, get_bytes=None):
    spec = SOURCES[name]
    raw = []
    for ds, cfg, split, where in spec["fetch"]:
        raw += fetch_rows(ds, cfg, split, where, get=get, start=spec.get("fetch_start", 0), pause=pause)
    if spec.get("zip"):
        raw += fetch_zip_json(*spec["zip"], get_bytes=get_bytes)
    if spec.get("row_filter"):
        k, v = spec["row_filter"]; raw = [r for r in raw if r.get(k) == v]
    raw_path = Path(raw_dir) / f"{name}_raw.jsonl"
    write_jsonl(raw_path, raw)
    items = spec["adapt"](raw)
    rows, drops = build_items(items, spec["cap_per_class"])
    corpus_texts = [json.loads(l)["text"] for c in train_corpora for l in open(c) if l.strip()]
    if corpus_texts:
        rows, nd = drop_near_duplicates(rows, corpus_texts)
        if nd:
            drops["near_duplicate_of_training_corpus"] = nd
    for r in rows:
        r.update(split="test", writer=f"human-{name}")
    tpath = Path(out_dir) / f"{name}_test_frozen.jsonl"
    write_jsonl(tpath, rows)
    manifest = {"source": name, "fetched_from": [list(f) for f in spec["fetch"]] + ([list(spec["zip"])] if spec.get("zip") else []), "raw_rows": len(raw), "raw_sha256": sha256_file(raw_path),
                "license_note": spec["license"], "label_provenance": spec["labels"], "domain": spec["domain"],
                "classes_present": sorted({r["category"] for r in rows}), "label_mapping_rule": "see adapter in probekit_external.py",
                "caps": {"per_class": spec["cap_per_class"], "neutral": "size of largest other class"}, "seed": SEED, "drops": drops,
                "n": len(rows), "class_counts": dict(Counter(r["category"] for r in rows)),
                "near_duplicate_rule": "dropped if >= 50% of 5-grams inside one training-corpus passage or exact text match",
                "testset_file": tpath.name, "testset_sha256": sha256_file(tpath),
                "note": "neither the raw data nor this derived test jsonl is committed (third-party terms unclear or share-alike); rebuild with `python probekit_external.py build --source <name>` and check testset_sha256"}
    (Path(out_dir) / f"{name}_test_frozen.manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    return manifest


def load_source(name, out_dir="data/testsets"):
    """Hash-verified load of an external test set (same guard as probekit_testset.load_frozen)."""
    mp = Path(out_dir) / f"{name}_test_frozen.manifest.json"
    m = json.loads(mp.read_text()); p = mp.with_name(m["testset_file"])
    if sha256_file(p) != m["testset_sha256"]:
        raise RuntimeError(f"external test set {p} changed since its manifest was written")
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()], m


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    b = sp.add_parser("build"); b.add_argument("--source", required=True, choices=list(SOURCES)); b.add_argument("--out-dir", default="data/testsets")
    b.add_argument("--train-corpus", nargs="*", default=["data/corpus/corpus_merged2.jsonl", "data/corpus/corpus_human_goemotions_pool.jsonl"])
    l = sp.add_parser("leakage"); l.add_argument("--out-dir", default="data/testsets"); l.add_argument("--corpus", nargs="+", required=True)
    l.add_argument("--sources", nargs="*", default=list(SOURCES))
    a = ap.parse_args()
    if a.cmd == "build":
        m = build_source(a.source, a.out_dir, train_corpora=a.train_corpus)
        print(json.dumps({k: m[k] for k in ("source", "raw_rows", "n", "class_counts", "drops", "testset_sha256")}, indent=1))
    else:
        ok = True
        for s in a.sources:
            if not (Path(a.out_dir) / f"{s}_test_frozen.manifest.json").exists():
                continue
            rows, _ = load_source(s, a.out_dir)
            for c in a.corpus:
                r = overlap([x["text"] for x in rows], [json.loads(q)["text"] for q in open(c) if q.strip()])
                print(f"{s} vs {c}: exact={r['exact']} containment>=0.5={r['containment>=0.5']} max={r['max_containment']}")
                ok &= r["exact"] == 0 and r["containment>=0.5"] == 0
        print("EXTERNAL LEAKAGE CHECK:", "PASS" if ok else "FAIL")
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
