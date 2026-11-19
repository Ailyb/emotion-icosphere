#!/usr/bin/env bash
# rebuild_human_data.sh -- rebuild every file that contains GoEmotions (Reddit) text from the ORIGINAL release.
# Those files are not stored in the repository (the data have no stated licence; see ../NOTICE.md). Only ids and hashes are.
#   1. fetch the original release into data/external/goemotions (train.tsv dev.tsv test.tsv emotions.txt, goemotions_*.csv):
#      https://github.com/google-research/google-research/tree/master/goemotions  (data/ folder; train/dev/test TSVs)
#   2. ./rebuild_human_data.sh        (deterministic; prints OK or MISMATCH for each file against data/DATA_HASHES.json)
set -euo pipefail
python probekit_testset.py build
python probekit_testset.py pool
cat data/corpus/corpus_v2.jsonl data/corpus/corpus_v3_qwen3.8-27b.jsonl data/corpus/corpus_human_goemotions_train.jsonl > data/corpus/corpus_merged2.jsonl
python adversarial_filter.py --corpus data/corpus/corpus_merged2.jsonl --out data/corpus/corpus_merged2_scored.jsonl
python - <<'PY'
import hashlib, json, sys
bad = 0
for f, h in json.load(open("data/DATA_HASHES.json"))["sha256"].items():
    got = hashlib.sha256(open(f, "rb").read()).hexdigest()
    print(("OK       " if got == h else "MISMATCH ") + f); bad += got != h
sys.exit(1 if bad else 0)
PY
