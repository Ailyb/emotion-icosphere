#!/usr/bin/env bash
# run_merged.sh -- merge corpora, mark hard passages, evaluate, compare estimators, robustness, confusion.
#
#   ./run_merged.sh data/corpus/corpus_v2.jsonl data/corpus/corpus_human_goemotions.jsonl [more.jsonl ...]
#
# Env: MODEL (default mlx-community/Llama-3.2-3B-Instruct), TAG (default merged), LAYERS (e.g. "8 10 12 ... 26" for deeper models), DRY=1 to only print the commands,
#      SKIP_EVAL=1 to resume after the estimator comparison (skips merge, hard-passage marking and eval_v3).
# Outputs go to data/eval_$TAG/<model>/ so earlier results are NOT overwritten.
# Reads data/hardset/hardset.jsonl and data/hardset/own_voice.jsonl as TEST ONLY (nothing here trains on them).
set -euo pipefail
[ "$#" -ge 1 ] || { echo "usage: $0 corpus1.jsonl [corpus2.jsonl ...]"; exit 1; }
MODEL="${MODEL:-mlx-community/Llama-3.2-3B-Instruct}"
TAG="${TAG:-merged}"
SLUG="${MODEL//\//__}"
W="data/eval_${TAG}/${SLUG}"
MERGED="data/corpus/corpus_${TAG}.jsonl"
SCORED="data/corpus/corpus_${TAG}_scored.jsonl"

run() { echo "+ $*"; if [ -z "${DRY:-}" ]; then "$@"; fi; }

if [ -z "${SKIP_EVAL:-}" ]; then
echo "== merge =="
echo "+ cat $* > $MERGED"
if [ -z "${DRY:-}" ]; then mkdir -p data/corpus "$W"; cat "$@" > "$MERGED"; python - "$MERGED" <<'PY'
import json, sys, collections
ids = [json.loads(l)["id"] for l in open(sys.argv[1]) if l.strip()]
d = [k for k, v in collections.Counter(ids).items() if v > 1]
assert not d, f"duplicate ids in merged corpus, e.g. {d[:3]}"
print(f"merged {len(ids)} passages, no duplicate ids")
PY
fi
echo "== mark hard passages =="
run python adversarial_filter.py --corpus "$MERGED" --out "$SCORED"
echo "== estimator comparison (extracts hidden states the first time: slow) =="
run python probekit_eval_v3.py --model "$MODEL" --corpus "$SCORED" --cache-dir "$W" --out "$W/v3" ${LAYERS:+--layers $LAYERS}
fi
for m in meandiff lda logreg; do
  echo "== robustness: $m =="
  run python make_config.py --model "$MODEL" --method "$m" --eval-dir "$W/v3"
  if [ -z "${DRY:-}" ]; then CFG=$(ls -t "$W"/v3/best_config_"${m}"_L*.json | head -1); else CFG="$W/v3/best_config_${m}_L<layer>.json"; fi
  run python probekit_robustness.py --model "$MODEL" --corpus "$SCORED" --cache-dir "$W" --config "$CFG" --out "$W/robustness_${m}" --allow-partial
done
echo "== confusion diagnostics =="
run python probekit_confusion.py --model "$MODEL" --corpus "$SCORED" --cache-dir "$W" --eval-dir "$W/v3"
echo
echo "Done. Send me these files:"
for f in "$W/v3/report_v3.md" "$W/v3/errors_v3.md" "$W/v3/confusion_v3.md" "$W/robustness_meandiff/robustness.md" "$W/robustness_lda/robustness.md" "$W/robustness_logreg/robustness.md"; do echo "  $f"; done
