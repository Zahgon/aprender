#!/usr/bin/env bash
# code_eval_nightly.sh - one nightly `apr code eval` receipt and its ranked gap list
# (#5288, APR-CODE-EVAL-001 §10, row E7 of #5342).
#
# Runs `apr code eval` over one suite on one backend and writes the receipt, the ranked
# list of failing tasks (contracts/apr-code-eval-gap-v1.yaml) and the evidence beside
# them. LAB only: it runs on an idle host at night, never on a PR, merge-queue or
# release path, and never blocks one.
#
#   bash scripts/code_eval_nightly.sh --tasks FILE --envs FILE --model FILE \
#        --backend cpu|gpu --suite NAME [--out DIR] [--n N] [--seed N] [--cutoff DATE]
#
# Writes DIR/receipt.json, DIR/gaps.md, DIR/meta.txt (apr version line, commit, model
# path + sha256, suite, task-set sha256, host, start/end UTC, exit) and DIR/run.log.
#
# Exit is apr's: 0 when the run wrote its receipt, whatever the pass rate (a failing
# task is a gap-list row, not a red night). No receipt is a red night. Bad arguments
# exit 2.
set -euo pipefail

usage() { printf 'usage: %s --tasks FILE --envs FILE --model FILE --backend cpu|gpu --suite NAME [--out DIR] [--n N] [--seed N] [--cutoff DATE]\n' "$0" >&2; exit 2; }

TASKS="" ENVS="" MODEL="" BACKEND="" SUITE="" OUT="" N=1 SEED=0 CUTOFF=""
while [ "$#" -gt 0 ]; do
  [ "$#" -ge 2 ] || usage
  case "$1" in
    --tasks) TASKS="$2" ;;
    --envs) ENVS="$2" ;;
    --model) MODEL="$2" ;;
    --backend) BACKEND="$2" ;;
    --suite) SUITE="$2" ;;
    --out) OUT="$2" ;;
    --n) N="$2" ;;
    --seed) SEED="$2" ;;
    --cutoff) CUTOFF="$2" ;;
    *) usage ;;
  esac
  shift 2
done
for f in "$TASKS" "$ENVS" "$MODEL"; do
  [ -n "$f" ] || usage
  [ -f "$f" ] || { printf '%s is not a file\n' "$f" >&2; exit 2; }
done
[ -n "$SUITE" ] || usage
case "$BACKEND" in
  cpu) BE_FLAG="--no-gpu" ;;
  gpu) BE_FLAG="" ;;
  *) usage ;;
esac
case "$N$SEED" in *[!0-9]*) usage ;; esac

cd "$(git rev-parse --show-toplevel)"
. scripts/apr_bin.sh || exit 1

MODEL_ID="$(basename "$MODEL")"
MODEL_ID="${MODEL_ID%.*}"
OUT="${OUT:-target/code-eval-nightly/$(date -u +%Y%m%dT%H%M%SZ)-$SUITE-$MODEL_ID-$BACKEND}"
mkdir -p "$OUT/work"

{
  printf 'suite %s\nissue 5288\n' "$SUITE"
  printf 'apr %s\n' "$("$APR" --version 2>&1 | head -n 1)"
  printf 'commit %s\n' "$(git rev-parse HEAD)"
  printf 'model %s\nmodel_sha256 %s\n' "$MODEL" "$(sha256sum "$MODEL" | cut -d' ' -f1)"
  printf 'tasks %s\ntasks_sha256 %s\n' "$TASKS" "$(sha256sum "$TASKS" | cut -d' ' -f1)"
  printf 'backend %s\nn %s\nseed %s\nhost %s\nstart %s\n' "$BACKEND" "$N" "$SEED" "$(hostname)" "$(date -u +%FT%TZ)"
} > "$OUT/meta.txt"

ARGS=(code eval --tasks "$TASKS" --envs "$ENVS" --model "$MODEL" --suite "$SUITE"
      --n "$N" --seed "$SEED" --work-dir "$OUT/work"
      --receipt "$OUT/receipt.json" --gaps "$OUT/gaps.md")
if [ -n "$BE_FLAG" ]; then ARGS+=("$BE_FLAG"); fi
if [ -n "$CUTOFF" ]; then ARGS+=(--cutoff "$CUTOFF"); fi

RC=0
"$APR" "${ARGS[@]}" > "$OUT/run.log" 2>&1 < /dev/null || RC=$?
if [ "$RC" -eq 0 ] && [ ! -s "$OUT/receipt.json" ]; then
  printf 'apr exited 0 but wrote no receipt\n' >> "$OUT/run.log"
  RC=1
fi

printf 'end %s\nexit %s\n' "$(date -u +%FT%TZ)" "$RC" >> "$OUT/meta.txt"
tail -n 3 "$OUT/run.log"
printf 'receipt %s/receipt.json\ngaps %s/gaps.md\n' "$OUT" "$OUT"
exit "$RC"
