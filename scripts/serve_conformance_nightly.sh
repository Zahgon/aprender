#!/usr/bin/env bash
# serve_conformance_nightly.sh - one agent-conformance receipt (#5349, EPIC 0.72 row R-6, S2).
#
# Starts `apr serve run MODEL` on one backend, runs every case of
# realizar::api::conformance through the serve_conformance example, and writes
# the receipt plus the evidence beside it. LAB only: it runs on an idle host at
# night, never on a PR, merge-queue or release path.
#
#   bash scripts/serve_conformance_nightly.sh --model <file.gguf> --backend cpu|gpu \
#        [--out DIR] [--port N] [--wait-s N] [--timeout-s N]
#
# Writes DIR/receipt.json (the example's cells), DIR/meta.txt (apr version line,
# model path + sha256, host, start/end UTC, server_up), DIR/run.log, DIR/server.log.
#
# Exit 0 only when every cell passes and none is missing. A server that never
# answers /health is NOT skipped: the example still runs and every cell reads
# not_measured, so the receipt exists and the exit is 1. Bad arguments exit 2.
set -euo pipefail

usage() { printf 'usage: %s --model FILE --backend cpu|gpu [--out DIR] [--port N] [--wait-s N] [--timeout-s N]\n' "$0" >&2; exit 2; }

MODEL="" BACKEND="" OUT="" PORT=18090 WAIT_S=180 TIMEOUT_S=300
while [ "$#" -gt 0 ]; do
  [ "$#" -ge 2 ] || usage
  case "$1" in
    --model) MODEL="$2" ;;
    --backend) BACKEND="$2" ;;
    --out) OUT="$2" ;;
    --port) PORT="$2" ;;
    --wait-s) WAIT_S="$2" ;;
    --timeout-s) TIMEOUT_S="$2" ;;
    *) usage ;;
  esac
  shift 2
done
[ -n "$MODEL" ] || usage
[ -f "$MODEL" ] || { printf 'model %s is not a file\n' "$MODEL" >&2; exit 2; }
case "$BACKEND" in
  cpu) BE_FLAG="--no-gpu" ;;
  gpu) BE_FLAG="--gpu" ;;
  *) usage ;;
esac
case "$PORT$WAIT_S$TIMEOUT_S" in *[!0-9]*) usage ;; esac

cd "$(git rev-parse --show-toplevel)"
. scripts/apr_bin.sh || exit 1

MODEL_ID="$(basename "$MODEL" .gguf)"
OUT="${OUT:-target/serve-conformance/$(date -u +%Y%m%dT%H%M%SZ)-$MODEL_ID-$BACKEND}"
mkdir -p "$OUT"

# The runner is built before the server starts, so its compile time never eats the server's wait.
cargo build --release -p aprender-serve --example serve_conformance > "$OUT/build.log" 2>&1 \
  || { printf 'serve_conformance did not build; see %s/build.log\n' "$OUT" >&2; exit 1; }

{
  printf 'suite apr-serve-conformance\nissue 5349\n'
  printf 'apr %s\n' "$("$APR" --version 2>&1 | head -n 1)"
  printf 'commit %s\n' "$(git rev-parse HEAD)"
  printf 'model %s\nmodel_sha256 %s\n' "$MODEL" "$(sha256sum "$MODEL" | cut -d' ' -f1)"
  printf 'backend %s\nhost %s\nstart %s\n' "$BACKEND" "$(hostname)" "$(date -u +%FT%TZ)"
} > "$OUT/meta.txt"

SPID=""
stop_server() {
  if [ -n "$SPID" ] && kill -0 "$SPID" 2>/dev/null; then
    kill "$SPID" 2>/dev/null || true
    wait "$SPID" 2>/dev/null || true
  fi
}
trap stop_server EXIT

"$APR" serve run "$MODEL" --port "$PORT" "$BE_FLAG" > "$OUT/server.log" 2>&1 < /dev/null &
SPID=$!
printf 'server_pid %s\n' "$SPID" >> "$OUT/meta.txt"

UP=0
for _ in $(seq 1 "$WAIT_S"); do
  if curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then UP=1; break; fi
  kill -0 "$SPID" 2>/dev/null || break
  sleep 1
done
printf 'server_up %s\n' "$UP" >> "$OUT/meta.txt"

RC=0
cargo run -q --release -p aprender-serve --example serve_conformance -- \
  --url "http://127.0.0.1:$PORT" --model "$MODEL_ID" --backend "$BACKEND" \
  --out "$OUT/receipt.json" --timeout-s "$TIMEOUT_S" > "$OUT/run.log" 2>&1 || RC=$?

printf 'end %s\nexit %s\n' "$(date -u +%FT%TZ)" "$RC" >> "$OUT/meta.txt"
tail -n 1 "$OUT/run.log"
printf 'receipt %s/receipt.json\n' "$OUT"
exit "$RC"
