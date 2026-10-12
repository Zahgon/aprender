#!/usr/bin/env bash
# apr_code_edit_verify.sh - does `apr code -p` finish a scripted edit-and-verify
# task on a real model, on CUDA? (#3719)
#
# One run is one cell: one model on one machine. The fixture under
# tests/fixtures/apr-code-edit-verify/ is a small shell project whose
# test_stats.sh fails; the fix is a one-line edit to stats.sh. task.txt asks
# the agent to make that fix and run the test.
#
# Nothing the agent SAYS counts as evidence:
#   - the edit is judged by diffing the working copy against the fixture;
#   - the test is re-run HERE, after the agent exits;
#   - "the agent ran the test" is read from EDIT_VERIFY_RUN_LOG, which
#     test_stats.sh appends to and which the agent's shell tool inherits. The
#     harness's own re-run unsets it, so it never appears in the log;
#   - the backend is read from the serve CHILD's own output. `apr code` has no
#     GPU flag: its driver spawns an `apr serve --gpu` child, and a requested
#     flag proves nothing. The driver pipes the child's output and shows it
#     only when startup fails, so APR_BIN points it at a wrapper that runs the
#     pinned binary and tees the child's stdout and stderr to files. The JSON
#     document's `backend` object (what each completion reported) must agree.
#
# The judge is the jq block at the bottom; it writes <out>/cell.json.
#
# Usage:
#   scripts/apr_code_edit_verify.sh --model FILE --out DIR
#       [--apr FILE] [--max-turns N] [--lock-wait SEC] [--timeout SEC]
#   DIR must not exist yet; the run writes every artifact into it.
#   --apr names a binary built elsewhere from this commit; its `apr --version`
#   sha must be HEAD's. Without it, scripts/apr_bin.sh resolves and proves one.
#
# Exit: 0 PASS; 1 FAIL (cell.json names the first failing mechanism);
#       2 decline (GPU lock not acquired, missing input); 3 usage error.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FIXTURE="$ROOT/tests/fixtures/apr-code-edit-verify"
# APR_GPU_LOCK exists so a dry run never queues on the shared lock.
GPU_LOCK="${APR_GPU_LOCK:-/tmp/apr-gpu.lock}"

MODEL=""
OUT=""
APR_GIVEN=""
MAX_TURNS=12
LOCK_WAIT=3600
TIMEOUT=900

usage() {
    sed -n '/^# Usage:/,/^# Exit:/p' "$0" | sed 's/^# \{0,1\}//' >&2
    exit 3
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --model) MODEL="${2:-}"; shift 2 ;;
        --out) OUT="${2:-}"; shift 2 ;;
        --apr) APR_GIVEN="${2:-}"; shift 2 ;;
        --max-turns) MAX_TURNS="${2:-}"; shift 2 ;;
        --lock-wait) LOCK_WAIT="${2:-}"; shift 2 ;;
        --timeout) TIMEOUT="${2:-}"; shift 2 ;;
        -h|--help) usage ;;
        *) printf 'unknown argument: %s\n' "$1" >&2; usage ;;
    esac
done

if [ -z "$MODEL" ] || [ -z "$OUT" ]; then
    usage
fi
if [ ! -f "$MODEL" ]; then
    printf 'DECLINE  model file not found: %s\n' "$MODEL" >&2
    exit 2
fi
if ! command -v jq >/dev/null 2>&1; then
    printf 'DECLINE  jq not found: the judge needs it\n' >&2
    exit 2
fi

if [ -n "$APR_GIVEN" ]; then
    APR="$(cd "$(dirname "$APR_GIVEN")" && pwd)/$(basename "$APR_GIVEN")"
    head_short="$(git -C "$ROOT" rev-parse --short=9 HEAD)"
    if ! "$APR" --version 2>/dev/null | grep -qF "($head_short"; then
        printf 'DECLINE  %s does not report HEAD %s: %s\n' "$APR" "$head_short" \
            "$("$APR" --version 2>&1 | head -1)" >&2
        exit 2
    fi
else
    # shellcheck source=scripts/apr_bin.sh
    . "$ROOT/scripts/apr_bin.sh" || exit 2
fi

if [ -e "$OUT" ]; then
    printf 'usage: --out %s already exists; give a fresh directory\n' "$OUT" >&2
    exit 3
fi
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
cp -R "$FIXTURE/project" "$OUT/project"

# The wrapper runs the pinned binary for the serve child and records what the
# child prints. tee writes back to the original streams, so the driver sees
# exactly what it would have seen without the wrapper.
wrapper="$OUT/apr-serve-child"
{
    printf '#!/usr/bin/env bash\n'
    printf 'exec %q "$@" > >(tee -a %q) 2> >(tee -a %q >&2)\n' \
        "$APR" "$OUT/serve-child.stdout" "$OUT/serve-child.stderr"
} > "$wrapper"
chmod +x "$wrapper"
: > "$OUT/serve-child.stdout"
: > "$OUT/serve-child.stderr"
: > "$OUT/agent-test-runs.log"

"$APR" --version > "$OUT/apr-version.txt" 2>&1
sha256sum "$MODEL" | cut -d' ' -f1 > "$OUT/model.sha256"
date -u +%FT%TZ > "$OUT/started.txt"

prompt="$(cat "$FIXTURE/task.txt")"

# The GPU call takes the shared lock directly and runs choom'd to 1000, so
# this run and never a CI job is the OOM victim. The lock-acquired marker,
# written once the lock is held, separates "never got the GPU" from "apr
# exited 75".
set +e
(
    cd "$OUT/project" &&
    flock -w "$LOCK_WAIT" -E 75 "$GPU_LOCK" choom -n 1000 -- \
        bash -c 'date -u +%FT%TZ > "$1"; shift; exec "$@"' _ "$OUT/lock-acquired" \
        env APR_BIN="$wrapper" EDIT_VERIFY_RUN_LOG="$OUT/agent-test-runs.log" \
        timeout "$TIMEOUT" \
        "$APR" code -p \
            --model "$MODEL" \
            --project "$OUT/project" \
            --output-format json \
            --emit-trace "$OUT/trace.jsonl" \
            --max-turns "$MAX_TURNS" \
            -- "$prompt"
) > "$OUT/stdout.json" 2> "$OUT/stderr.txt"
rc=$?
set -e
date -u +%FT%TZ > "$OUT/finished.txt"

# The independent re-run: the agent's report of the test is not the test.
set +e
(cd "$OUT/project" && env -u EDIT_VERIFY_RUN_LOG timeout 120 bash test_stats.sh) \
    > "$OUT/rerun.txt" 2>&1
test_rc=$?
set -e

diff -u "$FIXTURE/project/stats.sh" "$OUT/project/stats.sh" > "$OUT/stats.diff" || true
test_untouched=false
if cmp -s "$FIXTURE/project/test_stats.sh" "$OUT/project/test_stats.sh"; then
    test_untouched=true
fi
# A one-line edit: one line removed and one added, nothing else in the project.
removed="$(grep -c '^-[^-]' "$OUT/stats.diff" || true)"
added="$(grep -c '^+[^+]' "$OUT/stats.diff" || true)"
extra="$(cd "$OUT/project" && find . -type f ! -name stats.sh ! -name test_stats.sh | wc -l)"

# The serve child's own words: the line that says what it loaded on, and any
# fallback line.
child_backend="$(grep -ho 'gpu-layers: .*(backend=[a-z]*)' \
    "$OUT/serve-child.stdout" "$OUT/serve-child.stderr" | tail -1 || true)"
child_fallback="$(cat "$OUT/serve-child.stdout" "$OUT/serve-child.stderr" | grep -c "GPU->CPU FALLBACK" || true)"
agent_runs="$(grep -c 'test_stats.sh' "$OUT/agent-test-runs.log" || true)"

envelope='null'
if jq -e 'type == "object"' "$OUT/stdout.json" >/dev/null 2>&1; then
    envelope="$(jq -c . "$OUT/stdout.json")"
fi

jq -n \
    --arg model "$(basename "$MODEL")" \
    --arg model_sha256 "$(cat "$OUT/model.sha256")" \
    --arg apr_version "$(head -1 "$OUT/apr-version.txt")" \
    --argjson rc "$rc" \
    --argjson test_rc "$test_rc" \
    --argjson locked "$([ -s "$OUT/lock-acquired" ] && echo true || echo false)" \
    --argjson envelope "$envelope" \
    --arg child_backend "$child_backend" \
    --argjson child_fallback "${child_fallback:-0}" \
    --argjson test_untouched "$test_untouched" \
    --argjson removed "${removed:-0}" \
    --argjson added "${added:-0}" \
    --argjson extra "$extra" \
    --argjson agent_runs "${agent_runs:-0}" \
    --rawfile diff "$OUT/stats.diff" \
'
def first_fail:
  if ($locked | not) then "lock: GPU lock not acquired"
  elif $rc == 124 then "timeout: apr code did not exit in time"
  elif $envelope == null then "envelope: stdout is not one JSON document (rc \($rc))"
  elif $envelope.is_error != false then "agent: run ended in error (\($envelope.error.kind // $envelope.subtype // "unknown"))"
  elif $rc != 0 then "exit: apr code rc \($rc)"
  elif ($child_backend | test("backend=cuda") | not) then "backend: serve child printed no backend=cuda load line"
  elif $child_fallback > 0 then "backend: serve child printed a GPU->CPU FALLBACK line"
  elif ($envelope.backend.ran // null) != "gpu" then "backend: envelope backend.ran is \($envelope.backend.ran // "null"), not gpu"
  elif $envelope.backend.fell_back != false then "backend: envelope backend.fell_back is not false"
  elif ($test_untouched | not) then "edit: test_stats.sh was changed"
  elif $extra > 0 then "edit: the agent created \($extra) extra file(s)"
  elif ($removed != 1 or $added != 1) then "edit: stats.sh diff is -\($removed)/+\($added), not one line"
  elif $agent_runs < 1 then "verify: the agent never ran test_stats.sh"
  elif $test_rc != 0 then "verify: independent re-run of test_stats.sh rc \($test_rc)"
  else null end;
first_fail as $f
| {
    schema: "apr-code-edit-verify-cell-v1",
    verdict: (if $f == null then "PASS" else "FAIL" end),
    first_failing_mechanism: $f,
    model: $model, model_sha256: $model_sha256, apr_version: $apr_version,
    rc: $rc, test_rc: $test_rc,
    serve_child_backend_line: $child_backend,
    serve_child_fallback_lines: $child_fallback,
    envelope_backend: ($envelope.backend // null),
    num_turns: ($envelope.num_turns // null),
    duration_ms: ($envelope.duration_ms // null),
    agent_test_runs: $agent_runs,
    stats_diff: $diff
  }
' > "$OUT/cell.json"

jq -r '"\(.verdict)  \(.model)  \(.first_failing_mechanism // "all mechanisms")  [\(.serve_child_backend_line)]"' "$OUT/cell.json"
if [ ! -s "$OUT/lock-acquired" ]; then
    exit 2
fi
[ "$(jq -r .verdict "$OUT/cell.json")" = PASS ]
