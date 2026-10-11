#!/usr/bin/env bash
# nightly_checkout_c_test.sh — a scheduled nightly.yml run builds its own head_sha, never main's later tip (#5082)
#
# WHY. The train (scripts/release/nightly_train.sh) and the producers judge a night by the head_sha of that night's
#   scheduled nightly.yml run: main's head when GitHub CREATED the run. Scheduled runs are created hours after their
#   cron and start later still, so a gate job that checks out `main` builds and gates whatever merged in between,
#   and the night's "green" belongs to a commit nothing judged by that sha. The gate's checkout must therefore be
#   github.sha on a schedule; a dispatch keeps `main` (a dispatch from a PR branch must never ship that branch).
#
# THE CHECK. In the `gate:` job, the `ref:` of the first actions/checkout step is exactly
#     ${{ github.event_name == 'schedule' && github.sha || 'main' }}
#   Anything else, an absent ref included (it would check out the dispatching ref), is red.
#
# USAGE
#   bash scripts/tests/nightly_checkout_c_test.sh [WORKFLOW]   check WORKFLOW (default .github/workflows/nightly.yml)
#   bash scripts/tests/nightly_checkout_c_test.sh --mutants    each mutant of the real workflow must turn the check red
# EXIT 0 green · 1 red · 3 a caller error.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WF="$HERE/../../.github/workflows/nightly.yml"
WANT="\${{ github.event_name == 'schedule' && github.sha || 'main' }}"

# gate_ref FILE -> the ref: of the first checkout step of the gate: job, or "<none>"; rc 1 when there is no such step
gate_ref() {
    awk '
        /^  [A-Za-z0-9_-]+:[[:space:]]*$/ { injob = ($1 == "gate:"); step = 0; next }
        !injob { next }
        /^      - / { if (step) { print "<none>"; found = 1; exit } }
        /uses: actions\/checkout/ { step = 1; next }
        step && /^          ref:/ { sub(/^          ref:[[:space:]]*/, ""); print; found = 1; exit }
        END { if (!found && step) print "<none>"; if (!found && !step) exit 1 }
    ' "$1"
}

check() {
    local got
    [ -f "$1" ] || { echo "nightly_checkout_c_test: no workflow at $1" >&2; return 3; }
    got="$(gate_ref "$1")" || { echo "RED: the gate job of $1 has no actions/checkout step"; return 1; }
    if [ "$got" = "$WANT" ]; then
        echo "GREEN: the gate checks out $got"
        return 0
    fi
    echo "RED: the gate checks out ${got}, not $WANT: a late scheduled run would build main's later tip (#5082)"
    return 1
}

mutants() {
    local n=0 bad=0 name expr rc
    TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/nightly-checkout-c.XXXXXX")"
    trap 'rm -rf -- "${TMP_DIR:?}"' EXIT
    check "$WF" > /dev/null || { echo "BROKEN: the real workflow is red, so no mutant can be read"; return 1; }
    while IFS=$'\t' read -r name expr; do
        [ -n "$name" ] || continue
        n=$((n + 1))
        sed -E "$expr" "$WF" > "$TMP_DIR/$name.yml"
        if cmp -s "$WF" "$TMP_DIR/$name.yml"; then
            echo "BROKEN: $name changed nothing"; bad=$((bad + 1)); continue
        fi
        rc=0; check "$TMP_DIR/$name.yml" > /dev/null || rc=$?
        if [ "$rc" = 1 ]; then echo "killed: $name"; else echo "SURVIVED: $name (rc $rc)"; bad=$((bad + 1)); fi
    done <<'MUTANTS'
m1_checkout_main	s/ref: \$\{\{ github\.event_name == 'schedule' && github\.sha \|\| 'main' \}\}/ref: main/
m2_ref_dropped	/ref: \$\{\{ github\.event_name == 'schedule'/d
m3_dispatch_gets_sha	s/github\.event_name == 'schedule'/github.event_name == 'workflow_dispatch'/
m4_schedule_builds_main_tip	s/&& github\.sha \|\|/\&\& 'main' ||/
MUTANTS
    echo "--- $((n - bad))/$n mutants killed ---"
    [ "$bad" = 0 ] && [ "$n" -gt 0 ]
}

case "${1:-}" in
    --mutants) mutants ;;
    -h|--help) sed -n '2,17p' "$0" ;;
    "") check "$WF" ;;
    -*) echo "nightly_checkout_c_test: unknown option $1" >&2; exit 3 ;;
    *) check "$1" ;;
esac
