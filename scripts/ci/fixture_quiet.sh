#!/usr/bin/env bash
# fixture_quiet.sh -- run a case table whose fixture rows print GitHub workflow
# commands, so that none of them reaches the runner as a command (#4596, #4610).
#
# WHY THIS EXISTS
# ---------------
# `fat_driver.py self-test` plants negative controls that call the driver's real
# reporter, so it prints `::error::section a: failure`, `::error::section a:
# cancelled` and `::error::external job 'workspace-test': not completed within
# 12000s`. Printed bare, the runner made each one a job annotation, listed next
# to the job's real failure. On run 36410838844 a reviewer read the fixture's
# 12000s line as a real runner timeout, and that reading was acted on as a
# ruling. The shard had failed 63 minutes later, on something else.
#
# WHAT IT DOES
# ------------
# Each line the command prints (stdout and stderr) is shown with a `fixture| `
# prefix, between a label line and an end line, inside a
# `::stop-commands::<token>` ... `::<token>::` span. The prefix labels the text
# as fixture output. The span stops the runner from acting on any command form,
# including ones a prefix would not break. The token is 32 hex digits drawn
# fresh from /dev/urandom on every run, so a fixture line cannot end the span.
# The exit status is the command's own: a BAD row still turns the step red.
#
#   scripts/ci/fixture_quiet.sh <label> -- <command> [args...]
#   scripts/ci/fixture_quiet.sh --self-test     # case table, then the wiring check on this repo's workflows
#   scripts/ci/fixture_quiet.sh --check-wiring [WORKFLOW.yml...]
#
# The ci.yml step wraps the driver's case table with it; the step keeps its
# name and its verdict. Rust port of the driver: #4610 item 2, with a7.
#
# WHERE IT MAY BE WIRED (#5063)
# -----------------------------
# Only around `fat_driver.py self-test`. Wrapped around the driver's real `run`
# or `wait` step, it would turn real section failures into fixture text and the
# run would stay quiet. --check-wiring is RED on a self-test step that is not
# wrapped and on a run or wait step that is; --self-test runs it on this repo's
# workflows, so the x86-main step that runs the case table also runs the check.

set -euo pipefail

draw_token() {
    local t
    t=$(od -An -N16 -tx1 /dev/urandom | tr -d ' \n')
    [[ "$t" =~ ^[0-9a-f]{32}$ ]] || return 1
    printf '%s\n' "$t"
}

prefix_lines() {
    local line
    while IFS= read -r line || [ -n "$line" ]; do
        printf 'fixture| %s\n' "$line"
    done
}

# quiet LABEL -- CMD... : the wrapper itself. Returns the command's status; 2 on
# a usage error or no token; 1 if the command passed but its output was lost.
quiet() {
    local label="${1:-}"
    shift || true
    [ "${1:-}" = "--" ] && shift
    if [ -z "$label" ] || [ "$#" -eq 0 ]; then
        echo "fixture_quiet.sh: usage: fixture_quiet.sh LABEL -- COMMAND [ARGS...]" >&2
        return 2
    fi
    local tok
    if ! tok=$(draw_token); then
        echo "fixture_quiet.sh: could not draw a token from /dev/urandom" >&2
        return 2
    fi
    printf '%s: the lines marked fixture| are case-table text, not job errors (#4596)\n' "$label"
    printf '::stop-commands::%s\n' "$tok"
    local st rc
    set +e
    "$@" 2>&1 | prefix_lines
    st=("${PIPESTATUS[@]}")
    set -e
    rc=${st[0]}
    if [ "$rc" -eq 0 ] && [ "${st[1]}" -ne 0 ]; then
        rc=1
    fi
    printf '::%s::\n' "$tok"
    printf '%s: end of case-table text, rc=%s\n' "$label" "$rc"
    return "$rc"
}

# check_wiring WORKFLOW... -> one FAIL line per misplaced wrapper. rc 0 clean;
# 1 a FAIL; 2 a file not readable, or no fat_driver.py self-test step at all (a
# check that found nothing to check is not clean). A step is a `- ` item and the
# more-indented lines under it; comment lines and trailing ` #` comments are not
# read, so a comment naming the wrapper wraps nothing.
check_wiring() {
    local f out
    if [ "$#" -eq 0 ]; then
        echo "fixture_quiet.sh: --check-wiring needs a workflow file" >&2
        return 2
    fi
    for f in "$@"; do
        [ -r "$f" ] || { echo "FAIL $f: not readable"; return 2; }
    done
    if ! out=$(awk -f "$SCRIPT_DIR/fixture_quiet_wiring.awk" "$@"); then
        echo "FAIL awk exited non-zero on the workflows"
        return 2
    fi
    if grep -qx NONE <<<"$out"; then
        grep -vx NONE <<<"$out" || true
        echo "FAIL no fat_driver.py self-test step in: $*"
        return 2
    fi
    [ -z "$out" ] || { printf '%s\n' "$out"; return 1; }
}

# ---------------------------------------------------------------------------
# case table. Captured output is never echoed raw: it carries the span's open
# and close lines, and printing those here would stop this step's own commands.

span_ok() {   # OUT: exactly one open and one matching close, open first, and
              # no other line between them, or outside them, starts with `::`.
    local out="$1" open close n_open n_close
    n_open=$(grep -c '^::stop-commands::' <<<"$out" || true)
    [ "$n_open" -eq 1 ] || return 1
    open=$(grep -n '^::stop-commands::' <<<"$out" | head -n 1)
    local tok="${open#*::stop-commands::}"
    [[ "$tok" =~ ^[0-9a-f]{32}$ ]] || return 1
    n_close=$(grep -c "^::${tok}::\$" <<<"$out" || true)
    [ "$n_close" -eq 1 ] || return 1
    close=$(grep -n "^::${tok}::\$" <<<"$out" | head -n 1)
    [ "${open%%:*}" -lt "${close%%:*}" ] || return 1
    local others
    others=$(grep -c '^::' <<<"$out" || true)
    (( others == 2 ))
}

token_of() { grep '^::stop-commands::' <<<"$1" | head -n 1 | sed 's/^::stop-commands:://'; }

self_test() {
    local bad=0 rows=0 out rc
    row() {   # row LABEL CONDITION-RESULT(0 = as expected)
        rows=$((rows + 1))
        if [ "$2" -eq 0 ]; then
            printf 'ok   %s\n' "$1"
        else
            printf 'BAD  %s\n' "$1"
            bad=$((bad + 1))
        fi
    }
    t() { "$@" >/dev/null 2>&1 && echo 0 || echo 1; }

    rc=0; out=$(quiet fixture-row -- true) || rc=$?
    row "a passing command keeps rc 0" "$([ "$rc" -eq 0 ] && echo 0 || echo 1)"
    rc=0; out=$(quiet fixture-row -- false) || rc=$?
    row "a failing command keeps rc 1 (a BAD row still turns the step red)" "$([ "$rc" -eq 1 ] && echo 0 || echo 1)"
    rc=0; out=$(quiet fixture-row -- sh -c 'exit 3') || rc=$?
    row "any other status passes through unchanged (3)" "$([ "$rc" -eq 3 ] && echo 0 || echo 1)"

    rc=0; out=$(quiet fixture-row -- sh -c 'echo "::error::section a: failure"; echo "::error::on stderr" >&2; exit 1') || rc=$?
    row "the span opens once, closes once with the same token, open first" "$(t span_ok "$out")"
    row "a fixture ::error:: on stdout is shown prefixed, never bare" \
        "$(grep -qx 'fixture| ::error::section a: failure' <<<"$out" && ! grep -q '^::error::' <<<"$out" && echo 0 || echo 1)"
    row "a fixture ::error:: on stderr is shown prefixed, never bare" \
        "$(grep -qx 'fixture| ::error::on stderr' <<<"$out" && echo 0 || echo 1)"
    row "fixture lines sit inside the span" \
        "$(awk -v t="$(token_of "$out")" '$0 == "::stop-commands::" t {o=1; next} $0 == "::" t "::" {o=0; next} /^fixture[|] / && !o {bad=1} END {exit bad}' <<<"$out" && echo 0 || echo 1)"
    row "a label line names the text as case-table text" \
        "$(grep -q '^fixture-row: the lines marked fixture| are case-table text' <<<"${out%%$'\n'*}" && echo 0 || echo 1)"

    rc=0; out=$(quiet fixture-row -- sh -c 'echo "::stop-commands::guess"; echo "::guess::"; printf "::error::no newline"') || rc=$?
    row "a fixture that opens or closes its own span is only text" "$(t span_ok "$out")"
    row "a last line with no newline is still shown" \
        "$(grep -qx 'fixture| ::error::no newline' <<<"$out" && echo 0 || echo 1)"

    rc=0; out=$( prefix_lines() { cat >/dev/null; return 1; }; quiet fixture-row -- true ) || rc=$?
    row "a passing command whose output was lost is rc 1, not a pass" "$([ "$rc" -eq 1 ] && echo 0 || echo 1)"

    local a b
    a=$(token_of "$(quiet fixture-row -- true)"); b=$(token_of "$(quiet fixture-row -- true)")
    row "every run draws a fresh token" "$([ -n "$a" ] && [ "$a" != "$b" ] && echo 0 || echo 1)"

    rc=0; out=$(quiet fixture-row 2>/dev/null) || rc=$?
    row "no command is a usage error (rc 2), not a pass" "$([ "$rc" -eq 2 ] && echo 0 || echo 1)"
    rc=0; out=$(quiet "" -- true 2>/dev/null) || rc=$?
    row "an empty label is a usage error (rc 2)" "$([ "$rc" -eq 2 ] && echo 0 || echo 1)"

    # #5063: the wiring check. Each row is one workflow, read through a process
    # substitution so the table writes no file. w NAME TEXT -> check_wiring's rc.
    w() { local r=0; check_wiring <(printf '%s\n' "$2") > /dev/null || r=$?; echo "$r"; }
    local good='    steps:
      # #4596: fixture_quiet.sh shows them as labelled text.
      - name: fat_driver case table must still turn RED
        run: setsid --wait bash scripts/ci/fixture_quiet.sh "fat_driver self-test" -- python3 scripts/ci/fat_driver.py self-test
      - name: Sections
        env:
          A: b
        run: >-
          python3 scripts/ci/fat_driver.py run
          --sections a
      - name: wait
        id: wait
        if: always()
        run: python3 scripts/ci/fat_driver.py wait --results r.json'
    row "wiring: the shape ci.yml uses is clean" "$([ "$(w good "$good")" = 0 ] && echo 0 || echo 1)"
    row "wiring: RED on an unwrapped self-test" \
        "$([ "$(w r1 '      - name: t
        run: python3 scripts/ci/fat_driver.py self-test')" = 1 ] && echo 0 || echo 1)"
    row "wiring: RED on a wrapped run (one line)" \
        "$([ "$(w r2 "$good
      - run: bash scripts/ci/fixture_quiet.sh x -- python3 scripts/ci/fat_driver.py run --sections a")" = 1 ] && echo 0 || echo 1)"
    row "wiring: RED on a wrapped wait" \
        "$([ "$(w r3 "$good
      - name: w
        run: bash scripts/ci/fixture_quiet.sh x -- python3 scripts/ci/fat_driver.py wait")" = 1 ] && echo 0 || echo 1)"
    row "wiring: RED on a wrapped run folded over lines" \
        "$([ "$(w r4 "$good
      - name: s
        run: >-
          bash scripts/ci/fixture_quiet.sh x --
          python3 scripts/ci/fat_driver.py run
          --sections a")" = 1 ] && echo 0 || echo 1)"
    row "wiring: RED on a self-test whose only wrapper is a comment line" \
        "$([ "$(w r5 '      - name: t
        run: |
          # fixture_quiet.sh
          python3 scripts/ci/fat_driver.py self-test')" = 1 ] && echo 0 || echo 1)"
    row "wiring: RED on a self-test whose only wrapper is a trailing comment" \
        "$([ "$(w r6 '      - name: t
        run: python3 scripts/ci/fat_driver.py self-test # fixture_quiet.sh')" = 1 ] && echo 0 || echo 1)"
    row "wiring: a wrapped self-test does not cover the bare run in the next step" \
        "$([ "$(w c1 '      - run: bash scripts/ci/fixture_quiet.sh x -- python3 scripts/ci/fat_driver.py self-test
      - run: python3 scripts/ci/fat_driver.py run')" = 0 ] && echo 0 || echo 1)"
    row "wiring: no self-test step anywhere is not clean (rc 2)" \
        "$([ "$(w n1 '      - run: python3 scripts/ci/fat_driver.py run')" = 2 ] && echo 0 || echo 1)"
    rc=0; check_wiring "$REPO_ROOT/.github/workflows/absent-5063.yml" > /dev/null || rc=$?
    row "wiring: an unreadable workflow is not clean (rc 2)" "$([ "$rc" -eq 2 ] && echo 0 || echo 1)"
    rc=0; out=$(check_wiring "$REPO_ROOT"/.github/workflows/*.yml) || rc=$?
    [ "$rc" -eq 0 ] || printf '%s\n' "$out"
    row "wiring: this repo's workflows wrap every self-test and no run or wait" "$([ "$rc" -eq 0 ] && echo 0 || echo 1)"

    printf 'fixture_quiet.sh self-test: %s/%s rows as expected\n' "$((rows - bad))" "$rows"
    (( rows > 0 && bad == 0 ))
}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)

if [ "${1:-}" = "--self-test" ]; then
    self_test
elif [ "${1:-}" = "--check-wiring" ]; then
    shift
    if [ "$#" -eq 0 ]; then set -- "$REPO_ROOT"/.github/workflows/*.yml; fi
    check_wiring "$@"
else
    quiet "$@"
fi
