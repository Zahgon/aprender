#!/usr/bin/env bash
# coverage_floor_test.sh — case table for cov_floor_applied (scripts/lib/coverage_floor.sh, #4671).
#
# The dogfood coverage note must name the floor `make coverage-check` APPLIED. Each row's make
# output is the REAL verdict line of the `coverage` recipe, rendered by `make -n coverage` with the
# floor under test, so a reworded Makefile line turns these rows red instead of leaving the parser
# matching a string make no longer prints. The expected floor is make's own value, read with the
# same override, never a number written here.
#
# Usage: bash scripts/tests/coverage_floor_test.sh   exit 0 all rows pass, 1 a row failed,
#        2 the test could not run.
set -uo pipefail

ROOT=$(git -C "$(dirname "$0")" rev-parse --show-toplevel 2>/dev/null) || { echo "ENV  not inside a git checkout" >&2; exit 2; }
LIB="$ROOT/scripts/lib/coverage_floor.sh"
[ -f "$LIB" ] || { echo "ENV  $LIB is missing" >&2; exit 2; }
# shellcheck source=../lib/coverage_floor.sh
. "$LIB" || { echo "ENV  cannot source $LIB" >&2; exit 2; }

# render <MAKEFLAGS> <regex of one verdict line> <pct> — that verdict line as `make coverage` would print it.
render() {
  local line
  line=$(MAKEFLAGS=$1 make -s -n -C "$ROOT" coverage 2>/dev/null | grep -E "$2" | head -n 1 \
         | sed -e 's/^[[:space:]]*echo "//' -e 's/"; *\\*$//' -e "s/\${COV_PCT}/$3/g")
  [ -n "$line" ] || { echo "ENV  make -n coverage printed no line matching /$2/ (MAKEFLAGS=$1)" >&2; exit 2; }
  printf '%s\n' "$line"
}
# make_floor <MAKEFLAGS> — the floor make itself resolves under that override.
make_floor() {
  MAKEFLAGS=$1 make -s -n -C "$ROOT" coverage 2>/dev/null | grep -oE 'the enforced floor [0-9]+%' \
    | grep -oE '[0-9]+' | head -n 1
}

fails=0 rows=0
row() { # row <name> <make output> <expected floor, or NONE>
  local name=$1 out=$2 want=$3 got rc=0
  got=$(cov_floor_applied "$out") || rc=$?
  [ "$rc" -ne 0 ] && got="NONE${got:+ with output: $got}"
  rows=$((rows + 1))
  if [ -n "$want" ] && [ "$got" = "$want" ]; then
    printf 'ok    %-28s floor %s\n' "$name" "$want"
  else
    printf 'FAIL  %-28s want [%s] got [%s]\n' "$name" "$want" "$got"
    fails=$((fails + 1))
  fi
}

MK=$(make_floor "")
ZERO=$(make_floor "COV_FLOOR=0")
[ -n "$MK" ] && [ "$ZERO" = 0 ] || { echo "ENV  make did not resolve a floor (Makefile [$MK], MAKEFLAGS COV_FLOOR=0 [$ZERO])" >&2; exit 2; }
# The bug this table exists for: the note's old default (88) differed from make's floor. If the
# Makefile ever sets 88 again, row 1 could not tell the two apart, so say so instead of passing.
[ "$MK" != 88 ] || echo "NOTE  the Makefile floor is 88, the old shell default: row makefile-holds cannot catch #4671 itself"

TOTAL='TOTAL: 800000/900000 lines covered (91%)'
row makefile-holds       "$TOTAL"$'\n'"$(render "" 'holds the floor' 91)"                  "$MK"
row makefile-regression  "$TOTAL"$'\n'"$(render "" 'fell below the enforced floor' 80)"                       "$MK"
row makefile-threshold   "$TOTAL"$'\n'"$(render "" 'meets threshold' 97)"                  "$MK"
row makeflags-floor-0    "$TOTAL"$'\n'"$(render "COV_FLOOR=0" 'holds the floor' 91)"       "$ZERO"
row no-figure            $'Running coverage (95%+ threshold)...\n❌ coverage DID NOT MEASURE: no lcov.info was written. No coverage verdict.' NONE
# The banner names the THRESHOLD, never the floor; a parser that read it would report 95.
row banner-only          'Running coverage (95%+ threshold)...'                             NONE

echo "rows=$rows fails=$fails"
[ "$rows" -gt 0 ] || { echo "ENV  no rows ran" >&2; exit 2; }
[ "$fails" -eq 0 ]
