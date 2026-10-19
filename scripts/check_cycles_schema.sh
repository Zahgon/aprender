#!/usr/bin/env bash
# check_cycles_schema.sh -- cop cycle rows (cycles.jsonl) must carry fixes[] as
# objects with an id (#5736, operator ruling 5). A bare string in fixes[] is
# accepted by the writer and then breaks or silently skips the row downstream.
#
#   check_cycles_schema.sh FILE          one JSON row per line; exit 1 on a bad row
#   check_cycles_schema.sh --self-test   plant each falsifier; must PASS/FAIL as named
#
# Absent or empty fixes is valid. A violation prints the row number and the
# offending element.
set -euo pipefail

# One line per violation: "row N: <reason>: <element>".
violations() {
    jq -R -r --slurp '
      split("\n") | map(select(length > 0)) | to_entries[] |
      (.key + 1) as $n | .value as $raw |
      ($raw | try fromjson catch null) as $row |
      if $row == null or ($row | type) != "object" then
        "row \($n): not a JSON object: \($raw)"
      elif ($row | has("fixes")) | not then empty
      elif ($row.fixes | type) != "array" then
        "row \($n): fixes is not an array: \($row.fixes | tojson)"
      else
        $row.fixes | to_entries[] |
        if (.value | type) != "object" then
          "row \($n): fixes[\(.key)] is not an object: \(.value | tojson)"
        elif (.value | has("id")) | not then
          "row \($n): fixes[\(.key)] has no id: \(.value | tojson)"
        else empty end
      end' "$1"
}

check_file() {
    local out
    out=$(violations "$1")
    if [ -n "$out" ]; then
        printf '%s\n' "$out" >&2
        return 1
    fi
}

self_test() {
    local d rc=0
    d=$(mktemp -d)
    # shellcheck disable=SC2064
    trap "rm -rf -- '$d'" RETURN
    expect() { # name want(PASS|FAIL) json [needle]
        local f="$d/$1.jsonl" got=PASS err=""
        printf '%s\n' "$3" >"$f"
        err=$(check_file "$f" 2>&1) || got=FAIL
        if [ "$got" != "$2" ] || { [ -n "${4:-}" ] && [[ "$err" != *"$4"* ]]; }; then
            echo "SELF-TEST FAIL: $1 wanted $2${4:+ naming $4}, got $got: $err" >&2
            rc=1
        fi
    }
    expect string-element FAIL '{"fixes":["abc"]}' '"abc"'
    expect object-with-id PASS '{"fixes":[{"id":"x"}]}'
    expect empty-fixes PASS '{"fixes":[]}'
    expect no-fixes PASS '{"t":"x"}'
    expect object-without-id FAIL '{"fixes":[{"note":"no id"}]}' 'no id'
    expect fixes-not-array FAIL '{"fixes":"abc"}' 'not an array'
    expect not-json FAIL 'garbage' 'row 1'
    printf '%s\n%s\n' '{"fixes":[]}' '{"fixes":["bad"]}' >"$d/row2.jsonl"
    local out2
    out2=$(check_file "$d/row2.jsonl" 2>&1) || true
    [[ "$out2" == *"row 2"* ]] || {
        echo "SELF-TEST FAIL: row number not reported" >&2
        rc=1
    }
    [ "$rc" -eq 0 ] && echo "check_cycles_schema self-test: 8 cases ok"
    return "$rc"
}

case "${1:-}" in
    --self-test) self_test ;;
    "") echo "usage: $0 FILE | --self-test" >&2; exit 2 ;;
    *) check_file "$1" ;;
esac
