#!/usr/bin/env bash
# check_cycles_schema.sh -- cop cycle rows (cycles.jsonl) must be the canonical
# v1 shape of COPS.md (2), the shape the KPI report's loops.sh enforces (#5736,
# operator ruling 5). A malformed row (a bare string in fixes[], a legacy key)
# is otherwise accepted by the writer and breaks or silently skips downstream.
#
#   check_cycles_schema.sh FILE          one JSON row per line; exit 1 on a bad row
#   check_cycles_schema.sh --self-test   plant each falsifier; must PASS/FAIL as named
#
# Rules (the validation stage of loops.sh, same reason text, one line per defect
# as "FILE:LINENO: reason"; blank lines are skipped but still counted):
#   t              string, ISO UTC, seconds optional, fractional seconds ok
#   legacy keys    ts, signals, armed_stuck_60m, findings are not v1
#   repo_signals   absent/null, or an object of objects (no armed_stuck_60m inside)
#   fixes          absent/null, or an array of objects (extra keys, e.g. id, allowed)
#   merged_1h/24h  absent/null, or an object
#   reverts        absent/null, or an array of objects
set -euo pipefail

JQ_RULES='
  def tp: if type != "string" then null
          else (if test("T[0-9]{2}:[0-9]{2}Z$") then sub("Z$"; ":00Z") else . end)
               | sub("\\.[0-9]+Z$"; "Z") | (try fromdateiso8601 catch null) end;
  select(length > 0) | input_line_number as $n | . as $line
  | (try {v: ($line | fromjson)} catch {bad: 1}) as $p
  | (if $p.bad then ["not JSON"]
     elif ($p.v | type) != "object" then ["not a JSON object (\($p.v | type))"]
     else $p.v as $o | [
       (if ($o.t | type) != "string" then "missing string t"
        elif ($o.t | tp) == null then "t is not ISO UTC: \($o.t)" else empty end),
       (["ts", "signals", "armed_stuck_60m", "findings"][] as $k
        | select($o | has($k)) | "legacy key \($k) is not v1"),
       (if ($o.repo_signals // null) == null then empty
        elif ($o.repo_signals | type) != "object" then "repo_signals is not an object (\($o.repo_signals | type))"
        else
          ([$o.repo_signals | to_entries[] | select((.value | type) != "object") | .key]
           | if length > 0 then "repo_signals is not an object of objects (non-object values: \(join(",")))" else empty end),
          ([$o.repo_signals | to_entries[] | select((.value | type) == "object" and (.value | has("armed_stuck_60m"))) | .key]
           | if length > 0 then "legacy key armed_stuck_60m is not v1 (in repo_signals: \(join(",")))" else empty end)
        end),
       (if ($o.fixes // null) == null then empty
        elif ($o.fixes | type) != "array" then "fixes is not an array"
        else $o.fixes | to_entries[] | select((.value | type) != "object") | "fixes[\(.key)] is not an object (\(.value | type))" end),
       (["merged_1h", "merged_24h"][] as $k
        | select(($o[$k] // null) != null and ($o[$k] | type) != "object") | "\($k) is not an object (\($o[$k] | type))"),
       (if ($o.reverts // null) == null then empty
        elif ($o.reverts | type) != "array" then "reverts is not an array"
        else $o.reverts | to_entries[] | select((.value | type) != "object") | "reverts[\(.key)] is not an object (\(.value | type))" end)
     ] end)[] | "\($n): \(.)"'

check_file() {
    local out
    out=$(jq -R -r "$JQ_RULES" "$1" | sed "s|^|$1:|")
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
    local sig='"repo_signals":{"aprender":{"idle_workers":0}}'
    expect() { # name want(PASS|FAIL) json [needle]
        local f="$d/$1.jsonl" got=PASS err=""
        printf '%s\n' "$3" >"$f"
        err=$(check_file "$f" 2>&1) || got=FAIL
        if [ "$got" != "$2" ] || { [ -n "${4:-}" ] && [[ "$err" != *"$4"* ]]; }; then
            echo "SELF-TEST FAIL: $1 wanted $2${4:+ naming $4}, got $got: $err" >&2
            rc=1
        fi
    }
    expect v1-full PASS '{"t":"2026-10-19T11:30:15.5Z",'"$sig"',"merged_1h":{"aprender":1},"merged_24h":{"aprender":9},"reverts":[{"name":"c1"}],"fixes":[{"finding_t":"2026-10-19T07:00Z","live_t":null,"what":"w"}]}'
    expect minute-time PASS '{"t":"2026-10-19T11:00Z"}'
    expect fixes-object-with-id PASS '{"t":"2026-10-19T11:00Z","fixes":[{"id":"x"}]}'
    expect empty-and-null PASS '{"t":"2026-10-19T11:00Z","fixes":[],"reverts":null}'
    expect fixes-string FAIL '{"t":"2026-10-19T11:00Z","fixes":["abc"]}' 'fixes[0] is not an object (string)'
    expect fixes-not-array FAIL '{"t":"2026-10-19T11:00Z","fixes":"abc"}' 'fixes is not an array'
    expect reverts-string FAIL '{"t":"2026-10-19T11:00Z","reverts":["c1"]}' 'reverts[0] is not an object'
    expect merged-flat FAIL '{"t":"2026-10-19T11:00Z","merged_1h":3}' 'merged_1h is not an object (number)'
    expect flat-signals FAIL '{"t":"2026-10-19T11:00Z","repo_signals":{"idle_workers":0}}' 'non-object values: idle_workers'
    expect legacy-ts FAIL '{"ts":"2026-10-19T11:00Z"}' 'legacy key ts is not v1'
    expect legacy-signals FAIL '{"t":"2026-10-19T11:00Z","signals":{}}' 'legacy key signals is not v1'
    expect legacy-armed FAIL '{"t":"2026-10-19T11:00Z","repo_signals":{"a":{"armed_stuck_60m":1}}}' 'armed_stuck_60m is not v1 (in repo_signals: a)'
    expect bad-time FAIL '{"t":"yesterday"}' 't is not ISO UTC'
    expect not-json FAIL 'garbage' ':1: not JSON'
    printf '%s\n\n%s\n' '{"t":"2026-10-19T11:00Z"}' '{"t":"2026-10-19T11:15Z","fixes":["bad"]}' >"$d/lineno.jsonl"
    local out
    out=$(check_file "$d/lineno.jsonl" 2>&1) || true
    [[ "$out" == *"lineno.jsonl:3: fixes[0]"* ]] || {
        echo "SELF-TEST FAIL: blank lines must still count toward the row number, got: $out" >&2
        rc=1
    }
    [ "$rc" -eq 0 ] && echo "check_cycles_schema self-test: 15 cases ok"
    return "$rc"
}

case "${1:-}" in
    --self-test) self_test ;;
    "") echo "usage: $0 FILE | --self-test" >&2; exit 2 ;;
    *) check_file "$1" ;;
esac
