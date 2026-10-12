#!/usr/bin/env bash
# release_receipt.sh — one receipt per candidate commit C: the two long release steps, measured the night
# before, so the release window only checks a receipt and computes nothing (#5185).
#
# WHY. 0.71.0's window spent 58.5 min on the CRUX models smoke and up to 14.5 min on the publish dry run,
# on the cut sha, for the first time. Each red there cost a fix -> merge -> rerun loop. The 45-minute
# window (from 0.71.2) has no room for either, so both run nightly on C and land here, keyed by C.
#
#   --seal --commit C --models STATE REASON --publish STATE REASON --out DIR [--run-url URL]
#        write DIR/release-receipt-<C>.json (schema release-candidate-receipt/v1). STATE is
#        green | red | not_measured, as the lanes write it (models_nightly.sh --relay --measure crux;
#        release_lanes.sh measure publish-dryrun). verdict is GO only when BOTH are green.
#   --check --commit C FILE
#        the window's check: exit 0 and "RECEIPT GO at <C9>" only for a well-formed receipt for
#        exactly C with verdict GO; else exit 1 and "RECEIPT STOP at <C9>: <why>". A missing,
#        unreadable or other-commit receipt is STOP, never a pass, and never a re-run inside the window.
#   --self-test   the case table (no network, no cargo, no GPU)
#
# EXIT  0 sealed / GO · 1 STOP · 2 usage or a value this file refuses to write
set -uo pipefail
PROG=release_receipt
SCHEMA=release-candidate-receipt/v1

say() { printf '%s\n' "$*"; }
usage() { say "$PROG: $*" >&2; exit 2; }
is_sha() { [[ $1 =~ ^[0-9a-f]{40}$ ]]; }
is_state() { case $1 in green | red | not_measured) return 0 ;; esac; return 1; }

seal() { # seal C MSTATE MREASON PSTATE PREASON OUT URL
    local c=$1 ms=$2 mr=$3 ps=$4 pr=$5 out=$6 url=$7 verdict=NO-GO f
    is_sha "$c" || usage "--commit must be a 40-hex sha, got '$c'"
    is_state "$ms" || usage "models state must be green|red|not_measured, got '$ms'"
    is_state "$ps" || usage "publish state must be green|red|not_measured, got '$ps'"
    [ "$ms" = green ] && [ "$ps" = green ] && verdict=GO
    mkdir -p -- "$out" || usage "cannot create $out"
    f="$out/release-receipt-$c.json"
    jq -n --arg s "$SCHEMA" --arg c "$c" --arg at "$(date -u +%FT%TZ)" --arg url "$url" \
        --arg ms "$ms" --arg mr "$mr" --arg ps "$ps" --arg pr "$pr" --arg v "$verdict" \
        '{schema: $s, commit: $c, made_at: $at, run_url: $url,
          steps: {models: {state: $ms, reason: $mr}, publish_dryrun: {state: $ps, reason: $pr}},
          verdict: $v}' > "$f" || usage "cannot write $f"
    say "RECEIPT SEALED $verdict at ${c:0:9}: models $ms, publish-dryrun $ps ($f)"
}

check() { # check C FILE
    local c=$1 f=$2 why
    is_sha "$c" || usage "--commit must be a 40-hex sha, got '$c'"
    why=$(jq -r --arg s "$SCHEMA" --arg c "$c" '
        if .schema != $s then "schema is \(.schema // "absent"), not \($s)"
        elif .commit != $c then "the receipt is for \(.commit // "no commit"), not the cut sha"
        elif (.steps.models.state | type) != "string" or (.steps.publish_dryrun.state | type) != "string"
            then "a step state is missing"
        elif .verdict == "GO" and .steps.models.state == "green" and .steps.publish_dryrun.state == "green"
            then "GO"
        elif .verdict == "GO" then "verdict GO but a step is not green"
        else "verdict \(.verdict // "absent"): models \(.steps.models.state) (\(.steps.models.reason)), publish-dryrun \(.steps.publish_dryrun.state) (\(.steps.publish_dryrun.reason))"
        end' -- "$f" 2> /dev/null) || why="no readable receipt at $f"
    if [ "$why" = GO ]; then say "RECEIPT GO at ${c:0:9}"; return 0; fi
    say "RECEIPT STOP at ${c:0:9}: $why"; return 1
}

self_test() {
    local t fail=0 n=0 A B out rc
    t=$(mktemp -d) || exit 2
    A=$(printf 'a%.0s' {1..40}); B=$(printf 'b%.0s' {1..40})
    # case NAME WANT-RC CMD... : run CMD, compare its exit code
    case_() {
        local name=$1 want=$2; shift 2; n=$((n + 1))
        out=$("$@" 2>&1); rc=$?
        if [ "$rc" -ne "$want" ]; then fail=$((fail + 1)); say "  FAIL $name: rc $rc, want $want: $out"; fi
    }
    case_ seal-go 0 seal "$A" green ok green ok "$t/go" ""
    case_ check-go 0 check "$A" "$t/go/release-receipt-$A.json"
    case_ check-other-commit 1 check "$B" "$t/go/release-receipt-$A.json"
    case_ seal-models-red 0 seal "$A" red "lambda red" green ok "$t/mr" ""
    case_ check-models-red 1 check "$A" "$t/mr/release-receipt-$A.json"
    case_ seal-publish-nm 0 seal "$A" green ok not_measured "version published" "$t/pn" ""
    case_ check-publish-nm 1 check "$A" "$t/pn/release-receipt-$A.json"
    case_ seal-both-nm 0 seal "$A" not_measured x not_measured y "$t/nn" ""
    case_ check-both-nm 1 check "$A" "$t/nn/release-receipt-$A.json"
    case_ seal-bad-state 2 seal "$A" pass ok green ok "$t/bad" ""
    case_ seal-short-sha 2 seal "${A:0:9}" green ok green ok "$t/bad" ""
    case_ check-short-sha 2 check "${A:0:9}" "$t/go/release-receipt-$A.json"
    case_ check-missing 1 check "$A" "$t/none.json"
    printf 'not json' > "$t/junk.json"
    case_ check-junk 1 check "$A" "$t/junk.json"
    jq '.schema = "other/v1"' "$t/go/release-receipt-$A.json" > "$t/schema.json"
    case_ check-schema 1 check "$A" "$t/schema.json"
    # a hand-edited GO over a red step is STOP: the check recomputes, it does not trust the verdict word
    jq '.verdict = "GO"' "$t/mr/release-receipt-$A.json" > "$t/forged.json"
    case_ check-forged-go 1 check "$A" "$t/forged.json"
    # the sealed verdict word itself: GO only when both steps are green
    verdict_is() { [ "$(jq -r .verdict -- "$1")" = "$2" ]; }
    case_ verdict-go 0 verdict_is "$t/go/release-receipt-$A.json" GO
    case_ verdict-models-red 0 verdict_is "$t/mr/release-receipt-$A.json" NO-GO
    case_ verdict-publish-nm 0 verdict_is "$t/pn/release-receipt-$A.json" NO-GO
    case_ verdict-both-nm 0 verdict_is "$t/nn/release-receipt-$A.json" NO-GO
    jq 'del(.steps.publish_dryrun)' "$t/go/release-receipt-$A.json" > "$t/nostep.json"
    case_ check-no-step 1 check "$A" "$t/nostep.json"
    rm -rf -- "${t:?}"
    if [ "$fail" -ne 0 ]; then say "$PROG self-test: RED, $fail of $n cases failed"; exit 1; fi
    say "$PROG self-test: GREEN, $n of $n cases"
}

main() {
    local mode=${1:-} c= ms= mr= ps= pr= out= url= f=
    [ -n "$mode" ] || usage "usage: --seal | --check | --self-test"
    shift
    command -v jq > /dev/null || usage "jq is required"
    while [ $# -gt 0 ]; do
        case $1 in
            --commit) [ $# -ge 2 ] || usage "--commit needs a sha"; c=$2; shift 2 ;;
            --models) [ $# -ge 3 ] || usage "--models needs STATE REASON"; ms=$2; mr=$3; shift 3 ;;
            --publish) [ $# -ge 3 ] || usage "--publish needs STATE REASON"; ps=$2; pr=$3; shift 3 ;;
            --out) [ $# -ge 2 ] || usage "--out needs a DIR"; out=$2; shift 2 ;;
            --run-url) [ $# -ge 2 ] || usage "--run-url needs a URL"; url=$2; shift 2 ;;
            -*) usage "unknown flag $1" ;;
            *) f=$1; shift ;;
        esac
    done
    case $mode in
        --seal) [ -n "$out" ] || usage "--seal needs --out DIR"; seal "$c" "$ms" "$mr" "$ps" "$pr" "$out" "$url" ;;
        --check) [ -n "$f" ] || usage "--check needs a receipt FILE"; check "$c" "$f" ;;
        --self-test) self_test ;;
        *) usage "unknown mode $mode" ;;
    esac
}

main "$@"
