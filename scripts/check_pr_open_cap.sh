#!/usr/bin/env bash
# check_pr_open_cap.sh -- the case table for scripts/pr_open_cap.sh, the open-PR cap
# check behind scripts/gh_pr_create.sh (#5518). The cap was advisory; this makes the
# entry point refuse. Rows: cap-1 passes, the cap refuses (the planted 11th), a draft
# counts, an external repo is skipped, a failed or malformed count read is
# not_measured (L23: refuse nothing, say so), the refusal names the count and the oldest.
#
# Counts come from a fixture file (PR_OPEN_CAP_FIXTURE), so no network and no gh.
#
#   check_pr_open_cap.sh              the case table
#   check_pr_open_cap.sh --self-test  the mutants: wrong subjects, each must turn a row RED
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)" || exit 2
SUBJECT="${PR_OPEN_CAP_SUBJECT:-$ROOT/scripts/pr_open_cap.sh}"
CONTRACT="$ROOT/contracts/pr-open-cap-v1.yaml"
CAP="$(sed -n 's/^pr_open_cap: *\([0-9][0-9]*\).*$/\1/p' "$CONTRACT" 2>/dev/null | head -n 1)"
[ -n "$CAP" ] || { echo "no pr_open_cap in $CONTRACT" >&2; exit 2; }

TMP="$(mktemp -d)" || exit 2
trap 'rm -rf -- "${TMP:?}"' EXIT

# fixture N DRAFTS -- N open PRs (DRAFTS of them drafts), oldest is #100
fixture() {
    printf '{"total":%s,"drafts":%s,"oldest":{"number":100,"title":"oldest open"}}\n' "$1" "$2" > "$TMP/f.json"
}
run() { PR_OPEN_CAP_FIXTURE="$TMP/f.json" PR_OPEN_CAP_CONTRACT="$CONTRACT" bash "$SUBJECT" "$@"; }

fails=0
row() { # row WANT_RC DESCRIPTION GREP_PATTERN CMD...
    local want=$1 desc=$2 pat=$3; shift 3
    local out rc
    out="$("$@" 2>&1)"; rc=$?
    if [ "$rc" = "$want" ] && printf '%s' "$out" | grep -Eq -- "$pat"; then
        echo "ok    $desc"
    else
        echo "RED   $desc (rc=$rc want=$want)"; printf '%s\n' "$out" | sed 's/^/        /'; fails=$((fails + 1))
    fi
}

table() {
    echo "=== pr_open_cap.sh refuses the open at the cap (check_pr_open_cap.sh) ==="
    fixture $((CAP - 1)) 0
    row 0 "cap-1 open PRs: the cap-th PR is allowed" '^allowed' run --repo paiml/aprender
    fixture "$CAP" 0
    row 3 "cap open PRs: the next open is refused, naming count and oldest" "refused: $CAP open.*#100" run --repo paiml/aprender
    fixture $((CAP + 1)) 0
    row 3 "over the cap stays refused" '^refused' run --repo paiml/aprender
    fixture "$CAP" "$CAP"
    row 3 "drafts count toward the cap" '^refused' run --repo paiml/aprender
    fixture $((CAP + 5)) 0
    row 0 "an external repo is excluded" '^skipped' run --repo modelcontextprotocol/rust-sdk
    printf 'not json\n' > "$TMP/f.json"
    row 0 "a malformed count read is not_measured, nothing refused" '^not_measured' run --repo paiml/aprender
    rm -f "${TMP:?}/f.json"
    row 0 "a missing count read is not_measured, nothing refused" '^not_measured' run --repo paiml/aprender
}

if [ "${1:-}" = "--self-test" ]; then
    # each mutant is a wrong subject; the table must go RED on it
    mk() { printf '%s\n' "$2" > "$TMP/$1.sh"; }
    mk off_by_one '[ "$(jq .total "$PR_OPEN_CAP_FIXTURE")" -gt 10 ] && { echo "refused: 11 open #100"; exit 3; }; echo allowed; exit 0'
    mk never_refuses 'echo allowed; exit 0'
    mk refuses_unmeasured 'echo "refused: 0 open #0"; exit 3'
    mk drafts_free 'n=$(jq ".total-.drafts" "$PR_OPEN_CAP_FIXTURE" 2>/dev/null) || { echo not_measured; exit 0; }; [ "$n" -ge 10 ] && { echo "refused: $n open #100"; exit 3; }; echo allowed'
    bad=0
    for m in off_by_one never_refuses refuses_unmeasured drafts_free; do
        if PR_OPEN_CAP_SUBJECT="$TMP/$m.sh" bash "$0" >/dev/null 2>&1; then echo "MUTANT SURVIVED: $m"; bad=1; else echo "killed: $m"; fi
    done
    exit "$bad"
fi
table
[ "$fails" = 0 ] && { echo "all rows ok"; exit 0; }
echo "$fails row(s) RED"; exit 1
