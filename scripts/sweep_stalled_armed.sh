#!/usr/bin/env bash
# sweep_stalled_armed.sh -- find ARMED pull requests that are stalled, and say what to do.
# #5471: armed PRs sat BEHIND with a red head for 2.5 h outside the merge queue, because
# auto-merge never enqueues a red head and nothing re-ran it.
#
# THE RULE THIS ENCODES (cop ruling, 2026-10-18): a main-merge push voids the reviewers'
# seats (C336 #6), so BEHIND alone is never a reason to push. The merge queue tests the
# PR against main itself. So the decision table is:
#
#   in the merge queue                      -> leave   (it is being tested)
#   DIRTY (conflicts)                       -> merge-main  (the only push this sweep asks for)
#   red head, idle >= IDLE_MIN              -> rerun   (rerun --failed; no push, seats kept)
#   BEHIND, green or pending                -> leave   (BEHIND does not block queue entry)
#   anything else                           -> leave
#
# Unarmed PRs are out of scope. A failed or unreadable read is `unread`, never a decision.
#
# Usage
#   bash scripts/sweep_stalled_armed.sh               print one line per armed PR
#   bash scripts/sweep_stalled_armed.sh --apply       also run `gh run rerun --failed` for `rerun` rows
#   bash scripts/sweep_stalled_armed.sh --self-test   the case table, no network
# Exit: 0 ok, 2 environment (gh missing, read failed), 1 self-test failure.
set -uo pipefail

REPO="${REPO:-paiml/aprender}"
IDLE_MIN="${IDLE_MIN:-30}"

# decide <queued:0|1> <merge_state> <red:0|1> <idle_minutes>
decide() {
    local queued=$1 state=$2 red=$3 idle=$4
    if [ "$queued" = 1 ]; then echo leave; return; fi
    if [ "$state" = DIRTY ]; then echo merge-main; return; fi
    if [ "$red" = 1 ] && [ "$idle" -ge "$IDLE_MIN" ]; then echo rerun; return; fi
    echo leave
}

self_test() {
    local fail=0 n=0 want got
    while IFS=' ' read -r want q s r i; do
        n=$((n + 1))
        got=$(decide "$q" "$s" "$r" "$i")
        if [ "$got" != "$want" ]; then
            echo "FAIL case $n: want $want got $got ($q $s $r $i)"
            fail=1
        fi
    done <<'CASES'
leave 1 BEHIND 1 999
leave 1 DIRTY 1 999
merge-main 0 DIRTY 0 0
merge-main 0 DIRTY 1 999
rerun 0 BEHIND 1 30
rerun 0 BLOCKED 1 31
leave 0 BEHIND 1 29
leave 0 BEHIND 0 999
leave 0 CLEAN 0 999
leave 0 BLOCKED 0 120
CASES
    [ "$fail" = 0 ] && echo "ok $n cases"
    return "$fail"
}

# Mutation guard: a decide() that pushes for BEHIND must fail the table.
if [ "${1:-}" = --self-test ]; then
    self_test || exit 1
    # shellcheck disable=SC2317
    decide() { [ "$2" = BEHIND ] && { echo merge-main; return; }; echo leave; }
    if self_test >/dev/null; then echo "FAIL: the table did not catch a BEHIND-pushing decide()"; exit 1; fi
    echo "ok mutant (BEHIND pushes) caught"
    exit 0
fi

command -v gh >/dev/null 2>&1 || { echo "ENV: gh missing" >&2; exit 2; }
apply=0
[ "${1:-}" = --apply ] && apply=1
owner=${REPO%%/*}
name=${REPO##*/}
now=$(date -u +%s)

rows=$(gh api graphql -f query='query($o:String!,$n:String!){repository(owner:$o,name:$n){pullRequests(states:OPEN,first:100){nodes{number mergeStateStatus updatedAt autoMergeRequest{enabledAt} mergeQueueEntry{position} commits(last:1){nodes{commit{statusCheckRollup{contexts(first:100){nodes{__typename ... on CheckRun{conclusion detailsUrl} ... on StatusContext{state}}}}}}}}}}}' \
    -f o="$owner" -f n="$name" \
    --jq '.data.repository.pullRequests.nodes[]|select(.autoMergeRequest!=null or .mergeQueueEntry!=null)|[.number,(if .mergeQueueEntry!=null then 1 else 0 end),.mergeStateStatus,.updatedAt,([.commits.nodes[0].commit.statusCheckRollup.contexts.nodes[]?|select(.conclusion=="FAILURE" or .state=="FAILURE" or .state=="ERROR")|.detailsUrl // "x"]|first // "")]|@tsv') || { echo "ENV: graphql read failed (not_measured)" >&2; exit 2; }

[ -z "$rows" ] && { echo "no armed PRs"; exit 0; }
while IFS=$'\t' read -r pr queued state updated redurl; do
    upd=$(date -u -d "$updated" +%s 2>/dev/null) || { echo "$pr unread"; continue; }
    idle=$(((now - upd) / 60))
    red=0
    [ -n "$redurl" ] && red=1
    act=$(decide "$queued" "$state" "$red" "$idle")
    echo "$pr $act state=$state red=$red idle=${idle}m"
    if [ "$act" = rerun ] && [ "$apply" = 1 ]; then
        run=$(printf '%s' "$redurl" | sed -nE 's#.*/runs/([0-9]+)/.*#\1#p')
        [ -n "$run" ] && gh run rerun --failed "$run" -R "$REPO" >/dev/null 2>&1 && echo "$pr reran $run"
    fi
done <<<"$rows"
