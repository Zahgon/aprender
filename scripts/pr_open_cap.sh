#!/usr/bin/env bash
# pr_open_cap.sh -- refuse to open a PR when the repo is at its open-PR cap (#5518).
# Open PRs count with drafts and stale PRs. The cap is read from ONE place,
# contracts/pr-open-cap-v1.yaml; one GraphQL call reads the count.
#   pr_open_cap.sh [--repo OWNER/NAME]
# exit 0 allowed / skipped (external repo) / not_measured (read failed, nothing refused, L23)
# exit 3 refused: the repo already has `cap` open PRs, so one more would exceed it
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)" || exit 2
CONTRACT="${PR_OPEN_CAP_CONTRACT:-$ROOT/contracts/pr-open-cap-v1.yaml}"
REPO=paiml/aprender
[ "${1:-}" = "--repo" ] && REPO="${2:-}"
case "$REPO" in
    paiml/*) ;;
    *) echo "skipped: $REPO is an external repo, no cap"; exit 0 ;;
esac
cap="$(sed -n 's/^pr_open_cap: *\([0-9][0-9]*\).*$/\1/p' "$CONTRACT" 2>/dev/null | head -n 1)"
[ -n "$cap" ] || { echo "not_measured: no pr_open_cap in $CONTRACT"; exit 0; }

if [ -n "${PR_OPEN_CAP_FIXTURE:-}" ]; then
    json="$(cat -- "$PR_OPEN_CAP_FIXTURE" 2>/dev/null)"
else
    q='query($o:String!,$n:String!){repository(owner:$o,name:$n){pullRequests(states:OPEN,first:1,orderBy:{field:CREATED_AT,direction:ASC}){totalCount nodes{number title}}}}'
    json="$(gh api graphql -f query="$q" -f o="${REPO%%/*}" -f n="${REPO##*/}" \
        --jq '.data.repository.pullRequests|{total:.totalCount,oldest:.nodes[0]}' 2>/dev/null)"
fi
total="$(printf '%s' "$json" | jq -er '.total' 2>/dev/null)" \
    || { echo "not_measured: could not read the open-PR count for $REPO; nothing refused"; exit 0; }
oldest="$(printf '%s' "$json" | jq -r '.oldest|"#\(.number) \(.title)"' 2>/dev/null)"
if [ "$total" -ge "$cap" ]; then
    echo "refused: $total open PRs in $REPO (cap $cap, drafts included); oldest open: ${oldest:-unknown}. Land or close one first."
    exit 3
fi
echo "allowed: $total open PRs in $REPO (cap $cap)"
