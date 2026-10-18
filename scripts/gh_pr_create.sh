#!/usr/bin/env bash
# gh_pr_create.sh -- `gh pr create` behind the open-PR cap check (#5518). Drafts included:
# the check runs before gh sees any flag.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)" || exit 2
bash "$HERE/pr_open_cap.sh" || exit $?
exec gh pr create "$@"
