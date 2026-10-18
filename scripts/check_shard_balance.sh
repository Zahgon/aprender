#!/usr/bin/env bash
# check_shard_balance.sh — the once-only steps of workspace-test-shard sit where the
# three shards finish together, and every Σ read has its input on its own shard (#5486).
#
# The merge queue waits on the SLOWEST shard. On 2026-10-16 shard 1 averaged 23.7 min
# against 11.4 / 11.6 for shards 2 and 3, because every `matrix.shard == 1` step landed
# there. This guard sums ci/shard-step-costs.tsv (measured per-step seconds) per shard for
# the full tier and fails on a predicted spread over 120 s, on an unmeasured once-only
# step, on a stale cost row, on a Σ check that reads a junit or step log its shard never
# produced, and on a fan-in that no longer names the same three shards as the matrix.
# Rules B1-B3, S1, F1: scripts/ci/shard_balance.py.
#
# Text-only: reads ci/sections.yml, .github/workflows/ci.yml and the cost table.
#
#   bash scripts/check_shard_balance.sh
#   bash scripts/check_shard_balance.sh --self-test
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
case "${1:-}" in
    --help|-h) sed -n '2,16p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' ;;
    --self-test) python3 scripts/ci/shard_balance.py self-test ;;
    "") python3 scripts/ci/shard_balance.py check ;;
    *) echo "check_shard_balance: unknown argument $1" >&2; exit 2 ;;
esac
