#!/usr/bin/env bash
# Tests for stats.sh. Exit 0 when every test passes.
set -u
cd "$(dirname "$0")"
. ./stats.sh
printf '%s\n' "$(date -u +%FT%TZ) test_stats.sh" >> "${EDIT_VERIFY_RUN_LOG:-/dev/null}"

fail=0
check() {
    if [ "$2" = "$3" ]; then
        echo "PASS $1"
    else
        echo "FAIL $1: expected $3, got $2"
        fail=1
    fi
}

check test_mean_three "$(mean 2 4 6)" 4
check test_mean_one "$(mean 7)" 7
exit "$fail"
