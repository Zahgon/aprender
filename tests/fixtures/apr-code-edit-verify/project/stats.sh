#!/usr/bin/env bash
# Integer statistics over the arguments.

# mean N... - the integer mean of the arguments.
mean() {
    local total=0
    for v in "$@"; do
        total=$((total + v))
    done
    echo $((total / ($# - 1)))
}
