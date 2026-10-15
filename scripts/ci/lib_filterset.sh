#!/usr/bin/env bash
# lib_filterset.sh ROLE MODE -- the nextest filterset (-E) a workspace-test-shard lib
# role runs or lists. ROLE is libx (X64: the arch-divergent crates) or liba (ARM64:
# every other crate); the crate list is ci/arch-divergent-lib-crates.txt.
#   MODE run   libx -> the divergent crates       liba -> everything else
#   MODE list  libx -> all()  (the whole X64 universe, listed once by libx part 1)
#              liba -> everything else  (the ARM64 side of the per-arch equality check)
# Anything else, or an empty crate list, is a refusal (rc 2) with nothing on stdout.
set -euo pipefail

list="${LIB_FILTERSET_LIST:-ci/arch-divergent-lib-crates.txt}"
role="${1:-}"
mode="${2:-}"
[ -f "$list" ] || { printf 'REFUSE %s: no such file\n' "$list" >&2; exit 2; }
div=$(grep -E '^[a-z][a-z0-9_-]*$' "$list" | sed 's/.*/package(=&)/' | paste -sd'|' -)
[ -n "$div" ] || { printf 'REFUSE %s names no crate -- an empty split is not a split\n' "$list" >&2; exit 2; }
case "$role:$mode" in
    libx:run) printf '%s\n' "$div" ;;
    libx:list) printf 'all()\n' ;;
    liba:run|liba:list) printf 'not (%s)\n' "$div" ;;
    *) printf 'usage: %s libx|liba run|list\n' "$0" >&2; exit 2 ;;
esac
