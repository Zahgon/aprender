#!/usr/bin/env bash
# vacuity_2876_mutants.sh -- each guard fixed for #2876 (a gate that answered empty input with PASS)
# must turn its own --self-test red when its zero-input refusal is mutated away.
#
# Each row is: guard, mutant name, sed expression. The expression must change the file
# (else BROKEN), and the mutated guard's --self-test must exit non-zero (else SURVIVED).
# Usage: bash scripts/tests/vacuity_2876_mutants.sh     EXIT 0 all killed, 1 otherwise.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
td="$(mktemp -d "${TMPDIR:-/tmp}/vacuity-2876-mut.XXXXXX")"
trap 'rm -rf -- "${td:?}"' EXIT
mkdir -p "$td/scripts"   # one mutant at a time, under its guard's own name
n=0 bad=0
while IFS=$'\t' read -r guard name expr; do
    [ -n "$guard" ] || continue
    n=$((n + 1))
    src="$ROOT/scripts/$guard"
    bash "$src" --self-test > /dev/null 2>&1 || { echo "BROKEN: $guard's unmutated self-test is red"; bad=$((bad + 1)); continue; }
    sed -E "$expr" "$src" > "$td/scripts/$guard"
    if cmp -s "$src" "$td/scripts/$guard"; then echo "BROKEN: $name changed nothing"; bad=$((bad + 1)); continue; fi
    rc=0; bash "$td/scripts/$guard" --self-test > /dev/null 2>&1 || rc=$?
    if [ "$rc" != 0 ]; then echo "killed: $name"; else echo "SURVIVED: $name"; bad=$((bad + 1)); fi
done <<'MUTANTS'
check_build_rs_paths.sh	build_rs_zero_ok	s/^if \[ "\$checked" -eq 0 \]; then/if false; then/
check_exclude_anchored.sh	exclude_zero_ok	s/^if \[ "\$checked" -eq 0 \]; then/if false; then/
check_exclude_anchored.sh	exclude_git_error_is_empty	s/^if ! manifests="\$\(git ls-files '\*Cargo.toml' \| sort\)"; then/if ! manifests="$( (git ls-files '*Cargo.toml' || true) | sort)"; then/
check_build_rs_paths.sh	build_rs_git_error_is_empty	s/^if ! files=\$\(git ls-files (.*)\); then/if ! files=$(git ls-files \1 || true); then/
check_include_files.sh	include_zero_ok	s/^if \[ "\$checked" -eq 0 \]; then/if false; then/
check_book_cli_parity.sh	book_zero_ok	s/^  if \[ "\$total" -eq 0 \]; then/  if false; then/
check_book_cli_parity.sh	book_echo_count	s/^  total=.*/  total=$(echo "$cmds" | wc -l)/
MUTANTS
echo "--- $((n - bad))/$n mutants killed ---"
[ "$bad" = 0 ] && [ "$n" -gt 0 ]
