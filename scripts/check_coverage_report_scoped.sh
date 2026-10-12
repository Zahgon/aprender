#!/usr/bin/env bash
# scripts/check_coverage_report_scoped.sh -- every `llvm-cov report` must be SCOPED (#4023).
#
# This repo's root Cargo.toml is also a package (the `apr` facade). An unscoped `llvm-cov report`
# covers only that facade and writes an EMPTY lcov: it blanked coverage once before (the
# Makefile's single-phase note) and again in coverage-nightly run 35892421393. The older
# cargo-llvm-cov rejected `report --workspace` and 0.9.0 rejects `report --exclude`, so the
# accepted scope is the derived list from `aprender-ci-tools coverage-report-scope` (the Rust
# port, #4787) or the bash port scripts/coverage_report_scope.sh (#4715), or an explicit
# `-p`/`--package`. This refuses any `llvm-cov report` invocation line that has none. A call to
# the deleted scripts/coverage_report_scope.py (#4787, C301) expands to NOTHING, so it is flagged.
#
# Usage:  check_coverage_report_scoped.sh [file ...]   default: Makefile scripts/*.sh .github/workflows/*.yml
#         check_coverage_report_scoped.sh --self-test   the case table below
# Executed, never sourced.
set -euo pipefail

# A line invoking `llvm-cov report` (not a comment) is scoped if it carries one of these.
SCOPED_RE='coverage_report_scope\.sh([^[:alnum:]_]|$)|aprender-ci-tools([[:space:]]+--)?[[:space:]]+coverage-report-scope([^[:alnum:]_-]|$)|(^|[[:space:]])(-p|--package)[[:space:]=]'
INVOKE_RE='llvm-cov[[:space:]]+report([[:space:]]|$)'
COMMENT_RE='^[[:space:]]*@?#'

scan() {
  local bad=0 scanned=0 f n line
  for f in "$@"; do
    [ -f "$f" ] || continue
    scanned=$((scanned + 1))
    n=0
    while IFS= read -r line || [ -n "$line" ]; do
      n=$((n + 1))
      [[ "$line" =~ $COMMENT_RE ]] && continue
      [[ "$line" =~ $INVOKE_RE ]] || continue
      if ! [[ "$line" =~ $SCOPED_RE ]]; then
        echo "UNSCOPED llvm-cov report: $f:$n: ${line#"${line%%[![:space:]]*}"}"
        bad=$((bad + 1))
      fi
    done < "$f"
  done
  # Anti-vacuity: a scan that read nothing proves nothing (run from the wrong
  # directory, or every path missing). It must fail, not pass silently.
  if [ "$scanned" -eq 0 ]; then
    echo "check_coverage_report_scoped: scanned 0 files (none of: $*) -- refusing to pass vacuously"
    return 2
  fi
  if [ "$bad" -gt 0 ]; then
    echo "check_coverage_report_scoped: $bad unscoped \`llvm-cov report\` invocation(s). Scope each with"
    echo "  \$(cargo run -q --locked -p aprender-ci-tools -- coverage-report-scope [--exclude NAME]),"
    echo "  \$(scripts/coverage_report_scope.sh [--exclude NAME]) or -p/--package: an unscoped"
    echo "  report covers only the root facade and writes an empty lcov (#4023)."
    return 1
  fi
  return 0
}

self_test() {
  local dir rc fails=0
  dir=$(mktemp -d)
  trap 'rm -rf "${dir:?}"' RETURN
  # must_flag: the two shapes that blanked coverage, lookalikes of both helper names, and the
  # deleted .py (its `$(...)` expands to nothing)
  printf '\t@cargo-llvm-cov llvm-cov report --lcov --output-path x.info\n' > "$dir/unscoped.mk"
  printf '\t@cargo-llvm-cov llvm-cov report --workspace --lcov --output-path x.info\n' > "$dir/workspace.mk"
  printf '\t@cargo-llvm-cov llvm-cov report $$(scripts/coverage_report_scope.shx) --json\n' > "$dir/lookalike.mk"
  printf '\t@cargo-llvm-cov llvm-cov report $$(aprender-ci-tools coverage-report-scopes) --json\n' > "$dir/lookalike_bin.mk"
  printf '\t@cargo-llvm-cov llvm-cov report $$(python3 scripts/coverage_report_scope.py) --lcov\n' > "$dir/stale_py.mk"
  # must_pass: derived scope (bin via cargo run, installed bin, bash port), explicit -p,
  # --package=, a comment, and a non-report llvm-cov call
  printf '\t@cargo-llvm-cov llvm-cov report $$(cargo run -q --locked -p aprender-ci-tools -- coverage-report-scope --exclude g) --lcov\n' > "$dir/derived_bin.mk"
  printf 'R=$(cargo llvm-cov report $(aprender-ci-tools coverage-report-scope) --summary-only)\n' > "$dir/derived_bin_installed.mk"
  printf '\t@cargo-llvm-cov llvm-cov report $$(scripts/coverage_report_scope.sh --exclude g) --json\n' > "$dir/derived_sh.mk"
  printf '\t@cargo-llvm-cov llvm-cov report -p a -p b --summary-only\n' > "$dir/explicit.mk"
  printf '\t@cargo-llvm-cov llvm-cov report --package=a --lcov\n' > "$dir/package.mk"
  printf '\t@# an unscoped `llvm-cov report` is the defect\n' > "$dir/comment.mk"
  printf '\t@cargo-llvm-cov llvm-cov test --no-report --workspace --lib\n' > "$dir/test.mk"
  for c in unscoped workspace lookalike lookalike_bin stale_py; do
    rc=0; scan "$dir/$c.mk" > /dev/null || rc=$?
    [ "$rc" -eq 1 ] || { echo "SELF-TEST FAIL: $c must be flagged (rc=$rc)"; fails=$((fails + 1)); }
  done
  rc=0; scan "$dir/does-not-exist.mk" > /dev/null || rc=$?
  [ "$rc" -eq 2 ] || { echo "SELF-TEST FAIL: scanning 0 files must refuse (rc=$rc)"; fails=$((fails + 1)); }
  for c in derived_bin derived_bin_installed derived_sh explicit package comment test; do
    rc=0; scan "$dir/$c.mk" > /dev/null || rc=$?
    [ "$rc" -eq 0 ] || { echo "SELF-TEST FAIL: $c must pass (rc=$rc)"; fails=$((fails + 1)); }
  done
  if [ "$fails" -eq 0 ]; then echo "check_coverage_report_scoped self-test: 13/13 cases OK"; return 0; fi
  return 1
}

case "${1:-}" in
  --help|-h) echo "usage: $0 [file ...] | --self-test   (no args: scan Makefile scripts/*.sh .github/workflows/*.yml)" ;;
  --self-test) self_test ;;
  "")
    shopt -s nullglob
    files=()
    for f in Makefile scripts/*.sh .github/workflows/*.yml; do
      # this guard's own case table holds unscoped lines by design
      [ "$f" = "scripts/check_coverage_report_scoped.sh" ] || files+=("$f")
    done
    scan "${files[@]}"
    ;;
  *) scan "$@" ;;
esac
