#!/usr/bin/env bash
# FALSIFY-BOOK-CLI-PARITY-001 — every `apr <cmd>` has a chapter at book/src/cli/<cmd>.md.
# Per BOOK-CLOSEOUT-001 § Phase 4.
set -euo pipefail

# parity CMDS BOOK_CLI_DIR -> prints the coverage lines; returns 1 on a missing chapter, or on an
# empty command list (#2876): `apr --help` with no `Commands:` heading gave cmds="", the loop never
# ran, `echo "" | wc -l` counted 1, and the gate printed "Coverage: 1/1" and PASS.
parity() {
  local cmds=$1 dir=$2 cmd missing=0 total
  total=$(printf '%s\n' "$cmds" | grep -c . || true)
  if [ "$total" -eq 0 ]; then
    echo "FAIL: apr --help listed 0 subcommands under Commands: -- the gate read nothing (#2876)"
    echo "FALSIFY-BOOK-CLI-PARITY-001: FAIL"
    return 1
  fi
  for cmd in $cmds; do
    if [ ! -f "$dir/${cmd}.md" ]; then
      echo "FAIL: $dir/${cmd}.md does not exist (apr ${cmd} has no chapter)"
      missing=$((missing+1))
    fi
  done
  echo ""
  echo "Coverage: $((total - missing))/${total} CLI subcommands have a chapter (${missing} missing)"
  if [ "$missing" -gt 0 ]; then
    echo "FALSIFY-BOOK-CLI-PARITY-001: FAIL"
    return 1
  fi
  echo "FALSIFY-BOOK-CLI-PARITY-001: PASS"
}

self_test() {
  local td ok=0 bad=0
  td="$(mktemp -d "${TMPDIR:-/tmp}/book-cli-parity.XXXXXX")"
  trap 'rm -rf -- "${td:?}"' RETURN
  row() { # row LABEL GOT WANT
    if [ "$2" = "$3" ]; then printf 'ok    %s\n' "$1"; ok=$((ok + 1))
    else printf 'FAIL  %s: got [%s], want [%s]\n' "$1" "$2" "$3"; bad=$((bad + 1)); fi
  }
  rc() { local r=0; parity "$1" "$td" > "$td/out" 2>&1 || r=$?; echo "$r"; }
  : > "$td/run.md"; : > "$td/qa.md"
  row "an empty command list is RED, never Coverage: 1/1 PASS (#2876)" "$(rc '')" 1
  row "a blank-line-only command list is RED" "$(rc $'\n\n')" 1
  row "every command with a chapter is PASS" "$(rc $'run\nqa')" 0
  row "the coverage line counts the commands" "$(command grep -c '^Coverage: 2/2' "$td/out")" 1
  row "a command with no chapter is RED" "$(rc $'run\nserve')" 1
  printf 'check_book_cli_parity self-test: %s ok, %s bad\n' "$ok" "$bad"
  [ "$bad" -eq 0 ] && [ "$ok" -ge 5 ]
}

case "${1:-}" in
  --self-test) self_test; exit $? ;;
  -h|--help) echo "usage: bash scripts/check_book_cli_parity.sh [--self-test]"; exit 0 ;;
esac

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# This gate reads `apr --help` and asserts a chapter exists for every subcommand.
# It used to resolve the binary as `${APR:-/home/noah/.cargo/bin/apr}`, falling
# back to `$(which apr)` — one developer's home directory, then PATH. Both are
# the #2357 defect: the command list it compared the book against came from
# whatever binary happened to be installed, not from this commit. A chapter
# added for a NEW subcommand would have been reported missing, and a chapter for
# a DELETED one reported present, with no way to tell from the output.
. scripts/apr_bin.sh || exit 1

cmds=$("$APR" --help 2>&1 | awk '/^Commands:/{f=1; next} f && /^  [a-z]/{print $1}')

parity "$cmds" book/src/cli
