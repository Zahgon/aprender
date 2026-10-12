#!/usr/bin/env bash
# check_include_files.sh — Verify all include!() referenced files are tracked by git
#
# Prevents the CB-510 bug where .gitignore hid src/models/ part files from git,
# causing crates.io publishes to silently exclude required source files.
#
# Usage: ./scripts/check_include_files.sh
# Exit 0 if all OK, exit 1 if any include!() files are untracked/missing, or if none were found.
#
# Vacuity (#2876): the include!() list came from a grep in a process substitution, so a tree
# with no src/ or crates/ (or a grep error) read as zero directives and printed
# "OK: All 0 include!() files are tracked". Zero checked is now a FAIL.
#   ./scripts/check_include_files.sh --self-test   the case table, in fixture repos

set -uo pipefail
# guard_tree.sh probes `--help` to decide whether to run a self-test. Answer it before any work:
# a probe that fell through to the body ran this whole guard a second time, serially (#4046).
SELF="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/$(basename -- "${BASH_SOURCE[0]}")"
self_test() {
    local td ok=0 bad=0
    td="$(mktemp -d "${TMPDIR:-/tmp}/include-files.XXXXXX")"
    trap 'rm -rf -- "${td:?}"' RETURN
    cd -- "$td" || return 1
    export GIT_CEILING_DIRECTORIES="${td%/*}" GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_NOSYSTEM=1
    row() { # row LABEL GOT WANT
        if [ "$2" = "$3" ]; then printf 'ok    %s\n' "$1"; ok=$((ok + 1))
        else printf 'FAIL  %s: got [%s], want [%s]\n' "$1" "$2" "$3"; bad=$((bad + 1)); fi
    }
    run() { local rc=0; (cd -- "$td/$1" && bash "$SELF") > "$td/$1.out" 2>&1 || rc=$?; echo "$rc"; }
    fx() { mkdir -p "$td/$1/crates/c/src"; git init -q "$td/$1"; printf '%s\n' "$2" > "$td/$1/crates/c/src/lib.rs"; }
    fx none 'pub fn f() {}'; git -C none add -A
    row "a tree with no include!() is RED, never OK: All 0 (#2876)" "$(run none)" 1
    mkdir -p nosrc
    row "a directory with no src/ or crates/ is RED" "$(run nosrc)" 1
    fx good 'include!("part.rs");'; : > good/crates/c/src/part.rs; git -C good add -A
    row "a tracked include!() target is OK" "$(run good)" 0
    row "the OK line counts what it read" "$(command grep -c '^OK: All 1 include' good.out)" 1
    fx untracked 'include!("part.rs");'; git -C untracked add -A; : > untracked/crates/c/src/part.rs
    row "an untracked include!() target is RED (CB-510)" "$(run untracked)" 1
    printf 'check_include_files self-test: %s ok, %s bad\n' "$ok" "$bad"
    [ "$bad" -eq 0 ] && [ "$ok" -ge 5 ]
}
case "${1:-}" in
    -h|--help) printf 'usage: bash scripts/check_include_files.sh [--self-test] (no arguments: scans every include!() target)\n'; exit 0 ;;
    --self-test) self_test; exit $? ;;
esac

errors=0
checked=0

# Every tracked path, read ONCE: a `git ls-files --error-unmatch` per target was ~1,800 git
# forks and ~60 s of CI's guard-tree (#4429). A hit here is tracked; a miss still asks git
# the original question, so this only ever skips a git call whose answer is already known.
declare -A TRACKED=()
while IFS= read -r t; do TRACKED[$t]=1; done < <(git ls-files -- src crates)
norm() { # <path> -> the path with ./ and x/../ segments collapsed, lexically (as git would)
    local IFS=/ seg out=()
    for seg in $1; do
        case "$seg" in
            '' | .) ;;
            ..) if [ "${#out[@]}" -gt 0 ] && [ "${out[-1]}" != .. ]; then unset 'out[-1]'; else out+=(..); fi ;;
            *) out+=("$seg") ;;
        esac
    done
    printf '%s' "${out[*]}"
}

# Find all include!() directives in Rust source files
while IFS=: read -r file line content; do
    # Extract the included filename from include!("filename.rs")
    [[ $content =~ include!\([[:space:]]*\"([^\"]+)\"[[:space:]]*\) ]] || continue
    included=${BASH_REMATCH[1]}

    # Resolve relative to the directory containing the source file
    dir=${file%/*}
    resolved="$dir/$included"

    checked=$((checked + 1))

    # Check file exists
    if [ ! -f "$resolved" ]; then
        echo "MISSING: $resolved (referenced by $file:$line)"
        errors=$((errors + 1))
        continue
    fi

    # Check file is tracked by git (not gitignored)
    [ -n "${TRACKED[$(norm "$resolved")]+x}" ] && continue
    if ! git ls-files --error-unmatch "$resolved" >/dev/null 2>&1; then
        # Double-check: is it ignored?
        if git check-ignore -q "$resolved" 2>/dev/null; then
            echo "GITIGNORED: $resolved (referenced by $file:$line)"
        else
            echo "UNTRACKED: $resolved (referenced by $file:$line)"
        fi
        errors=$((errors + 1))
    fi
done < <(grep -rn 'include!(' src/ crates/ --include='*.rs' | grep -v '/target/' | grep -v '#\[')

if [ "$checked" -eq 0 ]; then
    echo "FAIL: 0 include!() directives found under src/ and crates/ in $PWD: the scan read nothing (#2876)"
    exit 1
fi

if [ "$errors" -gt 0 ]; then
    echo ""
    echo "FAIL: $errors include!() files are missing, gitignored, or untracked out of $checked checked"
    echo "Fix: git add the files and check .gitignore / Cargo.toml exclude patterns"
    exit 1
else
    echo "OK: All $checked include!() files are tracked by git"
    exit 0
fi
