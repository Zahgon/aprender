#!/usr/bin/env bash
#
# check_exclude_anchored.sh — CB-510 class guard, generalised.
#
# A `exclude = [...]` entry in Cargo.toml is a gitignore-style pattern. A bare
# directory name such as `"models/"` or `"tests/"` is NOT anchored to the package
# root: it matches at EVERY depth, so it silently swallows `src/models/` and
# `src/api/tests/` on the way to crates.io.
#
# This has now shipped twice:
#   * CB-510  — `"models/"` hid `src/models/` from git and from the package.
#   * v0.63.0 — `"tests/"` dropped 443 files from `aprender-serve` and 117 broken
#               `mod tests;` declarations from `aprender-train`. Both published
#               crates cannot compile their own test suites.
#
# The fix in both cases is a leading slash: `"/models/"`, `"/tests/"`.
#
# This guard fails when a non-anchored directory pattern in any `exclude` list
# matches a directory that actually exists under that crate's `src/`.
#
# Vacuity (#2876): a scan that read no manifest with a src/ tree is RED, never "OK - 0 checked".
#   `git ls-files` ran in a process substitution, so a root that is not a checkout (or one git
#   refuses) read as an empty list and the guard printed OK.
#
# Usage: bash scripts/check_exclude_anchored.sh [repo_root]
#        bash scripts/check_exclude_anchored.sh --self-test   the case table, in fixture repos

set -euo pipefail

SELF="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/$(basename -- "${BASH_SOURCE[0]}")"

self_test() {
    local td ok=0 bad=0
    td="$(mktemp -d "${TMPDIR:-/tmp}/exclude-anchored.XXXXXX")"
    trap 'rm -rf -- "${td:?}"' RETURN
    cd -- "$td" || return 1
    export GIT_CEILING_DIRECTORIES="${td%/*}" GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_NOSYSTEM=1
    row() { # row LABEL GOT WANT
        if [ "$2" = "$3" ]; then printf 'ok    %s\n' "$1"; ok=$((ok + 1))
        else printf 'FAIL  %s: got [%s], want [%s]\n' "$1" "$2" "$3"; bad=$((bad + 1)); fi
    }
    run() { local rc=0; bash "$SELF" "$td/$1" > "$td/$1.out" 2>&1 || rc=$?; echo "$rc"; }
    crate() { mkdir -p "$td/$1/c/src/$3"; printf '[package]\nname = "c"\nexclude = ["%s"]\n' "$2" > "$td/$1/c/Cargo.toml"; }
    mkdir -p empty/docs nogit
    git init -q empty; : > empty/docs/a.md; git -C empty add -A
    row "a checkout with no manifest is RED, never OK - 0 checked (#2876)" "$(run empty)" 1
    crate nogit /tests/ tests
    row "a root git cannot list is RED, not an empty list" "$(run nogit)" 1
    row "and it names the git failure, not an empty tree" "$(command grep -c 'git could not list' nogit.out)" 1
    crate good /tests/ tests; git init -q good; git -C good add -A
    row "an anchored exclude is OK" "$(run good)" 0
    row "the OK line counts what it read" "$(command grep -c 'OK - 1 manifest' good.out)" 1
    crate bad tests/ tests; git init -q bad; git -C bad add -A
    row "a bare tests/ that swallows src/tests is RED (v0.63.0)" "$(run bad)" 1
    printf 'check_exclude_anchored self-test: %s ok, %s bad\n' "$ok" "$bad"
    [ "$bad" -eq 0 ] && [ "$ok" -ge 6 ]
}

case "${1:-}" in
    --self-test) self_test; exit $? ;;
    -h | --help) echo "usage: bash scripts/check_exclude_anchored.sh [repo_root | --self-test]"; exit 0 ;;
esac

repo_root="${1:-$(git rev-parse --show-toplevel)}"
cd "$repo_root"

violations=0
checked=0

# Emit the quoted entries of the `exclude = [ ... ]` array of a Cargo.toml, one
# per line, with the surrounding quotes stripped. Comment lines are skipped.
extract_exclude_patterns() {
    manifest="$1"
    sed -n '/^exclude[[:space:]]*=[[:space:]]*\[/,/\]/p' "$manifest" \
        | sed 's/#.*//' \
        | grep -o '"[^"]*"' \
        | tr -d '"'
}

if ! manifests="$(git ls-files '*Cargo.toml' | sort)"; then
    printf 'check_exclude_anchored: FAIL - git could not list manifests under %s: the scan read nothing (#2876)\n' "$repo_root" >&2
    exit 1
fi

while IFS= read -r manifest; do
    [ -n "$manifest" ] || continue
    crate_dir="$(dirname "$manifest")"
    src_dir="$crate_dir/src"
    [ -d "$src_dir" ] || continue

    checked=$((checked + 1))

    while IFS= read -r pattern; do
        # Anchored (`/tests/`), glob (`*.rs`, `**/x`), and non-directory
        # patterns are all fine — only a bare `name/` is ambiguous.
        case "$pattern" in
            /*) continue ;;
            *\**) continue ;;
            */) : ;;
            *) continue ;;
        esac

        dir_name="${pattern%/}"
        # A pattern containing a slash in the middle (`contracts/foo/`) is
        # already specific enough to be unambiguous in practice.
        case "$dir_name" in
            */*) continue ;;
        esac

        matches="$(find "$src_dir" -type d -name "$dir_name" -print 2>/dev/null || true)"
        if [ -n "$matches" ]; then
            count="$(printf '%s\n' "$matches" | wc -l | tr -d ' ')"
            printf 'FAIL %s\n' "$manifest"
            printf '     exclude pattern "%s" is not root-anchored and matches %s director(y|ies) under %s:\n' \
                "$pattern" "$count" "$src_dir"
            printf '%s\n' "$matches" | sed 's/^/       /' | head -5
            if [ "$count" -gt 5 ]; then
                printf '       ... and %s more\n' "$((count - 5))"
            fi
            printf '     fix: write it as "/%s"\n\n' "$pattern"
            violations=$((violations + 1))
        fi
    done < <(extract_exclude_patterns "$manifest")
done <<< "$manifests"

if [ "$violations" -gt 0 ]; then
    printf 'check_exclude_anchored: %s violation(s) across %s manifest(s) with a src/ tree\n' \
        "$violations" "$checked" >&2
    printf 'A non-anchored exclude pattern removes source from the published crate.\n' >&2
    exit 1
fi

if [ "$checked" -eq 0 ]; then
    printf 'check_exclude_anchored: FAIL - 0 manifests with a src/ tree checked: the scan read nothing (#2876)\n' >&2
    exit 1
fi

printf 'check_exclude_anchored: OK - %s manifest(s) checked, no non-anchored exclude swallows a src/ directory\n' "$checked"
