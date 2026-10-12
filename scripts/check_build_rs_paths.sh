#!/usr/bin/env bash
# check_build_rs_paths.sh — Poka-Yoke: flag build.rs files that PANIC when
# a path outside the crate root is absent. Those break `cargo install`.
#
# Root cause this guards against (v0.31.1 release blocker):
#   crates/aprender-mcp/build.rs resolved `CARGO_MANIFEST_DIR/../../contracts/…`
#   and called `.unwrap_or_else(|e| panic!(...))` when the file was missing.
#   That file lives in the monorepo tree but NOT in the published tarball
#   (outside the package root). `cargo install aprender@0.31.1` panicked at
#   build time for every external user. v0.31.1 was yanked.
#
# Acceptable patterns (NOT flagged):
#   1. build.rs has `ALLOW_ESCAPE: <reason>` comment
#   2. build.rs has `.exists()` + `return`/`Ok(())` graceful fallback
#      (crates/aprender-core/build.rs does this for provable-contracts/)
#   3. Path join of `".."` is inside a `#[cfg(test)]` block
#
# Failure pattern (FLAGGED):
#   - build.rs joins `".."` AND calls `panic!`/`unwrap_or_else(|…| panic!)`
#     AND has no `.exists()` guard before the read.
#
# Complementary gate: check_package_verify.sh runs `cargo package` per crate
# which actually unpacks the tarball and compiles — the dynamic equivalent.
#
# Vacuity (#2876): a scan that read no build.rs is RED, never "OK: 0 checked". CI ran this
#   inside a container where git refused the host-owned checkout ("dubious ownership"), so
#   `git ls-files` failed unseen in a process substitution and the gate printed
#   "OK: 0 build.rs files checked" on main. A git failure or an empty list now fails by name.
#
# Usage: bash scripts/check_build_rs_paths.sh             scan this checkout
#        bash scripts/check_build_rs_paths.sh --self-test the case table, in fixture repos
# Exit 0 clean, exit 1 on any finding or an empty scan.

set -uo pipefail

SELF="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/$(basename -- "${BASH_SOURCE[0]}")"

self_test() {
    local td ok=0 bad=0 rc
    td="$(mktemp -d "${TMPDIR:-/tmp}/build-rs-paths.XXXXXX")"
    trap 'rm -rf -- "${td:?}"' RETURN
    cd -- "$td" || return 1
    export GIT_CEILING_DIRECTORIES="${td%/*}" GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_NOSYSTEM=1
    row() { # row LABEL GOT WANT
        if [ "$2" = "$3" ]; then printf 'ok    %s\n' "$1"; ok=$((ok + 1))
        else printf 'FAIL  %s: got [%s], want [%s]\n' "$1" "$2" "$3"; bad=$((bad + 1)); fi
    }
    run() { rc=0; (cd -- "$td/$1" && bash "$SELF") > "$td/$1.out" 2>&1 || rc=$?; echo "$rc"; }
    mkdir -p empty/c nogit good/c bad/c
    git init -q empty; : > empty/c/lib.rs; git -C empty add -A
    row "a checkout with no build.rs is RED, never OK: 0 checked (#2876)" "$(run empty)" 1
    row "a directory git cannot read is RED (the dubious-ownership run)" "$(run nogit)" 1
    row "and it names the git failure, not an empty tree" "$(command grep -c 'git ls-files could not list' nogit.out)" 1
    git init -q good; printf 'fn main() {}\n' > good/c/build.rs; git -C good add -A
    row "a clean build.rs is OK" "$(run good)" 0
    row "the OK line counts what it read" "$(command grep -c '^OK: 1 build.rs' good.out)" 1
    git init -q bad
    printf 'fn main() { let p = std::path::Path::new("..").join("x"); std::fs::read(p).unwrap_or_else(|e| panic!("{e}")); }\n' > bad/c/build.rs
    git -C bad add -A
    row "a build.rs that joins .. and panics is RED" "$(run bad)" 1
    printf 'check_build_rs_paths self-test: %s ok, %s bad\n' "$ok" "$bad"
    [ "$bad" -eq 0 ] && [ "$ok" -ge 6 ]
}

case "${1:-}" in
    --self-test) self_test; exit $? ;;
    -h | --help) echo "usage: bash scripts/check_build_rs_paths.sh [--self-test]"; exit 0 ;;
    "") : ;;
    *) echo "check_build_rs_paths: unknown argument $1" >&2; exit 3 ;;
esac

errors=0
checked=0
suspects=0

echo "Build.rs path safety gate (static)..."

# A failed git (not a checkout, or dubious ownership) must not read as an empty list (#2876).
if ! files=$(git ls-files '**/build.rs' 'build.rs'); then
    echo "FAIL: git ls-files could not list build.rs files here ($PWD): the scan read nothing (#2876)."
    exit 1
fi

while IFS= read -r build_rs; do
    [ -n "$build_rs" ] || continue
    checked=$((checked + 1))

    # Skip if no `..` path escape at all.
    if ! grep -qE '"\.\."' "$build_rs"; then
        continue
    fi
    suspects=$((suspects + 1))

    # Acceptable: explicit allow-escape annotation.
    if grep -q 'ALLOW_ESCAPE' "$build_rs"; then
        continue
    fi
    # Acceptable: graceful fallback — checks `.exists()` before reading.
    # Pattern covers both `if path.exists()` and `if !path.exists() { return; }`.
    if grep -qE '\.exists\(\)' "$build_rs"; then
        continue
    fi

    # Hard-coded absolute paths are always wrong.
    if grep -qE '"/(home|tmp|usr|var|opt|mnt|root)/' "$build_rs"; then
        echo "  FAIL: $build_rs — hard-coded absolute path (portability bug)"
        errors=$((errors + 1))
        continue
    fi

    # Suspect: has `..` join AND panics on read failure, no .exists() guard.
    if grep -qE 'panic!|unwrap_or_else\(\|.*\|\s*panic' "$build_rs"; then
        echo "  FAIL: $build_rs"
        echo "    joins \"..\" onto CARGO_MANIFEST_DIR AND panics on missing file;"
        echo "    this class broke \`cargo install\` in v0.31.1 (aprender-mcp)."
        echo "    Fix: either"
        echo "      (a) copy the file inside the crate + update build.rs path, OR"
        echo "      (b) add a \`.exists()\` guard with graceful fallback, OR"
        echo "      (c) annotate with '// ALLOW_ESCAPE: <reason>' if only used in dev."
        errors=$((errors + 1))
    fi
done <<< "$files"

echo ""
if [ "$checked" -eq 0 ]; then
    echo "FAIL: 0 build.rs files checked: the scan read nothing, so it proves nothing (#2876)."
    exit 1
fi
if [ "$errors" -eq 0 ]; then
    echo "OK: $checked build.rs files checked ($suspects with \`..\` joins all have graceful fallbacks)."
    exit 0
else
    echo "FAIL: $errors/$suspects suspect build.rs files WILL break \`cargo install\`."
    echo ""
    echo "v0.31.1 was yanked for exactly this class of bug. Do not ship without fixing."
    exit 1
fi
