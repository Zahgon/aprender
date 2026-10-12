#!/usr/bin/env bash
# check_release_state_dir.sh -- the release train's state directory may not sit under a
# .cargo config that carries a [patch] table (#5198).
#
# THE DEFECT. The state dir defaulted to <main checkout>/target/release-train/<T>, and the
# autopilot builds in its wt/ subdirectory. cargo reads .cargo/config.toml in every
# ancestor of its cwd, so the build there read the main checkout's gitignored
# .cargo/config.toml, whose [patch.crates-io] entries rewrite Cargo.lock, and the models
# step's `cargo build --locked` failed on a sha that builds clean outside the tree.
#
# THE RULES, run against scripts/release/lib_release_params.sh in a sandbox (HOME,
# XDG_STATE_HOME and CARGO_HOME all point into a mktemp dir; nothing outside it is read
# or written):
#   S1  the default state dir is outside the repository checkout
#   S2  a [patch] config (config.toml or the legacy config name) above the state dir
#       refuses, from release_params and from release_state_dir_check alike, RELEASE_AP too
#   S3  a config without [patch], a config at CARGO_HOME, and a clean dir all pass
#   S4  no absolute HOME/XDG_STATE_HOME and no RELEASE_AP refuses (never a relative dir)
# Zero rows run is ENV rc=2, never a pass.
#
#   check_release_state_dir.sh              run the rows (they ARE the check)
#   check_release_state_dir.sh --self-test  the same rows
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)" || exit 2
LIB="$ROOT/scripts/release/lib_release_params.sh"
[ -f "$LIB" ] || { printf 'ENV   %s: missing -- cannot judge, not a pass\n' "$LIB" >&2; exit 2; }
case "${1:-}" in ""|--self-test) ;; *) printf 'usage: %s [--self-test]\n' "$0" >&2; exit 2 ;; esac

SB=$(mktemp -d) || exit 2
trap 'rm -rf -- "${SB:?}"' EXIT
n=0 bad=0

# row <want_rc> <label> <body> -- body runs in a fresh bash with the lib sourced and a sandboxed env
row() {
    local want=$1 label=$2 body=$3 d rc
    n=$((n + 1))
    d="$SB/r$n"
    mkdir -p "$d/home/.cargo" "$d/state" || return 2
    env -i PATH="$PATH" HOME="$d/home" XDG_STATE_HOME="$d/state" CARGO_HOME="$d/home/.cargo" \
        D="$d" LIB="$LIB" ROOT="$ROOT" \
        bash -c '. "$LIB" || exit 9; '"$body" >"$d/out" 2>&1
    rc=$?
    if [ "$rc" = "$want" ]; then
        printf 'ok    %s (rc %s)\n' "$label" "$rc"
    else
        bad=1
        printf 'FAIL  %s: rc %s, want %s\n' "$label" "$rc" "$want"
        sed 's/^/        /' "$d/out"
    fi
}

PATCH=$'[patch.crates-io]\nfoo = { path = "x" }\n'

row 0 "S1 default state dir is outside the checkout" \
    'release_params 9.9.9 "$ROOT" || exit 3
     case "$AP/" in "$ROOT"/*) exit 1 ;; esac
     case "$AP" in "$XDG_STATE_HOME"/aprender/release-train/v9.9.9) exit 0 ;; *) exit 4 ;; esac'
row 0 "S1 HOME fallback when XDG_STATE_HOME is unset" \
    'unset XDG_STATE_HOME; release_params 9.9.9 || exit 3
     [ "$AP" = "$HOME/.local/state/aprender/release-train/v9.9.9" ]'
row 2 "S2 RELEASE_AP under a [patch] config.toml refuses" \
    'mkdir -p "$D/co/.cargo"; printf "%s" "'"$PATCH"'" > "$D/co/.cargo/config.toml"
     RELEASE_AP="$D/co/target/release-train/v9.9.9" release_params 9.9.9'
row 2 "S2 legacy .cargo/config name refuses" \
    'mkdir -p "$D/co/.cargo"; printf "%s" "'"$PATCH"'" > "$D/co/.cargo/config"
     RELEASE_AP="$D/co/a/b" release_params 9.9.9'
row 2 "S2 default state dir under a [patch] config refuses" \
    'mkdir -p "$D/state/.cargo"; printf "%s" "'"$PATCH"'" > "$D/state/.cargo/config.toml"
     release_params 9.9.9'
row 2 "S2 indented [patch] table refuses" \
    'mkdir -p "$D/co/.cargo" "$D/co/x"; printf "  [patch.\"https://github.com/x/y\"]\n" > "$D/co/.cargo/config.toml"
     release_state_dir_check "$D/co/x"'
row 0 "S3 a config without [patch] passes" \
    'mkdir -p "$D/co/.cargo"; printf "[build]\njobs = 4\n" > "$D/co/.cargo/config.toml"
     RELEASE_AP="$D/co/a" release_params 9.9.9'
row 0 "S3 a [patch] config at CARGO_HOME is exempt" \
    'printf "%s" "'"$PATCH"'" > "$CARGO_HOME/config.toml"
     RELEASE_AP="$HOME/a" release_params 9.9.9'
row 0 "S3 a clean dir passes" \
    'mkdir -p "$D/clean"; release_state_dir_check "$D/clean"'
row 2 "S4 no absolute base and no RELEASE_AP refuses" \
    'unset XDG_STATE_HOME; HOME=rel release_params 9.9.9'
row 2 "S4 a bad version still refuses" \
    'release_params 9.9'

[ "$n" -gt 0 ] || { printf 'ENV   no rows ran -- not a pass\n' >&2; exit 2; }
[ "$bad" = 0 ] || { printf 'check_release_state_dir: RED (%s rows)\n' "$n"; exit 1; }
printf 'check_release_state_dir: %s rows green\n' "$n"
