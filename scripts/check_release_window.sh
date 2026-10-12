#!/usr/bin/env bash
# check_release_window.sh — the 45-minute release window (#5185), as a case table plus mutants.
#
# SUBJECT. scripts/release/lib_window.sh (the inputs, the receipt fetch, the clock) and the lines of
# scripts/release/autopilot.sh that use it under AP_WINDOW=1. No network, no cargo, no GPU: `gh` is a
# stub on PATH that serves a fixture artifact list and a fixture download.
#
# ROWS. Each runs the real library function (or the real autopilot block, extracted) and compares its
# exit code and one output needle:
#   notes-present / notes-made / notes-empty-section / notes-no-changelog
#   fetch-ok / fetch-none / fetch-expired-only / fetch-list-fails / fetch-download-fails
#   models-line-ok / models-line-other-sha / models-line-ladder-mode
#   clock-within / clock-over-andon-once / clock-total-over
#   ap-models-reads-receipt   t1_models under AP_WIN=1 copies the receipt's bundle and measures nothing
#   ap-dryrun-reads-receipt   the tag step under AP_WIN=1 re-reads the receipt and never runs rc_publish_gate
#   ap-window-off-unchanged   AP_WINDOW unset: the window block never runs (no INPUTS, no WINDOW line)
# MUTANTS. Each must turn at least one row RED: notes never made, fetch takes an expired artifact, the
# models line ignores the sha, the andon never fires, the andon fires every step, the window models
# step falls through to a measurement.
#
#   check_release_window.sh              judge the tree
# EXIT 0 every row and every mutant behaves · 1 a row or a mutant is wrong · 2 cannot run
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)" || exit 2
LIB="$ROOT/scripts/release/lib_window.sh"
AP_SH="$ROOT/scripts/release/autopilot.sh"
[ -f "$LIB" ] && [ -f "$AP_SH" ] || { echo "check_release_window: subject missing" >&2; exit 2; }
command -v jq > /dev/null || { echo "check_release_window: jq is required" >&2; exit 2; }
T=$(mktemp -d) || exit 2
trap 'rm -rf -- "${T:?}"' EXIT
C=$(printf 'c%.0s' {1..40}); C=${C//c/a}; D=$(printf 'd%.0s' {1..40})

# the gh stub: `api` prints $GH_LIST (a file) through jq with the caller's --jq; `run download` copies
# $GH_DL (a dir) into -D unless GH_DL_FAIL=1; GH_API_FAIL=1 fails the list read.
mkdir -p "$T/bin"
cat > "$T/bin/gh" <<'STUB'
#!/usr/bin/env bash
case $1 in
  api) [ "${GH_API_FAIL:-0}" = 1 ] && exit 1
       q=''; while [ $# -gt 0 ]; do [ "$1" = --jq ] && q=$2; shift; done
       jq -r "$q" "$GH_LIST" ;;
  run) [ "${GH_DL_FAIL:-0}" = 1 ] && exit 1
       d=''; while [ $# -gt 0 ]; do [ "$1" = -D ] && d=$2; shift; done
       mkdir -p "$d" && cp -r "$GH_DL"/. "$d"/ ;;
  *) exit 1 ;;
esac
STUB
chmod +x "$T/bin/gh"

art() { # art FILE [expired]... -> an artifact list, one entry per arg "runid:expired:created"
    local f=$1; shift
    printf '%s\n' "$@" | jq -R 'split(":") | {workflow_run: {id: (.[0] | tonumber)}, expired: (.[1] == "true"), created_at: .[2]}' \
        | jq -s '{artifacts: .}' > "$f"
}
mkdir -p "$T/dl"; printf '{}' > "$T/dl/release-receipt-$C.json"

fail=0; n=0
row() { # row NAME WANT-RC NEEDLE CMD... (CMD runs in a subshell with lib + stub)
    local name=$1 want=$2 needle=$3 out rc; shift 3; n=$((n + 1))
    out=$( (PATH="$T/bin:$PATH"; . "$LIB_UNDER"; "$@") 2>&1 ); rc=$?
    if [ "$rc" -ne "$want" ] || { [ -n "$needle" ] && ! grep -qF -- "$needle" <<< "$out"; }; then
        fail=$((fail + 1)); printf '  RED %s: rc %s (want %s), needle "%s" in: %s\n' "$name" "$rc" "$want" "$needle" "${out//$'\n'/ | }"
    fi
}

clock() { # clock STEP-SECONDS... : open, sleepless -- SECONDS is advanced by assignment
    local s; win_open
    for s in "$@"; do SECONDS=$((SECONDS + s)); win_mark "step$s"; printf '%s\n' "${WIN_OUT[@]}"; done
    win_close; printf '%s\n' "${WIN_OUT[@]}"
}

judge() { # judge LIB AUTOPILOT -> rows, returns the number RED
    LIB_UNDER=$1; local ap=$2; fail=0; n=0
    # notes
    printf 'x\n' > "$T/notes-present"
    row notes-present 0 present win_notes /nonexistent 1.0.0 "$T/notes-present"
    printf '# Changelog\n\n## [1.2.3] - d\n- one\n\n## [1.2.2]\n- old\n' > "$T/CHANGELOG.md"
    rm -f "$T/notes-made"
    row notes-made 0 "made from" bash -c ". '$LIB_UNDER'; win_notes '$T/CHANGELOG.md' 1.2.3 '$T/notes-made' && grep -qx -- '- one' '$T/notes-made' && ! grep -q old '$T/notes-made'"
    rm -f "$T/notes-empty"
    row notes-empty-section 1 "section is empty" win_notes "$T/CHANGELOG.md" 9.9.9 "$T/notes-empty"
    row notes-no-changelog 1 "no " win_notes "$T/none.md" 1.2.3 "$T/notes-x"
    # fetch
    art "$T/l-ok" "11:false:2026-10-01" "22:false:2026-10-02" "33:true:2026-10-03"
    row fetch-ok 0 22 env GH_LIST="$T/l-ok" GH_DL="$T/dl" bash -c ". '$LIB_UNDER'; win_fetch o/r $C '$T/f1' && [ -f '$T/f1/release-receipt-$C.json' ]"
    printf '{"artifacts":[]}' > "$T/l-none"
    row fetch-none 1 "no unexpired" env GH_LIST="$T/l-none" GH_DL="$T/dl" bash -c ". '$LIB_UNDER'; win_fetch o/r $C '$T/f2'"
    art "$T/l-exp" "33:true:2026-10-03"
    row fetch-expired-only 1 "no unexpired" env GH_LIST="$T/l-exp" GH_DL="$T/dl" bash -c ". '$LIB_UNDER'; win_fetch o/r $C '$T/f3'"
    row fetch-list-fails 1 "not_measured" env GH_API_FAIL=1 GH_LIST="$T/l-ok" GH_DL="$T/dl" bash -c ". '$LIB_UNDER'; win_fetch o/r $C '$T/f4'"
    row fetch-download-fails 1 "could not be downloaded" env GH_DL_FAIL=1 GH_LIST="$T/l-ok" GH_DL="$T/dl" bash -c ". '$LIB_UNDER'; win_fetch o/r $C '$T/f5'"
    # models line
    printf 'MODELS lambda ...\nMODELS GO (CRUX smoke) on lambda and gx10 at %s: ok\n' "${C:0:9}" > "$T/m-ok.log"
    row models-line-ok 0 "MODELS GO (CRUX smoke)" win_models_line "$T/m-ok.log" "$C"
    row models-line-other-sha 1 "" win_models_line "$T/m-ok.log" "$D"
    printf 'MODELS GO on lambda and gx10 at %s: ladder\n' "${C:0:9}" > "$T/m-lad.log"
    row models-line-ladder-mode 1 "" win_models_line "$T/m-lad.log" "$C"
    # clock
    row clock-within 0 "within budget" bash -c ". '$LIB_UNDER'; $(declare -f clock); clock 600 600 \$(true) | grep -q ANDON && exit 1; clock 600 600"
    row clock-over-andon-once 0 "ANDON WINDOW over budget" bash -c ". '$LIB_UNDER'; $(declare -f clock); o=\$(clock 1500 1500 600 600); [ \"\$(grep -c ANDON <<< \"\$o\")\" = 1 ] && printf '%s\n' \"\$o\""
    row clock-total-over 0 "OVER budget" bash -c ". '$LIB_UNDER'; $(declare -f clock); clock 3000"
    # autopilot hooks, extracted
    local tm tag
    tm=$(awk '/^t1_models\(\) \{/,/^\}/; /^t1_models_receipt\(\) \{/,/^\}/' "$ap")
    mkdir -p "$T/ap/receipt/models-t1"; cp "$T/m-ok.log" "$T/ap/receipt/models-t1/models-t1.log"
    row ap-models-reads-receipt 0 "nightly receipt, 0 s" bash -c "
        . '$LIB_UNDER'; AP='$T/ap'; STATUS='$T/ap/STATUS'; MC=$C; AP_POLICY=1; AP_WIN=1; V=1.2.3
        say() { printf '%s\n' \"\$*\"; }; die() { echo \"STOP \$*\"; exit 1; }; rehearsal() { return 1; }
        $tm
        t1_models && [ -f '$T/ap/models-t1/models-t1.log' ]"
    tag=$(awk '/^if run_step tag; then$/,/^  printf .%s\\n. "\$MC" > "\$AP\/publish-dryrun-commit"$/' "$ap" | sed -n '/rm -f -- "${AP:?}\/publish-dryrun-commit"/,$p')
    row ap-dryrun-reads-receipt 0 "dryrun-ok" bash -c "
        AP='$T/ap'; MC=$C; AP_WIN=1; STATUS='$T/ap/STATUS'
        die() { echo \"STOP \$*\"; exit 1; }
        bash() { case \$1 in scripts/release/rc_publish_gate.sh) echo RAN-GATE; return 1 ;; *) [ \"\$2\" = --check ] && { echo 'RECEIPT GO'; return 0; }; return 1 ;; esac; }
        printf '{\"steps\":{\"publish_dryrun\":{\"state\":\"green\",\"reason\":\"ok\"}},\"run_url\":\"u\"}' > '$T/ap/receipt/release-receipt-$C.json'
        $tag
        printf '%s\n' \"\$MC\" > \"\$AP/publish-dryrun-commit\"
        grep -q RAN-GATE '$T/ap/publish-dryrun.log' && exit 1; echo dryrun-ok"
    local blk
    blk=$(awk '/^if \[ "\$\{AP_WINDOW:-\}" = 1 \]; then$/,/^fi$/' "$ap")
    row ap-window-off-unchanged 0 "off-ok" bash -c "
        unset AP_WINDOW; AP_WIN=0; say() { echo \"SAID \$*\"; }; die() { echo \"STOP \$*\"; exit 1; }
        [ -n \"\$(cat <<'B'
$blk
B
)\" ] || { echo no-block; exit 1; }
        $blk
        [ \"\$AP_WIN\" = 0 ] && echo off-ok"
    return "$fail"
}

# the tree
judge "$LIB" "$AP_SH" > "$T/tree.out"; tree_red=$?
cat "$T/tree.out"
[ "$tree_red" = 0 ] || { echo "check_release_window: RED, $tree_red of $n row(s) on the tree"; exit 1; }
rows=$n

# the mutants: each a sed over a copy of the subject; each must turn >= 1 row RED
mutant() { # mutant NAME FILE(lib|ap) SED
    local name=$1 which=$2 expr=$3 l=$LIB a=$AP_SH red
    if [ "$which" = lib ]; then l="$T/mut-lib.sh"; sed "$expr" "$LIB" > "$l"; else a="$T/mut-ap.sh"; sed "$expr" "$AP_SH" > "$a"; fi
    if cmp -s "$l" "$LIB" && cmp -s "$a" "$AP_SH"; then echo "  MUTANT $name did not apply"; return 1; fi
    judge "$l" "$a" > /dev/null; red=$?
    [ "$red" -gt 0 ] || { echo "  MUTANT $name SURVIVED"; return 1; }
}
MUTANTS=(
    'notes-never-made|lib|s/^    \[ -s "\$out" \] \&\& { echo "present"; return 0; }$/    [ -s "$out" ] || { echo "no $out"; return 1; }; echo present; return 0/'
    'fetch-takes-expired|lib|s/select(.expired | not)/select(true)/'
    'models-line-any-sha|lib|s/ at \${c:0:9}:/ at /'
    'andon-never|lib|s/"\$total" -gt "\$WIN_BUDGET_S"/"$total" -gt 999999/'
    'andon-every-step|lib|s/ \&\& \[ "\$WIN_ANDON" = 0 \]//'
    'window-models-measures|ap|/t1_models_receipt; return; fi/d'
    'window-dryrun-runs-gate|ap|s/^  if \[ "\${AP_WIN:-0}" = 1 \]; then$/  if false; then/'
)
mfail=0; m=0
for spec in "${MUTANTS[@]}"; do
    m=$((m + 1)); IFS='|' read -r name which expr <<< "$spec"
    mutant "$name" "$which" "$expr" || mfail=$((mfail + 1))
done
[ "$mfail" = 0 ] || { echo "check_release_window: RED, $mfail of $m mutant(s) not killed"; exit 1; }
echo "check_release_window: PASS  $rows row(s), $m mutant(s) killed: the window checks receipts, makes its inputs before it opens, and clocks every step against 45 min"
