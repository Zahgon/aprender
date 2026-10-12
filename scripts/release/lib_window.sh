# lib_window.sh — the 45-minute release window (#5185; operator 2026-10-12: "A release, i.e. frozen
# main, needs to release in 45 minutes; the end."; effective 0.71.2).
#
# The window CHECKS receipts and computes nothing long. Two steps that used to run inside it on the
# cut sha for the first time -- the CRUX models smoke (58.5 min at 0.71.0) and the publish dry run
# (up to 14.5 min) -- run each night on the candidate commit C instead (release-lanes-nightly.yml,
# job `receipt`), which uploads ONE artifact, release-receipt-<C>:
#   release-receipt-<C>.json   scripts/release/release_receipt.sh --seal (models + publish-dryrun states)
#   models-t1/                 the verified CRUX bundle of C (models-t1.log + the host receipts), when
#                              the relay found one
# Before the window opens, the inputs check proves every window input exists for the cut sha:
# release_notes.md (made here from the release commit's CHANGELOG [V] when absent: 0.71.1 lost 17m53s
# to a missing one AFTER models GO), and a GO receipt for exactly the cut sha. A missing receipt is never
# made inside the window: the window is not opened. Inside the window every step prints its elapsed
# time and the running total; past the budget one ANDON line is printed, once (the steps still run:
# an andon is a signal, not a stop, and nothing here re-runs a step).
#
# SOURCED, so option-neutral: no `set` here (scripts/check_sourced_libs_option_neutral.sh). Callers:
#   . scripts/release/lib_window.sh || exit 2

WIN_BUDGET_S=2700   # 45 min

# win_fetch REPO C DIR -> rc 0: DIR holds the newest unexpired release-receipt-<C> artifact.
# rc 1: stdout says why (no artifact, a failed read or download). Never a pass on a failed read.
win_fetch() {
    local repo=${1:?} c=${2:?} dir=${3:?} run
    run=$(gh api "repos/$repo/actions/artifacts?name=release-receipt-$c&per_page=30" \
        --jq '[.artifacts[] | select(.expired | not)] | sort_by(.created_at) | last | .workflow_run.id // empty' 2> /dev/null) \
        || { echo "the artifact list for release-receipt-${c:0:9} could not be read (not_measured)"; return 1; }
    [[ $run =~ ^[0-9]+$ ]] || { echo "no unexpired release-receipt-${c:0:9} artifact: the nightly has not sealed a receipt for the cut sha"; return 1; }
    rm -rf -- "${dir:?}"
    gh run download "$run" --repo "$repo" -n "release-receipt-$c" -D "$dir" > /dev/null 2>&1 \
        || { echo "run $run: release-receipt-${c:0:9} could not be downloaded"; return 1; }
    printf '%s\n' "$run"
}

# win_notes CHANGELOG V OUT -> rc 0 when OUT exists non-empty, made from CHANGELOG [V] if absent
# (prepare_bump.sh's own awk). rc 1 with a reason when neither exists.
win_notes() {
    local cl=${1:?} v=${2:?} out=${3:?}
    [ -s "$out" ] && { echo "present"; return 0; }
    [ -f "$cl" ] || { echo "no $out and no $cl to make it from"; return 1; }
    awk -v h="## [$v]" 'index($0, h) == 1 {f = 1; next} f && /^## \[/ {exit} f' "$cl" > "$out" \
        || { echo "cannot write $out"; return 1; }
    [ -s "$out" ] || { rm -f -- "${out:?}"; echo "the CHANGELOG [$v] section is empty"; return 1; }
    echo "made from $cl [$v]"
}

# win_models_line LOG C -> the MODELS GO (CRUX smoke) line at C's 9-char sha, rc 1 when there is none.
win_models_line() {
    local log=${1:?} c=${2:?} line
    line=$(grep -E "^MODELS GO \(CRUX smoke\) .* at ${c:0:9}:" "$log" 2> /dev/null | tail -n 1)
    [ -n "$line" ] || return 1
    printf '%s\n' "$line"
}

# The clock. win_open starts it; win_mark NEXT closes the step in progress and starts NEXT; win_close
# closes the last one. Each fills WIN_OUT with the lines to print (the caller prints them, so the
# clock's state stays in the caller's shell).
win_open() { WIN_T0=$SECONDS; WIN_STEP=''; WIN_STEP_T0=$SECONDS; WIN_ANDON=0; WIN_OUT=(); }
win_mark() {
    local next=$1 now=$SECONDS total
    WIN_OUT=()
    [ -n "${WIN_T0:-}" ] || return 0
    [ "$next" != "$WIN_STEP" ] || return 0
    total=$((now - WIN_T0))
    [ -z "$WIN_STEP" ] || WIN_OUT+=("WINDOW step $WIN_STEP seconds=$((now - WIN_STEP_T0)) total=${total}s of ${WIN_BUDGET_S}s")
    if [ "$total" -gt "$WIN_BUDGET_S" ] && [ "$WIN_ANDON" = 0 ]; then
        WIN_ANDON=1
        WIN_OUT+=("ANDON WINDOW over budget: ${total}s > ${WIN_BUDGET_S}s (45 min) at the end of ${WIN_STEP:-the inputs check}")
    fi
    WIN_STEP=$next; WIN_STEP_T0=$now
}
win_close() {
    local total
    win_mark ''
    [ -n "${WIN_T0:-}" ] || return 0
    total=$((SECONDS - WIN_T0))
    WIN_OUT+=("WINDOW total=${total}s of ${WIN_BUDGET_S}s ($([ "$total" -le "$WIN_BUDGET_S" ] && echo within || echo OVER) budget)")
    WIN_T0=''
}
