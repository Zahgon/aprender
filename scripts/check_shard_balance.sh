#!/usr/bin/env bash
# check_shard_balance.sh — the once-only steps of workspace-test-shard sit where the
# three shards finish together, and every Σ read has its input on its own shard (#5486).
#
# The merge queue waits on the SLOWEST shard. On 2026-10-16 shard 1 averaged 23.7 min
# against 11.4 / 11.6 for shards 2 and 3, because every `matrix.shard == 1` step landed
# there. This guard sums ci/shard-step-costs.tsv (measured per-step seconds) per shard
# for the FULL tier (what the merge queue runs) and refuses:
#   B1  a predicted spread (slowest - fastest shard) over the budget (120 s)
#   B2  a full-tier step gated to one shard with no measured cost (it cannot be placed)
#   B3  a cost row that names no step, or whose arity does not fit its step
#   S1  a Σ read on a shard where its input was never written: `--junit "$sig/X"` with
#       no earlier step on that shard copying X, or `--step "NAME"` where NAME does not
#       run on that shard (reads inside `if [ "$SHARDS" = 1 ]` are single-shard only)
#   F1  the fan-in's 3-shard assumption broken: matrix-pins, the ci.yml matrix and the
#       verdict job's `for n in ...` loop must name the same shards, and each universe
#       the verdict job reads from sigma-shard-K must be listed by a step on shard K
#
# bash + awk + jq, no interpreter (operator C301): ci/sections.yml's jobs are read by
# scripts/lib/ci_guard_yaml.awk, the reader the guard-step runner uses; ci.yml carries
# folded scalars outside that reader's subset, so its two job blocks are read as text.
#
#   bash scripts/check_shard_balance.sh [--root DIR] [--budget SECONDS]
#   bash scripts/check_shard_balance.sh --self-test
#
# Exit: 0 GREEN, 1 a finding, 2 unreadable input / usage.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
YAML_AWK="$HERE/lib/ci_guard_yaml.awk"
JOB="workspace-test-shard"
command -v jq > /dev/null 2>&1 || { echo "check_shard_balance: jq missing" >&2; exit 2; }

# top_block <file> <key> -- the top-level `key:` block (column 0 to the next column-0 key)
top_block() { awk -v k="$2:" '/^[^ \t#]/ { keep = ($0 == k || index($0, k " ") == 1) } keep' "$1"; }
# job_block <file> <job> -- one job of a top-level jobs: mapping, as text
job_block() {
    awk -v j="  $2:" '/^[^ \t#]/ { injobs = ($0 ~ /^jobs:[ \t]*$/); keep = 0; next }
        injobs && /^  [^ \t#]/ { keep = ($0 == j) } injobs && keep' "$1"
}

# check <root> <budget> -- prints the report and FAIL lines; rc 0 green, 1 findings, 2 unreadable
check() {
    local root="$1" budget="$2" sec="$1/ci/sections.yml" ci="$1/.github/workflows/ci.yml" costs="$1/ci/shard-step-costs.tsv"
    local f; for f in "$sec" "$ci" "$costs"; do [ -r "$f" ] || { echo "check_shard_balance: cannot read $f" >&2; return 2; }; done
    local jobs pins cimatrix fanin
    jobs="$(top_block "$sec" jobs | awk -f "$YAML_AWK")" || { echo "check_shard_balance: cannot read the jobs: block of $sec" >&2; return 2; }
    # matrix-pins is flow-map YAML outside the reader's subset: `- {shard: N, shards: M}` rows
    pins="$(top_block "$sec" matrix-pins | awk -v j="  $JOB:" '
        /^  [^ \t#]/ { keep = ($0 == j); next }
        keep && match($0, /shard: *[0-9]+, *shards: *[0-9]+/) { s = substr($0, RSTART, RLENGTH); gsub(/[^0-9,]/, "", s); print s }' |
        jq -R 'split(",") | {shard: (.[0]|tonumber), shards: (.[1]|tonumber)}' | jq -s .)"
    cimatrix="$(job_block "$ci" "$JOB" | awk '/^ +shard: *\[/ { sub(/^[^[]*\[/, ""); sub(/\].*/, ""); gsub(/ /, ""); print; exit }')"
    fanin="$(job_block "$ci" workspace-test)"
    jq -n -r --argjson jobs "$jobs" --argjson pins "$pins" --arg cimatrix "$cimatrix" --arg fanin "$fanin" \
          --rawfile costs "$costs" --argjson budget "$budget" --arg job "$JOB" '
      def nm: .name // .uses // "";
      ($pins | length) as $n | [range(1; $n + 1)] as $all
      | def shards: ((.if // "") | [scan("matrix\\.shard\\s*==\\s*([0-9]+)") | .[0] | tonumber]) as $m | if $m == [] then $all else $m end;
      def full: ((.if // "") | test("tier\\s*==\\s*\u0027(quick|reuse|none)\u0027") | not);
      ($jobs.jobs[$job].steps) as $steps
      | ($steps | map(nm)) as $names
      | ($costs | split("\n") | map(select(length > 0 and (startswith("#") | not)) | split("\t")
          | {key: .[0], value: (.[1] | split(",") | map(tonumber))}) | from_entries) as $cost
      # F1
      | [ (if ($pins | map(.shard) | sort) != $all or ($pins | any(.shards != $n)) then "F1 matrix-pins \($pins | tojson) is not shards 1..\($n) of \($n)" else empty end),
          (if $cimatrix != ($all | map(tostring) | join(",")) then "F1 ci.yml \($job) matrix shard [\($cimatrix)] != matrix-pins \($all | tojson)" else empty end),
          (("for n in " + ($all | map(tostring) | join(" "))) as $loop | if ($fanin | contains($loop)) then empty else "F1 the workspace-test verdict job does not loop `\($loop)` over the shards" end),
          ( ("lib.list.json", "explicit.list.txt") as $u
            | ([$steps[] | select((.run // "") | (contains("/" + $u) and contains(">"))) | shards[]] | unique) as $on
            | $fanin | [scan("sigma-shard-([0-9]+)/" + ($u | gsub("\\."; "\\."))) | .[0] | tonumber] | unique[]
            | select(. as $k | $on | index($k) | not)
            | "F1 the verdict job reads sigma-shard-\(.)/\($u), but no step on shard \(.) lists it" )
        ] as $f1
      # S1
      | [ range(0; $steps | length) as $i | $steps[$i] as $s
          | (($s.run // "") | gsub("if \\[ \"\\$SHARDS\" = 1 \\]; then[\\s\\S]*?\\n\\s*fi\\b"; "")) as $run
          | ($s | shards[]) as $k
          | ( ($run | [scan("--junit \"\\$sig/([\\w.-]+\\.junit\\.xml)\"") | .[0]][]) as $x
              | select([ $steps[0:$i][] | select(((.run // "") | contains("\"$RUNNER_TEMP/sigma/" + $x + "\"")) and (shards | index($k))) ] == [])
              | "S1 \u0027\($names[$i])\u0027 reads \($x) on shard \($k), where no earlier step wrote it" ),
            ( ($run | [scan("--step \"([^\"]+)\"") | .[0]][]) as $x
              | select([ range(0; $i) | select(($names[.] | startswith($x)) and ($steps[.] | shards | index($k))) ] == [])
              | "S1 \u0027\($names[$i])\u0027 reads the log of step \u0027\($x)\u0027 on shard \($k), where that step does not run" )
        ] as $s1
      # B1-B3
      | [ $steps[] | select(full) | {name: nm, on: shards} | . + {c: $cost[.name]} ] as $rows
      | [ $rows[] | select(.c == null and (.on | length) < $n) | "B2 \u0027\(.name)\u0027 runs on shard(s) \(.on | tojson) only and has no row in the cost table" ] as $b2
      | [ ($rows[] | select(.c != null and (.c | length) != 1 and ((.c | length) != $n or (.on | length) != $n))
          | "B3 \u0027\(.name)\u0027: \(.c | length) cost(s) for a step on \(.on | length) shard(s)"),
          ($cost | keys[] | select(. as $k | $names | index($k) | not) | "B3 cost row \u0027\(.)\u0027 names no step of \($job)") ] as $b3
      | [ $all[] as $k | [ $rows[] | select(.c != null and (.on | index($k)))
          | if (.c | length) == 1 then .c[0] elif (.c | length) == $n then .c[$k - 1] else 0 end ] | add // 0 ] as $tot
      | (($tot | max) - ($tot | min)) as $spread
      | [ if $spread > $budget then "B1 predicted shard spread \($spread) s > \($budget) s: move a once-only step off shard \(($tot | index($tot | max)) + 1)" else empty end ] as $b1
      | ($f1 + $s1 + $b2 + $b3 + $b1) as $bad
      | "predicted full-tier seconds per shard: " + ([range(0; $n) | "s\(. + 1)=\($tot[.])"] | join(", ")),
        "predicted spread: \($spread) s (budget \($budget) s)",
        ($bad[] | "FAIL " + .),
        "check_shard_balance: " + (if $bad == [] then "GREEN" else "RED" end)
    ' || return 2
}

# ── case table ───────────────────────────────────────────────────────────────
fixture() { # fixture <dir> <gpu_shard> <gpu_reader_shard> -- a 3-shard tree the check reads
    local d="$1" g="$2" r="$3"
    mkdir -p "$d/ci" "$d/.github/workflows"
    cat > "$d/ci/sections.yml" <<EOF
matrix-pins:
  $JOB:
    - {shard: 1, shards: 3}
    - {shard: 2, shards: 3}
    - {shard: 3, shards: 3}
jobs:
  $JOB:
    steps:
      - name: tier
      - name: lib
        if: steps.tier.outputs.tier == 'full'
        run: cp x "\$RUNNER_TEMP/sigma/lib.junit.xml"
      - name: gpu
        if: steps.tier.outputs.tier == 'full' && matrix.shard == $g
        run: cp x "\$RUNNER_TEMP/sigma/gpu.junit.xml"
      - name: Compute tests (x)
        if: steps.tier.outputs.tier == 'full' && matrix.shard == 1
      - name: sigma1
        if: steps.tier.outputs.tier == 'full' && matrix.shard == 1
        run: |
          list > /sigma/lib.list.json
          list > "\$sig/explicit.list.txt"
          if [ "\$SHARDS" = 1 ]; then
            sigma --junit "\$sig/lib.junit.xml"
          fi
          sigma --step "Compute tests"
      - name: sigma-gpu
        if: steps.tier.outputs.tier == 'full' && matrix.shard == $r
        run: sigma --junit "\$sig/gpu.junit.xml"
      - name: examples
        if: steps.tier.outputs.tier == 'full' && matrix.shard == 3
      - name: quick only
        if: steps.tier.outputs.tier == 'quick' && matrix.shard == 1
EOF
    cat > "$d/.github/workflows/ci.yml" <<'EOF'
jobs:
  workspace-test-shard:
    strategy:
      matrix:
        shard: [1, 2, 3]
    steps:
      - run: >-
          folded
  workspace-test:
    steps:
      - run: |
          for n in 1 2 3; do :; done
          x --list-json "$d/sigma-shard-1/lib.list.json"
          x --list "$d/sigma-shard-1/explicit.list.txt"
EOF
    printf 'lib\t100,100,100\tm\ngpu\t50\tm\nCompute tests (x)\t60\tm\nsigma1\t40\tm\nsigma-gpu\t5\tm\nexamples\t100\tm\n' > "$d/ci/shard-step-costs.tsv"
}

self_test() {
    local tmp n=0 bad=0 out got
    tmp="$(mktemp -d)" || exit 2
    # shellcheck disable=SC2064
    trap "rm -rf -- '$tmp'" EXIT
    row() { # row <label> <want: pass|B1|B2|B3|S1|F1> <budget> <mutation...>
        local label="$1" want="$2" budget="$3" d; shift 3
        n=$((n + 1)); d="$tmp/$n"
        fixture "$d" 2 2
        if [ "$#" -gt 0 ]; then (cd "$d" && "$@") || { echo "FAIL $label: mutation did not apply"; bad=$((bad + 1)); return; }; fi
        out="$(check "$d" "$budget" 2>&1)"
        got="$(printf '%s\n' "$out" | sed -n 's/^FAIL \([A-Z][0-9]\) .*/\1/p' | sort -u | tr '\n' ' ' | sed 's/ $//')"
        [ -n "$got" ] || got=pass
        if [ "$got" = "$want" ]; then echo "ok   $label: $want"
        else echo "FAIL $label: want $want, got $got"; printf '%s\n' "$out" | sed 's/^/     | /'; bad=$((bad + 1)); fi
    }
    sec=ci/sections.yml cy=.github/workflows/ci.yml tsv=ci/shard-step-costs.tsv
    row "balanced placement passes" pass 120
    row "GPU and its Σ read both on shard 1 is over budget" B1 40 \
        sed -i 's/matrix.shard == 2/matrix.shard == 1/' "$sec"
    row "a once-only step with no cost row" B2 120 sed -i '/^gpu\t/d' "$tsv"
    row "a cost row naming no step" B3 120 sh -c "printf 'gone\t9\tm\n' >> $tsv"
    row "a per-shard row with the wrong arity" B3 120 sed -i 's/^lib\t100,100,100/lib\t1,2/' "$tsv"
    row "GPU Σ read left on shard 1, GPU on shard 2" S1 120 \
        sed -i '/name: sigma-gpu/{n;s/matrix.shard == 2/matrix.shard == 1/}' "$sec"
    row "compute moved to shard 2, its --step read left on shard 1" S1 120 \
        sed -i '/name: Compute tests/{n;s/matrix.shard == 1/matrix.shard == 2/}' "$sec"
    row "a GPU read inside the single-shard block is not a 3-shard read" pass 120 \
        sed -i 's|^          fi$|            sigma --junit "$sig/gpu.junit.xml"\n          fi|' "$sec"
    row "the same GPU read outside the single-shard block" S1 120 \
        sed -i 's|^          sigma --step|          sigma --junit "$sig/gpu.junit.xml"\n          sigma --step|' "$sec"
    row "ci.yml matrix disagrees with matrix-pins" F1 120 sed -i 's/shard: \[1, 2, 3\]/shard: [1, 2]/' "$cy"
    row "the verdict job loops over two shards" F1 120 sed -i 's/for n in 1 2 3/for n in 1 2/' "$cy"
    row "universes listed on shard 2, read from sigma-shard-1" F1 120 \
        sed -i -e '/name: sigma1/{n;s/matrix.shard == 1/matrix.shard == 2/}' -e '/sigma --step/d' "$sec"
    echo "check_shard_balance self-test: $((n - bad))/$n rows ok"
    [ "$bad" -eq 0 ]
}

root="$(cd "$HERE/.." && pwd)" budget=120
case "${1:-}" in
    --help|-h) sed -n '2,25p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    --self-test) self_test; exit $? ;;
esac
while [ "$#" -gt 0 ]; do
    case "$1" in
        --root) root="${2:?--root needs a dir}"; shift 2 ;;
        --budget) budget="${2:?--budget needs seconds}"; shift 2 ;;
        *) echo "check_shard_balance: unknown argument $1" >&2; exit 2 ;;
    esac
done
out="$(check "$root" "$budget")"; rc=$?
printf '%s\n' "$out"
[ "$rc" -eq 2 ] && exit 2
case "$out" in *"check_shard_balance: GREEN"*) exit 0 ;; *) exit 1 ;; esac
