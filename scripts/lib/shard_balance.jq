# shard_balance.jq -- the rules of scripts/check_shard_balance.sh (#5486): B1-B3, S1, F1.
# Inputs (jq -n): $jobs (ci/sections.yml jobs as JSON), $pins (matrix-pins rows),
# $cimatrix (ci.yml shard list, "1,2,3"), $fanin (the workspace-test job text),
# $costs (ci/shard-step-costs.tsv), $budget (seconds), $job (the sharded job name).
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
