#!/usr/bin/env bash
# crux_ollama_parity.sh: apr serve vs ollama on the SAME GGUF, same host (#4220).
#
# One run = one model file. For each engine it streams the same four prompts
# (warmup, ~850 / ~4k / ~32k-token prefill probes, then 3 x 128-token decode)
# and writes one JSON row per request to <out>/{apr,ollama}.jsonl, plus
# provenance.txt and serve.trace (apr's GPU-path trace lines).
#
# Client timing, both engines: TTFT = request sent -> first content chunk;
# decode = (chunks - 1) / (t_last - t_first). ollama rows also carry its own
# prompt_eval/eval rates; apr prefill rates come from serve.log's
# `[qwen35] batched prefill` lines. Every ollama request sends keep_alive 0
# (the card is empty after each load), so ollama's client TTFT includes a model
# load and its server prefill rate is the fair number.
#
# Usage: APR=<apr binary> bash scripts/crux_ollama_parity.sh <model.gguf> <out-dir> [ctx]
# Run it under the GPU lock: flock /tmp/apr-gpu.lock bash scripts/crux_ollama_parity.sh ...
# --self-test runs the PARSE case table (no GPU, no server).
# Exit: 0 both engines answered every probe; 1 a probe failed; 2 ENV.
set -uo pipefail
export LC_ALL=C  # EPOCHREALTIME uses the locale decimal separator
case "${1:-}" in -h|--help) sed -n 2,20p "$0"; exit 0 ;; esac
PROG=crux_ollama_parity

# PARSE: stamped stream ("<epoch>\t<line>" per line) -> one row. apr sends OpenAI SSE (`data: {...}`,
# usage on the last chunk); ollama sends NDJSON with server timings on the `done` line.
PARSE='
    [inputs | (index("\t")) as $i | {t:(.[:$i]|tonumber), l:.[$i+1:]}
       | .l |= sub("^data: ?";"") | select(.l != "" and .l != "[DONE]")
       | .j = (try (.l|fromjson) catch null) | select(.j != null)] as $ev
    | [ $ev[] | {t, c:((.j.message.content // ([.j.choices[]?.delta.content // empty] | join(""))) // "")}
        | select(.c != "") ] as $ch
    | ([ $ev[] | .j.usage // empty ] | last) as $u
    | ([ $ev[] | select(.j.done == true) | .j ] | last // {}) as $f
    | ($t0|tonumber) as $t0 | ($ch|length) as $n
    | if $n == 0 then {label:$l, error:true} else
      {label:$l, chunks:$n, usage:$u, ttft_ms:(((($ch[0].t)-$t0)*1e4|round)/10),
       e2e_s:(((($e|tonumber)-$t0)*1e3|round)/1e3), text:([$ch[].c]|join(""))}
      + (if $n > 1 then {decode_tok_s:(((($n-1)/($ch[-1].t-$ch[0].t))*10|round)/10)} else {} end)
      + (if ($f.prompt_eval_duration // 0) > 0 then {prompt_tokens:$f.prompt_eval_count,
           srv_prefill_tok_s:((($f.prompt_eval_count/$f.prompt_eval_duration*1e9)*10|round)/10),
           load_ms:((($f.load_duration // 0)/1e5|round)/10)} else {} end)
      + (if ($f.eval_duration // 0) > 0 and ($f.eval_count // 0) > 1 then
           {srv_decode_tok_s:((($f.eval_count/$f.eval_duration*1e9)*10|round)/10)} else {} end)
      end'

self_test() {  # case table for PARSE: each row = fixture, expected jq predicate on the parsed row
  local d f rc=0; d=$(mktemp -d) || return 2
  printf '100.000\tdata: {"choices":[{"delta":{"role":"assistant"}}]}\n100.250\tdata: {"choices":[{"delta":{"content":"The"}}]}\n100.260\t\n100.300\tdata: {"choices":[{"delta":{"content":" wind"}}]}\n100.350\tdata: {"choices":[],"usage":{"prompt_tokens":30,"completion_tokens":2}}\n100.351\tdata: [DONE]\n' > "$d/apr"
  printf '200.000\t{"message":{"content":"A"},"done":false}\n200.100\t{"message":{"content":"B"},"done":false}\n200.300\t{"message":{"content":""},"done":true,"prompt_eval_count":800,"prompt_eval_duration":100000000,"eval_count":2,"eval_duration":20000000,"load_duration":2000000000}\n' > "$d/ollama"
  : > "$d/empty"
  printf '1.0\tdata: {"choices":[{"delta":{"role":"assistant"}}]}\n1.1\tdata: [DONE]\n' > "$d/nocontent"
  while IFS='|' read -r f t0 pred; do
    if jq -nRc --arg l "$f" --arg t0 "$t0" --arg e 1000 "$PARSE" "$d/$f" | jq -e "$pred" > /dev/null; then
      printf 'ok   %s\n' "$f"
    else
      printf 'FAIL %s: %s\n' "$f" "$pred"; rc=1
    fi
  done <<'ROWS'
apr|99.9|.chunks == 2 and .ttft_ms == 350 and .decode_tok_s == 20 and .text == "The wind" and .usage.prompt_tokens == 30
ollama|199.0|.chunks == 2 and .ttft_ms == 1000 and .decode_tok_s == 10 and .srv_prefill_tok_s == 8000 and .srv_decode_tok_s == 100 and .load_ms == 2000
empty|0|.error == true
nocontent|0|.error == true
ROWS
  rm -r "${d:?}"
  return "$rc"
}
if [ "${1:-}" = --self-test ]; then self_test; exit $?; fi

G="${1:?model.gguf}"; O="${2:?out-dir}"; CTX="${3:-40960}"
for t in curl jq ollama sha256sum; do
  command -v "$t" >/dev/null 2>&1 || { printf '%s: ENV - %s is missing\n' "$PROG" "$t" >&2; exit 2; }
done
[ -x "${APR:-}" ] || { printf '%s: ENV - set APR to the apr binary under test\n' "$PROG" >&2; exit 2; }
[ -f "$G" ] || { printf '%s: ENV - %s not found\n' "$PROG" "$G" >&2; exit 2; }
mkdir -p "$O" || exit 2
unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY
export no_proxy="127.0.0.1,localhost" NO_PROXY="127.0.0.1,localhost"
PORT="${CRUX_PORT:-18650}"
OM="crux4220-$(basename "$G" .gguf | tr 'A-Z.' 'a-z-')"
FAILS=0

{
  printf 'date: %s\n' "$(date -u +%FT%TZ)"
  printf 'apr: %s sha256=%s\n' "$("$APR" --version 2>&1 | head -n 1)" "$(sha256sum "$APR" | cut -c1-64)"
  printf 'gguf: %s sha256=%s\n' "$G" "$(sha256sum "$G" | cut -c1-64)"
  printf 'ollama: %s\n' "$(ollama --version 2>&1 | tail -n 1)"
  printf 'ctx: %s\n' "$CTX"
  nvidia-smi --query-gpu=name,driver_version,memory.used --format=csv,noheader 2>&1
  printf 'foreign compute apps:\n'; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>&1
  printf 'load: %s\n' "$(cut -d' ' -f1-3 /proc/loadavg)"
} > "$O/provenance.txt"

para="The history of numerical computing is a history of trading precision for speed, and every generation of hardware has renegotiated that trade in its own terms. "
# Each prompt opens with its own name so neither engine can serve it from a cached prefix of
# another (apr reused the p850 prefix inside p4k when they shared their first tokens).
mk_prompt() {  # $1=name $2=reps
  local i s="Document $1. "
  for ((i = 0; i < $2; i++)); do s+="$para"; done
  printf '%s\nSummarize the above in one sentence.' "$s" > "$O/$1.txt"
}
mk_prompt p850 29; mk_prompt p4k 140; mk_prompt p32k 1128
printf 'Write a long, detailed story about a lighthouse keeper. Keep going for many paragraphs.' > "$O/decode.txt"

# probe <engine apr|ollama> <prompt-file> <max-tokens> <label>: one streamed request -> one JSON row on stdout.
# The body goes through a file: the 32k prompt is past the argv limit (E2BIG).
# The read loop only stamps raw lines (no forks): parsing per line would cost more than one 0.8B decode step
# and the rate would measure this loop, not the server. One jq pass parses the stamped stream afterwards.
probe() {
  local eng=$1 pf=$2 n=$3 label=$4 url t0 now line row raw="$O/.raw-$1-$4" body="$O/.body-$1-$4.json"
  if [ "$eng" = apr ]; then
    url="http://127.0.0.1:$PORT/v1/chat/completions"
    jq -nc --rawfile p "$pf" --argjson n "$n" \
      '{model:"default",stream:true,temperature:0,max_tokens:$n,stream_options:{include_usage:true},messages:[{role:"user",content:$p}]}' > "$body" || return 1
  else
    url="http://127.0.0.1:11434/api/chat"
    jq -nc --rawfile p "$pf" --argjson n "$n" --arg m "$OM" --argjson ctx "$CTX" \
      '{model:$m,stream:true,think:false,keep_alive:0,options:{temperature:0,num_predict:$n,num_ctx:$ctx},messages:[{role:"user",content:$p}]}' > "$body" || return 1
  fi
  : > "$raw"
  t0=$EPOCHREALTIME
  while IFS= read -r line; do
    printf '%s\t%s\n' "$EPOCHREALTIME" "$line" >> "$raw"
  done < <(curl -sN --max-time 1800 -H 'Content-Type: application/json' --data-binary "@$body" "$url")
  now=$EPOCHREALTIME
  row=$(jq -nRc --arg l "$label" --arg t0 "$t0" --arg e "$now" "$PARSE" "$raw") || row="{\"label\":\"$label\",\"error\":true}"
  printf '%s\n' "$row"
  [[ $row != *'"error":true'* ]]
}
probe_all() {  # $1=engine $2=out.jsonl
  local p i
  probe "$1" "$O/decode.txt" 8 warmup >> "$2" || FAILS=$((FAILS + 1))
  for p in p850 p4k p32k; do probe "$1" "$O/$p.txt" 1 "ttft-$p" >> "$2" || FAILS=$((FAILS + 1)); done
  for i in 1 2 3; do probe "$1" "$O/decode.txt" 128 "decode-$i" >> "$2" || FAILS=$((FAILS + 1)); done
}

: > "$O/ollama.jsonl"; : > "$O/apr.jsonl"
# 1. ollama on the SAME file through a Modelfile FROM.
printf 'FROM %s\nPARAMETER num_ctx %s\nPARAMETER temperature 0\n' "$G" "$CTX" > "$O/Modelfile"
if ollama create "$OM" -f "$O/Modelfile" > "$O/ollama-create.log" 2>&1; then
  probe_all ollama "$O/ollama.jsonl"
  ollama stop "$OM" > /dev/null 2>&1 || true
  ollama rm "$OM" > /dev/null 2>&1 || true
else
  printf '%s: ollama create failed, see %s\n' "$PROG" "$O/ollama-create.log" >&2; FAILS=$((FAILS + 1))
fi
sleep 3
# 2. apr serve on the same file, all layers on the GPU, with the trace on.
"$APR" serve run "$G" --gpu-layers all --port "$PORT" --host 127.0.0.1 --context-length "$CTX" --trace > "$O/serve.log" 2>&1 &
SP=$!; printf '%s\n' "$SP" > "$O/serve.pid"
trap 'kill "$SP" 2>/dev/null || true' EXIT
up=0
for _ in $(seq 1 600); do
  curl -sf "http://127.0.0.1:$PORT/health" > /dev/null && { up=1; break; }
  kill -0 "$SP" 2>/dev/null || break
  sleep 1
done
if [ "$up" = 1 ]; then
  probe_all apr "$O/apr.jsonl"
else
  printf '%s: apr serve never became healthy, see %s\n' "$PROG" "$O/serve.log" >&2; FAILS=$((FAILS + 1))
fi
kill "$SP" 2>/dev/null || true; wait "$SP" 2>/dev/null || true; trap - EXIT
grep -E 'Backend|gpu-layers|batched prefill|fell_back|per-token|decode graph' "$O/serve.log" \
  | sed -E 's/[0-9]+(\.[0-9]+)? ?ms//g; s/[0-9]+ tokens in//; s/\([0-9.]+ tok\/s, //' \
  | sort | uniq -c | sort -rn | head -n 20 > "$O/serve.trace" || true
printf '%s: %s -> %s (%s failed probes)\n' "$PROG" "$(basename "$G")" "$O" "$FAILS"
[ "$FAILS" = 0 ]
