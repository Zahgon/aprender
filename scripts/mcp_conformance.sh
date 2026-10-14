#!/usr/bin/env bash
# R-7 (#5348): `apr mcp` against the MCP inspector and the official MCP
# conformance suite. GREEN only when every check below ran and passed.
#
#   scripts/mcp_conformance.sh <apr-binary> [npm-dir]
#
# Tools are external validators, pinned exactly and installed with
# --ignore-scripts into npm-dir (default ~/.cache/apr-mcp-conformance-npm):
#   @modelcontextprotocol/inspector   2.11.0  (inspector CLI, stdio)
#   @modelcontextprotocol/conformance 0.1.16  (server scenarios, HTTP)
#   supergateway                      3.4.3   (stdio -> streamable HTTP bridge)
#   ajv                               8.20.0  (JSON Schema 2020-12 meta-validation)
#
# Neither upstream tool validates a tool's inputSchema as JSON Schema: both
# check only the root (`type: "object"`). A property typed "strng" passes all
# of them (measured), so check 3 meta-validates every served inputSchema.
#
# Scenario set. `apr mcp` declares only the `tools` capability and none of
# the suite's fixture tools (test_simple_text, test_image_content, ...). A
# fixture scenario against it is either a FAILURE for a feature it does not
# claim, or a vacuous SUCCESS: tools-call-simple-text passes on the text of
# "Unknown tool: test_simple_text". Neither says anything about apr, so only
# the three scenarios that exercise apr's own surface run here.
# dns-rebinding-protection judges the HTTP bridge, not apr, and is out too.
set -euo pipefail

APR="${1:?usage: mcp_conformance.sh <apr-binary> [npm-dir]}"
NPM_DIR="${2:-${XDG_CACHE_HOME:-$HOME/.cache}/apr-mcp-conformance-npm}"
PORT="${MCP_CONFORMANCE_PORT:-18471}"
SCENARIOS=(server-initialize ping tools-list)

[ -x "$APR" ] || { echo "FAIL: not executable: $APR" >&2; exit 2; }
command -v node > /dev/null || { echo "FAIL: node not on PATH" >&2; exit 2; }
command -v jq > /dev/null || { echo "FAIL: jq not on PATH" >&2; exit 2; }

BIN="$NPM_DIR/node_modules/.bin"
if [ ! -x "$BIN/conformance" ] || [ ! -x "$BIN/mcp-inspector" ] || [ ! -x "$BIN/supergateway" ] \
    || [ ! -f "$NPM_DIR/node_modules/ajv/dist/2020.js" ]; then
    mkdir -p "$NPM_DIR"
    npm install --prefix "$NPM_DIR" --ignore-scripts --no-audit --no-fund \
        @modelcontextprotocol/inspector@2.11.0 \
        @modelcontextprotocol/conformance@0.1.16 \
        supergateway@3.4.3 ajv@8.20.0 > "$NPM_DIR/install.log" 2>&1 \
        || { echo "FAIL: npm install (see $NPM_DIR/install.log)" >&2; exit 2; }
fi

OUT="$(mktemp -d)"
GW_PID=""
cleanup() { if [ -n "$GW_PID" ]; then kill "$GW_PID" 2> /dev/null || true; fi; }
trap cleanup EXIT

echo "apr: $APR"
echo "sha256: $(sha256sum "$APR" | cut -d' ' -f1)"
echo "out: $OUT"
fails=0

# 1. Inspector CLI over stdio: tools/list must parse in the inspector's SDK.
# The inspector DROPS a tool whose schema its SDK rejects and still exits 0
# (measured: 9 tools served, 8 listed), so its count must equal the raw count
# read straight off `apr mcp`, never just be non-zero.
printf '%s\n' \
    '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"mcp_conformance","version":"0"}}}' \
    '{"jsonrpc":"2.0","method":"notifications/initialized"}' \
    '{"jsonrpc":"2.0","id":2,"method":"tools/list"}' \
    | timeout 30 "$APR" mcp 2> /dev/null | jq -c 'select(.id == 2) | .result.tools' > "$OUT/raw_tools.json" || true
raw_n="$(jq 'length' "$OUT/raw_tools.json" 2> /dev/null || true)"
if "$BIN/mcp-inspector" --cli "$APR" mcp --method tools/list > "$OUT/tools_list.json" 2> "$OUT/tools_list.err"; then
    n="$(jq '.tools | length' "$OUT/tools_list.json")"
    if [ -n "$raw_n" ] && [ "$raw_n" -ge 1 ] && [ "$n" -eq "$raw_n" ]; then
        echo "PASS inspector tools/list: $n of $raw_n tools"
    else
        echo "FAIL inspector tools/list: inspector kept ${n} of ${raw_n:-?} served tools"; fails=$((fails + 1))
    fi
else
    echo "FAIL inspector tools/list: $(head -c 300 "$OUT/tools_list.err")"; fails=$((fails + 1))
fi

# 2. Inspector CLI over stdio: a real tool call, not a fixture name.
if "$BIN/mcp-inspector" --cli "$APR" mcp --method tools/call --tool-name apr.version \
    > "$OUT/version.json" 2> "$OUT/version.err" \
    && [ "$(jq -r '.isError // false' "$OUT/version.json")" = "false" ] \
    && jq -er '.content[0].text | fromjson | .version' "$OUT/version.json" > /dev/null; then
    echo "PASS inspector tools/call apr.version"
else
    echo "FAIL inspector tools/call apr.version: $(head -c 300 "$OUT/version.json" "$OUT/version.err")"; fails=$((fails + 1))
fi

# 3. Every served inputSchema is a valid JSON Schema 2020-12 document (the MCP
# default dialect). Not an inspector check: see the header for why it exists.
if [ -n "$raw_n" ] && [ "$raw_n" -ge 1 ] && AJV_DIR="$NPM_DIR/node_modules/ajv" node -e '
    const Ajv = require(process.env.AJV_DIR + "/dist/2020.js").default;
    const ajv = new Ajv({ strict: true });
    const tools = JSON.parse(require("fs").readFileSync(process.argv[1], "utf8"));
    var bad = 0;
    for (const t of tools) {
        if (!ajv.validateSchema(t.inputSchema)) {
            bad++;
            console.log(t.name + ": " + ajv.errorsText(ajv.errors));
        }
    }
    process.exit(bad === 0 ? 0 : 1);
' "$OUT/raw_tools.json" > "$OUT/schema_check.log" 2>&1; then
    echo "PASS inputSchema meta-validation: $raw_n of $raw_n tools"
else
    echo "FAIL inputSchema meta-validation: $(head -c 300 "$OUT/schema_check.log")"; fails=$((fails + 1))
fi

# 4. Conformance suite over the bridge.
"$BIN/supergateway" --stdio "$APR mcp" --outputTransport streamableHttp \
    --port "$PORT" --logLevel none > "$OUT/gateway.log" 2>&1 &
GW_PID=$!
up=0
for _ in 1 2 3 4 5 6 7 8 9 10; do
    if (exec 3<> "/dev/tcp/127.0.0.1/$PORT") 2> /dev/null; then up=1; break; fi
    sleep 1
done
[ "$up" -eq 1 ] || { echo "FAIL: bridge did not listen on $PORT"; exit 1; }

for s in "${SCENARIOS[@]}"; do
    "$BIN/conformance" server --url "http://127.0.0.1:$PORT/mcp" --scenario "$s" \
        -o "$OUT/conf" > "$OUT/$s.log" 2>&1 || true
    checks="$(find "$OUT/conf" -path "*server-$s-*" -name checks.json | head -n 1)"
    if [ -z "$checks" ]; then
        echo "FAIL conformance $s: no checks.json"; fails=$((fails + 1)); continue
    fi
    total="$(jq 'length' "$checks")"
    ok="$(jq '[.[] | select(.status == "SUCCESS")] | length' "$checks")"
    if [ "$total" -ge 1 ] && [ "$ok" -eq "$total" ]; then
        echo "PASS conformance $s: $ok/$total"
    else
        echo "FAIL conformance $s: $ok/$total $(jq -r '[.[] | select(.status != "SUCCESS") | .errorMessage] | join("; ")' "$checks" | head -c 300)"
        fails=$((fails + 1))
    fi
done

if [ "$fails" -eq 0 ]; then
    echo "RESULT: GREEN (2 inspector checks + schema meta-validation + ${#SCENARIOS[@]} conformance scenarios)"
else
    echo "RESULT: RED ($fails failed)"
    exit 1
fi
