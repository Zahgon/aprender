#!/usr/bin/env bash
# ci_scope.sh -- decide whether a pull_request needs the fat CI jobs at all.
#
#   ci_scope.sh classify <draft:true|false>   < changed paths, one per line
#       prints `skip=1 reason=draft`, `skip=1 reason=docs-only` or `skip=0 reason=full`
#   ci_scope.sh --self-test                    case table + wiring checks on ci.yml
#
# WHY. CI demand: a draft PR and a PR that only touches inert documentation each cost
# a full x86-main + three workspace-test shards. Skipping happens INSIDE ci.yml, never
# with a workflow-level `paths-ignore`: a filtered-out workflow never reports, and the
# required checks (`ci / gate`, `gate`, `workspace-test`) would sit at "expected"
# forever. Here the verdict jobs still run and report success on a skipped scope.
#
# SCOPE. pull_request ONLY. merge_group and push never reach this script's verdict
# (the `scope` job is `if: github.event_name == 'pull_request'`), so a queue build or
# a main push is never skipped or cancelled by it. A draft gets a fresh run when it is
# marked ready (`ready_for_review` is in the trigger types).
#
# DOCS-ONLY is an ALLOWLIST of trees no test or gate reads, found by searching the
# repo for readers (docs/audits, docs/roadmaps, docs/specifications, docs/findings,
# docs/BEATS.md, README.md, CLAUDE.md are READ by gates and are therefore NOT here).
# Every changed path must match; any other path, an empty list or a `..` is `full`.
# Growing the list is a gate-content change: prove no reader first.
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 2
DOCS_ONLY_RE='^docs/(archive|postmortems|reports|qa-reports|images)/[^.][^[:space:]]*$'

classify() { # classify <draft> ; paths on stdin
    local draft=${1:-false} n=0 p
    if [ "$draft" = true ]; then cat >/dev/null; echo "skip=1 reason=draft"; return 0; fi
    while IFS= read -r p; do
        [ -n "$p" ] || continue
        n=$((n + 1))
        case "$p" in *..*) echo "skip=0 reason=full"; return 0 ;; esac
        printf '%s\n' "$p" | grep -Eq "$DOCS_ONLY_RE" || { echo "skip=0 reason=full"; return 0; }
    done
    [ "$n" -gt 0 ] && echo "skip=1 reason=docs-only" || echo "skip=0 reason=full"
}

self_test() {
    local fail=0 got
    t() { # t NAME WANT DRAFT PATHS...
        local name=$1 want=$2 draft=$3; shift 3
        got=$(printf '%s\n' "$@" | classify "$draft")
        [ "$got" = "$want" ] || { echo "FAIL $name: want [$want] got [$got]"; fail=1; }
    }
    t "draft skips" "skip=1 reason=draft" true crates/a/src/lib.rs
    t "draft, no paths" "skip=1 reason=draft" true
    t "docs archive only" "skip=1 reason=docs-only" false docs/archive/a.md docs/images/x.png
    t "postmortem only" "skip=1 reason=docs-only" false docs/postmortems/2026-10-19.md
    t "code" "skip=0 reason=full" false crates/a/src/lib.rs
    t "docs + code" "skip=0 reason=full" false docs/archive/a.md crates/a/src/lib.rs
    t "README is gated" "skip=0 reason=full" false README.md
    t "CLAUDE.md is gated" "skip=0 reason=full" false CLAUDE.md
    t "BEATS is gated" "skip=0 reason=full" false docs/BEATS.md
    t "roadmap entries are gated" "skip=0 reason=full" false docs/roadmaps/entries/PMAT-1.yaml
    t "specifications are gated" "skip=0 reason=full" false docs/specifications/a.md
    t "audits are read" "skip=0 reason=full" false docs/audits/surface_audit.csv
    t "findings ledger is gated" "skip=0 reason=full" false docs/findings/w1.jsonl
    t "workflow file" "skip=0 reason=full" false .github/workflows/ci.yml
    t "contracts" "skip=0 reason=full" false contracts/a-v1.yaml
    t "dot-dot escape" "skip=0 reason=full" false docs/archive/../../Cargo.toml
    t "hidden file under docs" "skip=0 reason=full" false docs/archive/.hidden
    t "prefix lookalike" "skip=0 reason=full" false docs/archive-x/a.md
    t "empty diff is full" "skip=0 reason=full" false
    local wf="$REPO_ROOT/.github/workflows/ci.yml" py
    py=$(python3 - "$wf" <<'PY'
import sys, yaml
w = yaml.safe_load(open(sys.argv[1]))
on = w.get(True, w.get("on"))
bad = []
pr = on.get("pull_request") or {}
if "paths-ignore" in pr or "paths" in pr: bad.append("pull_request has a paths filter (stuck required checks)")
if "ready_for_review" not in (pr.get("types") or []): bad.append("pull_request types lack ready_for_review")
for k in ("push", "merge_group"):
    v = on.get(k) or {}
    if "paths-ignore" in v or "paths" in v: bad.append(k + " has a paths filter")
j = w["jobs"]
sc = j.get("scope")
if not sc or sc.get("if") != "${{ github.event_name == 'pull_request' }}": bad.append("scope job missing or not pull_request-only")
for name in ("x86-main", "workspace-test-shard"):
    if "scope" not in (j[name].get("needs") or []) or "needs.scope.outputs.skip != '1'" not in str(j[name].get("if")): bad.append(name + " does not honour scope")
for name in ("ci-gate", "gate", "workspace-test"):
    if "scope" not in (j[name].get("needs") or []): bad.append(name + " does not need scope")
    if name != "workspace-test" and "needs.scope.outputs.skip" not in str(j[name]): bad.append(name + " does not read scope")
if "needs.scope.outputs.skip" not in str(j["workspace-test"]): bad.append("workspace-test does not read scope")
c = w.get("concurrency") or {}
if "pull_request" not in str(c.get("cancel-in-progress")): bad.append("concurrency no longer cancels superseded PR runs")
print("\n".join(bad))
PY
    ) || { echo "FAIL wiring: cannot read ci.yml"; fail=1; }
    [ -z "$py" ] || { printf 'FAIL wiring: %s\n' "$py"; fail=1; }
    [ "$fail" -eq 0 ] && echo "ci_scope: self-test ok"
    return "$fail"
}

case "${1:-}" in
    classify) shift; classify "${1:-false}" ;;
    --self-test) self_test ;;
    *) echo "usage: $0 classify <draft> | --self-test" >&2; exit 2 ;;
esac
