#!/usr/bin/env bash
# Poka-yoke: every inline self-hosted CI job MUST pin a discriminating runner
# label, so a job can never silently land on the wrong self-hosted runner.
#
# Root cause it guards (aprender#2269): GitHub auto-assigns self-hosted/Linux/X64
# as DEFAULT labels that cannot be removed, so a GPU dev runner (lambda-4090)
# matched the bare `[self-hosted, X64, Linux]` selector and the mutants job ran
# there — a box with no sovereign-ci registry — failing non-deterministically.
#
# Rule: any `runs-on:` that names `self-hosted` must ALSO name one of:
#   - clean-room  (the provisioned sovereign-ci pool: registry + cached image)
#   - a GPU label: cuda | gpu | rtx4090 | ada | blackwell | gb10
#   - a macOS label: apple-silicon | m4 | mini-builder
# Reusable-workflow jobs (`uses:`) have no `runs-on` and are naturally exempt.
# GitHub-hosted jobs (ubuntu-latest, …) don't name self-hosted and are exempt.
set -euo pipefail
cd "$(dirname "$0")/.."

# perf-solo (PERF-013, infra#338): one box, intel-clean-room-16, deliberately WITHOUT clean-room so
# general pool work cannot queue in front of a speed measurement. A one-box label discriminates.
# mini-builder (#4292, infra#1042): the mini's dedicated repo-level release runner, labelled
# self-hosted,macOS,ARM64,mini-builder and "used only via mini-builder so it never joins the
# general pool" (operator, via the cop). Also a one-box label.
# review-light (#4729): one runner on intel-clean-room-15, same unit and isolation as the
# clean-room pool, deliberately WITHOUT clean-room so build jobs cannot queue in front of
# pr-review-quorum `present` (a seconds-long check). A one-box label, like perf-solo.
DISCRIM='clean-room|cuda|gpu|rtx4090|ada|blackwell|gb10|apple-silicon|m4|perf-solo|mini-builder|review-light'

# #5247: match LABEL TOKENS, never the raw line. The old `grep -qE "$DISCRIM" <<< "$sel"` read
# the trailing YAML comment and matched substrings, so `[self-hosted, Linux, X64, mutants] #
# never the ARM64 gpu runner` passed on the word "gpu" in its comment, and `not-gpu` passed too.
# label_tokens: drop the comment (a `#` after whitespace), split on list/expression punctuation,
# one lower-cased token per line (GitHub matches labels case-insensitively).
label_tokens() {
  sed -E 's/[[:space:]]+#.*$//; s/[][{}(),"'"'"'[:space:]]+/\n/g' <<< "$1" | tr '[:upper:]' '[:lower:]'
}
# verdict <runs-on line> -> SKIP (not self-hosted), PASS or FAIL
verdict() {
  local toks
  toks=$(label_tokens "$1")
  grep -qx 'self-hosted' <<< "$toks" || { echo SKIP; return; }
  if grep -qxE "$DISCRIM" <<< "$toks"; then echo PASS; else echo FAIL; fi
}

# Case table, run on every invocation: a matcher that regresses fails the guard before it
# judges a single workflow. The first row is the #5247 defect and must stay FAIL.
cases_bad=0
while IFS=$'\t' read -r want sel; do
  [ -n "$want" ] || continue
  got=$(verdict "$sel")
  if [ "$got" != "$want" ]; then
    echo "::error::check_runner_labels case table: want ${want}, got ${got}: ${sel}"
    cases_bad=1
  fi
done <<'CASES'
FAIL	    runs-on: [self-hosted, Linux, X64, mutants] # never the ARM64 gpu runner
FAIL	    runs-on: [self-hosted, Linux, X64, mutants]   # clean-room is not this pool
FAIL	    runs-on: [self-hosted, Linux, X64, not-gpu]
FAIL	    runs-on: [self-hosted, Linux, X64, xm4y]
FAIL	    runs-on: [self-hosted, X64, Linux]
FAIL	    runs-on: [self-hosted, Linux, X64, clean-room-2]
PASS	    runs-on: [self-hosted, Linux, X64, clean-room]
PASS	    runs-on: [Self-Hosted, Linux, GPU]
PASS	    runs-on: [self-hosted, Linux, ARM64, gpu, docker] # infra#1468: off the main CI pool
PASS	    runs-on: [self-hosted, X64, Linux, build, perf-solo]   # perf-solo = one box
PASS	    runs-on: [self-hosted, review-light]
PASS	    runs-on: [self-hosted, macOS, ARM64, apple-silicon, "${{ inputs.runner_label }}"]
PASS	    runs-on: ${{ github.event.schedule == '*/10 0-5,20-23 * * *' && fromJSON('["self-hosted", "X64", "Linux", "clean-room"]') || fromJSON('["self-hosted", "Linux", "clean-room", "ARM64"]') }}
SKIP	    runs-on: ubuntu-latest # not self-hosted
CASES
if [ "$cases_bad" -ne 0 ]; then
  echo "FAIL: the label matcher disagrees with its case table; fix label_tokens/verdict first." >&2
  exit 1
fi
[ "${1:-}" = "--self-test" ] && { echo "✓ check_runner_labels: case table green"; exit 0; }

fail=0
while IFS=: read -r file line sel; do
  # Only inline self-hosted selectors.
  [ "$(verdict "$sel")" = FAIL ] || continue
  echo "::error file=${file},line=${line}::self-hosted job lacks a discriminating runner label (need clean-room or a GPU/macOS label): ${sel# }"
  fail=1
done < <(grep -rnE '^[[:space:]]*runs-on:.*self-hosted' .github/workflows/*.yml 2>/dev/null)

if [ "$fail" -ne 0 ]; then
  echo "FAIL: pin each self-hosted job to a provisioned pool (clean-room) or a GPU/macOS label." >&2
  echo "      Bare [self-hosted, X64, Linux] can land on ANY self-hosted runner (incl. GPU dev boxes)." >&2
  exit 1
fi
echo "✓ check_runner_labels: every self-hosted job pins a discriminating runner label"
