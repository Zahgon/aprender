#!/usr/bin/env python3
"""shard_balance.py -- where the once-only steps of workspace-test-shard sit (#5486).

The three workspace-test shards run in parallel, so the slowest one is the merge
queue's critical path. Steps gated `matrix.shard == N` run on one shard only; the
nextest lib and explicit-command steps are partitioned and cost a different amount
on each shard. This check sums the measured cost table (ci/shard-step-costs.tsv) per
shard, for the FULL tier (what the merge queue runs), and refuses:

  B1  a predicted spread (slowest - fastest shard) over the budget
  B2  a full-tier step gated to one shard with no measured cost (it cannot be placed)
  B3  a cost row that names no step (a stale table predicts nothing)
  S1  a Σ reader on a shard where its input was never written: a `--junit "$sig/X"`
      read where no earlier step on that shard copied X, or a `--step "NAME"` read
      where the step NAME does not run on that shard
  F1  the fan-in's 3-shard assumption broken: matrix-pins, the ci.yml matrix and the
      verdict job's `for n in ...` loop must name the same shards, and the shard the
      verdict job reads the universes from must be the shard that lists them

Reads text, builds nothing.

  python3 scripts/ci/shard_balance.py check [--budget SECONDS]
  python3 scripts/ci/shard_balance.py self-test
"""
from __future__ import annotations

import re
import sys

import yaml

JOB = "workspace-test-shard"
BUDGET = 120

NOT_FULL = re.compile(r"tier\s*==\s*'(quick|reuse|none)'")
ONE_SHARD = re.compile(r"matrix\.shard\s*==\s*(\d+)")
SINGLE_SHARD_BLOCK = re.compile(r'if \[ "\$SHARDS" = 1 \]; then.*?\n\s*fi\b', re.S)
JUNIT_WRITE = re.compile(r'"\$RUNNER_TEMP/sigma/([\w.-]+\.junit\.xml)"')
JUNIT_READ = re.compile(r'--junit "\$sig/([\w.-]+\.junit\.xml)"')
STEP_READ = re.compile(r'--step "([^"]+)"')


def shards_of(step: dict, n: int) -> set[int]:
    m = ONE_SHARD.findall(str(step.get("if") or ""))
    return {int(x) for x in m} if m else set(range(1, n + 1))


def full_tier(step: dict) -> bool:
    return not NOT_FULL.search(str(step.get("if") or ""))


def parse_costs(text: str) -> dict[str, list[int]]:
    costs = {}
    for line in text.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        name, secs, _src = line.split("\t")
        costs[name] = [int(x) for x in secs.split(",")]
    return costs


def check(sections: dict, ci: dict, costs: dict[str, list[int]], budget: int = BUDGET) -> tuple[list[str], list[str]]:
    """-> (failures, report lines)."""
    bad, out = [], []
    pins = sections["matrix-pins"][JOB]
    n = len(pins)
    want = list(range(1, n + 1))
    steps = sections["jobs"][JOB]["steps"]
    names = [s.get("name", s.get("uses", "")) for s in steps]

    # F1 -- the fan-in assumes these shards
    if sorted(p["shard"] for p in pins) != want or any(p["shards"] != n for p in pins):
        bad.append(f"F1 matrix-pins {pins} is not shards 1..{n} of {n}")
    ci_matrix = ci["jobs"][JOB]["strategy"]["matrix"]["shard"]
    if ci_matrix != want:
        bad.append(f"F1 ci.yml {JOB} matrix shard {ci_matrix} != matrix-pins {want}")
    fan_in = "\n".join(str(s.get("run", "")) for s in ci["jobs"]["workspace-test"]["steps"])
    loop = "for n in " + " ".join(map(str, want))
    if loop not in fan_in:
        bad.append(f"F1 the workspace-test verdict job does not loop `{loop}` over the shards")
    for universe in ("lib.list.json", "explicit.list.txt"):
        src = set(re.findall(r"sigma-shard-(\d+)/" + re.escape(universe), fan_in))
        writers = [i for i, s in enumerate(steps) if "/" + universe in str(s.get("run", "")) and ">" in str(s.get("run", ""))]
        on = set().union(*(shards_of(steps[i], n) for i in writers)) if writers else set()
        for k in src:
            if int(k) not in on:
                bad.append(f"F1 the verdict job reads sigma-shard-{k}/{universe}, but no step on shard {k} lists it")

    # S1 -- every Σ read has its input on the same shard, written before it
    for i, s in enumerate(steps):
        run = SINGLE_SHARD_BLOCK.sub("", str(s.get("run", "")))
        for k in shards_of(s, n):
            for x in JUNIT_READ.findall(run):
                if not any(x in JUNIT_WRITE.findall(str(steps[j].get("run", ""))) and k in shards_of(steps[j], n) for j in range(i)):
                    bad.append(f"S1 '{names[i]}' reads {x} on shard {k}, where no earlier step wrote it")
            for x in STEP_READ.findall(run):
                if not any(names[j].startswith(x) and k in shards_of(steps[j], n) for j in range(i)):
                    bad.append(f"S1 '{names[i]}' reads the log of step '{x}' on shard {k}, where that step does not run")

    # B1-B3 -- predicted full-tier time per shard
    total = [0] * n
    for i, s in enumerate(steps):
        if not full_tier(s):
            continue
        on = shards_of(s, n)
        c = costs.get(names[i])
        if c is None:
            if len(on) < n:
                bad.append(f"B2 '{names[i]}' runs on shard(s) {sorted(on)} only and has no row in the cost table")
            continue
        if len(c) == 1:
            for k in on:
                total[k - 1] += c[0]
        elif len(c) == n and len(on) == n:
            for k in on:
                total[k - 1] += c[k - 1]
        else:
            bad.append(f"B3 '{names[i]}': {len(c)} cost(s) for a step on {len(on)} shard(s)")
    for name in costs:
        if name not in names:
            bad.append(f"B3 cost row '{name}' names no step of {JOB}")
    spread = max(total) - min(total)
    out.append("predicted full-tier seconds per shard: " + ", ".join(f"s{k + 1}={t}" for k, t in enumerate(total)))
    out.append(f"predicted spread: {spread} s (budget {budget} s)")
    if spread > budget:
        bad.append(f"B1 predicted shard spread {spread} s > {budget} s: move a once-only step off shard {total.index(max(total)) + 1}")
    return bad, out


def _fixture(gpu_shard: int = 2, gpu_reader_shard: int = 2, extra: dict | None = None) -> tuple[dict, dict, dict]:
    steps = [
        {"name": "Decide the test tier"},
        {"name": "lib", "if": "steps.tier.outputs.tier == 'full'", "run": 'cp x "$RUNNER_TEMP/sigma/lib.junit.xml"'},
        {"name": "gpu", "if": f"steps.tier.outputs.tier == 'full' && matrix.shard == {gpu_shard}",
         "run": 'cp x "$RUNNER_TEMP/sigma/gpu.junit.xml"'},
        {"name": "Compute tests (x)", "if": "steps.tier.outputs.tier == 'full' && matrix.shard == 1"},
        {"name": "sigma1", "if": "steps.tier.outputs.tier == 'full' && matrix.shard == 1",
         "run": 'cargo nextest list > /sigma/lib.list.json\nlist > "$sig/explicit.list.txt"\n'
                'if [ "$SHARDS" = 1 ]; then\n  sigma --junit "$sig/lib.junit.xml"\nfi\nsigma --step "Compute tests"'},
        {"name": "sigma-gpu", "if": f"steps.tier.outputs.tier == 'full' && matrix.shard == {gpu_reader_shard}",
         "run": 'sigma --junit "$sig/gpu.junit.xml"'},
        {"name": "examples", "if": "steps.tier.outputs.tier == 'full' && matrix.shard == 3"},
        {"name": "quick only", "if": "steps.tier.outputs.tier == 'quick' && matrix.shard == 1"},
    ]
    if extra:
        steps.append(extra)
    sections = {"matrix-pins": {JOB: [{"shard": k, "shards": 3} for k in (1, 2, 3)]}, "jobs": {JOB: {"steps": steps}}}
    ci = {"jobs": {JOB: {"strategy": {"matrix": {"shard": [1, 2, 3]}}},
                   "workspace-test": {"steps": [{"run": 'for n in 1 2 3; do :; done\n'
                                                        '--list-json "$d/sigma-shard-1/lib.list.json"\n'
                                                        '--list "$d/sigma-shard-1/explicit.list.txt"'}]}}}
    costs = {"lib": [100, 100, 100], "gpu": [50], "Compute tests (x)": [60], "sigma1": [40], "sigma-gpu": [5], "examples": [100]}
    return sections, ci, costs


def self_test() -> int:
    fails = 0

    def row(label, sections, ci, costs, want_code, budget=BUDGET):
        nonlocal fails
        bad, _ = check(sections, ci, costs, budget)
        codes = sorted({b.split()[0] for b in bad})
        ok = codes == ([want_code] if want_code else [])
        fails += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {label}: want {want_code or 'pass'}, got {codes or 'pass'}")

    row("balanced placement passes", *_fixture(), None)
    row("GPU and its Σ read both on shard 1 is over budget", *_fixture(1, 1), "B1", 40)
    s, c, k = _fixture(); del k["gpu"]
    row("a once-only step with no cost row", s, c, k, "B2")
    s, c, k = _fixture(); k["gone"] = [9]
    row("a cost row naming no step", s, c, k, "B3")
    s, c, k = _fixture(); k["lib"] = [1, 2]
    row("a per-shard row with the wrong arity", s, c, k, "B3")
    row("GPU moved to shard 2, its Σ read left on shard 1", *_fixture(2, 1), "S1")
    s, c, k = _fixture(); s["jobs"][JOB]["steps"][3]["if"] = "steps.tier.outputs.tier == 'full' && matrix.shard == 2"
    row("compute moved to shard 2, its --step read left on shard 1", s, c, k, "S1")
    s, c, k = _fixture(); s["jobs"][JOB]["steps"][4]["run"] = s["jobs"][JOB]["steps"][4]["run"].replace("fi\n", 'sigma --junit "$sig/gpu.junit.xml"\nfi\n')
    row("a GPU read inside the single-shard block is not a 3-shard read", s, c, k, None)
    s, c, k = _fixture(); c["jobs"][JOB]["strategy"]["matrix"]["shard"] = [1, 2]
    row("ci.yml matrix disagrees with matrix-pins", s, c, k, "F1")
    s, c, k = _fixture(); c["jobs"]["workspace-test"]["steps"][0]["run"] = c["jobs"]["workspace-test"]["steps"][0]["run"].replace("1 2 3", "1 2")
    row("the verdict job loops over two shards", s, c, k, "F1")
    s, c, k = _fixture(); s["jobs"][JOB]["steps"][4]["if"] = "steps.tier.outputs.tier == 'full' && matrix.shard == 2"
    s["jobs"][JOB]["steps"][4]["run"] = "cargo nextest list > /sigma/lib.list.json\nlist > \"$sig/explicit.list.txt\""
    row("universes listed on shard 2, read from sigma-shard-1", s, c, k, "F1")
    row("a quick-tier-only step needs no cost row", *_fixture(), None)
    print(f"shard_balance self-test: {12 - fails}/12 rows ok")
    return 1 if fails else 0


def main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0 if argv else 2
    if argv[0] == "self-test":
        return self_test()
    if argv[0] != "check":
        print(f"shard_balance: unknown command {argv[0]}", file=sys.stderr)
        return 2
    budget = int(argv[argv.index("--budget") + 1]) if "--budget" in argv else BUDGET
    try:
        with open("ci/sections.yml") as f:
            sections = yaml.safe_load(f)
        with open(".github/workflows/ci.yml") as f:
            ci = yaml.safe_load(f)
        with open("ci/shard-step-costs.tsv") as f:
            costs = parse_costs(f.read())
    except (OSError, ValueError, yaml.YAMLError) as e:
        print(f"shard_balance: cannot read inputs: {e}", file=sys.stderr)
        return 2
    bad, out = check(sections, ci, costs, budget)
    print("\n".join(out))
    for b in bad:
        print(f"FAIL {b}")
    print("shard_balance: " + ("RED" if bad else "GREEN"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
