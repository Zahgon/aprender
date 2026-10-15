#!/usr/bin/env python3
"""Σ-executed invariant (#4433): the tests CI ran are exactly the tests it owes.

Operator ruling 2026-09-25: "Same test set as today (Σ executed unchanged)"; the
cop's condition on the ci.yml PR: "The test-id set across the 5 jobs must equal
today's set (by id, not by count), and a mutation that drops one test must turn
it RED."

Every mode compares two sets of test IDS and fails on any difference, in either
direction. The UNIVERSE is listed with the filter written here, in the Σ step --
never read back from the step that ran the tests -- so a narrowed run step (an
added --exclude, a --partition, a dropped fragment) cannot narrow its own yardstick.

  nextest  --junit J --list-json L    ids = "<binary-id> <test>"
  cargo    --log F --step S --list L --ignored I
                                      a `cargo test` step's "test X ... ok|FAILED"
                                      lines vs `-- --list` minus `-- --list --ignored`
  explicit --log F --step S --list L  the explicit-command runner's
                                      "::group::[i/N] <cmd>" lines vs its --list
  --self-test                         the case table: each mode must go RED on a
                                      dropped id, an extra id and an empty universe

--log/--step read the fat driver's section log (FAT_SECTION_LOG), sliced to the
step whose "##[step N] <name>" marker contains S.
"""
from __future__ import annotations

import json
import re
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


def compare(kind: str, executed: set, universe: set) -> int:
    missing = sorted(universe - executed)
    extra = sorted(executed - universe)
    print(f"Σ {kind}: universe={len(universe)} executed={len(executed)} "
          f"missing={len(missing)} extra={len(extra)}")
    if not universe:
        print(f"::error::Σ {kind}: the universe is EMPTY -- zero owed tests is broken wiring, not a pass")
        return 1
    for m in missing[:50]:
        print(f"::error::Σ {kind}: owed but NOT executed: {m}")
    for x in extra[:50]:
        print(f"::error::Σ {kind}: executed but not in the universe: {x}")
    if missing or extra:
        more = max(len(missing), len(extra)) - 50
        if more > 0:
            print(f"  ... and {more} more")
        return 1
    print(f"Σ {kind}: OK -- {len(universe)} ids, executed set == universe")
    return 0


def junit_ids(path: Path) -> set:
    ids = set()
    for tc in ET.parse(path).getroot().iter("testcase"):
        if tc.find("skipped") is not None:
            continue
        ids.add(f"{tc.get('classname')} {tc.get('name')}")
    return ids


def nextest_list_ids(path: Path) -> set:
    doc = json.loads(Path(path).read_text())
    ids = set()
    for bid, suite in (doc.get("rust-suites") or {}).items():
        for name, case in (suite.get("testcases") or {}).items():
            fm = (case.get("filter-match") or {}).get("status")
            if fm == "matches":
                ids.add(f"{bid} {name}")
    return ids


def divergent_crates(path: Path) -> set:
    return {l.strip() for l in Path(path).read_text().splitlines()
            if re.fullmatch(r"[a-z][a-z0-9_-]*", l.strip())}


def arch_equal(x64_list: Path, arm_list: Path, divergent: Path) -> int:
    """The ARM64 lib shards run every crate NOT in `divergent`. That is only
    sound if those crates list the same ids on both arches, so the claim is
    re-proven on every run: X64's universe minus the divergent crates' ids must
    equal the ARM64 list. A crate that grows an arch-specific test turns this RED
    until it is added to the divergent list."""
    div = divergent_crates(divergent)
    if not div:
        print("::error::Σ arch-equal: the divergent-crate list is EMPTY -- broken wiring, not a pass")
        return 1
    x64 = {i for i in nextest_list_ids(x64_list) if i.split(" ", 1)[0] not in div}
    return compare("arch-equal (X64 minus divergent == ARM64)", nextest_list_ids(arm_list), x64)


STEP_MARK = re.compile(r"^##\[step \d+\] (.*)$")
STEP_END = re.compile(r"^(success|failure) \(")


def step_labels(log: str):
    """(line, label) for every line; label is None unless it is a step marker
    that opens a step (a "success (..)"/"failure (..)" marker closes none)."""
    for line in log.splitlines():
        m = STEP_MARK.match(line)
        opens = m and not STEP_END.match(m.group(1))
        yield line, (m.group(1) if opens else None), bool(m)


def step_slice(log: str, step: str) -> str:
    """The text between the step's marker and the next step marker."""
    out, on = [], False
    for line, label, is_marker in step_labels(log):
        if on and label is not None:
            break
        if not on and label is not None and step in label:
            on = True
        elif on:
            out.append(line)
    if not on:
        raise SystemExit(f"Σ: no step marker containing {step!r} in the section log")
    return "\n".join(out)


def cargo_ids(text: str) -> set:
    # libtest prints `test X - should panic ... ok` for #[should_panic] tests; the id is X.
    return {m.group(1) for m in re.finditer(
        r"^test (\S+)(?: - should panic)? \.\.\. (?:ok|FAILED)\b", text, re.M)}


def cargo_list_ids(text: str) -> set:
    return {m.group(1) for m in re.finditer(r"^(\S+): test$", text, re.M)}


def explicit_ids(text: str) -> set:
    return {m.group(1).strip() for m in re.finditer(r"^::group::\[\d+/\d+\] (.*)$", text, re.M)}


def parse_opts(rest):
    # --junit and --log repeat: a sharded run passes one per shard and the
    # executed set is their UNION, compared with the one universe.
    opts, many = {}, {"--junit": [], "--log": []}
    for k, v in zip(rest[::2], rest[1::2]):
        if k in many:
            many[k].append(v)
        else:
            opts[k] = v
    return opts, many


def run_logs(mode, opts, many) -> int:
    slices = [step_slice(Path(f).read_text(errors="replace"), opts["--step"]) for f in many["--log"]]
    if mode == "cargo":
        uni = cargo_list_ids(Path(opts["--list"]).read_text()) - \
            cargo_list_ids(Path(opts["--ignored"]).read_text())
        return compare(opts.get("--kind", "cargo"), set().union(*map(cargo_ids, slices)), uni)
    if mode == "explicit":
        uni = {l.strip() for l in Path(opts["--list"]).read_text().splitlines() if l.strip()}
        return compare(opts.get("--kind", "explicit"), set().union(*map(explicit_ids, slices)), uni)
    print(f"Σ: unknown mode {mode!r}", file=sys.stderr)
    return 2


def run(argv) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    if argv[0] == "--self-test":
        return self_test()
    mode, (opts, many) = argv[0], parse_opts(argv[1:])
    if mode == "nextest":
        executed = set().union(*(junit_ids(Path(j)) for j in many["--junit"]))
        return compare(opts.get("--kind", "nextest"), executed,
                       nextest_list_ids(Path(opts["--list-json"])))
    if mode == "arch-equal":
        return arch_equal(Path(opts["--x64-list"]), Path(opts["--arm-list"]),
                          Path(opts["--divergent"]))
    return run_logs(mode, opts, many)


# --------------------------------------------------------------------------
JUNIT = """<?xml version="1.0"?><testsuites><testsuite name="s">
{cases}</testsuite></testsuites>"""


def write_junit(d, name, cases):
    body = "\n".join(f'<testcase classname="{b}" name="{n}"/>' for b, n in cases)
    (d / name).write_text(JUNIT.format(cases=body))
    return str(d / name)


def write_listing(d, name, ids, extra=None):
    s = {}
    for b, n in ids:
        s.setdefault(b, {"testcases": {}})["testcases"][n] = {"filter-match": {"status": "matches"}}
    for b, n in extra or ():
        s.setdefault(b, {"testcases": {}})["testcases"][n] = {
            "filter-match": {"status": "mismatch", "reason": "ignored"}}
    (d / name).write_text(json.dumps({"rust-suites": s}))
    return str(d / name)


def log_with(step_body, step="Compute tests"):
    return ("##[step 3] Earlier step\ntest z::other ... ok\n##[step 3] success (1s)\n"
            f"##[step 4] {step} (x)\n{step_body}\n##[step 4] success (2s)\n"
            "##[step 5] Later\ntest z::later ... ok\n")


U = [("aprender-core", "a::one"), ("aprender-core", "a::two"), ("apr-cli", "c::three")]


def rows_nextest(d, rows):
    skipped = [("aprender-core", "t::skipped_by_filter")]
    for label, executed, universe, want in (
        ("nextest: executed == universe", U, U, 0),
        ("nextest: MUTANT drops one test", U[:-1], U, 1),
        ("nextest: an extra test", U + [("x", "y")], U, 1),
        ("nextest: empty universe", [], [], 1),
    ):
        j = write_junit(d, "j.xml", executed)
        lst = write_listing(d, "l.json", universe, skipped)
        rows.append((label, want, run(["nextest", "--junit", j, "--list-json", lst])))
    # a skipped testcase is not executed
    cases = "\n".join([f'<testcase classname="{b}" name="{n}"/>' for b, n in U[:-1]]
                      + ['<testcase classname="apr-cli" name="c::three"><skipped/></testcase>'])
    (d / "j.xml").write_text(JUNIT.format(cases=cases))
    rc = run(["nextest", "--junit", str(d / "j.xml"), "--list-json", str(d / "l.json")])
    rows.append(("nextest: a <skipped/> case is not executed", 1, rc))


def rows_cargo(d, rows):
    (d / "list.txt").write_text("b::one: test\nb::two: test\nb::ign: test\n\n3 tests\n")
    (d / "ign.txt").write_text("b::ign: test\n")
    for label, body, want in (
        ("cargo: executed == universe", "test b::one ... ok\ntest b::two ... FAILED\ntest b::ign ... ignored", 0),
        ("cargo: MUTANT drops one test", "test b::one ... ok", 1),
        ("cargo: a should_panic test counts as executed",
         "test b::one ... ok\ntest b::two - should panic ... ok", 0),
        ("cargo: other steps' tests do not count", "test b::one ... ok\n", 1),
    ):
        (d / "sec.log").write_text(log_with(body))
        rc = run(["cargo", "--log", str(d / "sec.log"), "--step", "Compute tests",
                  "--list", str(d / "list.txt"), "--ignored", str(d / "ign.txt")])
        rows.append((label, want, rc))


def rows_explicit(d, rows):
    (d / "ex.txt").write_text("cargo test -p a --test one\ncargo test -p b --test two\n")
    for label, body, want in (
        ("explicit: every fragment ran", "::group::[1/2] cargo test -p a --test one\n::endgroup::\n"
         "::group::[2/2] cargo test -p b --test two\n::endgroup::", 0),
        ("explicit: MUTANT drops one fragment", "::group::[1/2] cargo test -p a --test one\n::endgroup::", 1),
    ):
        (d / "sec.log").write_text(log_with(body, "Integration tests"))
        rc = run(["explicit", "--log", str(d / "sec.log"), "--step", "Integration tests",
                  "--list", str(d / "ex.txt")])
        rows.append((label, want, rc))


def rows_shards(d, rows):
    # Sharded: one junit / one log per shard, the executed set is the union.
    write_listing(d, "l.json", U)
    for label, parts, want in (
        ("nextest shards: union of three partitions == universe", [U[:1], U[1:2], U[2:]], 0),
        ("nextest shards: MUTANT a shard's partition lost", [U[:1], U[1:2]], 1),
        ("nextest shards: no junit at all", [], 1),
    ):
        args = ["nextest", "--list-json", str(d / "l.json")]
        for i, part in enumerate(parts):
            args += ["--junit", write_junit(d, f"s{i}.xml", part)]
        rows.append((label, want, run(args)))
    g1 = "::group::[1/1] cargo test -p a --test one\n::endgroup::"
    g2 = "::group::[1/1] cargo test -p b --test two\n::endgroup::"
    for label, bodies, want in (
        ("explicit shards: union of the shard logs == list", [g1, g2], 0),
        ("explicit shards: MUTANT a shard's log lost", [g1], 1),
    ):
        args = ["explicit", "--step", "Integration tests", "--list", str(d / "ex.txt")]
        for i, body in enumerate(bodies):
            (d / f"s{i}.log").write_text(log_with(body, "Integration tests"))
            args += ["--log", str(d / f"s{i}.log")]
        rows.append((label, want, run(args)))


def rows_missing_step(d, rows):
    try:
        (d / "sec.log").write_text(log_with(""))
        run(["explicit", "--log", str(d / "sec.log"), "--step", "No such step", "--list", str(d / "ex.txt")])
        rows.append(("a missing step marker refuses", 1, 0))
    except SystemExit:
        rows.append(("a missing step marker refuses", 1, 1))


def rows_arch(d, rows):
    # arch-equal: ARM64 ids == X64 ids outside the divergent crates.
    (d / "div.txt").write_text("# c\nneon-crate\n")
    (d / "empty.txt").write_text("# none\n")
    common = [("a", "t1"), ("a", "t2"), ("b", "t3")]
    x64 = write_listing(d, "x.json", common + [("neon-crate", "avx2_only")])
    for label, arm, div, want in (
        ("arch-equal: ARM == X64 minus divergent", common, "div.txt", 0),
        ("arch-equal: MUTANT ARM lacks a shared test", common[:-1], "div.txt", 1),
        ("arch-equal: MUTANT ARM has an extra test", common + [("a", "t_neon")], "div.txt", 1),
        ("arch-equal: ARM ran a divergent crate", common + [("neon-crate", "avx2_only")], "div.txt", 1),
        ("arch-equal: an empty divergent list is refused", common, "empty.txt", 1),
    ):
        rc = run(["arch-equal", "--x64-list", x64, "--arm-list", write_listing(d, "a.json", arm),
                  "--divergent", str(d / div)])
        rows.append((label, want, rc))


def self_test() -> int:
    rows = []
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        for build in (rows_nextest, rows_cargo, rows_explicit, rows_shards, rows_missing_step, rows_arch):
            build(d, rows)
    bad = 0
    print("\n--- Σ-executed case table ---")
    for label, want, got in rows:
        ok = (want == 0) == (got == 0)
        bad += not ok
        print(f"{'ok  ' if ok else 'BAD '} want={'PASS' if want == 0 else 'RED '} got_rc={got}  {label}")
    print(f"Σ self-test: {len(rows) - bad}/{len(rows)} rows as expected")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(run(sys.argv[1:]))
