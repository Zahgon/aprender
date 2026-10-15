# `smoke` — the harness self-test suite

Eight hand-written tasks, two per kind (docs/specifications/apr-code-eval-v1.md §3),
licensed MIT like the rest of this crate. Every nightly runs it first: if `smoke` does
not validate (null edit fails, gold edit resolves, three gold runs agree), the harness is
broken and no model score that night means anything.

| id | kind | what the gold edit does |
|---|---|---|
| `smoke-fn-saturating-sum` | function | implements a saturating byte sum |
| `smoke-fn-run-length` | function | implements run-length encoding over characters |
| `smoke-edit-fizzbuzz` | edit | checks 15 before 3 and 5 |
| `smoke-edit-leap-year` | edit | adds the Gregorian century rule |
| `smoke-repo-config-comments` | repo | skips comments and blank lines in a two-module parser |
| `smoke-repo-word-frequency` | repo | lowercases counts and orders ties alphabetically, across two files |
| `smoke-term-sort-inbox` | terminal | `solve.sh` files `inbox/` by extension |
| `smoke-term-manifest` | terminal | `solve.sh` writes a sorted size manifest of `data/` |

Every task is a Rust crate named `smoke` whose hidden tests live in `tests/hidden.rs`;
a terminal task's hidden tests run `sh solve.sh` in a fixture directory and check the
final state. `envs.json` pins every task to the same `rust:1.93-slim` image by digest
and runs the hidden tests offline, one thread, as libtest JSON.

`tasks.jsonl` is the source of truth, one task per line. Checked by
`cargo test -p aprender-code-eval --lib smoke` (load, kinds, environments, gold applies,
test lists match the hidden tests, two seeded runs are byte-identical) and, with docker
and the pinned image present, `-- --ignored smoke` (the suite validates in the real
sandbox and two real runs are the same run).
