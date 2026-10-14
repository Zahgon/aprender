# coverage_floor.sh — read the coverage floor that `make coverage-check` APPLIED,
# from make's own verdict line (#4671). Sourced by scripts/dogfood.sh.
#
# WHY. The dogfood coverage note printed `${COV_FLOOR:-88}` from its own shell,
# where COV_FLOOR is never set, while make enforced `COV_FLOOR := 89`. The receipt
# named a floor that did not decide its verdict, and the PASS note named none, so a
# floor overridden through MAKEFLAGS also read "floor met". The only floor that
# decided the row is the one make printed; anything else is a guess.
#
# Sourced library: option-neutral (no `set`), fails by return status.
#
#   cov_floor_applied <make output>   prints the floor make applied (e.g. 89), or
#                                     nothing and returns 1 when make never reached
#                                     its verdict (no floor was applied).
# Case table: scripts/tests/coverage_floor_test.sh
cov_floor_applied() {
  local f
  # The three verdict lines of the `coverage` recipe (Makefile): REGRESSION,
  # holds-the-floor, and meets-threshold. Each names the floor it applied.
  f=$(printf '%s\n' "$1" | grep -oE '(the enforced floor|holds the floor|, floor) [0-9]+%' \
      | grep -oE '[0-9]+' | tail -n 1)
  [ -n "$f" ] || return 1
  printf '%s\n' "$f"
}
