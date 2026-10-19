# #5736: cop cycles.jsonl must be the canonical v1 shape (COPS.md (2), as loops.sh enforces); plants 15 falsifiers.
bash scripts/check_cycles_schema.sh --self-test
