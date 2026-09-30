"""
Re-applies the REAL production deterministic_fixes() (imported directly
from taxonomy_map_document.py, never reimplemented) to every already-
labeled row's saved result. This is safe and free -- deterministic_fixes()
is a pure function of (result, text) with no LLM calls, so re-running it
against already-collected data picks up the new multi_turn reverse-check
fix (2026-09-20) WITHOUT spending any new Gemma calls or re-labeling
anything. Verified idempotent/safe on the real data first (see
overnight_builder1_log.md): re-running the full function against the real
98,436-doc batch produces exactly the expected 724 interaction_mode
changes and zero exceptions, nothing else shifts unexpectedly.
"""
import sys
import json
import copy

sys.path.insert(0, "/projects/data/datasets/code_data/sai_rupesh/taxonomy")
from taxonomy_map_document import deterministic_fixes

IN_FILE = "sft_output_sample_combined_98436_clean.jsonl"
OUT_FILE = "sft_output_sample_combined_98436_clean.jsonl"  # in place -- see main() for the safe-write pattern


def main():
    rows = []
    n_changed = 0
    n_errors = 0
    with open(IN_FILE) as f:
        for line in f:
            d = json.loads(line)
            r = d.get("result")
            if r:
                before = copy.deepcopy(r)
                try:
                    deterministic_fixes(r, d.get("full_text", ""))
                except Exception:
                    n_errors += 1
                else:
                    if r != before:
                        n_changed += 1
            rows.append(d)

    tmp_file = IN_FILE + ".tmp"
    with open(tmp_file, "w", encoding="utf-8") as f:
        for d in rows:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    import os
    os.replace(tmp_file, OUT_FILE)  # atomic swap, no risk of a half-written file if interrupted

    print(f"rows processed: {len(rows)}")
    print(f"rows with a field changed by re-running deterministic_fixes(): {n_changed}")
    print(f"errors: {n_errors}")
    print(f"DONE: rewrote {OUT_FILE} in place")


if __name__ == "__main__":
    main()
