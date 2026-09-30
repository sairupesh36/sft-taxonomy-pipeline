"""
Builds the final, cleaned-up training file for tonight's retrain, combining
BOTH fixes made this session:
  1. sft_output_sample_combined_98436_fixed.jsonl (from relabel_severe_98436.py)
     -- the 1,341 severely-truncated rows re-labeled (1,330 of them) or
     excluded (ok=False, the 11 unrecoverable ones) using the fixed
     to_full_text().
  2. The 3,209 rows with a silently-dropped grounding/context column
     (found by the validator agent, verified real) -- excluded from
     training (ok=False) rather than retroactively patched, since
     re-attaching the exact right context to an already-sampled row risks a
     rushed matching bug; the sampler itself is fixed for future rows (see
     sample_from_sft_output.py).

Also excludes the 642 rows found by the validator's second lineage bug --
Tool_Use chat-format rows where an assistant's tool-call turn had
content=None and was silently dropped by sft_normalize.to_messages(),
leaving a tool result appear with no explanation. Same treatment as the
context-drop rows: exclude the already-sampled ones (retroactively
re-synthesizing the exact right tool call for an already-labeled row risks
a rushed mismatch), fix the sampler for new rows (done, see
sample_from_sft_output.py's patch_null_content_tool_turns()).

Rows are never physically deleted -- excluded rows are kept but marked
ok=False, which fasttext_baseline.py's load_rows() already treats as
"skip this row" (`if not d.get("ok"): continue`), so no other script needs
to change to respect the exclusion.
"""
import json

FIXED_FILE = "sft_output_sample_combined_98436_fixed.jsonl"
CONTEXT_DROP_UIDS_FILE = "context_drop_affected_uids.json"
TOOL_CALL_DROP_UIDS_FILE = "tool_call_dropped_affected_uids.json"
OUT_FILE = "sft_output_sample_combined_98436_clean.jsonl"


def main():
    with open(CONTEXT_DROP_UIDS_FILE) as f:
        drop_uids = set(json.load(f))
    print(f"context-drop exclusion list: {len(drop_uids)} uids")
    with open(TOOL_CALL_DROP_UIDS_FILE) as f:
        tool_call_drop_uids = set(json.load(f))
    print(f"tool-call-drop exclusion list: {len(tool_call_drop_uids)} uids")
    overlap = drop_uids & tool_call_drop_uids
    if overlap:
        print(f"NOTE: {len(overlap)} uids appear in BOTH exclusion lists (counted once)")

    n_total = 0
    n_context_excluded = 0
    n_tool_call_excluded = 0
    n_already_excluded_severe = 0
    n_ok_final = 0

    with open(FIXED_FILE) as f, open(OUT_FILE, "w", encoding="utf-8") as out_f:
        for line in f:
            d = json.loads(line)
            n_total += 1
            uid = d["uid"]
            if uid in drop_uids or uid in tool_call_drop_uids:
                if d.get("ok"):
                    if uid in drop_uids:
                        n_context_excluded += 1
                    if uid in tool_call_drop_uids:
                        n_tool_call_excluded += 1
                reasons = []
                if uid in drop_uids:
                    reasons.append("context_column_dropped")
                if uid in tool_call_drop_uids:
                    reasons.append("tool_call_turn_dropped")
                d["ok"] = False
                d["exclude_reason"] = ",".join(reasons)
            elif not d.get("ok"):
                n_already_excluded_severe += 1
            if d.get("ok"):
                n_ok_final += 1
            out_f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(f"total rows written: {n_total}")
    print(f"newly excluded for dropped-context bug: {n_context_excluded}")
    print(f"newly excluded for tool-call-turn-dropped bug: {n_tool_call_excluded}")
    print(f"already excluded (unrecoverable severe-truncation rows): {n_already_excluded_severe}")
    print(f"final ok=True rows usable for training: {n_ok_final}")
    print(f"DONE: wrote {OUT_FILE}")


if __name__ == "__main__":
    main()
