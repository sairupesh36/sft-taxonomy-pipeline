"""
Merges the newly-labeled weak-category-v2 batch into the clean 98,436-doc
training file, producing the next combined training file for retraining.
Same schema (uid/src_file/orig_domain/orig_family/orig_source/ok/result/
full_text) so this is a straight concatenation, no reshaping needed.

Dedup safety check included even though sample_weak_categories_v2 already
deduped against existing_content_hashes.json at SAMPLE time -- this merge
step checks for uid collisions between the two files (belt-and-braces,
cheap to check, would only ever fire if something upstream went wrong).
"""
import json

CLEAN_FILE = "sft_output_sample_combined_98436_clean.jsonl"
WEAK_V2_MAPPED_FILE = "sft_output_sample_weak_categories_v2_mapped.jsonl"


def main():
    existing_uids = set()
    rows = []
    with open(CLEAN_FILE) as f:
        for line in f:
            d = json.loads(line)
            rows.append(d)
            existing_uids.add(d["uid"])

    n_added = 0
    n_skipped_dup = 0
    n_skipped_not_ok = 0
    with open(WEAK_V2_MAPPED_FILE) as f:
        for line in f:
            d = json.loads(line)
            if d["uid"] in existing_uids:
                n_skipped_dup += 1
                continue
            if not d.get("ok"):
                n_skipped_not_ok += 1
                # still keep the row on disk (consistent with how severe-
                # truncation/context-drop rows are kept-but-excluded elsewhere),
                # just don't count it as new usable data
            rows.append(d)
            existing_uids.add(d["uid"])
            n_added += 1

    out_file = f"sft_output_sample_combined_{len(rows)}.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for d in rows:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    n_ok_total = sum(1 for d in rows if d.get("ok"))
    print(f"merged {n_added} new rows ({n_skipped_dup} uid-collisions skipped, "
          f"{n_skipped_not_ok} of the added rows failed Gemma/ok=False)")
    print(f"total rows: {len(rows)}, usable (ok=True): {n_ok_total}")
    print(f"DONE: wrote {out_file}")


if __name__ == "__main__":
    main()
