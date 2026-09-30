"""
Dry-run (no LLM calls) for the truncation remediation plan: for every
"severe" row in the 98,436-doc combined training file (no real [USER]/
[ASSISTANT] content survived truncation), try to recover its ORIGINAL raw
messages from the source files, so a fixed full_text can be regenerated and
re-labeled by Gemma once the router is back up.

Two id schemes exist depending on which round contributed the row:
  - original 15,722-doc batch: uid = "{src_file}:{line_number}" -> look up
    in ../sft_output_sample/{src_file} at that line.
  - later batch2/3/4 additions: uid = the raw row's uuid hash -> look up by
    scanning the matching raw batch file for that uuid.
"""
import json
import glob

COMBINED_FILE = "sft_output_sample_combined_98436.jsonl"
ORIGINAL_SAMPLE_DIR = "../sft_output_sample"
RAW_BATCH_FILES = ["sft_output_sample_batch2.jsonl", "sft_output_sample_batch3.jsonl", "sft_output_sample_batch4.jsonl"]


def is_severe(full_text):
    return "[USER]:" not in full_text and "[ASSISTANT]:" not in full_text


def find_severe_rows():
    rows = []
    with open(COMBINED_FILE) as f:
        for line in f:
            d = json.loads(line)
            if is_severe(d.get("full_text", "")):
                rows.append(d)
    return rows


def build_uuid_index():
    index = {}
    for fn in RAW_BATCH_FILES:
        with open(fn) as f:
            for line in f:
                d = json.loads(line)
                index[d["uuid"]] = d["messages"]
    return index


def recover_messages(row, uuid_index):
    uid = row["uid"]
    if ":" in uid and uid.split(":")[0].endswith(".jsonl"):
        # original-batch style: "{src_file}:{line_number}"
        src_file, line_num = uid.rsplit(":", 1)
        line_num = int(line_num)
        path = f"{ORIGINAL_SAMPLE_DIR}/{src_file}"
        try:
            with open(path) as f:
                for i, line in enumerate(f):
                    if i == line_num:
                        return json.loads(line)["messages"]
        except FileNotFoundError:
            return None
        return None
    else:
        return uuid_index.get(uid)


def main():
    severe = find_severe_rows()
    print(f"severe rows found: {len(severe)}")

    uuid_index = build_uuid_index()
    print(f"uuid index built from raw batch files: {len(uuid_index)} entries")

    recovered = 0
    not_recovered = []
    for row in severe:
        messages = recover_messages(row, uuid_index)
        if messages is not None:
            recovered += 1
        else:
            not_recovered.append(row["uid"])

    print(f"\nrecovered original messages: {recovered}/{len(severe)}")
    if not_recovered:
        print(f"NOT recovered ({len(not_recovered)}): {not_recovered[:10]}{'...' if len(not_recovered) > 10 else ''}")


if __name__ == "__main__":
    main()
