"""
Samples real documents from the 55 real files (~10.5M rows) found by
scan_chat_template_files.py -- files that store a genuine, complete
conversation as one rendered chat-template string in a single text column,
a shape sft_normalize.py's plan()/to_messages() doesn't recognize at all
(mode=None, "no_adapter", whole file silently skipped today). Uses the NEW
chat_template_parsers.py (this folder, not DEDUP_PIPELINE) to split that
text into real turns, then hands the result to the REAL production
S.canonicalize_turns()/S.build() exactly like every other sampler in this
project -- so a sampled record here has the identical shape as one from
sample_from_sft_output.py.

READ-ONLY against OUTPUT. Dedupes against existing_content_hashes.json.
Caps rows per file (each of these 55 is its own genuinely distinct real
dataset, same "family" concept as sample_from_sft_output.py) so a few
1M-row shards (the ykarout code-reasoning ChatML files) can't crowd out
the other 50-odd smaller, more topically-diverse files.
"""
import os
import sys
import json
import random

sys.path.insert(0, "/projects/data/datasets/translation_data/SFT/DEDUP_PIPELINE")
import sft_schema as S
import pyarrow.parquet as pq

sys.path.insert(0, "/projects/data/datasets/code_data/sai_rupesh/taxonomy/classifier_experiments")
from chat_template_parsers import parse_chat_template

FOUND_FILES_JSON = "/projects/data/datasets/code_data/sai_rupesh/taxonomy/classifier_experiments/chat_template_files_found.json"
EXISTING_HASHES_FILE = "/projects/data/datasets/code_data/sai_rupesh/taxonomy/classifier_experiments/existing_content_hashes.json"
OUT_FILE = "/projects/data/datasets/code_data/sai_rupesh/taxonomy/classifier_experiments/sft_output_sample_chat_template.jsonl"
OUTPUT_DIR = "/projects/data/datasets/translation_data/SFT/OUTPUT"

MAX_ROWS_PER_FILE = 500
BATCH_SIZE = 500

# One real file (datafreak/MATH-Llama2-train) has a MALFORMED rendering of
# its own 'text' column (missing the closing [/INST] tag entirely -- checked
# directly, not assumed: 0/100 real rows parsed), but it also has clean,
# already-separate 'user_message'/'assistant_message' columns sitting right
# there unused (production plan() doesn't recognize those exact names, only
# 'user'/'question'/etc.). Rather than force the text-template parser onto a
# genuinely broken rendering, this one named file uses its own clean columns
# directly. Narrow, evidence-based, single-file override -- same pattern as
# every other named-file-list fix tonight, not a generic rule.
COLUMN_OVERRIDES = {
    "Maths/datafreak__MATH-Llama2-train.parquet": {"user_col": "user_message", "assistant_col": "assistant_message"},
}


def rows_from_override(pf, override):
    for batch in pf.iter_batches(batch_size=BATCH_SIZE):
        rows = batch.to_pylist()
        for r in rows:
            u = (r.get(override["user_col"]) or "").strip()
            a = (r.get(override["assistant_col"]) or "").strip()
            if not u or not a:
                continue
            yield [{"role": "user", "content": u}, {"role": "assistant", "content": a}]


def rows_from_template(pf, text_col):
    for batch in pf.iter_batches(batch_size=BATCH_SIZE):
        rows = batch.to_pylist()
        for r in rows:
            text = r.get(text_col)
            if not text:
                continue
            _, turns = parse_chat_template(text)
            if turns:
                yield turns


def sample_file(entry, rng, seen_hashes):
    path = entry["path"]
    rel = os.path.relpath(path, OUTPUT_DIR)
    source = os.path.basename(path)[:-len(".parquet")].replace("__", "/")
    domain = entry["domain"]

    try:
        pf = pq.ParquetFile(path)
    except Exception:
        return

    override = COLUMN_OVERRIDES.get(rel)
    gen = rows_from_override(pf, override) if override else rows_from_template(pf, entry["text_col"])

    taken = 0
    idx = 0
    try:
        for turns in gen:
            if taken >= MAX_ROWS_PER_FILE:
                break
            i = idx
            idx += 1
            try:
                msgs = S.canonicalize_turns(turns)
                rec, err = S.build(msgs, file_name=rel, row_idx=i, source=source,
                                    domain=domain, adapter="chat_template_" + entry["template"])
            except Exception:
                continue
            if err or rec is None:
                continue
            if rec["content_hash"] in seen_hashes:
                continue
            seen_hashes.add(rec["content_hash"])
            taken += 1
            yield rec
    finally:
        pf.close()


def main():
    with open(FOUND_FILES_JSON) as f:
        found = json.load(f)
    with open(EXISTING_HASHES_FILE) as f:
        seen_hashes = set(json.load(f))
    print(f"loaded {len(found)} chat-template files, {len(seen_hashes)} existing hashes to dedup against", flush=True)

    rng = random.Random(11)
    rng.shuffle(found)

    kept = 0
    per_file_counts = {}
    with open(OUT_FILE, "w", encoding="utf-8") as out_f:
        for entry in found:
            n_this_file = 0
            for rec in sample_file(entry, rng, seen_hashes):
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out_f.flush()
                kept += 1
                n_this_file += 1
            per_file_counts[entry["path"]] = n_this_file
            print(f"  {entry['template']:12s} {os.path.basename(entry['path']):55s} -> {n_this_file} rows  (total so far: {kept})", flush=True)

    print(f"\nDONE: {kept} new rows -> {OUT_FILE}", flush=True)
    zero_files = [p for p, c in per_file_counts.items() if c == 0]
    if zero_files:
        print(f"{len(zero_files)} files yielded 0 rows (parser didn't match any real row -- worth a look, not necessarily wrong):")
        for p in zero_files:
            print(f"  {p}")


if __name__ == "__main__":
    main()
