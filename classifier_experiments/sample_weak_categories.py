"""
Targeted content-based sampler for the categories the classifier is measurably
weakest on (real per-class F1 = 0 for 7 domains, very low for 5 task_families --
see classifier_experiments/keywords_weak_categories.py for how these were found
and grounded). Unlike the earlier samplers, this one reads actual document TEXT
(not just schema/metadata) and searches for topic keywords, because these
categories are scattered thinly across many different OUTPUT folders rather than
concentrated in any one -- confirmed by manually pulling real examples of each
before writing this (a veterinary document showed up in a science-abstract file,
an automotive one in a finance file, etc).

READ-ONLY against OUTPUT, same as every other sampler in this project. Dedupes
against all existing labeled docs via content_hash.
"""
import os
import re
import sys
import glob
import json
import random
import collections

sys.path.insert(0, "/projects/data/datasets/translation_data/SFT/DEDUP_PIPELINE")
import sft_schema as S
import sft_normalize as N
import pyarrow.parquet as pq

sys.path.insert(0, "/projects/data/datasets/code_data/sai_rupesh/taxonomy/classifier_experiments")
from keywords_weak_categories import DOMAIN_KEYWORDS, TASK_FAMILY_KEYWORDS
# Same dropped-context-column fix applied to sample_from_sft_output.py
# (2026-09-20, see overnight_builder1_log.md) -- reused here, not
# reimplemented, so any weak-category row pulled from one of the 57 known-
# affected files (several are Finance/Biology sources this script also
# scans) gets its context spliced back in instead of silently losing it too.
from sample_from_sft_output import (
    load_context_drop_map, splice_dropped_context,
    load_tool_call_drop_files, patch_null_content_tool_turns,
)

OUTPUT_DIR = "/projects/data/datasets/translation_data/SFT/OUTPUT"
EXCLUDE_FILE = "/projects/data/datasets/translation_data/SFT/DEDUP_PIPELINE/benchmarks_exclude.txt"
EXISTING_HASHES_FILE = "/projects/data/datasets/code_data/sai_rupesh/taxonomy/classifier_experiments/existing_content_hashes.json"
OUT_FILE = "/projects/data/datasets/code_data/sai_rupesh/taxonomy/classifier_experiments/sft_output_sample_weak_categories_v2.jsonl"

# 2026-09-20: bumped from 400 -- round 1 (12 categories) still left several
# well under cap (ranking=150, code_review=115, system_design=113) because
# real supply is genuinely limited for some, but round 2 adds ~32 MORE
# categories (see keywords_weak_categories.py), several with under-20 real
# examples in the current training set -- raising the cap gives every
# category a real shot at more volume without hurting the ones that are
# already scarce (they'll just hit their natural ceiling and drop out, same
# self-balancing behavior the docstring above describes).
CAP_PER_CATEGORY = 800
MAX_ROWS_SCANNED_PER_FILE = 800  # bounds worst-case cost per shard file
BATCH_SIZE = 500
_SHARD_SUFFIX = re.compile(r"_\d+$")

CONTEXT_DROP_MAP = {}  # populated in main()
TOOL_CALL_DROP_FILES = set()  # populated in main()
ALL_CATS = {**DOMAIN_KEYWORDS, **TASK_FAMILY_KEYWORDS}
PATTERNS = {cat: re.compile("|".join(re.escape(k) for k in kws), re.IGNORECASE) for cat, kws in ALL_CATS.items()}


def load_excludes():
    if not os.path.exists(EXCLUDE_FILE):
        return set()
    with open(EXCLUDE_FILE) as fh:
        return {l.strip() for l in fh if l.strip() and not l.startswith("#")}


def matches(text, cat_counts):
    hits = []
    for cat, pat in PATTERNS.items():
        if cat_counts[cat] >= CAP_PER_CATEGORY:
            continue
        if pat.search(text):
            hits.append(cat)
    return hits


def scan_file(f, domain_name, rng, seen_hashes, cat_counts, out_f, rows_scanned_total):
    schema_pf = None
    try:
        schema_pf = pq.ParquetFile(f)
        cols = list(schema_pf.schema_arrow.names)
    except Exception:
        return 0
    finally:
        if schema_pf is not None:
            schema_pf.close()
    p = N.plan(cols)
    if p["mode"] is None:
        return 0

    rel = os.path.relpath(f, OUTPUT_DIR)
    source = os.path.basename(f)[:-len(".parquet")].replace("__", "/")
    dropped_cols = CONTEXT_DROP_MAP.get(rel)
    is_tool_call_drop_file = p["mode"] == "chat" and rel in TOOL_CALL_DROP_FILES
    scanned_here = 0
    kept_here = 0
    pf = None
    try:
        pf = pq.ParquetFile(f)
        idx = 0
        for batch in pf.iter_batches(batch_size=BATCH_SIZE):
            if scanned_here >= MAX_ROWS_SCANNED_PER_FILE:
                break
            rows = batch.to_pylist()
            for r in rows:
                i = idx
                idx += 1
                scanned_here += 1
                if scanned_here >= MAX_ROWS_SCANNED_PER_FILE:
                    break
                if is_tool_call_drop_file:
                    try:
                        coerced = N.coerce_chat(r.get(p["col"]))
                    except Exception:
                        coerced = None
                    if coerced:
                        patch_null_content_tool_turns(coerced)
                        r = dict(r)
                        r[p["col"]] = coerced
                try:
                    raw = N.to_messages(r, p)
                except Exception:
                    continue
                if not raw:
                    continue
                if dropped_cols:
                    raw = splice_dropped_context(raw, r, dropped_cols)
                try:
                    msgs = S.canonicalize_turns(raw)
                except Exception:
                    continue
                user_text = " ".join(m["content"] for m in msgs if m["role"] == "user")
                if not user_text.strip():
                    continue
                hits = matches(user_text, cat_counts)
                if not hits:
                    continue
                try:
                    rec, err = S.build(msgs, file_name=rel, row_idx=i, source=source,
                                        domain=domain_name, adapter=p["mode"])
                except Exception:
                    continue
                if err or rec is None:
                    continue
                if rec["content_hash"] in seen_hashes:
                    continue
                seen_hashes.add(rec["content_hash"])
                rec["matched_categories"] = hits
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out_f.flush()
                kept_here += 1
                for cat in hits:
                    cat_counts[cat] += 1
    except Exception:
        pass
    finally:
        if pf is not None:
            pf.close()
    return scanned_here


def all_categories_full(cat_counts):
    return all(cat_counts[cat] >= CAP_PER_CATEGORY for cat in ALL_CATS)


def main():
    global CONTEXT_DROP_MAP, TOOL_CALL_DROP_FILES
    CONTEXT_DROP_MAP = load_context_drop_map()
    print(f"loaded {len(CONTEXT_DROP_MAP)} known context-drop-affected files to splice-fix", flush=True)
    TOOL_CALL_DROP_FILES = load_tool_call_drop_files()
    print(f"loaded {len(TOOL_CALL_DROP_FILES)} known tool-call-drop-affected files to patch", flush=True)
    rng = random.Random(7)
    excludes = load_excludes()
    with open(EXISTING_HASHES_FILE) as f:
        seen_hashes = set(json.load(f))
    print(f"loaded {len(seen_hashes)} existing hashes to dedup against", flush=True)
    print(f"target categories: {list(ALL_CATS.keys())}", flush=True)

    domains = sorted([d for d in os.listdir(OUTPUT_DIR) if os.path.isdir(os.path.join(OUTPUT_DIR, d))])
    cat_counts = collections.Counter()
    rows_scanned_total = 0

    with open(OUT_FILE, "w", encoding="utf-8") as out_f:
        round_num = 0
        while not all_categories_full(cat_counts):
            round_num += 1
            any_progress = False
            for d in domains:
                if all_categories_full(cat_counts):
                    break
                files = glob.glob(os.path.join(OUTPUT_DIR, d, "*.parquet"))
                files = [f for f in files if os.path.relpath(f, OUTPUT_DIR) not in excludes]
                rng.shuffle(files)
                # only look at a slice of files per round per domain, so all domains get scanned each round
                slice_size = max(1, len(files) // 20) if files else 0
                start = ((round_num - 1) * slice_size) % max(1, len(files))
                chunk = files[start:start + slice_size] or files[:5]
                for f in chunk:
                    if all_categories_full(cat_counts):
                        break
                    scanned = scan_file(f, d, rng, seen_hashes, cat_counts, out_f, rows_scanned_total)
                    rows_scanned_total += scanned
                    if scanned:
                        any_progress = True
            print(f"round {round_num}: scanned {rows_scanned_total:,} rows total, counts so far: {dict(cat_counts)}", flush=True)
            if not any_progress:
                print("no more files to scan, stopping", flush=True)
                break
            if round_num > 25:
                print("hit round safety limit, stopping", flush=True)
                break

    total_kept = sum(1 for _ in open(OUT_FILE))
    print(f"\nDONE: {total_kept} matched rows -> {OUT_FILE}", flush=True)
    print("final per-category counts:", flush=True)
    for cat, c in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat:45s} {c}", flush=True)


if __name__ == "__main__":
    main()
