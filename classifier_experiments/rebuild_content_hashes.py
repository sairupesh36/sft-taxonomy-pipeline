"""
existing_content_hashes.json (dated Sep 16 07:20) predates batch4's ~20k new
rows and the weak-categories batch, so it's stale -- running a new sampler
pass against it as-is risks re-picking (duplicating) rows already used in
the current 98,436-doc training set. Rebuilds it fresh from every raw
sample file's own content_hash field (real production hash, not
recomputed), so any NEW sampling run correctly dedupes against everything
already in the training pool, not just the first 65,017 rows.
"""
import json
import glob

RAW_FILES = (
    glob.glob("../sft_output_sample/*.jsonl")
    + [
        "sft_output_sample_batch2.jsonl",
        "sft_output_sample_batch3.jsonl",
        "sft_output_sample_batch4.jsonl",
        "sft_output_sample_weak_categories.jsonl",
    ]
)
OUT_FILE = "existing_content_hashes.json"


def main():
    hashes = set()
    for fn in RAW_FILES:
        n = 0
        try:
            with open(fn) as f:
                for line in f:
                    d = json.loads(line)
                    h = d.get("content_hash")
                    if h:
                        hashes.add(h)
                        n += 1
        except FileNotFoundError:
            print(f"  MISSING: {fn}")
            continue
        print(f"  {fn}: {n} hashes")

    print(f"\ntotal unique hashes: {len(hashes)} (was 65,017 before rebuild)")
    with open(OUT_FILE, "w") as f:
        json.dump(sorted(hashes), f)
    print(f"DONE: wrote {OUT_FILE}")


if __name__ == "__main__":
    main()
