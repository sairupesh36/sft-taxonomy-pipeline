#!/usr/bin/env python3
"""
Stage 0 (no LLM): collapse every raw discovery CSV into deduped, counted
(L1) / (L1,L2) / (L1,L2,L3) value tables per axis, ready for LLM clustering.
Sources combined: extracted_{domains,tasks}_shard_*.csv (tulu3, 19 sources)
                   agentic_results/agentic_{domains,tasks}_shard_*.csv (18 sources)
"""
import csv
import glob
import json
from collections import Counter
from pathlib import Path

BASE_DIR = Path("/projects/data/datasets/code_data/sai_rupesh/taxonomy")
OUT_DIR = BASE_DIR / "unique_values"
OUT_DIR.mkdir(parents=True, exist_ok=True)

AXES = {
    "domain": {
        "cols": ("domain", "subdomain", "sub_subdomain"),
        "sources": [
            str(BASE_DIR / "extracted_domains_shard_*.csv"),
            str(BASE_DIR / "agentic_results" / "agentic_domains_shard_*.csv"),
        ],
    },
    "task": {
        "cols": ("task", "subtask", "sub_subtask"),
        "sources": [
            str(BASE_DIR / "extracted_tasks_shard_*.csv"),
            str(BASE_DIR / "agentic_results" / "agentic_tasks_shard_*.csv"),
        ],
    },
}


def load_rows(patterns):
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            with open(path, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    yield row


def main():
    for axis, cfg in AXES.items():
        l1_col, l2_col, l3_col = cfg["cols"]
        l1_counts = Counter()
        l2_counts = Counter()  # key: (l1, l2)
        l3_counts = Counter()  # key: (l1, l2, l3)
        total_rows = 0

        for row in load_rows(cfg["sources"]):
            v1 = (row.get(l1_col) or "").strip()
            v2 = (row.get(l2_col) or "").strip()
            v3 = (row.get(l3_col) or "").strip()
            if not v1:
                continue
            total_rows += 1
            l1_counts[v1] += 1
            if v2:
                l2_counts[(v1, v2)] += 1
                if v3:
                    l3_counts[(v1, v2, v3)] += 1

        (OUT_DIR / f"{axis}_l1.json").write_text(json.dumps(
            [{"value": v, "count": c} for v, c in l1_counts.most_common()], indent=1))
        (OUT_DIR / f"{axis}_l2.json").write_text(json.dumps(
            [{"l1": v[0], "value": v[1], "count": c} for v, c in l2_counts.most_common()], indent=1))
        (OUT_DIR / f"{axis}_l3.json").write_text(json.dumps(
            [{"l1": v[0], "l2": v[1], "value": v[2], "count": c} for v, c in l3_counts.most_common()], indent=1))

        print(f"[{axis}] rows={total_rows:,}  unique_l1={len(l1_counts):,}  "
              f"unique_l2_pairs={len(l2_counts):,}  unique_l3_triples={len(l3_counts):,}")


if __name__ == "__main__":
    main()
