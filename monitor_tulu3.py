#!/usr/bin/env python3
"""
Live Monitor for Resumed Tulu-3 5k-Capped Taxonomy Extraction (19 Sources)
Tracks: extracted_tasks_shard_*.csv and extracted_domains_shard_*.csv
"""

import glob
import os
import sys
import time
import csv
from collections import Counter
from pathlib import Path

BASE_DIR = Path("/projects/data/datasets/code_data/sai_rupesh/taxonomy")

TULU3_SOURCES = [
    ("ai2-adapt-dev/coconot_converted", 5000),
    ("ai2-adapt-dev/evol_codealpaca_heval_decontaminated", 5000),
    ("ai2-adapt-dev/flan_v2_converted", 5000),
    ("ai2-adapt-dev/no_robots_converted", 5000),
    ("ai2-adapt-dev/numinamath_tir_math_decontaminated", 5000),
    ("ai2-adapt-dev/oasst1_converted", 5000),
    ("ai2-adapt-dev/personahub_code_v2_34999", 5000),
    ("ai2-adapt-dev/personahub_ifdata_manual_seed_v3_29980", 5000),
    ("ai2-adapt-dev/personahub_math_v5_regen_149960", 5000),
    ("ai2-adapt-dev/tulu_hard_coded_repeated_10", 240),
    ("ai2-adapt-dev/tulu_v3.9_aya_100k", 5000),
    ("ai2-adapt-dev/tulu_v3.9_open_math_2_gsm8k_50k", 5000),
    ("ai2-adapt-dev/tulu_v3.9_personahub_math_interm_algebra_20k", 5000),
    ("ai2-adapt-dev/tulu_v3.9_sciriff_10k", 5000),
    ("ai2-adapt-dev/tulu_v3.9_synthetic_finalresp_wildguardmixtrain_decontaminated_50k", 5000),
    ("ai2-adapt-dev/tulu_v3.9_table_gpt_5k", 5000),
    ("ai2-adapt-dev/tulu_v3.9_wildchat_100k", 5000),
    ("ai2-adapt-dev/tulu_v3.9_wildjailbreak_decontaminated_50k", 5000),
    ("allenai/tulu-3-sft-personas-math-grade", 5000),
]

def scan_csvs(pattern):
    counts = Counter()
    total = 0
    for f in sorted(BASE_DIR.glob(pattern)):
        if f.exists() and f.stat().st_size > 0:
            with open(f, "r", encoding="utf-8", errors="ignore") as fp:
                r = csv.reader(fp)
                next(r, None)
                for row in r:
                    if len(row) >= 2:
                        counts[row[1].strip()] += 1
                        total += 1
    return counts, total

def main():
    while True:
        task_counts, total_tasks = scan_csvs("extracted_tasks_shard_*.csv")
        dom_counts, total_doms   = scan_csvs("extracted_domains_shard_*.csv")

        os.system("clear")
        print("=" * 115)
        print("     🚀 TULU-3 TAXONOMY RESUME MONITOR: 5K CAPPED EXTRACTION (19 SOURCES)")
        print("=" * 115)
        print(f"{'#':<3} | {'Dataset Source':<55} | {'Target':<6} | {'Tasks':<12} | {'Domains':<12} | {'Status'}")
        print("-" * 115)

        total_target = sum(target for _, target in TULU3_SOURCES)
        capped_tasks_done = 0
        capped_doms_done = 0

        for idx, (src, target) in enumerate(TULU3_SOURCES, 1):
            t_done = task_counts.get(src, 0)
            d_done = dom_counts.get(src, 0)
            capped_tasks_done += min(t_done, target)
            capped_doms_done += min(d_done, target)

            t_pct = (min(t_done, target) / target) * 100
            d_pct = (min(d_done, target) / target) * 100

            status = "✅ MET" if (t_done >= target and d_done >= target) else "⏳ IN PROGRESS"
            if t_done == 0 and d_done == 0:
                status = "⏸️ QUEUED"

            short_src = src.replace("ai2-adapt-dev/", "")
            print(f"{idx:<3} | {short_src:<55} | {target:<6} | {t_done:<5} ({t_pct:5.1f}%) | {d_done:<5} ({d_pct:5.1f}%) | {status}")

        print("=" * 115)
        task_overall_pct = (capped_tasks_done / total_target) * 100
        dom_overall_pct  = (capped_doms_done / total_target) * 100
        print(f"OVERALL 5K CAP PROGRESS:")
        print(f"  • Tasks Completed toward Cap:   {capped_tasks_done:,} / {total_target:,} ({task_overall_pct:.1f}%) [Total Extracted: {total_tasks:,}]")
        print(f"  • Domains Completed toward Cap: {capped_doms_done:,} / {total_target:,} ({dom_overall_pct:.1f}%) [Total Extracted: {total_doms:,}]")
        print("=" * 115)
        print(f"Updated at {time.strftime('%Y-%m-%d %H:%M:%S')}. Refreshing every 10s... (Press Ctrl+C to exit)")

        if capped_tasks_done >= total_target and capped_doms_done >= total_target:
            print("\n🎉 ALL 19 SOURCES HAVE MET THE 5K CAP FOR BOTH TASKS AND DOMAINS!")
            break

        time.sleep(10)

if __name__ == "__main__":
    main()
