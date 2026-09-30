#!/usr/bin/env python3
"""
Live Monitor for Agentic 17-Source Taxonomy Extraction (Tasks & Domains)
Tracks: agentic_tasks_shard_*.csv and agentic_domains_shard_*.csv
"""

import glob
import os
import sys
import time
import csv
from collections import Counter
from pathlib import Path

BASE_DIR = Path("/projects/data/datasets/code_data/sai_rupesh/taxonomy")
RES_DIR  = BASE_DIR / "agentic_results"

AGENTIC_SOURCES = [
    ("agent_flan_agent_instruct_react", 1731),
    ("agent_flan_agent_instruct_tflan", 1731),
    ("agent_flan_toolbench_instruct_3k", 3000),
    ("agent_flan_toolbench_negative", 761),
    ("agent_flan_toolbench_react_10p", 2288),
    ("agent_flan_toolbench_tflan_60p", 5000),
    ("agent_flan_toolbench_tflan_cot", 5000),
    ("agentinstruct_analytical_reasoning", 5000),
    ("agentinstruct_code_agent", 5000),
    ("agentinstruct_rag_agent", 5000),
    ("agentinstruct_tool_use", 5000),
    ("agentinstruct_webagent_flow", 5000),
    ("nemotron_interactive_agent", 5000),
    ("nemotron_search_agent", 5000),
    ("nemotron_tool_calling", 5000),
    ("swe_bench_software_engineering_agent", 5000),
    ("toolace_complex_tools", 5000),
]

def scan_csvs(pattern):
    counts = Counter()
    total = 0
    for f in sorted(RES_DIR.glob(pattern)):
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
        task_counts, total_tasks = scan_csvs("agentic_tasks_shard_*.csv")
        dom_counts, total_doms   = scan_csvs("agentic_domains_shard_*.csv")

        os.system("clear")
        print("=" * 115)
        print("     🚀 AGENTIC TAXONOMY EXTRACTION MONITOR: 17 SOURCES (5K CAP)")
        print("=" * 115)
        print(f"{'#':<3} | {'Agentic Source Name':<45} | {'Target':<6} | {'Tasks':<14} | {'Domains':<14} | {'Status'}")
        print("-" * 115)

        total_target = sum(target for _, target in AGENTIC_SOURCES)
        capped_tasks = 0
        capped_doms  = 0

        for idx, (src, target) in enumerate(AGENTIC_SOURCES, 1):
            t_done = task_counts.get(src, 0)
            d_done = dom_counts.get(src, 0)
            capped_tasks += min(t_done, target)
            capped_doms  += min(d_done, target)

            t_pct = (min(t_done, target) / target) * 100
            d_pct = (min(d_done, target) / target) * 100

            status = "✅ MET" if (t_done >= target and d_done >= target) else "⏳ IN PROGRESS"
            if t_done == 0 and d_done == 0:
                status = "⏸️ QUEUED"

            print(f"{idx:<3} | {src:<45} | {target:<6} | {t_done:<5} ({t_pct:5.1f}%)   | {d_done:<5} ({d_pct:5.1f}%)   | {status}")

        print("=" * 115)
        task_overall_pct = (capped_tasks / total_target) * 100
        dom_overall_pct  = (capped_doms / total_target) * 100
        print(f"OVERALL 17-SOURCE PROGRESS:")
        print(f"  • Tasks Extracted:   {capped_tasks:,} / {total_target:,} ({task_overall_pct:.1f}%)")
        print(f"  • Domains Extracted: {capped_doms:,} / {total_target:,} ({dom_overall_pct:.1f}%)")
        print("=" * 115)
        print(f"Refreshed at {time.strftime('%Y-%m-%d %H:%M:%S')}. Press Ctrl+C to exit.")

        if capped_tasks >= total_target and capped_doms >= total_target:
            print("\n🎉 ALL 17 AGENTIC SOURCES COMPLETED FOR BOTH TASKS AND DOMAINS!")
            break

        time.sleep(10)

if __name__ == "__main__":
    main()
