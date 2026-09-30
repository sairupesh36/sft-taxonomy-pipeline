#!/usr/bin/env python3
"""
⚡ Antigravity Unified 36-Source Taxonomy Discovery Monitor
Tracks all 19 Tulu-3 SFT Sources + 17 Agentic SFT Sources (159,751 Total Target)
Displays live open-ended hierarchical discovery extractions across both Tasks and Domains.
"""

import os
import sys
import time
import subprocess
from collections import Counter
from pathlib import Path

BASE_DIR = Path("/projects/data/datasets/code_data/sai_rupesh/taxonomy")
RES_DIR  = BASE_DIR / "discovery_results"

# 36 Unified Sources (19 Tulu-3 + 17 Agentic)
SOURCES_INFO = [
    # ── Tulu-3 Mixture (19 Sources) ──────────────────────────────────────────
    ("coconot_converted", "ai2-adapt-dev/coconot_converted", 5000, "Tulu-3"),
    ("evol_codealpaca", "ai2-adapt-dev/evol_codealpaca_heval_decontaminated", 5000, "Tulu-3"),
    ("flan_v2", "ai2-adapt-dev/flan_v2_converted", 5000, "Tulu-3"),
    ("no_robots", "ai2-adapt-dev/no_robots_converted", 5000, "Tulu-3"),
    ("numinamath", "ai2-adapt-dev/numinamath_tir_math_decontaminated", 5000, "Tulu-3"),
    ("oasst1", "ai2-adapt-dev/oasst1_converted", 5000, "Tulu-3"),
    ("personahub_code", "ai2-adapt-dev/personahub_code_v2_34999", 5000, "Tulu-3"),
    ("personahub_ifdata", "ai2-adapt-dev/personahub_ifdata_manual_seed_v3_29980", 5000, "Tulu-3"),
    ("personahub_math", "ai2-adapt-dev/personahub_math_v5_regen_149960", 5000, "Tulu-3"),
    ("tulu_hard_coded", "ai2-adapt-dev/tulu_hard_coded_repeated_10", 240, "Tulu-3"),
    ("tulu_aya_100k", "ai2-adapt-dev/tulu_v3.9_aya_100k", 5000, "Tulu-3"),
    ("tulu_open_math_2", "ai2-adapt-dev/tulu_v3.9_open_math_2_gsm8k_50k", 5000, "Tulu-3"),
    ("personahub_algebra", "ai2-adapt-dev/tulu_v3.9_personahub_math_interm_algebra_20k", 5000, "Tulu-3"),
    ("tulu_sciriff_10k", "ai2-adapt-dev/tulu_v3.9_sciriff_10k", 5000, "Tulu-3"),
    ("synthetic_wildguard", "ai2-adapt-dev/tulu_v3.9_synthetic_finalresp_wildguardmixtrain_decontaminated_50k", 5000, "Tulu-3"),
    ("table_gpt_5k", "ai2-adapt-dev/tulu_v3.9_table_gpt_5k", 5000, "Tulu-3"),
    ("tulu_wildchat_100k", "ai2-adapt-dev/tulu_v3.9_wildchat_100k", 5000, "Tulu-3"),
    ("wildjailbreak", "ai2-adapt-dev/tulu_v3.9_wildjailbreak_decontaminated_50k", 5000, "Tulu-3"),
    ("personas_math_grade", "allenai/tulu-3-sft-personas-math-grade", 5000, "Tulu-3"),

    # ── Agentic Mixture (17 Sources) ─────────────────────────────────────────
    ("agent_flan_instruct_react", "agent_flan_agent_instruct_react", 1731, "Agentic"),
    ("agent_flan_instruct_tflan", "agent_flan_agent_instruct_tflan", 1731, "Agentic"),
    ("agent_flan_toolbench_react", "agent_flan_toolbench_react_10p", 2288, "Agentic"),
    ("agent_flan_toolbench_tflan", "agent_flan_toolbench_tflan_60p", 5000, "Agentic"),
    ("agent_flan_toolbench_cot", "agent_flan_toolbench_tflan_cot", 5000, "Agentic"),
    ("agent_flan_toolbench_3k", "agent_flan_toolbench_instruct_3k", 3000, "Agentic"),
    ("agent_flan_negative", "agent_flan_toolbench_negative", 761, "Agentic"),
    ("toolace_complex_tools", "toolace_complex_tools", 5000, "Agentic"),
    ("nemotron_interactive", "nemotron_interactive_agent", 5000, "Agentic"),
    ("nemotron_search", "nemotron_search_agent", 5000, "Agentic"),
    ("nemotron_tool_calling", "nemotron_tool_calling", 5000, "Agentic"),
    ("agentinstruct_tool_use", "agentinstruct_tool_use", 5000, "Agentic"),
    ("agentinstruct_webagent", "agentinstruct_webagent_flow", 5000, "Agentic"),
    ("agentinstruct_code", "agentinstruct_code_agent", 5000, "Agentic"),
    ("agentinstruct_rag", "agentinstruct_rag_agent", 5000, "Agentic"),
    ("agentinstruct_analytical", "agentinstruct_analytical_reasoning", 5000, "Agentic"),
    ("swe_bench_engineering", "swe_bench_software_engineering_agent", 5000, "Agentic"),
]

TOTAL_TARGET = sum(item[2] for item in SOURCES_INFO)  # 159,751 rows

C_RESET  = "\033[0m"
C_BOLD   = "\033[1m"
C_GREEN  = "\033[1;32m"
C_YELLOW = "\033[1;33m"
C_CYAN   = "\033[1;36m"
C_MAGENTA= "\033[1;35m"
C_DIM    = "\033[2m"

def get_counts(prefix):
    """Fast stream parsing of source column in discovery_results"""
    cmd = (
        f"kubectl exec $(kubectl get pods -l 'app in (taxonomy-discovery-tasks, taxonomy-discovery-domains)' "
        f"--field-selector=status.phase=Running -o jsonpath='{{.items[0].metadata.name}}' 2>/dev/null) -- "
        f"awk -F',' 'NR>1 {{print $2}}' /projects/data/datasets/code_data/sai_rupesh/taxonomy/discovery_results/discovery_{prefix}_shard_*.csv 2>/dev/null"
    )
    try:
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=8)
        cnt = Counter()
        for line in res.stdout.splitlines():
            s = line.strip()
            if s and s != "source":
                cnt[s] += 1
        return cnt
    except Exception:
        return Counter()

def render():
    t_cnt = get_counts("tasks")
    d_cnt = get_counts("domains")

    t_capped = sum(min(t_cnt.get(full, 0), cap) for _, full, cap, _ in SOURCES_INFO)
    d_capped = sum(min(d_cnt.get(full, 0), cap) for _, full, cap, _ in SOURCES_INFO)

    print("\033[2J\033[H", end="")
    w = 104
    print("=" * w)
    print(f"{C_BOLD}{C_CYAN} 🚀 UNIFIED 36-SOURCE TAXONOMY DISCOVERY MONITOR (159,751 SAMPLES) {C_RESET}".center(w + 14))
    print("=" * w)
    print(f" Datasets Covered: 19 Tulu-3 SFT (90,240) + 17 Agentic SFT (69,511) | 288 SGLang GPU Slots")
    print(f" Tasks Progress:   {C_GREEN}{t_capped:,} / {TOTAL_TARGET:,} ({(t_capped/TOTAL_TARGET*100):.1f}%){C_RESET}")
    print(f" Domains Progress: {C_GREEN}{d_capped:,} / {TOTAL_TARGET:,} ({(d_capped/TOTAL_TARGET*100):.1f}%){C_RESET}")
    print("-" * w)
    
    hdr = f" {'#':<2} │ {'Sub-Dataset Source Name':<28} │ {'Set':<7} │ {'Target':>6} │ {'Tasks (5k)':>14} │ {'Domains (5k)':>14} │ {'Status':<12}"
    print(hdr)
    print("─" * w)

    for idx, (short_name, full_name, cap, set_type) in enumerate(SOURCES_INFO, 1):
        t_val = t_cnt.get(full_name, 0)
        d_val = d_cnt.get(full_name, 0)

        t_eff = min(t_val, cap)
        d_eff = min(d_val, cap)

        t_pct = (t_eff / cap * 100) if cap else 0.0
        d_pct = (d_eff / cap * 100) if cap else 0.0

        t_str = f"{t_eff:,} ({t_pct:4.0f}%)"
        d_str = f"{d_eff:,} ({d_pct:4.0f}%)"

        if t_eff >= cap and d_eff >= cap:
            st = f"{C_GREEN}COMPLETED   {C_RESET}"
        elif t_val > 0 or d_val > 0:
            st = f"{C_YELLOW}DISCOVERING {C_RESET}"
        else:
            st = f"{C_DIM}PENDING     {C_RESET}"

        row_text = f" {idx:<2} │ {short_name:<28} │ {set_type:<7} │ {cap:>6,d} │ {t_str:>14} │ {d_str:>14} │ {st}"
        print(row_text)

    print("=" * w)
    print(f"{C_DIM} Auto-refreshes every 4s | Live Cluster Stream | Press Ctrl+C to detach{C_RESET}")

def main():
    try:
        while True:
            render()
            time.sleep(4.0)
    except KeyboardInterrupt:
        print(f"\n{C_GREEN}✓ Monitor detached.{C_RESET} Discovery pods continue running.")

if __name__ == "__main__":
    main()
