#!/bin/bash
# 2026-09-25: full stats for all 10 datasets -- o200k tokens, paths on every sheet, per-file context length
cd /projects/data/datasets/code_data/sai_rupesh/taxonomy/sft_audit
WORKERS=110 python3 sft_stats.py > stats_run_o200k_v2.log 2>&1
echo ALLDONE > run_stats_o200k_v2.done
