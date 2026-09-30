#!/bin/bash
# 2026-09-25: after the all-10 audit finishes, refresh sft_stats for all 10 datasets
cd /projects/data/datasets/code_data/sai_rupesh/taxonomy/sft_audit
while [ ! -f run_synthetic_and_audit10.done ]; do sleep 60; done
WORKERS=48 python3 sft_stats.py > stats_run_all10.log 2>&1
echo ALLDONE > run_stats_all10.done
