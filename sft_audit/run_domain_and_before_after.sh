#!/bin/bash
# 2026-09-25: after the o200k stats run: hf_domain row-level domain pass, then the before/after workbook,
# then the domain workbook (both read the stats per-file table).
T=/projects/data/datasets/code_data/sai_rupesh/taxonomy
cd $T/sft_audit
while [ ! -f run_stats_o200k_v2.done ]; do sleep 60; done
cd $T/sft_loss_report/v2
WORKERS=100 python3 hf_domain_domainwise.py count > hf_domain_count.log 2>&1
python3 build_before_after_all10.py > build_before_after_all10.log 2>&1
cd $T/sft_audit
python3 sft_domain_stats.py > domain_stats_run.log 2>&1
echo ALLDONE > run_domain_and_before_after.done
