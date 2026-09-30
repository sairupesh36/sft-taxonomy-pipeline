#!/bin/bash
# 2026-09-25 (restart): o200k stats for all 10 -> hf_domain row-level domain pass -> before/after workbook
# -> domain workbook. Started with setsid/nohup so it survives the Claude session ending.
T=/projects/data/datasets/code_data/sai_rupesh/taxonomy
cd $T/sft_audit
WORKERS=122 python3 sft_stats.py > stats_run_o200k_v3.log 2>&1 || { echo STATS_FAILED > run_stats_domain_all.done; exit 1; }
echo "stats done $(date)" >> run_stats_domain_all.progress
cd $T/sft_loss_report/v2
WORKERS=100 python3 hf_domain_domainwise.py count > hf_domain_count.log 2>&1
echo "hf domain pass done $(date)" >> $T/sft_audit/run_stats_domain_all.progress
python3 build_before_after_all10.py > build_before_after_all10.log 2>&1
echo "before/after done $(date)" >> $T/sft_audit/run_stats_domain_all.progress
cd $T/sft_audit
python3 sft_domain_stats.py > domain_stats_run.log 2>&1
echo "domain stats done $(date)" >> run_stats_domain_all.progress
echo ALLDONE > run_stats_domain_all.done
