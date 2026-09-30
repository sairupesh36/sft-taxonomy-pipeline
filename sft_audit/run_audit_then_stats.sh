#!/bin/bash
# 2026-09-25: datasets 9/10 now point at the traces_team final folders. Audit those two first,
# then rerun the full stats -> hf domain -> before/after -> domain pipeline on all 10.
cd /projects/data/datasets/code_data/sai_rupesh/taxonomy/sft_audit
rm -f run_audit_then_stats.progress run_stats_domain_all.progress run_stats_domain_all.done
echo "audit start $(date)" >> run_audit_then_stats.progress
WORKERS=110 python3 sft_audit.py sft_datasets_sft_openai_ready sft_synthetic_data_openai_ready > audit_traces_team_9_10.log 2>&1
echo "audit done rc=$? $(date)" >> run_audit_then_stats.progress
./run_stats_domain_all.sh
echo "pipeline done $(date)" >> run_audit_then_stats.progress
