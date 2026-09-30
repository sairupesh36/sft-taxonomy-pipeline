#!/bin/bash
# 2026-09-25: clean traces_team/synthetic_data_openai_sft into taxonomy, then audit all 10 datasets
cd /projects/data/datasets/code_data/sai_rupesh/taxonomy/sft_audit
python3 sft_clean.py /projects/data/datasets/traces_team/synthetic_data_openai_sft /projects/data/datasets/code_data/sai_rupesh/taxonomy/sft_synthetic_data_openai_ready > clean_synthetic.log 2>&1
# wait for the all-9 audit another session started at 23:49, so two audits don't fight for the disk
while pgrep -f "python3 sft_audit.py" > /dev/null; do sleep 30; done
python3 sft_audit.py > audit_run_all10.log 2>&1
echo ALLDONE > run_synthetic_and_audit10.done
