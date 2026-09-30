#!/bin/bash
cd /projects/data/datasets/code_data/sai_rupesh/taxonomy/hf_agentic_pipeline/scripts
T=/projects/data/datasets/code_data/sai_rupesh/taxonomy
L=../logs
while [ ! -f $L/final_checks2.done ]; do sleep 30; done
python3 fix_think_issues.py $T/hf-agentic-data --backup-dir $T/hf_agentic_pipeline/backup_before_think_fix > $L/think_fix_hf.log 2>&1
python3 fix_think_issues.py $T/sft_43_language_wise_clean --backup-dir $T/sft_43_language_wise_clean/_backup_before_think_fix > $L/think_fix_sft43.log 2>&1
python3 fix_think_issues.py $T/traces_v1_final --backup-dir $T/sft_traces_v1/data/backup_before_think_fix > $L/think_fix_v1.log 2>&1
AUDIT_WORKERS=64 python3 audit.py $T/hf-agentic-data final_hf_agentic_audit.json > $L/final_hf_agentic_audit.log 2>&1
AUDIT_WORKERS=64 python3 audit.py $T/sft_43_language_wise_clean final_sft43_audit.json > $L/final_sft43_audit.log 2>&1
AUDIT_WORKERS=64 python3 audit.py $T/traces_v1_final final_v1_audit.json > $L/final_v1_audit.log 2>&1
echo ALLDONE > $L/think_fix_chain.done
