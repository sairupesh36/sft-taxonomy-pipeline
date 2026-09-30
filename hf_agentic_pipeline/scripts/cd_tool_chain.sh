#!/bin/bash
cd /projects/data/datasets/code_data/sai_rupesh/taxonomy/hf_agentic_pipeline/scripts
T=/projects/data/datasets/code_data/sai_rupesh/taxonomy
L=../logs
python3 fix_countdown_and_tools.py $T/traces_v1_final --backup-dir $T/sft_traces_v1/data/backup_before_countdown_tool_fix > $L/cd_tool_fix_v1.log 2>&1
python3 fix_countdown_and_tools.py $T/hf-agentic-data --backup-dir $T/hf_agentic_pipeline/backup_before_tool_fix > $L/cd_tool_fix_hf.log 2>&1
AUDIT_WORKERS=64 python3 audit.py $T/traces_v1_final final_v1_audit.json > $L/final_v1_audit.log 2>&1
AUDIT_WORKERS=64 python3 audit.py $T/hf-agentic-data final_hf_agentic_audit.json > $L/final_hf_agentic_audit.log 2>&1
echo ALLDONE > $L/cd_tool_chain.done
