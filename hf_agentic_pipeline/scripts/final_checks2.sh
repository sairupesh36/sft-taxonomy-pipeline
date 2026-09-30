#!/bin/bash
cd /projects/data/datasets/code_data/sai_rupesh/taxonomy/hf_agentic_pipeline/scripts
T=/projects/data/datasets/code_data/sai_rupesh/taxonomy
AUDIT_WORKERS=64 python3 audit.py $T/hf-agentic-data final_hf_agentic_audit.json > ../logs/final_hf_agentic_audit.log 2>&1
AUDIT_WORKERS=64 python3 audit.py $T/sft_43_language_wise_clean final_sft43_audit.json > ../logs/final_sft43_audit.log 2>&1
while pgrep -f "to_openai_format.py" >/dev/null; do sleep 30; done
AUDIT_WORKERS=64 python3 audit.py $T/traces_v1_final final_v1_audit.json > ../logs/final_v1_audit.log 2>&1
echo ALLDONE > ../logs/final_checks2.done
