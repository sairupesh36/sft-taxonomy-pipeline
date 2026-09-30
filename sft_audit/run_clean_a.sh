#!/bin/bash
cd /projects/data/datasets/code_data/sai_rupesh/taxonomy/sft_audit
WORKERS=32 python3 sft_clean.py /projects/data/datasets/code_data/sai_rupesh/taxonomy/posttraining_openai_sft/A_native_openai /projects/data/datasets/code_data/sai_rupesh/taxonomy/sft_posttraining_a_native_openai_ready > clean_a.log 2>&1
python3 sft_audit.py > audit_run_a.log 2>&1
echo ALLDONE > run_clean_a.done
