#!/bin/bash
cd /projects/data/datasets/code_data/sai_rupesh/taxonomy/sft_audit
WORKERS=32 python3 sft_clean.py /projects/data/datasets/code_data/sai_rupesh/taxonomy/posttraining_openai_sft/B1_marker_mapped /projects/data/datasets/code_data/sai_rupesh/taxonomy/sft_posttraining_b1_marker_mapped_openai_ready > clean_b1.log 2>&1
python3 sft_audit.py > audit_run_b1.log 2>&1
echo ALLDONE > run_clean_b1.done
