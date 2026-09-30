#!/bin/bash
cd /projects/data/datasets/code_data/sai_rupesh/taxonomy/sft_audit
WORKERS=40 python3 sft_clean.py /projects/data/datasets/traces_team/datasets_sft_openai_sft/by_dataset /projects/data/datasets/code_data/sai_rupesh/taxonomy/sft_datasets_sft_openai_ready > clean_datasets_sft.log 2>&1
echo ALLDONE > run_clean_datasets_sft.done
