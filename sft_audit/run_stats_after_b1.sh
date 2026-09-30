#!/bin/bash
cd /projects/data/datasets/code_data/sai_rupesh/taxonomy/sft_audit
while [ ! -f run_clean_b1.done ]; do sleep 60; done
WORKERS=48 python3 sft_stats.py > stats_run.log 2>&1
echo ALLDONE > run_stats.done
