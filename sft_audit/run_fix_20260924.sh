#!/bin/bash
cd /projects/data/datasets/code_data/sai_rupesh/taxonomy/sft_audit
T=/projects/data/datasets/code_data/sai_rupesh/taxonomy
python3 fixes/fix_20260924_shapes.py $T/sft_43_language_wise_clean --backup-dir $T/sft_43_language_wise_clean/_backup_before_shape_fix > fixes/fix_20260924_sft43.log 2>&1
python3 fixes/fix_20260924_shapes.py $T/traces_v1_final --backup-dir $T/sft_traces_v1/data/backup_before_shape_fix > fixes/fix_20260924_v1.log 2>&1
python3 fixes/fix_20260924_shapes.py $T/hf-agentic-data --backup-dir $T/hf_agentic_pipeline/backup_before_shape_fix > fixes/fix_20260924_hf.log 2>&1
python3 sft_audit.py > audit_run.log 2>&1
echo ALLDONE > run_fix_20260924.done
