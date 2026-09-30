#!/bin/bash
# 2026-09-25: drop think_not_closed rows from 5 datasets (only files the audit flagged, backups in
# <dataset>/_backup_before_rule_drop/), then audit all 10, then stats with real o200k token counts.
T=/projects/data/datasets/code_data/sai_rupesh/taxonomy
A=$T/sft_audit/reports/audit_20260925_002933.json
cd $T/sft_audit
for d in sft_hf_traces_data_openai_ready sft_datasets_sft_openai_ready sft_hf_agentic_data_openai_ready sft_posttraining_b1_marker_mapped_openai_ready sft_code_math_text_openai_ready; do
  WORKERS=48 python3 fixes/drop_failing_rows.py $T/$d $A >> fixes/drop_think_not_closed_20260925.log 2>&1
done
python3 sft_audit.py > audit_run_after_drop.log 2>&1
WORKERS=110 python3 sft_stats.py > stats_run_o200k.log 2>&1
echo ALLDONE > run_drop_think_and_stats_o200k.done
