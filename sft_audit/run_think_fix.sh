#!/bin/bash
# 2026-09-24: B2 re-converted with R1 self-talk moved into <think>; A + B2 re-cleaned with the new
# think_not_closed rule; audit on all datasets. Old versions kept under posttraining_openai_sft/_old_*.
set -e
T=/projects/data/datasets/code_data/sai_rupesh/taxonomy
P=$T/posttraining_openai_sft
cd $T
mv $P/B2_heuristic_mapped $P/_old_B2_heuristic_mapped_v1_nothink
mv $P/_convert_log_b2.jsonl $P/_old_convert_log_b2_v1.jsonl
python3 convert_posttraining_b2.py 48 > $P/b2_convert_v2.log 2>&1
cd $T/sft_audit
WORKERS=48 python3 sft_clean.py $P/B2_heuristic_mapped $T/_new_b2_openai_ready > clean_b2_v2.log 2>&1
WORKERS=32 python3 sft_clean.py $P/A_native_openai $T/_new_a_openai_ready > clean_a_v2.log 2>&1
mv $T/sft_posttraining_b2_heuristic_mapped_openai_ready $P/_old_final_b2_v1_nothink
mv $T/_new_b2_openai_ready $T/sft_posttraining_b2_heuristic_mapped_openai_ready
mv $T/sft_posttraining_a_native_openai_ready $P/_old_final_a_v1
mv $T/_new_a_openai_ready $T/sft_posttraining_a_native_openai_ready
python3 sft_audit.py > audit_run_think_fix.log 2>&1
echo ALLDONE > run_think_fix.done
