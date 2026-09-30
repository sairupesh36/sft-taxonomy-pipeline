#!/bin/bash
cd /projects/data/datasets/code_data/sai_rupesh/taxonomy/sft_audit
python3 sft_clean.py /projects/data/datasets/traces_team/code_math_text_translations_openai_sft /projects/data/datasets/code_data/sai_rupesh/taxonomy/sft_code_math_text_openai_ready > clean_code_math_text.log 2>&1
python3 sft_clean.py /projects/data/datasets/traces_team/RL_reference_data_from_SFT_openai_sft /projects/data/datasets/code_data/sai_rupesh/taxonomy/sft_rl_reference_data_openai_ready > clean_rl_reference.log 2>&1
while [ ! -f run_rule_drop.done ]; do sleep 30; done
python3 sft_audit.py > audit_run_final.log 2>&1
echo ALLDONE > run_reclean.done
