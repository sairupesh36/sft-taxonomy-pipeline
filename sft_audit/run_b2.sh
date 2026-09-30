#!/bin/bash
# B2: convert (heuristic split) -> clean -> register -> audit all datasets
T=/projects/data/datasets/code_data/sai_rupesh/taxonomy
cd $T
python3 convert_posttraining_b2.py 48 > posttraining_openai_sft/b2_convert.log 2>&1
cd $T/sft_audit
WORKERS=32 python3 sft_clean.py $T/posttraining_openai_sft/B2_heuristic_mapped $T/sft_posttraining_b2_heuristic_mapped_openai_ready > clean_b2.log 2>&1
grep -q sft_posttraining_b2_heuristic_mapped_openai_ready sft_audit.py || \
  sed -i 's|^    "sft_posttraining_a_native_openai_ready": f"{T}/sft_posttraining_a_native_openai_ready",|&\n    "sft_posttraining_b2_heuristic_mapped_openai_ready": f"{T}/sft_posttraining_b2_heuristic_mapped_openai_ready",|' sft_audit.py
python3 sft_audit.py > audit_run_b2.log 2>&1
echo ALLDONE > run_b2.done
