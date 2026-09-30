#!/bin/bash
cd /projects/data/datasets/code_data/sai_rupesh/taxonomy/sft_audit
for d in sft_hf_domain_data_openai_ready sft_hf_traces_data_openai_ready sft_hf_agentic_data_openai_ready sft_code_math_text_openai_ready sft_rl_reference_data_openai_ready; do
  echo "== $d" >> rule_drop.log
  python3 fixes/drop_failing_rows.py /projects/data/datasets/code_data/sai_rupesh/taxonomy/$d >> rule_drop.log 2>&1
done
python3 sft_audit.py > audit_run.log 2>&1
echo ALLDONE > run_rule_drop.done
