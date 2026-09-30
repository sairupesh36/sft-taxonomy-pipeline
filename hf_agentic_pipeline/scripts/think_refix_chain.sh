#!/bin/bash
# restore every file the first think-fix changed from its backup, then
# re-run the corrected fix (answer-inside-think rows are unwrapped, not dropped)
cd /projects/data/datasets/code_data/sai_rupesh/taxonomy/hf_agentic_pipeline/scripts
T=/projects/data/datasets/code_data/sai_rupesh/taxonomy
L=../logs
restore() {  # backup_dir final_dir
  (cd "$1" && find . -name "*.jsonl" -print) | while read f; do cp -p "$1/$f" "$2/$f"; echo "restored $f"; done
}
restore $T/hf_agentic_pipeline/backup_before_think_fix $T/hf-agentic-data > $L/think_restore.log
restore $T/sft_43_language_wise_clean/_backup_before_think_fix $T/sft_43_language_wise_clean >> $L/think_restore.log
restore $T/sft_traces_v1/data/backup_before_think_fix $T/traces_v1_final >> $L/think_restore.log
python3 fix_think_issues.py $T/hf-agentic-data > $L/think_fix_hf.log 2>&1
python3 fix_think_issues.py $T/sft_43_language_wise_clean > $L/think_fix_sft43.log 2>&1
python3 fix_think_issues.py $T/traces_v1_final > $L/think_fix_v1.log 2>&1
AUDIT_WORKERS=64 python3 audit.py $T/hf-agentic-data final_hf_agentic_audit.json > $L/final_hf_agentic_audit.log 2>&1
AUDIT_WORKERS=64 python3 audit.py $T/sft_43_language_wise_clean final_sft43_audit.json > $L/final_sft43_audit.log 2>&1
AUDIT_WORKERS=64 python3 audit.py $T/traces_v1_final final_v1_audit.json > $L/final_v1_audit.log 2>&1
echo ALLDONE > $L/think_refix_chain.done
