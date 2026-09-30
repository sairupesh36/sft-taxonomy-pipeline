#!/bin/bash
# waits for build_dedup_domain_folders.py and hf_before_rowdomain.py, then builds the before/after dedup workbook
cd /projects/data/datasets/code_data/sai_rupesh/taxonomy/sft_audit
until grep -q "^DONE" build_dedup_domain_folders.log && grep -q "^DONE" hf_before_rowdomain.log; do sleep 30; done
python3 dedup_domain_excel.py > dedup_domain_excel.log 2>&1
echo "EXCEL_DONE rc=$? $(date)" >> dedup_domain_excel.log
