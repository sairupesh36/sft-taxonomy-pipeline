#!/bin/bash
NEEDLE='small cap stocks perform vs. large cap stocks (like Dow constituents) during bear trends'
cd /projects/data/datasets/code_data/sai_rupesh/taxonomy/sft_43_language_wise/en
ls *.jsonl | xargs -P 30 -I{} sh -c 'r=$(grep -F -m 2 -c "'"$NEEDLE"'" {}); echo "{} $r"' > /projects/data/datasets/code_data/sai_rupesh/taxonomy/lc_work/grep_needle.out 2>&1
echo FINISHED >> /projects/data/datasets/code_data/sai_rupesh/taxonomy/lc_work/grep_needle.out
