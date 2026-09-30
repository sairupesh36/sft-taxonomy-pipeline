#!/bin/bash
NEEDLE='small cap stocks perform vs. large cap stocks (like Dow constituents) during bear trends'
OUT=/projects/data/datasets/code_data/sai_rupesh/taxonomy/lc_work/grep_special.out; : > $OUT
for d in no_natural_language uncertain; do
  cd /projects/data/datasets/code_data/sai_rupesh/taxonomy/sft_43_language_wise/$d
  ls *.jsonl | xargs -P 30 -I{} sh -c 'grep -F "'"$NEEDLE"'" {} | grep -F "3rd Friday" | head -c 400 | sed "s|^|'$d'/{} :: |"; true' >> $OUT 2>&1
done
echo FINISHED >> $OUT
