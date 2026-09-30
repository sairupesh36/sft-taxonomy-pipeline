#!/bin/bash
# One-screen status of every running SFT data job.
# Live view:  watch -n 30 bash /projects/data/datasets/code_data/sai_rupesh/taxonomy/hf_agentic_pipeline/status.sh
T=/projects/data/datasets/code_data/sai_rupesh/taxonomy
L=$T/hf_agentic_pipeline/logs
C=$T/sft_output_data_coverage

run() { pgrep -f "$1" >/dev/null && echo "RUNNING" || echo "stopped"; }
prog() { grep -o "^\[[0-9]*/[0-9]*\]" "$1" 2>/dev/null | tail -1; }
fin() { grep -q "^TOTAL\|^done\|^sampled\|^# hf-agentic" "$1" 2>/dev/null && echo "DONE"; }

echo "=== $(date '+%H:%M:%S') ==="
echo
echo "1) OpenAI-format pass (v1 + hf-agentic)   [$(run to_openai_format.py)] $(prog $L/to_openai_format.log) $(fin $L/to_openai_format.log)"
for t in $T/traces_v1_final/*.oaitmp $T/hf-agentic-data/*.oaitmp; do
  [ -f "$t" ] && echo "     writing: $(basename ${t%.oaitmp})  $(du -h "$t" | cut -f1) of $(du -h "${t%.oaitmp}" | cut -f1)"
done
echo
echo "2) Final checks chain                    [$(run final_checks.sh)]"
for s in final_hf_agentic_audit final_v1_audit final_sft43_audit final_validate final_report; do
  f=$L/$s.log
  if [ -f $f ]; then echo "     $s: $(prog $f) $(fin $f)"; else echo "     $s: waiting"; fi
done
[ -f $L/final_checks.done ] && echo "     >>> ALL FINAL CHECKS DONE"
echo
echo "3) SFT/OUTPUT/data coverage scan         [$(run coverage.py)] $(tail -1 $C/coverage.log 2>/dev/null)"
[ -f $C/coverage_stats.json ] && echo "     >>> coverage stats written"
echo
echo "CPU load: $(cut -d' ' -f1-3 /proc/loadavg)   free disk: $(df -h /projects/data | awk 'NR==2{print $4}')"
