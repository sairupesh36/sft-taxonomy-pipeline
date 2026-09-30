"""Stage 6 -- final numbers. Reads hf-agentic-data/ and the stage JSON
reports, changes nothing, writes reports/STATS.md."""

import json
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(__file__))
from common import OUT_DIR, PIPE_DIR, REPORT_DIR  # noqa: E402


def stats(fname):
    p = os.path.join(OUT_DIR, fname)
    rows = tools = msgs = tool_calls = 0
    with open(p) as f:
        for line in f:
            d = json.loads(line)
            rows += 1
            tools += "tools" in d
            msgs += len(d["messages"])
            tool_calls += sum(len(m.get("tool_calls") or []) for m in d["messages"])
    return fname, dict(rows=rows, bytes=os.path.getsize(p), rows_with_tools=tools,
                       avg_msgs=round(msgs / rows, 1) if rows else 0, tool_calls=tool_calls)


def load(name):
    p = os.path.join(REPORT_DIR, name)
    return json.load(open(p)) if os.path.exists(p) else {}


def main():
    files = sorted(f for f in os.listdir(OUT_DIR) if f.endswith(".jsonl"))
    with ProcessPoolExecutor(max_workers=32) as ex:
        per = dict(ex.map(stats, files))
    tot = Counter()
    for v in per.values():
        for k in ("rows", "bytes", "rows_with_tools", "tool_calls"):
            tot[k] += v[k]
    s1, s3, s4, s4b, s5 = (load(n) for n in ("stage1_conversion_report.json", "stage3_audit_report.json",
                                            "stage4_cleanup_report.json", "stage4b_dedup_report.json",
                                            "stage5_glm_validation_report.json"))
    L = ["# hf-agentic-data -- final stats", "",
         f"- files: {len(files)}",
         f"- rows: {tot['rows']:,}",
         f"- size: {tot['bytes'] / 1e9:.1f} GB",
         f"- rows with a `tools` list: {tot['rows_with_tools']:,}",
         f"- total tool calls: {tot['tool_calls']:,}", ""]
    if s1:
        L += ["## Stage 1 -- convert",
              f"- source rows read: {s1['rows_in']:,}, rows written: {s1['rows_out']:,}",
              f"- datasets converted: {s1['datasets_converted']}, skipped: {len(s1['datasets_skipped'])}", ""]
    if s4:
        L += ["## Stage 4 -- cleanup", f"- {s4.get('totals')}", ""]
    if s4b:
        L += ["## Stage 4b -- dedup", f"- duplicate rows removed: {s4b.get('rows_dropped', 0):,}", ""]
    if s3:
        L += ["## Stage 3 -- final audit", f"- rows audited: {s3.get('total_rows', 0):,}",
              f"- violations: {s3.get('violation_counts')}", ""]
    if s5:
        L += ["## Stage 5 -- GLM template check",
              f"- sampled: {s5.get('total_sampled')}, fully OK: {s5.get('ok')}",
              f"- problems: {s5.get('error_counts')}", ""]
    L += ["## Per file", "", "| file | rows | GB | rows with tools | avg msgs | tool calls |", "|---|---:|---:|---:|---:|---:|"]
    for f, v in sorted(per.items(), key=lambda x: -x[1]["rows"]):
        L.append(f"| {f} | {v['rows']:,} | {v['bytes'] / 1e9:.2f} | {v['rows_with_tools']:,} | {v['avg_msgs']} | {v['tool_calls']:,} |")
    if s1:
        L += ["", "## Skipped datasets", ""] + [f"- **{k}** -- {v}" for k, v in sorted(s1["datasets_skipped"].items())]
    open(os.path.join(PIPE_DIR, "STATS.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L[:30]))


if __name__ == "__main__":
    main()
