"""Read-only scan of traces_v1_final/ for two bug classes first found while
building hf-agentic-data:
  (a) tool_calls[].function.arguments containing null-valued keys -- the
      parquet struct-union artifact (every key seen anywhere in the column
      shows up as null in every row), e.g. Nanbeige__ToolMind.
  (b) assistant messages with empty content AND no tool_calls -- what the
      v1 converter produces from the legacy OpenAI `function_call` field,
      which it never read.
Changes nothing; writes one JSON report.
"""
import json, os, sys
from concurrent.futures import ProcessPoolExecutor, as_completed

SRC = "/projects/data/datasets/code_data/sai_rupesh/taxonomy/traces_v1_final"
OUT = "/projects/data/datasets/code_data/sai_rupesh/taxonomy/hf_agentic_pipeline/reports/v1_null_args_empty_assistant_scan.json"


def scan(fname):
    rows = null_arg_rows = null_arg_calls = total_calls = empty_asst_rows = empty_asst_msgs = 0
    ex_null = ex_empty = None
    with open(os.path.join(SRC, fname)) as f:
        for ln, line in enumerate(f):
            rows += 1
            msgs = json.loads(line)["messages"]
            row_null = row_empty = False
            for m in msgs:
                if m.get("role") != "assistant":
                    continue
                tcs = m.get("tool_calls") or []
                for tc in tcs:
                    total_calls += 1
                    args = (tc.get("function") or {}).get("arguments")
                    if isinstance(args, dict) and any(v is None for v in args.values()):
                        null_arg_calls += 1
                        row_null = True
                        if ex_null is None:
                            ex_null = {"line": ln, "call": tc}
                if not tcs and not (m.get("content") or "").strip() and not m.get("reasoning_content"):
                    empty_asst_msgs += 1
                    row_empty = True
                    if ex_empty is None:
                        ex_empty = {"line": ln}
            null_arg_rows += row_null
            empty_asst_rows += row_empty
    return fname, dict(rows=rows, total_calls=total_calls, null_arg_calls=null_arg_calls,
                       null_arg_rows=null_arg_rows, empty_asst_msgs=empty_asst_msgs,
                       empty_asst_rows=empty_asst_rows, ex_null=ex_null, ex_empty=ex_empty)


def main():
    files = sorted(f for f in os.listdir(SRC) if f.endswith(".jsonl"))
    res = {}
    with ProcessPoolExecutor(max_workers=32) as ex:
        futs = [ex.submit(scan, f) for f in files]
        for i, fu in enumerate(as_completed(futs), 1):
            name, r = fu.result()
            if r["null_arg_calls"] or r["empty_asst_msgs"]:
                res[name] = r
            print(f"[{i}/{len(files)}] {name} null_calls={r['null_arg_calls']} empty_asst={r['empty_asst_msgs']}", flush=True)
    tot = {k: sum(v[k] for v in res.values()) for k in ("null_arg_calls", "null_arg_rows", "empty_asst_msgs", "empty_asst_rows")}
    json.dump({"totals": tot, "files_affected": len(res), "per_file": res}, open(OUT, "w"), indent=1, default=str)
    print("TOTALS", tot, "files affected", len(res))


if __name__ == "__main__":
    main()
