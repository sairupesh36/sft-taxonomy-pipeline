"""How many labels does the real production run actually assign per row, for domain and task_family?"""
import glob, json, collections
from multiprocessing import Pool
ROOT = "/projects/data/datasets/code_data/sai_rupesh/taxonomy/sft_43_language_wise_classified"
def scan(path):
    dc = collections.Counter(); tc = collections.Counter(); ex3 = []
    with open(path, "rb") as f:
        for line in f:
            r = json.loads(line)
            dc[len(r.get("domain") or [])] += 1
            tc[len(r.get("task_family") or [])] += 1
            if len(r.get("domain") or []) >= 3 and len(ex3) < 3:
                ex3.append((r["domain"], next((m["content"][:150] for m in r["messages"] if m["role"] == "user"), "")))
    return dc, tc, ex3
if __name__ == "__main__":
    files = glob.glob(ROOT + "/*.jsonl")
    dc = collections.Counter(); tc = collections.Counter(); ex3 = []
    with Pool(64) as p:
        for d, t, e in p.imap_unordered(scan, files, chunksize=2):
            dc.update(d); tc.update(t); ex3 += e[:3 - len(ex3)] if len(ex3) < 3 else []
    n = sum(dc.values())
    print(f"total rows {n:,}")
    print("\ndomain label count distribution:")
    for k, v in sorted(dc.items()): print(f"  {k} labels: {v:,} ({100*v/n:.3f}%)")
    print("\ntask_family label count distribution:")
    for k, v in sorted(tc.items()): print(f"  {k} labels: {v:,} ({100*v/n:.3f}%)")
    print("\nexamples with 3+ domain labels:", ex3)
