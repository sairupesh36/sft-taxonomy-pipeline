"""Same breakdown as turn_shape_scan.py but with tool-calling rows set aside first."""
import glob, json, collections, time
from multiprocessing import Pool
ROOT = "/projects/data/datasets/code_data/sai_rupesh/taxonomy/sft_43_language_wise_clean"
def scan(path):
    c = collections.Counter(); uc = collections.Counter(); tool_rows = 0
    with open(path, "rb") as f:
        for line in f:
            r = json.loads(line); m = r["messages"]
            if any(x.get("tool_calls") for x in m): tool_rows += 1; continue
            nu = sum(x["role"] == "user" for x in m); uc[min(nu, 5)] += 1
            c["single_turn (1 user message)" if nu == 1 else "multi_turn (2+ user messages)"] += 1
    return c, uc, tool_rows
if __name__ == "__main__":
    files = sorted(f for f in glob.glob(ROOT + "/*/*.jsonl") if not f.split("/")[-2].startswith("_")); t0 = time.time()
    tot = collections.Counter(); uc = collections.Counter(); tool_rows = 0
    with Pool(64) as p:
        for c, u, tr in p.imap_unordered(scan, files, chunksize=1): tot.update(c); uc.update(u); tool_rows += tr
    n = sum(tot.values()); print(f"{time.time()-t0:.0f}s | rows WITHOUT any tool call: {n:,}  |  rows WITH a tool call (set aside): {tool_rows:,}")
    print("\nof the non-tool rows:"); [print(f"  {k:32s} {v:>12,} ({100*v/n:.2f}%)") for k, v in tot.most_common()]
    print("\nuser-turn count among non-tool rows:"); [print(f"  {k}{'+' if k==5 else ''} user turn(s): {v:,} ({100*v/n:.2f}%)") for k, v in sorted(uc.items())]
