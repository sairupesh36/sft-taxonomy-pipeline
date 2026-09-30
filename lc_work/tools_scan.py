"""Every tool-call row in the output: does it have a tools schema, or does the text define the tools some other way?"""
import glob, json, os, collections, time
from multiprocessing import Pool
ROOT = "/projects/data/datasets/code_data/sai_rupesh/taxonomy/sft_43_language_wise_clean"
def scan(path):
    c = collections.Counter(); names_no_tools = collections.Counter(); arg_keys = collections.Counter()
    with open(path, "rb") as f:
        for line in f:
            if b'"tool_calls"' not in line: continue
            r = json.loads(line); m = r["messages"]
            names = {tc["function"]["name"] for x in m for tc in (x.get("tool_calls") or [])}
            c["tool_rows"] += 1
            if r.get("tools"): c["with_tools_field"] += 1; continue
            c["no_tools_field"] += 1
            txt = " ".join(x["content"] for x in m if x["role"] in ("system", "user") and isinstance(x.get("content"), str))
            if names == {"python"}:
                c["no_tools__only_python"] += 1
                for x in m:
                    for tc in x.get("tool_calls") or []:
                        try: arg_keys[tuple(sorted(json.loads(tc["function"]["arguments"])))] += 1
                        except Exception: arg_keys["unparseable"] += 1
            elif all(n in txt for n in names): c["no_tools__defined_in_text"] += 1
            else:
                c["no_tools__undefined_other"] += 1
                for n in names: names_no_tools[n] += 1
    return c, names_no_tools, arg_keys
if __name__ == "__main__":
    files = sorted(f for f in glob.glob(ROOT + "/*/*.jsonl") if not f.split("/")[-2].startswith("_")); t0 = time.time()
    tot = collections.Counter(); nn = collections.Counter(); ak = collections.Counter()
    with Pool(64) as p:
        for c, n, a in p.imap_unordered(scan, files, chunksize=1): tot.update(c); nn.update(n); ak.update(a)
    print(f"{time.time()-t0:.0f}s |", dict(tot)); print("python-only rows: argument key sets:", ak.most_common(4)); print("other tools called with NO definition anywhere (top 12):", nn.most_common(12))
