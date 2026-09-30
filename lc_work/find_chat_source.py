import os, sys, json
sys.path.insert(0, "/projects/data/datasets/code_data/sai_rupesh/taxonomy/pylibs")
import pyarrow.parquet as pq
from multiprocessing import Pool
OUT = "/projects/data/datasets/translation_data/SFT/OUTPUT"
Q = "small cap stocks perform vs. large cap stocks"; A = "Saturday after the 3rd Friday"
inv = [r for r in json.load(open("output_column_inventory.json")) if "err" not in r]
def scan(r):
    if r["mode"] != "chat": return []
    col = next(c for c in r["cols"] if c.lower() in ("messages","conversations","conversation","turns","dialogue","exchanges"))
    hits = []
    try:
        pf = pq.ParquetFile(os.path.join(OUT, r["path"])); idx = 0
        for b in pf.iter_batches(batch_size=20000, columns=[col]):
            for i, x in enumerate(b.column(0).to_pylist()):
                s = x if isinstance(x, str) else json.dumps(x, ensure_ascii=False)
                if Q in s:
                    hits.append((r["path"], idx + i, A in s, "<think>" in s, s[:0]))
            idx += b.num_rows
    except Exception as e:
        return [(r["path"], -1, False, False, "ERR " + str(e)[:50])]
    return hits
if __name__ == "__main__":
    files = [r for r in inv if r["mode"] == "chat" and r["domain"] in sys.argv[1:]]
    print("chat-mode files to search:", len(files), "rows:", sum(r["rows"] for r in files), flush=True)
    with Pool(48) as p: res = p.map(scan, files, chunksize=1)
    hits = [h for x in res for h in x]; print("files/rows containing the question:", len(hits))
    for h in hits[:20]: print(f"  {h[0][:70]:70s} row {h[1]:>8}  has_options_answer={h[2]}  has_think={h[3]}  {h[4]}")
