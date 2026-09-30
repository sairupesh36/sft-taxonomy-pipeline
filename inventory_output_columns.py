"""READ-ONLY. Schema/metadata-only inventory of the original OUTPUT parquet files
(no row data read): for every file, which columns exist, which the production
normalizer (sft_normalize.plan) actually consumes, and which are ignored.
Writes results only under this project folder."""
import os, sys, glob, json, collections
from multiprocessing import Pool
sys.path.insert(0, "/projects/data/datasets/code_data/sai_rupesh/taxonomy/pylibs")
sys.path.insert(0, "/projects/data/datasets/translation_data/SFT/DEDUP_PIPELINE")
import pyarrow.parquet as pq
import sft_normalize as N
OUT = "/projects/data/datasets/translation_data/SFT/OUTPUT"

def consumed(p):
    used = set()
    def walk(v):
        if isinstance(v, str): used.add(v)
        elif isinstance(v, (list, tuple, set)):
            for x in v: walk(x)
    for k, v in p.items():
        if k != "mode": walk(v)
    return used

def one(path):
    try:
        pf = pq.ParquetFile(path)
        cols = list(pf.schema_arrow.names); nrows = pf.metadata.num_rows
        pf.close()
    except Exception as e:
        return dict(path=path, err=str(e)[:80])
    p = N.plan(cols)
    used = consumed(p) if p.get("mode") else set()
    low = {c.lower(): c for c in cols}
    used_l = {u.lower() for u in used}
    return dict(path=os.path.relpath(path, OUT), domain=os.path.relpath(path, OUT).split("/")[0],
                rows=nrows, mode=p.get("mode"), cols=cols,
                unused=[c for c in cols if c.lower() not in used_l])

if __name__ == "__main__":
    files = glob.glob(OUT + "/*/*.parquet")
    print("files:", len(files), flush=True)
    with Pool(48) as pool:
        res = pool.map(one, files, chunksize=64)
    json.dump(res, open("output_column_inventory.json", "w"))
    print("saved", len(res))
