import os, sys, glob
sys.path.insert(0, "/projects/data/datasets/code_data/sai_rupesh/taxonomy/pylibs")
import pyarrow.parquet as pq, pyarrow.compute as pc, pyarrow as pa
from multiprocessing import Pool
NEEDLE = "small cap stocks perform vs. large cap stocks"
NEEDLE2 = "Saturday after the 3rd Friday"
def scan(path):
    hits = []
    try:
        pf = pq.ParquetFile(path); cols = [f.name for f in pf.schema_arrow if pa.types.is_string(f.type) or pa.types.is_large_string(f.type)]
        if not cols: return hits
        for b in pf.iter_batches(batch_size=50000, columns=cols):
            t = pa.Table.from_batches([b])
            for c in cols:
                m = pc.match_substring(t[c], NEEDLE)
                if pc.any(m).as_py():
                    for i in pc.indices_nonzero(m).to_pylist()[:3]: hits.append((path, {k: (str(t[k][i].as_py())[:2000]) for k in cols}))
    except Exception as e:
        pass
    return hits
if __name__ == "__main__":
    files = sorted(f for d in sys.argv[1:] for f in glob.glob(f"/projects/data/datasets/translation_data/SFT/OUTPUT/{d}/*.parquet"))
    print("files:", len(files), flush=True)
    with Pool(48) as p: res = p.map(scan, files, chunksize=1)
    found = [h for r in res for h in r]
    print("hits:", len(found))
    for path, row in found:
        blob = " ".join(row.values())
        print(f"{os.path.basename(path)[:60]:60s} has_think={'<think>' in blob}  answer_is_options_text={'3rd Friday' in blob}  cols={list(row)[:5]}")
