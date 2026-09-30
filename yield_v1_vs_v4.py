"""READ-ONLY. For every OUTPUT parquet file: sample rows (up to 3 row groups x 100 rows),
run the production v1 normalizer and our v4 normalizer through the same
to_messages -> canonicalize_turns -> validate path, record success rate. Then
estimated convertible rows = file_rows x success_rate."""
import os, sys, json, glob, random
from multiprocessing import Pool
sys.path.insert(0, "/projects/data/datasets/code_data/sai_rupesh/taxonomy/pylibs")
sys.path.insert(0, "/projects/data/datasets/translation_data/SFT/DEDUP_PIPELINE")
sys.path.insert(1, "/projects/data/datasets/code_data/sai_rupesh/taxonomy")
import pyarrow.parquet as pq
import sft_schema as S
import sft_normalize as N1
import sft_normalize_v4 as N4
OUT = "/projects/data/datasets/translation_data/SFT/OUTPUT"

def rate(N, p, rows):
    ok = n = 0
    for r in rows:
        n += 1
        try:
            raw = N.to_messages(r, p)
            if not raw: continue
            if S.validate(S.canonicalize_turns(raw)) is None: ok += 1
        except Exception:
            pass
    return ok, n

def one(path):
    rel = os.path.relpath(path, OUT)
    res = dict(path=rel, domain=rel.split('/')[0])
    try:
        pf = pq.ParquetFile(path); cols = list(pf.schema_arrow.names); res['rows'] = pf.metadata.num_rows
        ng = pf.num_row_groups
        rows = []
        for gi in sorted({0, ng // 2, ng - 1}):
            for b in pf.iter_batches(batch_size=100, row_groups=[gi]):
                rows += b.to_pylist(); break
        pf.close()
        p1, p4 = N1.plan(cols), N4.plan(cols)
        res['mode1'], res['mode4'] = p1.get('mode'), p4.get('mode')
        res['n'] = len(rows)
        res['ok1'] = rate(N1, p1, rows)[0] if p1.get('mode') else 0
        res['ok4'] = rate(N4, p4, rows)[0] if p4.get('mode') else 0
    except Exception as e:
        res['err'] = str(e)[:70]
    return res

if __name__ == '__main__':
    files = glob.glob(OUT + '/*/*.parquet'); random.seed(1); random.shuffle(files)
    with Pool(56) as pool: out = pool.map(one, files, chunksize=8)
    json.dump(out, open('yield_v1_vs_v4.json', 'w')); print('done', len(out), flush=True)
