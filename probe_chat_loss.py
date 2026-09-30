"""READ-ONLY. For every 'chat'-mode OUTPUT file, read a small batch and check what the
production normalizer would throw away: extra keys inside messages (tool_calls, name,
tool_call_id...), empty-content assistant turns (dropped), tool-role turns, a `tools` column."""
import os, sys, json, glob, collections
from multiprocessing import Pool
sys.path.insert(0, "/projects/data/datasets/code_data/sai_rupesh/taxonomy/pylibs")
sys.path.insert(0, "/projects/data/datasets/translation_data/SFT/DEDUP_PIPELINE")
import pyarrow.parquet as pq
import sft_normalize as N
OUT = "/projects/data/datasets/translation_data/SFT/OUTPUT"
inv = {r['path']: r for r in json.load(open('output_column_inventory.json')) if 'err' not in r}

def probe(rel):
    r = inv[rel]; path = os.path.join(OUT, rel)
    res = dict(path=rel, rows=r['rows'], domain=r['domain'], n=0, extra_keys=collections.Counter(),
               empty_asst=0, tool_role=0, cols=r['cols'])
    try:
        pf = pq.ParquetFile(path); p = N.plan(r['cols'])
        col = p['col']
        for b in pf.iter_batches(batch_size=200, columns=[col]):
            for row in b.to_pylist()[:200]:
                raw = N.coerce_chat(row.get(col))
                if not raw: continue
                res['n'] += 1
                seen_empty = seen_tool = False
                for t in raw:
                    if isinstance(t, dict):
                        for k, v in t.items():
                            if k not in ('role','content','from','value','text','message','speaker','sender') and v not in (None, '', [], {}):
                                res['extra_keys'][k] += 1
                        role = str(t.get('role') or t.get('from') or '').lower()
                        c = t.get('content') or t.get('value') or t.get('text') or t.get('message') or ''
                        if role in ('assistant','gpt') and not str(c).strip(): seen_empty = True
                        if role in ('tool','function','observation','tool_response'): seen_tool = True
                res['empty_asst'] += seen_empty; res['tool_role'] += seen_tool
            break
        pf.close()
    except Exception as e:
        res['err'] = str(e)[:60]
    res['extra_keys'] = dict(res['extra_keys'])
    return res

if __name__ == '__main__':
    chat = [k for k, r in inv.items() if r['mode'] == 'chat']
    with Pool(48) as pool: out = pool.map(probe, chat, chunksize=4)
    json.dump(out, open('chat_loss_probe.json', 'w'))
    print('probed', len(out))
