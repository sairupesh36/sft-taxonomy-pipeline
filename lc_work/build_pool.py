"""Random pool of English target rows, scored by the current domain/task classifiers (top-2 labels + confidence)."""
import sys, os, json, random, re, hashlib
sys.path.insert(0, "/projects/data/datasets/code_data/sai_rupesh/taxonomy/pylibs")
sys.path.insert(0, "/projects/data/datasets/code_data/sai_rupesh/taxonomy")
sys.path.insert(0, "/projects/data/datasets/code_data/sai_rupesh/taxonomy/classifier_experiments")
from multiprocessing import Pool
EN = "/projects/data/datasets/code_data/sai_rupesh/taxonomy/sft_43_language_wise/en"
CE = "/projects/data/datasets/code_data/sai_rupesh/taxonomy/classifier_experiments"
PER = 1600

def sample_shard(fn):
    rnd = random.Random(hash(fn) % 100000 + 77); path = f"{EN}/{fn}"; size = os.path.getsize(path); out = []; tries = 0
    with open(path, "rb") as f:
        while len(out) < PER and tries < PER * 4:
            tries += 1; f.seek(rnd.randint(0, size - 1)); f.readline(); line = f.readline()
            if not line: continue
            try: d = json.loads(line)
            except Exception: continue
            if d.get("messages"): out.append(dict(shard=fn.split("_openai")[0], messages=d["messages"]))
    return out

if __name__ == "__main__":
    with Pool(24) as p: chunks = p.map(sample_shard, sorted(os.listdir(EN)))
    rows = [r for c in chunks for r in c]; print("sampled", len(rows), flush=True)
    import numpy as np
    _o = np.array
    def _c(*a, **k):
        if k.get("copy") is False: k["copy"] = None
        return _o(*a, **k)
    np.array = _c
    import fasttext
    from map_batch2 import to_full_text
    seen = {hashlib.md5(r["full_text"].encode()).hexdigest() for r in json.load(open("lc_work/target_run_80.json"))}
    md = fasttext.load_model(f"{CE}/domain_fasttext.bin"); mt = fasttext.load_model(f"{CE}/task_family_fasttext.bin")
    clean = lambda t: re.sub(r"\s+", " ", t.replace("\n", " ").replace("\t", " ")).strip()
    n = 0
    with open("lc_work/pool_scored.jsonl", "w") as out:
        for r in rows:
            full = to_full_text(r["messages"]); h = hashlib.md5(full.encode()).hexdigest()
            if h in seen or not full.strip(): continue
            seen.add(h); t = clean(full)
            dl, dp = md.predict(t, k=2); tl, tp = mt.predict(t, k=2)
            out.write(json.dumps(dict(h=h, shard=r["shard"], full_text=full,
                d1=dl[0][9:], dc1=round(float(dp[0]), 3), d2=dl[1][9:] if len(dl) > 1 else None, dc2=round(float(dp[1]), 3) if len(dp) > 1 else 0,
                t1=tl[0][9:], tc1=round(float(tp[0]), 3), t2=tl[1][9:] if len(tl) > 1 else None, tc2=round(float(tp[1]), 3) if len(tp) > 1 else 0)) + "\n"); n += 1
    print("pool written", n, flush=True)
