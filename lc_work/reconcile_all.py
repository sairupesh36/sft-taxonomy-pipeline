import os, json, glob, subprocess, sys
from multiprocessing import Pool
def nl(p):
    if not os.path.exists(p): return 0
    return int(subprocess.run(["wc","-l",p],capture_output=True,text=True).stdout.split()[0])
def job(rel): return rel, nl(rel), nl(f"_rejects_context/{rel}")
def kept_of(s):
    for k in ("rows_kept","kept","rows_out"):
        if k in s: return s[k]
    return 0
if __name__ == "__main__":
    files = sorted(glob.glob("*/*.jsonl")); exp = {}; nometa = 0
    for rel in files:
        mp = f"_meta/{rel}.json"
        if os.path.exists(mp): exp[rel] = kept_of(json.load(open(mp))["stats"])
        else: nometa += 1
    with Pool(48) as p: res = p.map(job, files, chunksize=8)
    same = dropped_ok = 0; redone = []; bad = []; total_written = 0; total_rec = 0; total_rej = 0
    for rel, kept, rej in res:
        total_written += kept; total_rej += rej
        e = exp.get(rel)
        if e is None: bad.append((rel, "no meta", kept, rej)); continue
        total_rec += e
        if kept == e and rej == 0: same += 1
        elif kept == e and rej > 0: redone.append(rel)            # rows are back, drops not applied
        elif kept + rej == e: dropped_ok += 1
        else: bad.append((rel, e, kept, rej))
    print(f"files {len(files)} | no meta {nometa}")
    print(f"  untouched, count equals filter record:            {same}")
    print(f"  shortened only by post-pass drops (explained):    {dropped_ok}")
    print(f"  restored by killed rerun (post-pass not applied): {len(redone)}")
    print(f"  UNEXPLAINED:                                      {len(bad)}  {bad[:5]}")
    print(f"rows: filter recorded {total_rec:,} | written now {total_written:,} | rows sitting in the redone chunks that the post-pass would drop: {sum(r for rel,k,r in res if rel in set(redone)):,}")
    json.dump({"redone": redone, "bad": bad}, open("../lc_work/reconcile_all.json", "w"))
