import os, json, subprocess, glob, collections
from multiprocessing import Pool
ROOT = "/projects/data/datasets/code_data/sai_rupesh/taxonomy/sft_43_language_wise_clean"
def count(rel):
    r = subprocess.run(["wc", "-l", os.path.join(ROOT, rel)], capture_output=True, text=True)
    return rel, int(r.stdout.split()[0])
if __name__ == "__main__":
    metas = {}
    for mp in glob.glob(ROOT + "/_meta/**/*.json", recursive=True):
        rel = os.path.relpath(mp, ROOT + "/_meta")[:-5]; d = json.load(open(mp)); metas[rel] = d["stats"].get("rows_kept", 0)
    with Pool(64) as p: counts = dict(p.map(count, list(metas), chunksize=8))
    diff = {k: (metas[k], counts[k]) for k in metas if metas[k] != counts[k]}
    print(f"files compared: {len(metas)} | files whose written lines differ from the recorded kept count: {len(diff)}")
    print(f"sum recorded {sum(metas.values()):,} | sum written {sum(counts.values()):,} | gap {sum(metas.values()) - sum(counts.values()):,}")
    if "--fix" in __import__("sys").argv:
        for k in diff:
            os.remove(os.path.join(ROOT, "_meta", k + ".json"))
        json.dump(sorted(diff), open("/projects/data/datasets/code_data/sai_rupesh/taxonomy/lc_work/bad_chunks.json", "w"))
        print(f"removed the records of {len(diff)} damaged chunks so the filter re-does them")
    by = collections.Counter()
    for k, (m, c) in diff.items(): by[k.split('/')[0]] += m - c
    print("gap by folder:", dict(by.most_common(6)))
    for k, (m, c) in sorted(diff.items(), key=lambda kv: -(kv[1][0] - kv[1][1]))[:6]: print(f"  {k}: recorded {m:,} written {c:,} (missing {m-c:,})")
