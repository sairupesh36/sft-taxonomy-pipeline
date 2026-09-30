"""Is every output file consistent with its recorded meta?  (files without meta, meta without file, line count != rows_kept)"""
import glob, json, os, subprocess, collections
from multiprocessing import Pool
ROOT = "/projects/data/datasets/code_data/sai_rupesh/taxonomy/sft_43_language_wise_clean"
def wc(p): return p, int(subprocess.run(["wc", "-l", p], capture_output=True, text=True).stdout.split()[0])
if __name__ == "__main__":
    files = {os.path.relpath(f, ROOT) for f in glob.glob(ROOT + "/*/*.jsonl") if not f.split("/")[-2].startswith("_")}
    metas = {}
    for mp in glob.glob(ROOT + "/_meta/*/*.json"):
        rel = os.path.relpath(mp, ROOT + "/_meta")[:-5]; metas[rel] = json.load(open(mp))["stats"]
    no_meta = sorted(files - set(metas)); no_file = sorted(set(metas) - files)
    with Pool(64) as p: counts = dict(p.map(wc, [os.path.join(ROOT, f) for f in files], chunksize=8))
    diff = {}
    for f in files & set(metas):
        rec = metas[f].get("rows_kept", 0); got = counts[os.path.join(ROOT, f)]
        if rec != got: diff[f] = (rec, got)
    print(f"output files {len(files):,} | meta records {len(metas):,}")
    print(f"files WITHOUT a meta record: {len(no_meta)} | meta records without a file: {len(no_file)}")
    print(f"files whose line count != recorded kept: {len(diff)} | rows recorded {sum(v[0] for v in diff.values()):,} vs written {sum(v[1] for v in diff.values()):,}")
    print("total rows written now:", f"{sum(counts.values()):,}", "| total recorded kept:", f"{sum(m.get('rows_kept',0) for m in metas.values()):,}")
    print("examples no_meta:", no_meta[:4]); print("examples diff:", list(diff.items())[:4])
    json.dump({"no_meta": no_meta, "diff": diff}, open("/projects/data/datasets/code_data/sai_rupesh/taxonomy/lc_work/state_check.json", "w"))
