"""Re-run the real filter (same settings as the full run) on ONLY the chunks whose meta record I deleted.
Outputs go to lc_work/meta_rebuild (scratch); only the small _meta JSON files are copied into the real output folder."""
import json, os, shutil, sys, time
from multiprocessing import Pool
sys.path.insert(0, "/projects/data/datasets/code_data/sai_rupesh/taxonomy")
import sft_clean_filter as F
T = "/projects/data/datasets/code_data/sai_rupesh/taxonomy"
IN = f"{T}/sft_43_language_wise"; OUT = f"{T}/sft_43_language_wise_clean"; SCR = f"{T}/lc_work/meta_rebuild"
cfg = json.load(open(f"{OUT}/_filter_report.json"))["settings"]
nometa = json.load(open(f"{T}/lc_work/reconcile.json"))["nometa"]
tasks = []
for rel in nometa:
    lang, name = rel.split("/"); cb = 256 << 20
    if "__p" in name:
        base, p = name[:-6].rsplit("__p", 1); i = int(p)
    else:                                   # small source file: one chunk, output keeps the source name
        base, i = name[:-6], 0
    src = f"{IN}/{lang}/{base}.jsonl"
    n = max(1, -(-os.path.getsize(src) // cb))
    tasks.append((src, IN, SCR, cfg, i*cb, (i+1)*cb if i < n-1 else 1 << 62, i, n))
if __name__ == "__main__":
    shutil.rmtree(SCR, ignore_errors=True); t0 = time.time(); done = 0
    with Pool(64) as pool:
        for r in pool.imap_unordered(F.run_chunk, tasks, chunksize=1):
            done += 1
            if done % 50 == 0 or done == len(tasks): print(f"[{done}/{len(tasks)}] {(time.time()-t0)/60:.1f} min", flush=True)
    copied = 0
    for rel in nometa:
        src = f"{SCR}/_meta/{rel}.json"; dst = f"{OUT}/_meta/{rel}.json"
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy(src, dst); copied += 1
    print(f"copied {copied} meta records back into the real output folder", flush=True)
    shutil.rmtree(SCR, ignore_errors=True)
    print("DONE rebuild", flush=True)
