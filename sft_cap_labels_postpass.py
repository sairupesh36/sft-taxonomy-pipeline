#!/usr/bin/env python3
"""One-time fix for rows that were classified before the multi-label cap existed. The saved output
only stores the label STRINGS (alphabetically sorted), not their scores, so a row with 3+ labels
can't be trimmed by just slicing the list -- we don't know which ones the model was most confident
about. This re-predicts ONLY on the small fraction of rows that actually have >2 domain or >2
task_family labels (0.014% / 0.023% of the 502.8M rows -- checked directly, see lc_work/
label_count_check.py), using the model's own fresh scores to keep its top-2 real picks. Everything
else is untouched. Atomic per-file rewrite (temp file + rename), like every other post-pass in this
project. Dry-run unless --apply."""
import argparse, glob, json, os, sys, time, collections
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "classifier_experiments"))
import sft_classify_domain_task as C
from fasttext_predict_utils import predict_capped

def work(args):
    path, threshold, apply = args
    dm, tm = C.get_models()
    n_dom = n_tf = rows = 0
    tmp = path + ".cap_tmp"; out = open(tmp, "wb") if apply else None
    with open(path, "rb") as f:
        for line in f:
            rows += 1
            need = False
            if apply:
                r = json.loads(line)
                if len(r.get("domain") or []) > 2:
                    text = C.clean_text(C.to_full_text(r["messages"])); r["domain"] = predict_capped(dm, text, threshold); n_dom += 1; need = True
                if len(r.get("task_family") or []) > 2:
                    text = C.clean_text(C.to_full_text(r["messages"])); r["task_family"] = predict_capped(tm, text, threshold); n_tf += 1; need = True
                out.write((json.dumps(r, ensure_ascii=False) + "\n").encode("utf-8") if need else line)
            else:
                r = json.loads(line)
                n_dom += len(r.get("domain") or []) > 2
                n_tf += len(r.get("task_family") or []) > 2
    if out:
        out.close()
        if n_dom or n_tf:
            os.replace(tmp, path)
        else:
            os.remove(tmp)
    return path, n_dom, n_tf, rows

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/projects/data/datasets/code_data/sai_rupesh/taxonomy/sft_43_language_wise_classified")
    ap.add_argument("--threshold", type=float, default=0.8)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--workers", type=int, default=48)
    a = ap.parse_args()
    files = sorted(f for f in glob.glob(os.path.join(a.root, "*.jsonl")))
    t0 = time.time(); total_d = total_t = total_rows = 0; touched = 0
    with Pool(a.workers) as p:
        for i, (path, nd, nt, rows) in enumerate(p.imap_unordered(work, [(f, a.threshold, a.apply) for f in files], chunksize=1), 1):
            total_d += nd; total_t += nt; total_rows += rows; touched += bool(nd or nt)
            if i % 1000 == 0: print(f"[{i}/{len(files)}] {time.time()-t0:.0f}s  fixed_so_far domain={total_d:,} task_family={total_t:,}", flush=True)
    print(f"{'APPLIED' if a.apply else 'DRY-RUN'}: rows_scanned={total_rows:,}  domain rows with >2 labels: {total_d:,}  "
          f"task_family rows with >2 labels: {total_t:,}  files touched: {touched:,}  ({time.time()-t0:.0f}s)")
    rp = os.path.join(a.root, "_classify_report.json")
    if a.apply and os.path.exists(rp):
        d = json.load(open(rp)); d.setdefault("post_steps", []).append(f"multi_label_cap_applied: domain={total_d}, task_family={total_t} rows re-capped at 2 (sft_cap_labels_postpass.py)")
        json.dump(d, open(rp + ".tmp", "w"), indent=1, ensure_ascii=False); os.replace(rp + ".tmp", rp)
