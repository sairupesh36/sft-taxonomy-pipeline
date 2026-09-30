"""In-place pass for datasets that are ALREADY cleaned: drop every row that
fails any sft_audit.py rule (all checks except INFO_CHECKS). Same gate as
step 6 of sft_clean.py. Only files with at least one failing row are
rewritten; each is backed up first to <dataset>/_backup_before_rule_drop/
(the "_" prefix keeps the audit and cleaner from reading it).

    python3 drop_failing_rows.py DATASET_DIR [AUDIT_JSON]

With AUDIT_JSON (an sft_audit report) only the files that report lists with
a rule failure are opened -- much faster on big datasets. .jsonl and
.jsonl.gz both supported (2026-09-25).
"""

import gzip
import json
import os
import shutil
import sys
from collections import Counter
from multiprocessing import Pool

sys.path.insert(0, "/projects/data/datasets/code_data/sai_rupesh/taxonomy/sft_audit")
import sft_audit as AUDIT  # noqa: E402


def run(args):
    top, rel = args
    path = os.path.join(top, rel)
    c = Counter()
    tmp = path + ".droptmp"
    gz = path.endswith(".gz")
    with (gzip.open(path, "rt") if gz else open(path)) as fi, \
            (gzip.open(tmp, "wt", compresslevel=3) if gz else open(tmp, "w")) as fo:
        for line in fi:
            if not line.strip():
                continue
            failed = AUDIT.check_row(json.loads(line)) - AUDIT.INFO_CHECKS
            if failed:
                c["rows_dropped"] += 1
                for f in failed:
                    c[f"fails:{f}"] += 1
            else:
                fo.write(line)
    if c["rows_dropped"]:
        b = os.path.join(top, "_backup_before_rule_drop", rel)
        os.makedirs(os.path.dirname(b), exist_ok=True)
        shutil.copy2(path, b)
        os.replace(tmp, path)
    else:
        os.remove(tmp)
    return rel, dict(c)


def main():
    top = sys.argv[1].rstrip("/")
    jobs = []
    for r, dirs, fs in os.walk(top):
        dirs[:] = [d for d in dirs if not d.startswith("_")]
        jobs += [(top, os.path.relpath(os.path.join(r, f), top)) for f in fs if f.endswith((".jsonl", ".jsonl.gz"))]
    if len(sys.argv) > 2:
        rep = json.load(open(sys.argv[2]))
        name = next(k for k, v in AUDIT.DATASETS.items() if v.rstrip("/") == top)
        rules = set(AUDIT.CHECKS) - AUDIT.INFO_CHECKS
        bad = {rel for rel, v in rep["datasets"][name]["per_file"].items() if set(v["checks"]) & rules}
        jobs = [j for j in jobs if j[1] in bad]
        print(f"{name}: {len(jobs)} files with rule failures in {os.path.basename(sys.argv[2])}", flush=True)
    tot = Counter()
    with Pool(int(os.environ.get("WORKERS", 64))) as pool:
        for rel, c in pool.imap_unordered(run, jobs):
            tot.update(c)
            if c:
                print(rel, c, flush=True)
    print("TOTAL", dict(tot), flush=True)


if __name__ == "__main__":
    main()
