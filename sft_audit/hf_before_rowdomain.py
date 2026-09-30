"""Rows per domain of OUR final sft_hf_domain_data (pre-dedup), row by row, with the SAME per-row rule as
build_dedup_domain_folders.py (prompt key -> source-folder domain). No tokenizing, rows only. Read-only.

    python3 hf_before_rowdomain.py   -> reports/hf_domain_before_rowdomain_per_file.jsonl (resumes)
"""
import json, os, sys
from collections import Counter
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_dedup_domain_folders as Bd   # noqa: E402
from sft_audit import DATASETS            # noqa: E402

ROOT = DATASETS[Bd.HF]
OUT = os.path.join(HERE, "reports", "hf_domain_before_rowdomain_per_file.jsonl")


def one(rel):
    Bd._init()
    c = Counter()
    with open(os.path.join(ROOT, rel), "rb") as f:
        for line in f:
            if line.strip():
                c[Bd.hf_row_domain(line)] += 1
    return rel, dict(c)


def main():
    files = []
    for r, dirs, fs in os.walk(ROOT):
        dirs[:] = [x for x in dirs if not x.startswith("_")]
        files += [os.path.relpath(os.path.join(r, f), ROOT) for f in fs if f.endswith(".jsonl")]
    done = {json.loads(l)["file"] for l in open(OUT)} if os.path.exists(OUT) else set()
    todo = sorted((f for f in files if f not in done), key=lambda f: -os.path.getsize(os.path.join(ROOT, f)))
    print(len(files), "files,", len(todo), "to do", flush=True)
    with Pool(int(os.environ.get("WORKERS", 40))) as pool, open(OUT, "a") as fo:
        for i, (rel, c) in enumerate(pool.imap_unordered(one, todo), 1):
            fo.write(json.dumps({"file": rel, "domains": c}) + "\n"); fo.flush()
            if i % 200 == 0:
                print(f"[{i}/{len(todo)}]", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
