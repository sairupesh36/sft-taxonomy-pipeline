"""Stage 4b -- exact-duplicate removal across ALL output files (new vs v1).

Added because the v1 data turned out to hold 22 byte-identical duplicate
files (41,709 rows) that nothing in its pipeline looked for, and because
this source has several datasets built from the same upstream data
(WildChat vs WildChat-1M, the two LiteCoder releases, dpo-mix vs
UltraFeedback...).

A row is a duplicate if its canonical JSON (messages + tools, keys sorted)
hashes the same as a row already kept. Files are processed in a fixed
order (sorted by name) so the result is reproducible; the first copy wins.
Pass 1 hashes every row in parallel; pass 2 rewrites only files that lose
rows.
"""

import hashlib
import json
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(__file__))
from common import OUT_DIR, REPORT_DIR  # noqa: E402


def key(line):
    d = json.loads(line)
    return hashlib.md5(json.dumps(d, sort_keys=True, ensure_ascii=False).encode()).digest()


def hash_file(fname):
    with open(os.path.join(OUT_DIR, fname)) as f:
        return fname, [key(l) for l in f]


def rewrite(args):
    fname, keep_mask = args
    path = os.path.join(OUT_DIR, fname)
    tmp = path + ".deduptmp"
    with open(path) as fi, open(tmp, "w") as fo:
        for line, k in zip(fi, keep_mask):
            if k:
                fo.write(line)
    if any(keep_mask):
        os.replace(tmp, path)
    else:
        os.remove(tmp)
        os.remove(path)
    return fname


def main():
    files = sorted(f for f in os.listdir(OUT_DIR) if f.endswith(".jsonl"))
    with ProcessPoolExecutor(max_workers=32) as ex:
        hashes = dict(ex.map(hash_file, files))
    seen = set()
    masks = {}
    dropped = Counter()
    dup_of = Counter()
    first_file = {}
    for f in files:
        m = []
        for h in hashes[f]:
            if h in seen:
                m.append(False)
                dropped[f] += 1
                dup_of[(f, first_file[h])] += 1
            else:
                seen.add(h)
                first_file[h] = f
                m.append(True)
        masks[f] = m
    todo = [(f, masks[f]) for f in files if dropped[f]]
    with ProcessPoolExecutor(max_workers=16) as ex:
        list(ex.map(rewrite, todo))
    total_in = sum(len(v) for v in hashes.values())
    rep = {"rows_in": total_in, "rows_dropped": sum(dropped.values()),
           "rows_out": total_in - sum(dropped.values()),
           "dropped_per_file": dict(dropped.most_common()),
           "duplicate_of": {f"{a} <- {b}": n for (a, b), n in dup_of.most_common()}}
    json.dump(rep, open(f"{REPORT_DIR}/stage4b_dedup_report.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != "duplicate_of"}, indent=1)[:3000])


if __name__ == "__main__":
    main()
