#!/usr/bin/env python3
"""Run the domain + task_family fastText classifiers (lc_work/models/*_v2_AB.bin) over the
CLEAN sft output and write each row back out with two added fields: "domain" and "task_family"
(each a list of 0-2 label strings, empty list if the model was too unsure to guess at all --
see fasttext_predict_utils.predict_with_fallback). Input rows and their order are untouched
beyond that; nothing is dropped, filtered, or deduplicated here.

Same chunked / resumable / atomic-write shape as sft_clean_filter.py: big files are split into
byte-range chunks, each chunk's finished output + a _meta/<name>.json record are written
atomically, and a chunk whose meta already exists is skipped on resume.

Usage:
  python3 sft_classify_domain_task.py --selftest
  python3 sft_classify_domain_task.py --in sft_43_language_wise_clean/en --in sft_43_language_wise_clean/uncertain \
      --in sft_43_language_wise_clean/no_natural_language --out sft_43_language_wise_classified \
      --threshold 0.8 --workers 32
"""
import argparse, glob, json, os, re, sys, time, collections
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "classifier_experiments"))

import numpy as np
_orig_np_array = np.array                                   # same fasttext/numpy>=2.0 compat shim as fasttext_baseline.py
def _np_array_compat(*a, **k):
    if k.get("copy") is False:
        k["copy"] = None
    return _orig_np_array(*a, **k)
np.array = _np_array_compat

from fasttext_predict_utils import predict_capped

MAX_CHARS = 3500                                              # same budget as map_batch2.to_full_text (the labeling pipeline this model was trained to match)
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lc_work", "models")
DOMAIN_MODEL_PATH = os.path.join(MODEL_DIR, "domain_fasttext_v2_AB.bin")
TASK_MODEL_PATH = os.path.join(MODEL_DIR, "task_family_fasttext_v2_AB.bin")
_MODELS = {}                                                  # loaded once per worker process, not per chunk


def get_models():
    if not _MODELS:
        import fasttext
        _MODELS["domain"] = fasttext.load_model(DOMAIN_MODEL_PATH)
        _MODELS["task_family"] = fasttext.load_model(TASK_MODEL_PATH)
    return _MODELS["domain"], _MODELS["task_family"]


def to_full_text(messages, max_chars=MAX_CHARS):
    """Identical logic to classifier_experiments/map_batch2.py's to_full_text: real conversation
    content ([USER]/[ASSISTANT]/[TOOL]) always kept in full first, the system prompt truncated or
    dropped to make room -- this is the exact feature shape the model was trained on."""
    system_parts = [f"[{m['role'].upper()}]: {m.get('content') or ''}" for m in messages if m["role"] == "system"]
    other_parts = [f"[{m['role'].upper()}]: {m.get('content') or ''}" for m in messages if m["role"] != "system"]
    other_text = "\n".join(other_parts)
    system_text = "\n".join(system_parts)
    if not other_text:
        return system_text[:max_chars] + " ...[truncated]" if len(system_text) > max_chars else system_text
    if len(other_text) > max_chars:
        return other_text[:max_chars] + " ...[truncated]"
    remaining = max_chars - len(other_text) - 1
    if system_text and remaining > 0:
        if len(system_text) > remaining:
            system_text = system_text[:remaining] + " ...[truncated]"
        return system_text + "\n" + other_text
    return other_text


def clean_text(text):
    """Identical to fasttext_baseline.py's clean_text -- must match training-time preprocessing."""
    text = text.replace("\n", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text.replace("__label__", "")


def classify_row(row, threshold):
    text = clean_text(to_full_text(row["messages"]))
    domain_model, task_model = get_models()
    domain = predict_capped(domain_model, text, threshold=threshold)
    task_family = predict_capped(task_model, text, threshold=threshold)
    row["domain"] = domain
    row["task_family"] = task_family
    return row


def out_name(rel, idx, nchunks):
    base = rel.replace("/", "__")
    return base if nchunks == 1 else f"{base}__c{idx:04d}"


def run_chunk(args):
    path, root, out_dir, threshold, start, end, idx, nchunks = args
    rel = os.path.relpath(path, root)
    name = out_name(rel, idx, nchunks)
    meta = os.path.join(out_dir, "_meta", name + ".json")
    if os.path.exists(meta):
        d = json.load(open(meta))
        return name, d["stats"], d["domain_counts"], d["task_family_counts"], True
    outp = os.path.join(out_dir, name); os.makedirs(os.path.dirname(outp), exist_ok=True); tmp = outp + ".tmp"
    st = collections.Counter(); dc = collections.Counter(); tc = collections.Counter(); t0 = time.time()
    with open(tmp, "w", encoding="utf-8", buffering=1 << 22) as fout, open(path, "rb", buffering=1 << 22) as f:
        if start > 0:
            f.seek(start - 1); f.readline()
        while f.tell() < end:
            l = f.readline()
            if not l:
                break
            st["rows_in"] += 1
            try:
                row = json.loads(l)
                row = classify_row(row, threshold)
            except Exception:
                st["errors"] += 1
                continue
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            st["rows_out"] += 1
            dc.update(row["domain"] or ["__NONE__"])
            tc.update(row["task_family"] or ["__NONE__"])
    os.replace(tmp, outp)
    st["seconds"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(meta), exist_ok=True)
    json.dump({"stats": dict(st), "domain_counts": dict(dc), "task_family_counts": dict(tc)}, open(meta + ".tmp", "w"))
    os.replace(meta + ".tmp", meta)
    return name, dict(st), dict(dc), dict(tc), False


def gather(paths):
    files = []
    for p in paths:
        files += sorted(glob.glob(os.path.join(p, "**", "*.jsonl"), recursive=True)) if os.path.isdir(p) else [p]
    return files


def selftest():
    print("loading models ...", flush=True)
    dm, tm = get_models()
    cases = [
        ([{"role": "user", "content": "Prove that the square root of 2 is irrational."},
          {"role": "assistant", "content": "Suppose for contradiction sqrt(2) = a/b in lowest terms..."}], "mathematics"),
        ([{"role": "user", "content": "Write a Python function that reverses a linked list."},
          {"role": "assistant", "content": "def reverse_list(head):\n    prev = None\n    while head:\n        ..."}], "software_engineering"),
    ]
    for msgs, expect in cases:
        row = classify_row({"messages": msgs}, threshold=0.8)
        print(f"  expect~{expect:20s} -> domain={row['domain']} task_family={row['task_family']}")
        assert expect in row["domain"], (expect, row["domain"])
    empty = classify_row({"messages": [{"role": "user", "content": "asdkj alksjd"}, {"role": "assistant", "content": "??"}]}, threshold=0.999)
    assert empty["domain"] == [] and empty["task_family"] == [], "an impossible threshold must fall back to empty, never crash"
    print("SELFTEST OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", action="append", default=[])
    ap.add_argument("--out")
    ap.add_argument("--threshold", type=float, default=0.8)
    ap.add_argument("--chunk-mb", type=int, default=256)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--sample", type=int, default=0, help="dry run: classify the first N rows of each input file, print examples, write nothing")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    files = gather(a.inp)
    if not files:
        sys.exit("no input files")
    if a.sample:
        for p in a.inp:                                       # sample from EACH input dir separately, not just the first files overall
            cand = [f for f in sorted(glob.glob(os.path.join(p, "**", "*.jsonl"), recursive=True)) if os.path.getsize(f) > 0]
            for fp in cand[:2]:
                print(f"\n=== {fp}")
                shown = 0
                with open(fp, "rb") as f:
                    for l in f:
                        if shown >= a.sample:
                            break
                        row = classify_row(json.loads(l), a.threshold)
                        u = next((m["content"] for m in row["messages"] if m["role"] == "user" and isinstance(m.get("content"), str)), "")
                        print(f"  domain={row['domain']!s:60s} task_family={row['task_family']!s:45s} | {u[:90]!r}")
                        shown += 1
        return
    if not a.out:
        sys.exit("--out is required unless --sample is used")
    root = a.inp[0] if len(a.inp) == 1 else os.path.commonpath([os.path.dirname(f) for f in files])
    cb = a.chunk_mb * (1 << 20); tasks = []
    for f in files:
        size = os.path.getsize(f); n = max(1, -(-size // cb))
        tasks += [(f, root, a.out, a.threshold, i * cb, (i + 1) * cb if i < n - 1 else 1 << 62, i, n) for i in range(n)]
    print(f"{len(files)} input files -> {len(tasks)} chunks, {a.workers} workers", flush=True)
    tot = collections.Counter(); dc = collections.Counter(); tc = collections.Counter(); done = skipped = 0; t0 = time.time()
    with Pool(a.workers) as pool:
        for name, st, d, t, resumed in pool.imap_unordered(run_chunk, tasks, chunksize=1):
            done += 1; skipped += resumed
            if not resumed:
                tot.update({k: v for k, v in st.items() if k != "seconds"}); dc.update(d); tc.update(t)
            if done % 50 == 0 or done == len(tasks):
                el = time.time() - t0
                print(f"[{done}/{len(tasks)} chunks, {skipped} resumed] {time.strftime('%H:%M:%S')}  rows_out {tot['rows_out']:,}  elapsed {el/60:.1f} min", flush=True)
    if a.out:                                                 # totals over EVERY finished chunk, including earlier runs
        tot = collections.Counter(); dc = collections.Counter(); tc = collections.Counter()
        for mp in glob.glob(os.path.join(a.out, "_meta", "**", "*.json"), recursive=True):
            d = json.load(open(mp)); tot.update(d["stats"]); dc.update(d["domain_counts"]); tc.update(d["task_family_counts"])
    n = max(1, tot.get("rows_out", 1))
    print(f"\nrows_in {tot.get('rows_in', 0):,}  rows_out {tot.get('rows_out', 0):,}  errors {tot.get('errors', 0):,}")
    print(f"rows with NO domain label at all: {dc.get('__NONE__', 0):,} ({100*dc.get('__NONE__', 0)/n:.2f}%)")
    print(f"rows with NO task_family label at all: {tc.get('__NONE__', 0):,} ({100*tc.get('__NONE__', 0)/n:.2f}%)")
    print("\ntop 10 domains:"); [print(f"  {k:40s} {v:>12,}") for k, v in sorted(dc.items(), key=lambda kv: -kv[1])[:10] if k != "__NONE__"]
    print("\ntop 10 task_families:"); [print(f"  {k:40s} {v:>12,}") for k, v in sorted(tc.items(), key=lambda kv: -kv[1])[:10] if k != "__NONE__"]
    json.dump({"totals": dict(tot), "domain_counts": dict(dc), "task_family_counts": dict(tc), "threshold": a.threshold},
               open(os.path.join(a.out, "_classify_report.json"), "w"), indent=1, ensure_ascii=False)
    print("report ->", os.path.join(a.out, "_classify_report.json"))


if __name__ == "__main__":
    main()
