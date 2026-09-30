#!/usr/bin/env python3
"""Drop SFT rows whose question points at content that is NOT in the row (missing-context rows).

Why: some source datasets keep the real input in a separate column (DNA sequence, image, protein...) and
the text only carries a placeholder such as <DNA> or <image>. After conversion the answer refers to
something the model can never see, so the row teaches guessing/hallucination.

Rule (checked on USER messages only, tag must be bare, not inside backticks):
    <image> <DNA> <RNA> <protein> <smiles> <mol> <seq>
(<img>/<audio>/<video> are NOT used: they are ordinary HTML tags in code questions -> false positives.)

Design for scale: step 1 = one fast byte-regex over each whole file (C speed, no JSON parsing);
step 2 = only files with a hit are parsed and rewritten (atomic temp file + rename).
Files without a hit are never touched. Dry-run by default; use --apply to modify.
(Finding NEW lost-context sources is a separate, report-only job: lc_work/context_probe3.py flags prompts
repeated many times with different answers. Those are suspects for a human to review, never auto-dropped.)
"""
import argparse, collections, glob, hashlib, json, mmap, os, re, sys, time
from multiprocessing import Pool

TAGS = "image|DNA|RNA|protein|smiles|mol|seq"
BYTE_RE = re.compile(rb"(?<![`\w/])<(?:%s)>(?![`\w])" % TAGS.encode(), re.I)
STR_RE = re.compile(r"(?<![`\w/])<(%s)>(?![`\w])" % TAGS, re.I)


def bad_tag(msgs):
    for m in msgs:
        if m.get("role") == "user" and isinstance(m.get("content"), str):
            t = STR_RE.search(m["content"])
            if t:
                return t.group(1).lower()
    return None


def scan_file(args):
    path, apply, rejects_dir = args
    with open(path, "rb") as f:
        try:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        except ValueError:          # empty file
            return path, 0, 0, {}, []
        has_hit = BYTE_RE.search(mm) is not None
        mm.close()
    if not has_hit:
        return path, 0, 0, {}, False
    rows = 0; dropped = collections.Counter(); kept = 0
    tmp = path + ".ctx_tmp"; rej = None
    if has_hit and apply:
        lang_dir = os.path.join(rejects_dir, path.split("/")[-2])
        os.makedirs(lang_dir, exist_ok=True)
        rej = open(os.path.join(lang_dir, os.path.basename(path)), "w", encoding="utf-8")
        out = open(tmp, "w", encoding="utf-8")
    else:
        out = None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows += 1
            if has_hit:
                d = json.loads(line)
                tag = bad_tag(d["messages"])
                if tag:
                    dropped[tag] += 1
                    if rej: rej.write(line)
                    continue
            kept += 1
            if out: out.write(line)
    if out:
        out.close(); rej.close()
        os.replace(tmp, path)
    return path, rows, kept, dict(dropped), has_hit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--report", default=None)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    files = sorted(f for f in glob.glob(os.path.join(a.root, "*", "*.jsonl")) if not f.split("/")[-2].startswith("_"))
    if a.limit: files = files[:a.limit]
    rejects_dir = os.path.join(a.root, "_rejects_context")
    t0 = time.time()
    tot_rows = tot_kept = 0; by_tag = collections.Counter(); by_lang = collections.Counter(); touched = []
    with Pool(a.workers) as pool:
        for i, (p, rows, kept, dropped, has_hit) in enumerate(pool.imap_unordered(scan_file, [(f, a.apply, rejects_dir) for f in files], chunksize=1), 1):
            tot_rows += rows; tot_kept += kept
            if dropped:
                lang = p.split("/")[-2]
                for t, c in dropped.items(): by_tag[t] += c; by_lang[lang] += c
                touched.append((os.path.basename(p), sum(dropped.values()), dropped))
            if i % 500 == 0: print(f"[{i}/{len(files)}] {time.time()-t0:.0f}s dropped so far {sum(by_tag.values()):,}", flush=True)
    rep = {"mode": "APPLIED" if a.apply else "DRY-RUN", "files_scanned": len(files), "rows_in_files_with_hits": tot_rows,
           "rows_dropped": sum(by_tag.values()), "by_tag": dict(by_tag), "by_language": dict(by_lang),
           "files_with_drops": len(touched), "top_files": sorted(touched, key=lambda x: -x[1])[:25], "seconds": round(time.time() - t0)}
    print(json.dumps(rep, indent=1, default=str))
    if a.report:
        json.dump(rep, open(a.report, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
