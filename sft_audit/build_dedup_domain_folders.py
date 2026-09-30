"""Split the fuzzy-DEDUPED SFT data (Himanshu's run over our 10 final datasets) into one folder per domain
(user ask 2026-09-25). Read-only on the dedup data; writes only under OUT.

    python3 build_dedup_domain_folders.py            # build (resumes: files already in the manifest are skipped)

OUT/<Domain>/<dataset>/<same relative path as in the dedup output>.jsonl
Domain = the same rough rules as the rows-only domain workbook (sft_audit/domain_rules.py):
  * sft_hf_domain_data: PER ROW -- the row's prompt key is looked up in our final hf_domain key -> domain map
    (sft_loss_report/v2/hf_final_key_domain.npy, domain of the source folder the prompt came from);
    keys not found -> "Untraced (hf_domain rows not matched to a source)".
  * every other dataset: per FILE, from its source (dataset / parquet) name.
Rows are copied byte for byte (no re-serialising). A folder name cannot hold "/", so
"NLP Tasks (QA / Summarization / Classification)" becomes "NLP Tasks (QA - Summarization - Classification)".
Every finished input file is appended to OUT/_manifest.jsonl (rows per domain + output paths).
"""
import json, os, re, sys, time
from collections import Counter
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import domain_rules as DR          # noqa: E402

SRC = "/projects/data/datasets/code_data/Himanshu_sharma/Sft/fuzzy_dedup_data"
OUT = "/projects/data/datasets/code_data/sai_rupesh/taxonomy/sft_final_deduped_domain_wise"
HF = "sft_hf_domain_data_openai_ready"
UNTRACED = "Untraced (hf_domain rows not matched to a source)"
# dedup folder name -> our dataset name (sft_audit.DATASETS)
DS = {"datasets_sft_openai_sft": "sft_datasets_sft_openai_ready", "synthetic_data_openai_sft": "sft_synthetic_data_openai_ready"}
MANIFEST = os.path.join(OUT, "_manifest.jsonl")


def dirname(domain):
    return domain.replace(" / ", " - ").replace("/", "-")


def clean_rel(ds, rel):
    """dedup output path -> the path shape source_of() expects (drop by_dataset/ for datasets_sft, part suffixes)."""
    if ds == "sft_datasets_sft_openai_ready" and rel.startswith("by_dataset/"):
        rel = rel[len("by_dataset/"):]
    return re.sub(r"(-\d{5})+(\.jsonl(\.gz)?)$", r"\2", rel)


def file_domain(ds, rel):
    src = DR.source_of(ds, clean_rel(ds, rel))
    return src, DR.domain_of(src)


_H = None


def _init():
    global _H
    if _H is None:
        sys.path.insert(0, os.path.join(HERE, "..", "sft_loss_report", "v2"))
        import hf_domain_domainwise as H   # prompt key + final key -> domain map (read-only)
        H._init()
        _H = H


def hf_row_domain(line):
    import numpy as np
    d = json.loads(line)
    pre = [m for m in d["messages"] if m["role"] in ("system", "user")]
    k = _H._A.k64("\n".join(f'{m["role"]}:{_H._A.S.norm(m["content"])}' for m in pre))
    i = int(np.searchsorted(_H._FU, np.uint64(k)))
    if i < len(_H._FU) and _H._FU[i] == k:
        dom = int(_H._DOM[i])
        if dom >= 0:
            return _H.DOMS[dom]
    return UNTRACED


def work(job):
    folder, rel = job
    ds = DS.get(folder, folder)
    src_path = os.path.join(SRC, folder, rel)
    rows = Counter(); outs = {}
    if ds == HF:
        _init()
        handles = {}
        try:
            with open(src_path, "rb") as f:
                for line in f:
                    if not line.strip():
                        continue
                    dom = hf_row_domain(line)
                    if dom == "Untraced":
                        dom = UNTRACED
                    if dom not in handles:
                        p = os.path.join(OUT, dirname(dom), ds, rel)
                        os.makedirs(os.path.dirname(p), exist_ok=True)
                        handles[dom] = open(p + ".tmp", "wb"); outs[dom] = p
                    handles[dom].write(line if line.endswith(b"\n") else line + b"\n")
                    rows[dom] += 1
        finally:
            for h in handles.values():
                h.close()
        for dom, p in outs.items():
            os.replace(p + ".tmp", p)
        src = "(row-level)"
    else:
        src, dom = file_domain(ds, rel)
        p = os.path.join(OUT, dirname(dom), ds, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        n = 0
        with open(src_path, "rb") as f, open(p + ".tmp", "wb") as g:
            while True:
                b = f.read(64 << 20)
                if not b:
                    break
                n += b.count(b"\n"); g.write(b)
        os.replace(p + ".tmp", p)
        rows[dom] = n; outs[dom] = p
    return {"folder": folder, "dataset": ds, "rel": rel, "src_path": src_path, "source": src,
            "rows": dict(rows), "out": outs}


def main():
    os.makedirs(OUT, exist_ok=True)
    done = set()
    if os.path.exists(MANIFEST):
        done = {(d["folder"], d["rel"]) for d in map(json.loads, open(MANIFEST))}
    jobs = []
    for folder in sorted(os.listdir(SRC)):
        if folder.startswith("_") or not os.path.isdir(os.path.join(SRC, folder)):
            continue
        root = os.path.join(SRC, folder)
        for r, dirs, fs in os.walk(root):
            for f in fs:
                if f.endswith(".jsonl"):
                    rel = os.path.relpath(os.path.join(r, f), root)
                    if (folder, rel) not in done:
                        jobs.append((folder, rel))
    jobs.sort(key=lambda j: -os.path.getsize(os.path.join(SRC, *j)))
    print(f"{len(jobs)} files to do ({len(done)} already done)", flush=True)
    t = time.time()
    with Pool(int(os.environ.get("WORKERS", 48))) as pool, open(MANIFEST, "a") as fo:
        for i, rec in enumerate(pool.imap_unordered(work, jobs), 1):
            fo.write(json.dumps(rec) + "\n"); fo.flush()
            if i % 200 == 0 or i == len(jobs):
                print(f"[{i}/{len(jobs)}] {time.time() - t:.0f}s", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
