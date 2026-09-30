"""Parallel version: pull real samples for every domain (49) and task_family (63) from the 517M-row
classified output, scanning all files concurrently instead of one at a time."""
import glob, json, os, random, collections
from multiprocessing import Pool
N_PER_LABEL = 5
ROOT = "/projects/data/datasets/code_data/sai_rupesh/taxonomy/sft_43_language_wise_classified"
ALL_DOMAINS = sorted(k for k in json.load(open(ROOT + "/_classify_report.json"))["domain_counts"] if k != "__NONE__")
ALL_TF = sorted(k for k in json.load(open(ROOT + "/_classify_report.json"))["task_family_counts"] if k != "__NONE__")

def prev(row):
    u = next((m["content"] for m in row["messages"] if m["role"] == "user" and isinstance(m.get("content"), str)), "")
    a = next((m["content"] for m in row["messages"] if m["role"] == "assistant" and isinstance(m.get("content"), str)), "")
    return u.replace("\n", " ")[:260], a.replace("\n", " ")[:160]

def scan(fn):
    dh = collections.defaultdict(list); th = collections.defaultdict(list)
    with open(fn, "rb") as f:
        for line in f:
            try: row = json.loads(line)
            except Exception: continue
            for d in (row.get("domain") or []):
                if len(dh[d]) < N_PER_LABEL: dh[d].append(prev(row) + (row.get("domain"), row.get("task_family")))
            for t in (row.get("task_family") or []):
                if len(th[t]) < N_PER_LABEL: th[t].append(prev(row) + (row.get("domain"), row.get("task_family")))
    return dh, th

if __name__ == "__main__":
    files = [f for f in glob.glob(ROOT + "/*.jsonl") if os.path.getsize(f) > 200_000]
    random.seed(21); random.shuffle(files)
    dom_hits = collections.defaultdict(list); tf_hits = collections.defaultdict(list)
    with Pool(96) as p:
        for i, (dh, th) in enumerate(p.imap_unordered(scan, files, chunksize=1), 1):
            for d, v in dh.items():
                if len(dom_hits[d]) < N_PER_LABEL: dom_hits[d] += v[:N_PER_LABEL - len(dom_hits[d])]
            for t, v in th.items():
                if len(tf_hits[t]) < N_PER_LABEL: tf_hits[t] += v[:N_PER_LABEL - len(tf_hits[t])]
            if i % 500 == 0:
                covered = sum(1 for d in ALL_DOMAINS if len(dom_hits[d]) >= N_PER_LABEL) + sum(1 for t in ALL_TF if len(tf_hits[t]) >= N_PER_LABEL)
                print(f"[{i}/{len(files)}] categories fully covered so far: {covered}/{len(ALL_DOMAINS)+len(ALL_TF)}", flush=True)
    missing_d = [d for d in ALL_DOMAINS if len(dom_hits[d]) < N_PER_LABEL]
    missing_t = [t for t in ALL_TF if len(tf_hits[t]) < N_PER_LABEL]
    print(f"DONE. domains still short: {missing_d}"); print(f"task_families still short: {missing_t}")
    json.dump({"domain": dom_hits, "task_family": tf_hits}, open("lc_work/all_category_samples.json", "w"), ensure_ascii=False, indent=1)
    print("saved -> lc_work/all_category_samples.json")
