"""Pull real samples for EVERY domain (49) and task_family (63) from the actual 517M-row classified
output, for a genuine per-category correctness read -- not a cherry-picked handful. Output layout
is flat (lang__filename.jsonl, not nested folders -- the bug that broke the earlier attempt at this)."""
import glob, json, os, random, collections
N_PER_LABEL = 5
ROOT = "/projects/data/datasets/code_data/sai_rupesh/taxonomy/sft_43_language_wise_classified"
random.seed(21)
files = [f for f in glob.glob(ROOT + "/*.jsonl") if os.path.getsize(f) > 200_000]
random.shuffle(files)

dom_hits = collections.defaultdict(list); tf_hits = collections.defaultdict(list)
ALL_DOMAINS = sorted(k for k in json.load(open(ROOT + "/_classify_report.json"))["domain_counts"] if k != "__NONE__")
ALL_TF = sorted(k for k in json.load(open(ROOT + "/_classify_report.json"))["task_family_counts"] if k != "__NONE__")

def prev(row):
    u = next((m["content"] for m in row["messages"] if m["role"] == "user" and isinstance(m.get("content"), str)), "")
    a = next((m["content"] for m in row["messages"] if m["role"] == "assistant" and isinstance(m.get("content"), str)), "")
    return u.replace("\n", " ")[:260], a.replace("\n", " ")[:160]

n_read = 0
for fn in files:
    if all(len(dom_hits[d]) >= N_PER_LABEL for d in ALL_DOMAINS) and all(len(tf_hits[t]) >= N_PER_LABEL for t in ALL_TF):
        break
    with open(fn, "rb") as f:
        for line in f:
            n_read += 1
            try: row = json.loads(line)
            except Exception: continue
            for d in (row.get("domain") or []):
                if len(dom_hits[d]) < N_PER_LABEL: dom_hits[d].append(prev(row) + (row.get("domain"), row.get("task_family")))
            for t in (row.get("task_family") or []):
                if len(tf_hits[t]) < N_PER_LABEL: tf_hits[t].append(prev(row) + (row.get("domain"), row.get("task_family")))

print(f"read {n_read:,} rows from {files.index(fn)+1} files")
missing_d = [d for d in ALL_DOMAINS if len(dom_hits[d]) < N_PER_LABEL]
missing_t = [t for t in ALL_TF if len(tf_hits[t]) < N_PER_LABEL]
print(f"domains still short of {N_PER_LABEL}: {missing_d}")
print(f"task_families still short of {N_PER_LABEL}: {missing_t}")
json.dump({"domain": dom_hits, "task_family": tf_hits}, open("lc_work/all_category_samples.json", "w"), ensure_ascii=False, indent=1)
print("saved -> lc_work/all_category_samples.json")
