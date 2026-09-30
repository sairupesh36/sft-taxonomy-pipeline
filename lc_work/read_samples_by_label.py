"""Collect real example rows per domain label and per task_family label from the classified output,
for manual reading. Two passes: (1) random-file sampling for common labels, (2) targeted grep for
labels that didn't turn up enough hits in the random sample (the rare tail)."""
import glob, json, os, random, subprocess, collections, sys
random.seed(11)
ROOT = "/projects/data/datasets/code_data/sai_rupesh/taxonomy/sft_43_language_wise_classified"
N_PER_LABEL = 4
files = [f for f in glob.glob(ROOT + "/*/*.jsonl") if os.path.getsize(f) > 200_000]
random.shuffle(files)

def prev_of(row):
    u = next((m["content"] for m in row["messages"] if m["role"] == "user" and isinstance(m.get("content"), str)), "")
    a = next((m["content"] for m in row["messages"] if m["role"] == "assistant" and isinstance(m.get("content"), str)), "")
    return u.replace("\n", " ")[:220], a.replace("\n", " ")[:130]

dom_hits = collections.defaultdict(list); tf_hits = collections.defaultdict(list)
n_read = 0
for fn in files[:260]:
    size = os.path.getsize(fn)
    with open(fn, "rb") as f:
        f.seek(random.randint(0, max(0, size - 2_000_000))); f.readline()
        for i, l in enumerate(f):
            if i > 4000: break
            try: row = json.loads(l)
            except Exception: continue
            n_read += 1
            for d in (row.get("domain") or []):
                if len(dom_hits[d]) < N_PER_LABEL: dom_hits[d].append(row)
            for t in (row.get("task_family") or []):
                if len(tf_hits[t]) < N_PER_LABEL: tf_hits[t].append(row)
print(f"random-sample pass: read {n_read:,} rows from 260 files; domains hit {len(dom_hits)}, task_families hit {len(tf_hits)}", flush=True)

ALL_DOMAINS = sorted(json.load(open(ROOT + "/_classify_report.json"))["domain_counts"])
ALL_TF = sorted(json.load(open(ROOT + "/_classify_report.json"))["task_family_counts"])
missing_d = [d for d in ALL_DOMAINS if d != "__NONE__" and len(dom_hits[d]) < N_PER_LABEL]
missing_t = [t for t in ALL_TF if t != "__NONE__" and len(tf_hits[t]) < N_PER_LABEL]
print(f"labels needing a targeted grep (rare tail): {len(missing_d)} domains, {len(missing_t)} task_families", flush=True)

def grep_fill(label, key, hits, kind):
    need = N_PER_LABEL - len(hits[label])
    if need <= 0: return
    pat = f'"{key}": \\[[^]]*"{label}"'
    try:
        out = subprocess.run(["grep", "-rlE", "-m", "1", pat, "--include=*.jsonl", ROOT], capture_output=True, text=True, timeout=90).stdout.split()
    except Exception:
        out = []
    for fp in out[:6]:
        if len(hits[label]) >= N_PER_LABEL: break
        try:
            r = subprocess.run(["grep", "-mE", str(need + 2), pat, fp], capture_output=True, text=True, timeout=60).stdout.splitlines()
        except Exception:
            r = []
        for line in r:
            if len(hits[label]) >= N_PER_LABEL: break
            try:
                row = json.loads(line)
                if label in (row.get(key) or []): hits[label].append(row)
            except Exception: pass

for d in missing_d: grep_fill(d, "domain", dom_hits, "domain")
for t in missing_t: grep_fill(t, "task_family", tf_hits, "task_family")
print(f"after targeted grep: domains covered {sum(1 for d in ALL_DOMAINS if d!='__NONE__' and dom_hits[d])}/{len(ALL_DOMAINS)-1}, "
      f"task_families covered {sum(1 for t in ALL_TF if t!='__NONE__' and tf_hits[t])}/{len(ALL_TF)-1}", flush=True)

out = {"domain": {k: [prev_of(r) + (r.get("domain"), r.get("task_family")) for r in v] for k, v in dom_hits.items()},
       "task_family": {k: [prev_of(r) + (r.get("domain"), r.get("task_family")) for r in v] for k, v in tf_hits.items()}}
json.dump(out, open("/projects/data/datasets/code_data/sai_rupesh/taxonomy/lc_work/label_samples.json", "w"), indent=1, ensure_ascii=False)
print("saved -> lc_work/label_samples.json")
