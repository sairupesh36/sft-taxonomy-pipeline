"""Quick proof-of-concept: can BGE-M3 (already live in the cluster) correctly zero-shot classify
real documents by cosine similarity against each category's own official description alone --
NO training examples needed at all? Tests exactly the failure mode fastText structurally can't
fix (categories with 0-1 real training examples)."""
import json, requests, numpy as np, pandas as pd, random

EP = "http://172.17.99.1:30080/embed"
def embed(texts):
    out = []
    for i in range(0, len(texts), 32):
        r = requests.post(EP, json={"inputs": texts[i:i+32]}, timeout=60)
        out += r.json()
    return np.array(out)

dv = pd.read_excel("taxonomy_tree_final.xlsx", sheet_name="domain_values", keep_default_na=False, na_values=[])
tv = pd.read_excel("taxonomy_tree_final.xlsx", sheet_name="task_family_values", keep_default_na=False, na_values=[])
dom_names = dv["domain"].tolist(); dom_desc = (dv["domain"] + ": " + dv["description"]).tolist()
task_names = tv["task_family"].tolist(); task_desc = (tv["task_family"] + ": " + tv["description"]).tolist()
print(f"embedding {len(dom_desc)} domain + {len(task_desc)} task_family descriptions...", flush=True)
dom_emb = embed(dom_desc); task_emb = embed(task_desc)

def load(path, axis, n):
    out = []
    for line in open(path):
        d = json.loads(line)
        if d.get("ok") and d["result"].get(axis): out.append((d["result"][axis], d["full_text"][:800]))
    random.Random(3).shuffle(out)
    return out[:n]

test_d = load("lc_work/labeled_T.jsonl", "domain", 200)
test_t = load("lc_work/labeled_T.jsonl", "task_family", 200)
print(f"testing on {len(test_d)} real domain docs, {len(test_t)} real task_family docs...", flush=True)
doc_emb_d = embed([t for _, t in test_d])
doc_emb_t = embed([t for _, t in test_t])

def cos(a, b):
    an = a / np.linalg.norm(a, axis=-1, keepdims=True); bn = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return an @ bn.T

def eval_top(labels_names, cat_emb, docs, doc_emb, k=2):
    sims = cos(doc_emb, cat_emb)
    top1_hit = top2_hit = 0
    for i, (true, _) in enumerate(docs):
        order = np.argsort(-sims[i])
        picks = [labels_names[j] for j in order[:k]]
        top1_hit += labels_names[order[0]] in true
        top2_hit += any(p in true for p in picks)
    return top1_hit / len(docs), top2_hit / len(docs)

d1, d2 = eval_top(dom_names, dom_emb, test_d, doc_emb_d)
t1, t2 = eval_top(task_names, task_emb, test_t, doc_emb_t)
print(f"\nZERO-SHOT (no training examples at all, pure cosine similarity vs official description):")
print(f"  domain:       top-1 accuracy={d1:.3f}  top-2 accuracy={d2:.3f}")
print(f"  task_family:  top-1 accuracy={t1:.3f}  top-2 accuracy={t2:.3f}")
