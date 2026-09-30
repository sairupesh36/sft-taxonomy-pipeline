"""Real test: GLiNER2.5-Decide on our actual domain/task_family taxonomy, real documents."""
import time, json, sys
sys.path.insert(0, "classifier_experiments")
import gliner2
import fasttext_baseline as B

t0 = time.time()
model = gliner2.GLiNER2.from_pretrained("fastino/GLiNER2.5-Decide")
print(f"model loaded in {time.time()-t0:.1f}s", flush=True)

import pandas as pd
dv = pd.read_excel("taxonomy_tree_final.xlsx", sheet_name="domain_values", keep_default_na=False, na_values=[])
dom_labels = {row["domain"]: row["description"] for _, row in dv.iterrows()}
print(f"{len(dom_labels)} domain labels with real descriptions", flush=True)

def load(path, axis, n):
    out = []
    for line in open(path):
        d = json.loads(line)
        if d.get("ok") and d["result"].get(axis): out.append((d["result"][axis], d["full_text"][:1500]))
    import random; random.Random(3).shuffle(out)
    return out[:n]

test = load("lc_work/labeled_T.jsonl", "domain", 8)
print(f"testing on {len(test)} real target_test docs", flush=True)

tasks = {"domain": {"labels": dom_labels, "multi_label": True, "cls_threshold": 0.4}}

t0 = time.time()
correct1 = correct2 = 0
    t1 = time.time()
for true, text in test:
    r = model.classify_text(text, tasks, threshold=0.4)
    pred = r.get("domain", [])
    if isinstance(pred, dict): pred = list(pred.keys())
    pred = pred if isinstance(pred, list) else [pred]
    hit = any(p in true for p in pred)
    print(f"  true={true} pred={pred}  ({time.time()-t1:.1f}s)"); t1 = time.time()
    correct1 += hit
el = time.time() - t0
print(f"\n{correct1}/{len(test)} at least one overlap | {len(test)/el:.2f} docs/sec (single doc calls, CPU)")
