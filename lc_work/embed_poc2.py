"""V2: instead of matching against the category's raw description alone, build each category's
PROTOTYPE embedding by averaging a handful of real confirmed example documents (few-shot, not
zero-shot) -- the standard fix for weak description-only zero-shot matching. Falls back to the
description alone only for categories with 0 real examples (the ones fastText can never learn)."""
import json, requests, numpy as np, random, collections

EP = "http://172.17.99.1:30080/embed"
def embed(texts):
    out = []
    for i in range(0, len(texts), 32):
        r = requests.post(EP, json={"inputs": texts[i:i+32]}, timeout=60)
        out += r.json()
    return np.array(out)

def load(path, axis):
    out = []
    for line in open(path):
        d = json.loads(line)
        if d.get("ok") and d["result"].get(axis): out.append((d["result"][axis], d["full_text"][:800]))
    return out

CE = "classifier_experiments"; W = "lc_work"
random.seed(5)
train_sources = [f"{CE}/sft_output_sample_combined_98436_clean.jsonl", f"{CE}/sft_output_sample_weak_categories_v2_mapped.jsonl",
                  f"{W}/labeled_A.jsonl", f"{W}/labeled_B.jsonl", f"{W}/labeled_weak_candidates.jsonl"]

def cos(a, b):
    an = a / np.linalg.norm(a, axis=-1, keepdims=True); bn = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return an @ bn.T

for axis in ("domain", "task_family"):
    by_label = collections.defaultdict(list)
    for p in train_sources:
        for labels, text in load(p, axis):
            for l in labels:
                if len(by_label[l]) < 20: by_label[l].append(text)   # up to 20 real examples per class = the "few-shot" prototype
    labels_names = sorted(by_label)
    print(f"\n{axis}: {len(labels_names)} classes have >=1 real example; support: min={min(len(v) for v in by_label.values())} max={max(len(v) for v in by_label.values())}")
    all_ex_texts = [t for l in labels_names for t in by_label[l]]
    all_emb = embed(all_ex_texts)
    idx = 0; proto = {}
    for l in labels_names:
        n = len(by_label[l]); proto[l] = all_emb[idx:idx+n].mean(axis=0); idx += n
    cat_emb = np.stack([proto[l] for l in labels_names])

    test = load(f"{W}/labeled_T.jsonl", axis); random.shuffle(test); test = test[:300]
    doc_emb = embed([t for _, t in test])
    sims = cos(doc_emb, cat_emb)
    top1 = top2 = 0
    for i, (true, _) in enumerate(test):
        order = np.argsort(-sims[i])
        picks = [labels_names[j] for j in order[:2]]
        top1 += labels_names[order[0]] in true
        top2 += any(p in true for p in picks)
    print(f"  FEW-SHOT PROTOTYPE (avg of up to 20 real examples/class) on {len(test)} real target_test docs: top-1={top1/len(test):.3f}  top-2={top2/len(test):.3f}")
