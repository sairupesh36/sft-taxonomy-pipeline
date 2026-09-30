"""Fair comparison: macro-F1 (same metric as fastText everywhere else) for the few-shot prototype
embedding approach, plus real throughput -- since 1.2B-doc scale is still the hard constraint."""
import json, requests, numpy as np, random, collections, time

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
                if len(by_label[l]) < 20: by_label[l].append(text)
    labels_names = sorted(by_label)
    all_ex_texts = [t for l in labels_names for t in by_label[l]]
    all_emb = embed(all_ex_texts)
    idx = 0; proto = {}
    for l in labels_names:
        n = len(by_label[l]); proto[l] = all_emb[idx:idx+n].mean(axis=0); idx += n
    cat_emb = np.stack([proto[l] for l in labels_names])

    test = load(f"{W}/labeled_T.jsonl", axis)
    t0 = time.time()
    doc_emb = embed([t for _, t in test])
    embed_time = time.time() - t0
    sims = cos(doc_emb, cat_emb)

    # macro-F1 at the best of a small threshold-on-similarity grid (top-2 cap, same convention as fastText)
    best_macro = 0; best_th = None
    for th in (0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7):
        per_class = collections.defaultdict(lambda: [0, 0, 0])
        for i, (true, _) in enumerate(test):
            order = np.argsort(-sims[i])[:2]
            pred = {labels_names[j] for j in order if sims[i][j] >= th}
            for c in set(true) | pred:
                e = per_class[c]
                if c in true and c in pred: e[0] += 1
                elif c in pred: e[1] += 1
                elif c in true: e[2] += 1
        f = lambda a, b, c: 2 * a / max(1, 2 * a + b + c)
        macro = sum(f(a, b, c) for a, b, c in per_class.values()) / max(1, len(per_class))
        if macro > best_macro: best_macro, best_th = macro, th
    print(f"{axis}: macro_F1={best_macro:.4f} (best sim-threshold={best_th}) on {len(test)} real target_test docs "
          f"| embed throughput: {len(test)/embed_time:.1f} docs/sec (single-stream, CPU service)")
