"""Does giving each LABEL its own threshold (instead of one shared threshold for all labels)
improve macro-F1? For each class, sweep a range of thresholds using ONLY that class's own raw
score (independent of what other classes do -- valid because loss="ova" gives every label an
independent yes/no score, not a shared softmax), pick whichever threshold gives that class its own
best F1, then average across classes. Compared directly against the current best single shared
threshold, on the same real target_test set."""
import sys, os, json, random, collections
sys.path.insert(0, "/projects/data/datasets/code_data/sai_rupesh/taxonomy/pylibs")
sys.path.insert(0, "/projects/data/datasets/code_data/sai_rupesh/taxonomy/classifier_experiments")
import numpy as np
_orig = np.array
def _c(*a, **k):
    if k.get("copy") is False: k["copy"] = None
    return _orig(*a, **k)
np.array = _c
import fasttext
import fasttext_baseline as B
W = "/projects/data/datasets/code_data/sai_rupesh/taxonomy/lc_work"

def load(path, axis):
    out = []
    for line in open(path):
        d = json.loads(line)
        if d.get("ok") and d["result"].get(axis): out.append((d["result"][axis], B.clean_text(d["full_text"])))
    return out

GRID = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

for axis in ("domain", "task_family"):
    model = fasttext.load_model(f"{W}/models/{axis}_fasttext_v2_AB.bin")
    test = load(f"{W}/labeled_T.jsonl", axis)
    # get EVERY label's raw score per doc once (threshold=0.0 -> everything), reuse for every threshold in the grid
    all_scores = []  # list of (true_label_set, {label: score})
    for labels, text in test:
        labs, scores = model.predict(text, k=-1, threshold=0.0)
        sd = {l.replace("__label__", ""): float(s) for l, s in zip(labs, scores)}
        all_scores.append((set(labels), sd))
    all_labels = sorted({l for _, sd in all_scores for l in sd} | {l for labs, _ in all_scores for l in labs})

    def f1_at(th_map):
        per_class = collections.defaultdict(lambda: [0, 0, 0])
        for true, sd in all_scores:
            pred = {l for l, s in sd.items() if s >= th_map.get(l, 0.8)}
            for c in true | pred:
                e = per_class[c]
                if c in true and c in pred: e[0] += 1
                elif c in pred: e[1] += 1
                elif c in true: e[2] += 1
        f = lambda a, b, c: 2 * a / max(1, 2 * a + b + c)
        rows = {c: f(a, b, cc) for c, (a, b, cc) in per_class.items()}
        return rows

    # 1) best SINGLE shared threshold (sweep, report the best)
    best_shared = max(GRID, key=lambda th: sum(f1_at({l: th for l in all_labels}).values()) / max(1, len(f1_at({l: th for l in all_labels}))))
    shared_rows = f1_at({l: best_shared for l in all_labels})
    shared_macro = sum(shared_rows.values()) / len(shared_rows)

    # 2) PER-CLASS best threshold: for each label, sweep independently on this SAME test set to find its own best F1
    per_class_th = {}
    for lab in all_labels:
        best_f1, best_th = -1, 0.8
        for th in GRID:
            rows = f1_at({**{l: best_shared for l in all_labels}, lab: th})  # vary only this label, rest at the shared-best
            if rows.get(lab, 0) > best_f1: best_f1, best_th = rows.get(lab, 0), th
        per_class_th[lab] = best_th
    per_class_rows = f1_at(per_class_th)
    per_class_macro = sum(per_class_rows.values()) / len(per_class_rows)

    print(f"\n########## {axis}")
    print(f"  best SHARED threshold = {best_shared}  -> macro_F1 = {shared_macro:.4f}")
    print(f"  PER-CLASS thresholds  -> macro_F1 = {per_class_macro:.4f}   (delta {per_class_macro - shared_macro:+.4f})")
    changed = sorted(((l, per_class_th[l], shared_rows.get(l,0), per_class_rows.get(l,0)) for l in all_labels if per_class_th[l] != best_shared and per_class_rows.get(l,0) != shared_rows.get(l,0)), key=lambda x: -(x[3]-x[2]))
    print(f"  classes whose OWN threshold differs from the shared one and actually changed its F1: {len(changed)}/{len(all_labels)}")
    for l, th, f_old, f_new in changed[:15]:
        print(f"    {l:42s} own_threshold={th:<5} F1 {f_old:.3f} -> {f_new:.3f}")
