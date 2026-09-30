"""Real macro-F1 (per-class F1, unweighted average) for the current v2_AB models, on the SAME
three eval sets arms.py already used (clean_val, weak_val, target_test) -- so this is a like-for-like
comparison against the micro-F1 numbers already saved in lc_work/results/arm_*_AB.json, not a new
sample. Multi-label macro-F1 = for each class, treat it as its own binary classification over all
docs (present in true set? present in pred set?), get that class's own P/R/F1, then average across
CLASSES (not docs) -- this is what exposes rare-class collapse that micro-F1's doc-pooled tp/fp/fn
hides."""
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
from fasttext_predict_utils import predict_with_fallback

CE = "/projects/data/datasets/code_data/sai_rupesh/taxonomy/classifier_experiments"
W = "/projects/data/datasets/code_data/sai_rupesh/taxonomy/lc_work"
MODELS = {"domain": f"{W}/models/domain_fasttext_v2_AB.bin", "task_family": f"{W}/models/task_family_fasttext_v2_AB.bin"}

def load(path, axis):
    out = []
    for line in open(path):
        d = json.loads(line)
        if d.get("ok") and d["result"].get(axis):
            out.append((d["result"][axis], B.clean_text(d["full_text"])))
    return out

def macro_and_micro(model, examples, threshold=0.5):
    per_class = collections.defaultdict(lambda: [0, 0, 0])   # label -> [tp, fp, fn]
    tp = fp = fn = 0
    for labels, text in examples:
        pred = predict_with_fallback(model, text, threshold)
        true = set(labels)
        tp += len(pred & true); fp += len(pred - true); fn += len(true - pred)
        for c in true | pred:
            e = per_class[c]
            if c in true and c in pred: e[0] += 1
            elif c in pred: e[1] += 1
            elif c in true: e[2] += 1
    f = lambda a, b, c: 2 * a / max(1, 2 * a + b + c)
    micro_f1 = f(tp, fp, fn)
    rows = []
    for c, (a, b, cc) in per_class.items():
        support = a + cc                                     # true occurrences of this class in the eval set
        rows.append((c, support, f(a, b, cc)))
    macro_f1 = sum(r[2] for r in rows) / len(rows) if rows else 0.0
    return micro_f1, macro_f1, sorted(rows, key=lambda r: r[2])

for axis in ("domain", "task_family"):
    print(f"\n########## {axis}  (model: {MODELS[axis]})")
    model = fasttext.load_model(MODELS[axis])
    ex = load(f"{CE}/sft_output_sample_combined_98436_clean.jsonl", axis); random.Random(42).shuffle(ex)
    nv = int(len(ex) * 0.15); clean_val = ex[:nv]
    weak = load(f"{CE}/sft_output_sample_weak_categories_v2_mapped.jsonl", axis); random.Random(42).shuffle(weak)
    nw = int(len(weak) * 0.15); weak_val = weak[:nw]
    target_test = load(f"{W}/labeled_T.jsonl", axis)
    for name, s in (("clean_val", clean_val), ("weak_val", weak_val), ("target_test", target_test)):
        micro, macro, rows = macro_and_micro(model, s)
        n_classes_scored = len(rows); zero_f1 = sum(1 for c, sup, f1 in rows if f1 == 0.0)
        print(f"  {name:14s} n={len(s):5d}  micro_F1={micro:.4f}   macro_F1={macro:.4f}   "
              f"classes_with_any_support={n_classes_scored}   classes_scoring_ZERO_F1={zero_f1}")
        if name == "target_test":                             # the one real, unbiased sample -- show its worst classes in detail
            print("    worst-scoring classes on target_test (label, support_in_this_100-1000-doc_set, F1):")
            for c, sup, f1 in rows[:15]:
                print(f"      {c:42s} support={sup:>4d}  F1={f1:.3f}")
    json.dump({"axis": axis, "macro_micro_by_set": {name: {"micro": m, "macro": ma} for name, (m, ma) in
               {n: macro_and_micro(model, s)[:2] for n, s in (("clean_val", clean_val), ("weak_val", weak_val), ("target_test", target_test))}.items()}},
              open(f"{W}/macro_f1_{axis}.json", "w"), indent=1)
