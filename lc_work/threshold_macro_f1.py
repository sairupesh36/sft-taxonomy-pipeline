"""Macro-F1 AND micro-F1 across a sweep of thresholds, for a given pair of domain/task_family
models, on 3 real eval sets (clean_val, weak_val, target_test -- the last one is the unbiased,
real-production-distribution sample, most representative of what a chosen threshold will actually
do in the field). Uses predict_capped (the same top-2-cap function now used in production) so this
sweep reflects exactly what will actually run, not the old uncapped behavior.
Usage: python3 threshold_macro_f1.py --domain-model PATH --task-model PATH [--label RUN_NAME]
"""
import sys, os, json, random, argparse, collections
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
from fasttext_predict_utils import predict_capped

CE = "/projects/data/datasets/code_data/sai_rupesh/taxonomy/classifier_experiments"
W = "/projects/data/datasets/code_data/sai_rupesh/taxonomy/lc_work"
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

def load(path, axis):
    out = []
    for line in open(path):
        d = json.loads(line)
        if d.get("ok") and d["result"].get(axis):
            out.append((d["result"][axis], B.clean_text(d["full_text"])))
    return out

def macro_micro(model, examples, threshold):
    per_class = collections.defaultdict(lambda: [0, 0, 0]); tp = fp = fn = 0
    for labels, text in examples:
        pred = set(predict_capped(model, text, threshold)); true = set(labels)
        tp += len(pred & true); fp += len(pred - true); fn += len(true - pred)
        for c in true | pred:
            e = per_class[c]
            if c in true and c in pred: e[0] += 1
            elif c in pred: e[1] += 1
            elif c in true: e[2] += 1
    f = lambda a, b, c: 2 * a / max(1, 2 * a + b + c)
    p = lambda a, b: a / max(1, a + b)
    rows = [(c, a + cc, f(a, b, cc)) for c, (a, b, cc) in per_class.items()]
    macro = sum(r[2] for r in rows) / len(rows) if rows else 0.0
    return dict(micro_p=round(p(tp, fp), 4), micro_r=round(p(tp, fn), 4), micro_f1=round(f(tp, fp, fn), 4),
                macro_f1=round(macro, 4), n_classes=len(rows), zero_f1=sum(1 for r in rows if r[2] == 0.0))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain-model", required=True); ap.add_argument("--task-model", required=True)
    ap.add_argument("--label", default="run")
    a = ap.parse_args()
    models = {"domain": fasttext.load_model(a.domain_model), "task_family": fasttext.load_model(a.task_model)}
    sets = {}
    for axis in ("domain", "task_family"):
        ex = load(f"{CE}/sft_output_sample_combined_98436_clean.jsonl", axis); random.Random(42).shuffle(ex)
        nv = int(len(ex) * 0.15); clean_val = ex[:nv]
        weak = load(f"{CE}/sft_output_sample_weak_categories_v2_mapped.jsonl", axis); random.Random(42).shuffle(weak)
        nw = int(len(weak) * 0.15); weak_val = weak[:nw]
        target_test = load(f"{W}/labeled_T.jsonl", axis)
        sets[axis] = {"clean_val": clean_val, "weak_val": weak_val, "target_test": target_test}
    results = {}
    for axis in ("domain", "task_family"):
        print(f"\n########## {axis}  ({a.label})")
        results[axis] = {}
        for name, s in sets[axis].items():
            print(f"  -- {name} (n={len(s)})")
            results[axis][name] = {}
            for th in THRESHOLDS:
                r = macro_micro(models[axis], s, th)
                results[axis][name][th] = r
                print(f"     threshold={th:.1f}  micro_F1={r['micro_f1']:.4f}  macro_F1={r['macro_f1']:.4f}  "
                      f"P={r['micro_p']:.3f}  R={r['micro_r']:.3f}  zero_F1_classes={r['zero_f1']}/{r['n_classes']}")
    json.dump(results, open(f"{W}/threshold_sweep_{a.label}.json", "w"), indent=1)
    print(f"\nsaved -> lc_work/threshold_sweep_{a.label}.json")
