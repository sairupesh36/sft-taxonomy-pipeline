"""
Per-class precision/recall/F1 diagnostic, run against the current best models
and their matching validation split. Reveals categories the overall micro F1
hides (a category can be at 0 while overall looks fine).
"""
import sys
import json
import collections
import numpy as np
import fasttext

_orig_np_array = np.array
def _np_array_compat(*args, **kwargs):
    if kwargs.get("copy") is False:
        kwargs["copy"] = None
    return _orig_np_array(*args, **kwargs)
np.array = _np_array_compat

from fasttext_predict_utils import predict_with_fallback

WORK_DIR = "/projects/data/datasets/code_data/sai_rupesh/taxonomy/classifier_experiments"


def load_val(path):
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split(" ", )
            labels = []
            i = 0
            while i < len(parts) and parts[i].startswith("__label__"):
                labels.append(parts[i][len("__label__"):])
                i += 1
            text = " ".join(parts[i:])
            rows.append((labels, text))
    return rows


def per_class(axis, model_path, val_path, threshold=0.5):
    model = fasttext.load_model(model_path)
    val = load_val(val_path)
    tp = collections.Counter()
    fp = collections.Counter()
    fn = collections.Counter()
    support = collections.Counter()
    for labels, text in val:
        # Uses the same predict_with_fallback() (confidence-floored top-1
        # fallback, 2026-09-20) that training/tuning eval now uses -- a
        # per-class diagnostic measured against a different prediction
        # function than what's actually deployed would be misleading.
        pred_set = predict_with_fallback(model, text, threshold)
        true_set = set(labels)
        for l in true_set:
            support[l] += 1
        for l in pred_set & true_set:
            tp[l] += 1
        for l in pred_set - true_set:
            fp[l] += 1
        for l in true_set - pred_set:
            fn[l] += 1

    all_labels = sorted(support, key=lambda l: -support[l])
    results = []
    for l in all_labels:
        p = tp[l] / (tp[l] + fp[l]) if (tp[l] + fp[l]) else 0.0
        r = tp[l] / (tp[l] + fn[l]) if (tp[l] + fn[l]) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        results.append({"label": l, "support": support[l], "precision": round(p, 3), "recall": round(r, 3), "f1": round(f1, 3)})
    return results


def main():
    for axis, model_name in [("domain", "domain_fasttext_optuna_best.bin"), ("task_family", "task_family_fasttext_optuna_best.bin")]:
        model_path = f"{WORK_DIR}/{model_name}"
        val_path = f"{WORK_DIR}/{axis}_val.txt"
        print(f"\n=== {axis} per-class F1 (model={model_name}) ===")
        results = per_class(axis, model_path, val_path)
        weak = [r for r in results if r["f1"] < 0.5]
        for r in results:
            flag = "  <-- WEAK" if r["f1"] < 0.5 else ""
            print(f"  {r['label']:45s} support={r['support']:5d}  P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f}{flag}")
        print(f"\n{axis}: {len(weak)}/{len(results)} labels below F1=0.5")
        with open(f"{WORK_DIR}/per_class_{axis}_diagnostic.json", "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
