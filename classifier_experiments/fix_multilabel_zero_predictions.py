"""
Fixes the known "multi-label under-prediction" gap documented in this
folder's CLAUDE.md: FastText's threshold-based predict() can return ZERO
labels for a document that has real labels, when the model splits
confidence between 2 plausible classes and neither individually clears the
single global threshold. Since every real document in this dataset has at
least one true label (rows with empty label lists are filtered out before
training -- see fasttext_baseline.py's `if not labels: continue`), a
zero-label prediction is always wrong and never the right call.

Fix: after the normal threshold-based predict(), if it returns nothing,
fall back to the single highest-scoring label (top-1) instead of returning
empty. This never changes a prediction that already has >=1 label above
threshold -- it only replaces "predict nothing" with "predict the model's
best single guess," which can only help recall on exactly the failure case
described above.

Measures the real effect (before/after) on the actual held-out validation
sets for both axes using the current tuned optuna models + thresholds,
rather than assuming it helps.
"""
import json
import numpy as np
import fasttext

_orig_np_array = np.array
def _np_array_compat(*args, **kwargs):
    if kwargs.get("copy") is False:
        kwargs["copy"] = None
    return _orig_np_array(*args, **kwargs)
np.array = _np_array_compat

WORK_DIR = "/projects/data/datasets/code_data/sai_rupesh/taxonomy/classifier_experiments"
AXES = ["domain", "task_family"]


def load_val(axis):
    val = []
    with open(f"{WORK_DIR}/{axis}_val.txt") as f:
        for line in f:
            parts = line.strip().split(" ")
            labels = [p.replace("__label__", "") for p in parts if p.startswith("__label__")]
            text = " ".join(p for p in parts if not p.startswith("__label__"))
            val.append((labels, text))
    return val


def predict_with_fallback(model, text, threshold):
    pred_labels, pred_scores = model.predict(text, k=-1, threshold=threshold)
    if pred_labels:
        return {p.replace("__label__", "") for p in pred_labels}
    # nothing cleared the threshold -- fall back to the single best guess
    # instead of predicting nothing (a real document always has >=1 label)
    top_labels, top_scores = model.predict(text, k=1)
    return {p.replace("__label__", "") for p in top_labels}


def score(model, val, threshold, use_fallback):
    tp = fp = fn = 0
    exact = 0
    zero_pred_count = 0
    for labels, text in val:
        if use_fallback:
            pred_set = predict_with_fallback(model, text, threshold)
        else:
            pred_labels, _ = model.predict(text, k=-1, threshold=threshold)
            pred_set = {p.replace("__label__", "") for p in pred_labels}
            if not pred_set:
                zero_pred_count += 1
        true_set = set(labels)
        tp += len(pred_set & true_set)
        fp += len(pred_set - true_set)
        fn += len(true_set - pred_set)
        if pred_set == true_set:
            exact += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1, exact / len(val), zero_pred_count


def main():
    # NOTE: the optuna-tuned models (domain_fasttext_optuna_best.bin etc.,
    # timestamped Sep 16 12:35/13:12) predate the CURRENT train/val split
    # files (regenerated Sep 16 15:54-15:56 for the 98,436-doc baseline
    # retrain). Scoring the old optuna models against the new val.txt is an
    # invalid, contaminated comparison -- some "val" rows may have been in
    # that old model's own training set. Use the plain baseline models
    # instead (domain_fasttext.bin / task_family_fasttext.bin), which ARE
    # timestamped together with the current split files, so this is a real
    # held-out comparison. Default threshold 0.5, matching what
    # fasttext_baseline.py itself uses (these were never optuna-tuned).
    for axis in AXES:
        threshold = 0.5
        model_path = f"{WORK_DIR}/{axis}_fasttext.bin"
        print(f"\n=== axis: {axis}  (threshold={threshold:.4f}, model={model_path}) ===")

        model = fasttext.load_model(model_path)
        val = load_val(axis)

        p0, r0, f10, e0, zero_count = score(model, val, threshold, use_fallback=False)
        print(f"BEFORE (no fallback): precision={p0:.3f} recall={r0:.3f} f1={f10:.3f} exact={e0:.3f}")
        print(f"  zero-label predictions: {zero_count}/{len(val)} ({100*zero_count/len(val):.2f}%)")

        p1, r1, f11, e1, _ = score(model, val, threshold, use_fallback=True)
        print(f"AFTER  (top-1 fallback): precision={p1:.3f} recall={r1:.3f} f1={f11:.3f} exact={e1:.3f}")
        print(f"  delta F1: {f11-f10:+.4f}   delta exact-match: {e1-e0:+.4f}")


if __name__ == "__main__":
    main()
