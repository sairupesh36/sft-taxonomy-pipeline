"""
FastText multi-label baseline for the taxonomy axes, trained on Gemma's own
mapped output (sft_output_sample_mapped.jsonl, 15,722 docs).

Goal: get a cheap, CPU-only floor score BEFORE building the embedding-model
version, so we know what we're trying to beat. Only run on the two axes with
enough per-class data to be meaningful (domain, task_family) -- subdomain and
task_subfamily are too thin per-class right now (see class-count check).
"""
import json
import random
import re
import numpy as np
import fasttext

from fasttext_predict_utils import predict_with_fallback

# The installed fasttext package calls np.array(probs, copy=False) inside
# predict(), which numpy>=2.0 now raises on instead of silently copying (a
# real upstream fasttext/numpy version mismatch, not our bug). Patch
# np.array in this process only so copy=False falls back to "copy if
# needed" (numpy's old behavior), since fasttext can't be fixed upstream.
_orig_np_array = np.array
def _np_array_compat(*args, **kwargs):
    if kwargs.get("copy") is False:
        kwargs["copy"] = None
    return _orig_np_array(*args, **kwargs)
np.array = _np_array_compat

# 2026-09-20: switched to the CLEANED version of this file -- same 98,436
# rows, but 1,341 severely-truncated rows are re-labeled (or excluded if
# unrecoverable) using the fixed to_full_text(), and 3,209 rows with a
# silently-dropped context/passage column are excluded (see
# overnight_builder1_log.md for the full story). Rows are marked ok=False
# rather than removed, so load_rows()'s existing `if not d.get("ok")` skip
# is all that's needed -- no other logic here changes.
DATA_PATH = "/projects/data/datasets/code_data/sai_rupesh/taxonomy/classifier_experiments/sft_output_sample_combined_98436_clean.jsonl"
WORK_DIR = "/projects/data/datasets/code_data/sai_rupesh/taxonomy/classifier_experiments"
AXES = ["domain", "task_family"]
SEED = 42


def clean_text(text):
    text = text.replace("\n", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text).strip()
    # fasttext treats "__label__" as a control token -- strip if it ever
    # appears literally inside document text so it can't corrupt a line.
    text = text.replace("__label__", "")
    return text


def load_rows():
    rows = []
    with open(DATA_PATH) as f:
        for line in f:
            d = json.loads(line)
            if not d.get("ok"):
                continue
            rows.append(d)
    return rows


def make_fasttext_line(labels, text):
    label_str = " ".join(f"__label__{l}" for l in labels)
    return f"{label_str} {text}\n"


def train_and_eval(axis, rows):
    print(f"\n=== axis: {axis} ===")
    examples = []
    for d in rows:
        labels = d["result"].get(axis, [])
        if not labels:
            continue
        text = clean_text(d["full_text"])
        examples.append((labels, text))

    random.Random(SEED).shuffle(examples)
    n_val = max(1, int(len(examples) * 0.15))
    val = examples[:n_val]
    train = examples[n_val:]
    print(f"train={len(train)} val={len(val)}")

    train_path = f"{WORK_DIR}/{axis}_train.txt"
    val_path = f"{WORK_DIR}/{axis}_val.txt"
    with open(train_path, "w") as f:
        for labels, text in train:
            f.write(make_fasttext_line(labels, text))
    with open(val_path, "w") as f:
        for labels, text in val:
            f.write(make_fasttext_line(labels, text))

    model = fasttext.train_supervised(
        input=train_path,
        epoch=25,
        lr=0.5,
        wordNgrams=2,
        dim=100,
        loss="ova",
        thread=4,
    )

    # Multi-label eval: micro precision/recall/F1 over the label sets,
    # plus exact-set-match rate.
    tp = fp = fn = 0
    exact = 0
    for labels, text in val:
        pred_set = predict_with_fallback(model, text, threshold=0.5)
        true_set = set(labels)
        tp += len(pred_set & true_set)
        fp += len(pred_set - true_set)
        fn += len(true_set - pred_set)
        if pred_set == true_set:
            exact += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    exact_match_rate = exact / len(val)

    print(f"  micro precision: {precision:.3f}")
    print(f"  micro recall:    {recall:.3f}")
    print(f"  micro F1:        {f1:.3f}")
    print(f"  exact-set match: {exact_match_rate:.3f}  ({exact}/{len(val)})")

    model_path = f"{WORK_DIR}/{axis}_fasttext.bin"
    model.save_model(model_path)
    print(f"  saved model -> {model_path}")

    return {
        "axis": axis,
        "train_n": len(train),
        "val_n": len(val),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_match_rate": exact_match_rate,
    }


def main():
    rows = load_rows()
    print(f"loaded {len(rows)} ok rows")
    results = []
    for axis in AXES:
        results.append(train_and_eval(axis, rows))

    print("\n=== summary ===")
    for r in results:
        print(f"{r['axis']:15s} F1={r['f1']:.3f}  exact_match={r['exact_match_rate']:.3f}  (n={r['train_n']}+{r['val_n']})")

    with open(f"{WORK_DIR}/fasttext_baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
