"""
Embedding-model + linear-head baseline, evaluated on the SAME train/val split
and SAME metrics as fasttext_baseline.py, so the two are directly comparable.

Approach: use a frozen pretrained sentence-embedding model to turn each doc
into a vector (no training of the big model -- just a forward pass, CPU-only,
no GPU needed). Then train one small multi-label classifier (logistic
regression per label, via sklearn's MultiOutputClassifier) on top of those
frozen vectors. This is the standard "linear probe" setup that MTEB's own
Classification score is designed to measure.
"""
import sys

import os
os.environ["HF_HOME"] = "/projects/data/datasets/code_data/sai_rupesh/taxonomy/classifier_experiments/hf_model_cache"
os.environ["TRANSFORMERS_CACHE"] = os.environ["HF_HOME"]
os.environ["SENTENCE_TRANSFORMERS_HOME"] = os.environ["HF_HOME"]

import json
import random
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer

DATA_PATH = "/projects/data/datasets/code_data/sai_rupesh/taxonomy/sft_output_sample_mapped.jsonl"
WORK_DIR = "/projects/data/datasets/code_data/sai_rupesh/taxonomy/classifier_experiments"
AXES = ["domain", "task_family"]
SEED = 42  # same seed as fasttext_baseline.py -> identical split


def clean_text(text):
    text = text.replace("\n", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text).strip()
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


def train_and_eval(axis, rows, model, model_name):
    print(f"\n=== axis: {axis}  |  model: {model_name} ===")
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

    train_labels = [labels for labels, _ in train]
    train_texts = [text for _, text in train]
    val_labels = [labels for labels, _ in val]
    val_texts = [text for _, text in val]

    mlb = MultiLabelBinarizer()
    y_train = mlb.fit_transform(train_labels)
    # unseen-in-train labels in val simply can't be predicted -- expected,
    # same limitation any classifier has.
    known = set(mlb.classes_)
    y_val = mlb.transform([[l for l in labels if l in known] for labels in val_labels])

    print("encoding train embeddings...")
    X_train = model.encode(train_texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True)
    print("encoding val embeddings...")
    X_val = model.encode(val_texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True)

    clf = MultiOutputClassifier(LogisticRegression(max_iter=1000), n_jobs=4)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_val)

    tp = int((y_pred & y_val).sum())
    fp = int((y_pred & (1 - y_val)).sum())
    fn = int(((1 - y_pred) & y_val).sum())
    exact = int((y_pred == y_val).all(axis=1).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    exact_match_rate = exact / len(val)

    print(f"  micro precision: {precision:.3f}")
    print(f"  micro recall:    {recall:.3f}")
    print(f"  micro F1:        {f1:.3f}")
    print(f"  exact-set match: {exact_match_rate:.3f}  ({exact}/{len(val)})")

    return {
        "axis": axis,
        "model": model_name,
        "train_n": len(train),
        "val_n": len(val),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_match_rate": exact_match_rate,
    }


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "sentence-transformers/all-MiniLM-L6-v2"
    rows = load_rows()
    print(f"loaded {len(rows)} ok rows")
    print(f"loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    results = []
    for axis in AXES:
        results.append(train_and_eval(axis, rows, model, model_name))

    print("\n=== summary ===")
    for r in results:
        print(f"{r['axis']:15s} F1={r['f1']:.3f}  exact_match={r['exact_match_rate']:.3f}  (n={r['train_n']}+{r['val_n']})  model={r['model']}")

    safe_name = model_name.replace("/", "__")
    with open(f"{WORK_DIR}/embedding_baseline_results_{safe_name}.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
