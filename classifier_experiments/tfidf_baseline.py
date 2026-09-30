"""
TF-IDF + linear classifier baseline -- the "dead simple, cheap" sanity floor
requested alongside the embedding-model re-attempt. No neural network, no
GPU, no pretrained weights: just word/ngram counts + logistic regression.

Evaluated on the EXACT SAME train/val split and metrics as fasttext_baseline.py
(same DATA_PATH, same SEED, same per-axis filtering/shuffle/15% split logic,
same micro precision/recall/F1 + exact-match-rate formulas) so all three
methods (FastText, TF-IDF+LR, gte-base+LR) are directly comparable.
"""
import json
import random
import re
import time
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

DATA_PATH = "/projects/data/datasets/code_data/sai_rupesh/taxonomy/classifier_experiments/sft_output_sample_combined_98436_clean.jsonl"
WORK_DIR = "/projects/data/datasets/code_data/sai_rupesh/taxonomy/classifier_experiments"
AXES = ["domain", "task_family"]
SEED = 42  # identical seed to fasttext_baseline.py / embedding_baseline.py -> identical split


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


def train_and_eval(axis, rows):
    print(f"\n=== axis: {axis} (TF-IDF + LogisticRegression) ===", flush=True)
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
    print(f"train={len(train)} val={len(val)}", flush=True)

    train_labels = [labels for labels, _ in train]
    train_texts = [text for _, text in train]
    val_labels = [labels for labels, _ in val]
    val_texts = [text for _, text in val]

    mlb = MultiLabelBinarizer()
    y_train = mlb.fit_transform(train_labels)
    known = set(mlb.classes_)
    y_val = mlb.transform([[l for l in labels if l in known] for labels in val_labels])

    print("fitting TF-IDF vectorizer on train text...", flush=True)
    t0 = time.time()
    vec = TfidfVectorizer(
        max_features=100_000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )
    X_train = vec.fit_transform(train_texts)
    X_val = vec.transform(val_texts)
    print(f"  vectorized: X_train={X_train.shape} X_val={X_val.shape} ({time.time()-t0:.1f}s)", flush=True)

    print("training OneVsRest logistic regression...", flush=True)
    t0 = time.time()
    # Tried n_jobs=-1 with loky (default): crashed with "WRITEBACKIFCOPY base
    # is read-only" -- loky memory-maps the big sparse matrix as read-only
    # across worker processes, and liblinear needs to mutate it in place.
    # Tried n_jobs=-1 with the threading backend instead: that avoided the
    # memmap issue but then SEGFAULTED (exit code 139) -- liblinear's
    # underlying C code is not safe to run this many-way concurrently inside
    # one process's threads. Both are real, reproduced failures on this box,
    # not guesses. Falling back to the simple, guaranteed-correct option:
    # fit the ~100 per-label classifiers one at a time, single-threaded, no
    # parallel backend at all. Slower, but this is the "dead simple baseline"
    # anyway -- correctness matters more than shaving minutes off it.
    clf = OneVsRestClassifier(
        LogisticRegression(max_iter=1000, solver="liblinear"),
        n_jobs=1,
    )
    clf.fit(X_train, y_train)
    train_time = time.time() - t0
    print(f"  trained in {train_time:.1f}s", flush=True)

    t0 = time.time()
    y_pred = clf.predict(X_val)
    predict_time = time.time() - t0
    docs_per_sec = len(val) / predict_time if predict_time > 0 else float("inf")

    y_pred = np.asarray(y_pred)
    y_val = np.asarray(y_val)

    tp = int((y_pred & y_val).sum())
    fp = int((y_pred & (1 - y_val)).sum())
    fn = int(((1 - y_pred) & y_val).sum())
    exact = int((y_pred == y_val).all(axis=1).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    exact_match_rate = exact / len(val)

    print(f"  micro precision: {precision:.3f}", flush=True)
    print(f"  micro recall:    {recall:.3f}", flush=True)
    print(f"  micro F1:        {f1:.3f}", flush=True)
    print(f"  exact-set match: {exact_match_rate:.3f}  ({exact}/{len(val)})", flush=True)
    print(f"  predict throughput: {docs_per_sec:.1f} docs/sec (val set, single process)", flush=True)

    return {
        "axis": axis,
        "method": "tfidf_logreg",
        "train_n": len(train),
        "val_n": len(val),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_match_rate": exact_match_rate,
        "train_time_sec": train_time,
        "predict_docs_per_sec": docs_per_sec,
        "vocab_size": len(vec.vocabulary_),
    }


def main():
    rows = load_rows()
    print(f"loaded {len(rows)} ok rows", flush=True)
    results = []
    for axis in AXES:
        results.append(train_and_eval(axis, rows))

    print("\n=== summary ===", flush=True)
    for r in results:
        print(f"{r['axis']:15s} F1={r['f1']:.3f}  exact_match={r['exact_match_rate']:.3f}  (n={r['train_n']}+{r['val_n']})", flush=True)

    with open(f"{WORK_DIR}/tfidf_baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
