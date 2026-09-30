"""
Train a linear (logistic regression) classifier head on top of FROZEN
gte-base-en-v1.5 embeddings (produced separately by embed_gte_base_98k.py,
since encoding is the slow CPU-bound part and we don't want to redo it every
time we tweak the classifier head).

Uses the EXACT SAME train/val split logic, seed, and metric formulas as
fasttext_baseline.py / tfidf_baseline.py so all methods are directly
comparable on the same 98,436-doc dataset.
"""
import json
import random
import re
import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer

# Using the CLEAN file (validator-found bugs fixed: 4,003 rows with
# dropped context/tool-call turns excluded via ok=False; 1,330 rows have
# corrected full_text/labels from the truncation relabel) -- same file
# fasttext_baseline.py now uses, for a fair, apples-to-apples comparison.
# Row order/uids are identical to the original dirty file used for the
# main embedding pass (verified: 0 mismatches across all 98,436 lines), so
# embedding row index i still lines up with this file's row i.
DATA_PATH = "/projects/data/datasets/code_data/sai_rupesh/taxonomy/classifier_experiments/sft_output_sample_combined_98436_clean.jsonl"
WORK_DIR = "/projects/data/datasets/code_data/sai_rupesh/taxonomy/classifier_experiments"
# The 1,330 rows whose full_text changed in the clean file were re-encoded
# separately (patch_gte_embeddings_for_relabel.py) rather than redoing the
# whole multi-hour job -- use the patched array if it exists, otherwise
# fall back to the raw one (with a loud warning, since that means those
# 1,330 rows' embeddings don't match their current text/labels).
import os as _os
_patched = f"{WORK_DIR}/gte_base_embeddings_98436_patched.npy"
EMB_PATH = _patched if _os.path.exists(_patched) else f"{WORK_DIR}/gte_base_embeddings_98436.npy"
META_PATH = f"{WORK_DIR}/gte_base_meta_98436.jsonl"
AXES = ["domain", "task_family"]
SEED = 42  # identical seed -> identical split as fasttext_baseline.py


def load_rows():
    # IMPORTANT: keep each row's RAW line number (its position in the file),
    # not its position after filtering. The embeddings array is indexed by
    # raw line number (embed_gte_base_98k.py embedded EVERY row of the
    # original dirty file -- confirmed separately that 100% of those rows
    # were ok=True there, so no compaction happened on that side). This
    # clean file now has 4,003 rows flipped to ok=False that were embedded
    # anyway (their vectors just won't be used) -- if we used a compacted,
    # post-filter position here instead of the raw line number, every row
    # after the first excluded one would silently point at the WRONG
    # document's embedding. Caught this before running anything, not after.
    rows = []
    with open(DATA_PATH) as f:
        for raw_idx, line in enumerate(f):
            d = json.loads(line)
            if not d.get("ok"):
                continue
            rows.append((raw_idx, d))
    return rows


def train_and_eval(axis, rows, embs):
    print(f"\n=== axis: {axis} (gte-base-en-v1.5 + LogisticRegression) ===", flush=True)
    # (labels, row_index_into_embs) -- row_index here is the RAW line number
    # from load_rows(), which is what actually lines up with the embeddings
    # array (see the comment in load_rows()).
    examples = []
    for raw_idx, d in rows:
        labels = d["result"].get(axis, [])
        if not labels:
            continue
        examples.append((labels, raw_idx))

    random.Random(SEED).shuffle(examples)
    n_val = max(1, int(len(examples) * 0.15))
    val = examples[:n_val]
    train = examples[n_val:]
    print(f"train={len(train)} val={len(val)}", flush=True)

    train_labels = [labels for labels, _ in train]
    train_idx = np.array([i for _, i in train])
    val_labels = [labels for labels, _ in val]
    val_idx = np.array([i for _, i in val])

    X_train = embs[train_idx]
    X_val = embs[val_idx]

    mlb = MultiLabelBinarizer()
    y_train = mlb.fit_transform(train_labels)
    known = set(mlb.classes_)
    y_val = mlb.transform([[l for l in labels if l in known] for labels in val_labels])

    print("training MultiOutput LogisticRegression on frozen embeddings...", flush=True)
    t0 = time.time()
    # tfidf_baseline.py hit two real crashes trying to parallelize this kind
    # of per-label classifier fit (loky: "WRITEBACKIFCOPY base is read-only"
    # on the shared matrix; threading backend: segfault, exit 139, from
    # liblinear's C code not being safe to run this concurrently). This data
    # is dense (not sparse) and much smaller per-label (768 dims), and uses
    # lbfgs not liblinear, but to not risk losing hours of embedding work to
    # the same class of crash, staying single-threaded here too -- it's fast
    # enough on 768-dim vectors regardless.
    clf = MultiOutputClassifier(LogisticRegression(max_iter=2000), n_jobs=1)
    clf.fit(X_train, y_train)
    train_time = time.time() - t0
    print(f"  trained in {train_time:.1f}s", flush=True)

    t0 = time.time()
    y_pred = np.asarray(clf.predict(X_val))
    predict_time = time.time() - t0
    docs_per_sec_classify_only = len(val) / predict_time if predict_time > 0 else float("inf")

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
    print(f"  classify-only throughput (embeddings already computed): {docs_per_sec_classify_only:.1f} docs/sec", flush=True)

    return {
        "axis": axis,
        "method": "gte_base_en_v1.5_logreg",
        "train_n": len(train),
        "val_n": len(val),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_match_rate": exact_match_rate,
        "train_time_sec": train_time,
        "classify_only_docs_per_sec": docs_per_sec_classify_only,
    }


def main():
    if EMB_PATH.endswith("_patched.npy"):
        print(f"using PATCHED embeddings (1,330 relabeled rows re-encoded): {EMB_PATH}", flush=True)
    else:
        print(
            f"WARNING: patched embeddings not found, using raw {EMB_PATH} -- "
            "1,330 rows with corrected text/labels (per the clean file) will "
            "still have their OLD, pre-fix embeddings. Run "
            "patch_gte_embeddings_for_relabel.py first for a fully fair "
            "comparison.",
            flush=True,
        )
    print("loading rows...", flush=True)
    rows = load_rows()
    print(f"{len(rows)} ok rows (clean file)", flush=True)

    print(f"loading embeddings from {EMB_PATH}...", flush=True)
    embs = np.load(EMB_PATH)
    print(f"embeddings shape: {embs.shape}", flush=True)
    # These counts are EXPECTED to differ now: embs has one row per document
    # in the original dirty file (all 98,436 were ok=True there), while
    # `rows` only has the clean file's still-ok=True subset (~94,433) -- the
    # ~4,003 excluded rows were embedded but just won't be indexed into.
    # Sanity-check the bound instead of exact equality.
    max_raw_idx = max(raw_idx for raw_idx, _ in rows)
    assert max_raw_idx < embs.shape[0], (
        f"a clean-file row index ({max_raw_idx}) falls outside the embeddings "
        f"array ({embs.shape[0]} rows) -- alignment assumption broken"
    )

    results = []
    for axis in AXES:
        results.append(train_and_eval(axis, rows, embs))

    print("\n=== summary ===", flush=True)
    for r in results:
        print(f"{r['axis']:15s} F1={r['f1']:.3f}  exact_match={r['exact_match_rate']:.3f}  (n={r['train_n']}+{r['val_n']})", flush=True)

    with open(f"{WORK_DIR}/gte_base_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
