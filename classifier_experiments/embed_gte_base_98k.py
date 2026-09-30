"""
Encode every 'ok' row of the current 98,436-doc combined training file into a
gte-base-en-v1.5 embedding vector, ONCE, so both the domain and task_family
classifier heads can reuse the same vectors (most docs have labels for both
axes, so encoding once instead of per-axis roughly halves the work).

Chunked + checkpointed on purpose: a single CPU-only encode pass over 98k
docs takes a few hours (measured ~7.4 docs/sec at batch_size=128 on this
box), so if the process dies partway we don't want to lose everything and
start over.

Loads rows the SAME way fasttext_baseline.py does (same file, same "ok"
filter, same row order) so the row index here lines up 1:1 with what a
training script sees if it also calls this same load_rows().
"""
import os
os.environ["HF_HOME"] = "/projects/data/datasets/code_data/sai_rupesh/taxonomy/.hf_cache"
os.environ["TRANSFORMERS_CACHE"] = os.environ["HF_HOME"]
os.environ["SENTENCE_TRANSFORMERS_HOME"] = os.environ["HF_HOME"]

import json
import re
import time
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

DATA_PATH = "/projects/data/datasets/code_data/sai_rupesh/taxonomy/classifier_experiments/sft_output_sample_combined_98436.jsonl"
WORK_DIR = "/projects/data/datasets/code_data/sai_rupesh/taxonomy/classifier_experiments"
CKPT_DIR = f"{WORK_DIR}/gte_embed_chunks"
FINAL_EMB = f"{WORK_DIR}/gte_base_embeddings_98436.npy"
FINAL_META = f"{WORK_DIR}/gte_base_meta_98436.jsonl"
MAX_CHARS = 3000  # same truncation length as the earlier build_embeddings.py attempt
CHUNK_SIZE = 5000
BATCH_SIZE = 128

os.makedirs(CKPT_DIR, exist_ok=True)
torch.set_num_threads(os.cpu_count())


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


def main():
    print("loading rows...", flush=True)
    rows = load_rows()
    n = len(rows)
    print(f"{n:,} ok rows", flush=True)

    print("loading gte-base-en-v1.5 (trust_remote_code)...", flush=True)
    model = SentenceTransformer(
        "Alibaba-NLP/gte-base-en-v1.5",
        trust_remote_code=True,
        cache_folder=os.environ["HF_HOME"],
    )

    with open(FINAL_META, "w") as f:
        for i, d in enumerate(rows):
            f.write(json.dumps({"idx": i, "uid": d["uid"]}) + "\n")

    n_chunks = (n + CHUNK_SIZE - 1) // CHUNK_SIZE
    t_start = time.time()
    for c in range(n_chunks):
        ckpt_path = f"{CKPT_DIR}/chunk_{c:05d}.npy"
        if os.path.exists(ckpt_path):
            continue
        lo, hi = c * CHUNK_SIZE, min((c + 1) * CHUNK_SIZE, n)
        texts = [clean_text(rows[i]["full_text"])[:MAX_CHARS] for i in range(lo, hi)]
        t0 = time.time()
        embs = model.encode(
            texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        np.save(ckpt_path, embs.astype(np.float32))
        dt = time.time() - t0
        elapsed = time.time() - t_start
        done = hi
        rate = done / elapsed if elapsed > 0 else 0
        eta_min = (n - done) / rate / 60 if rate > 0 else float("inf")
        print(
            f"chunk {c+1}/{n_chunks} rows {lo}-{hi} done in {dt:.1f}s "
            f"({len(texts)/dt:.2f} docs/sec)  overall {done}/{n} "
            f"rate={rate:.2f} docs/sec  ETA={eta_min:.1f}min",
            flush=True,
        )

    print("all chunks encoded, concatenating...", flush=True)
    parts = []
    for c in range(n_chunks):
        parts.append(np.load(f"{CKPT_DIR}/chunk_{c:05d}.npy"))
    full = np.concatenate(parts, axis=0)
    assert full.shape[0] == n, f"{full.shape[0]} != {n}"
    np.save(FINAL_EMB, full)
    print(f"saved {FINAL_EMB} shape={full.shape}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
