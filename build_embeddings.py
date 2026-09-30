#!/usr/bin/env python3
"""
Step 1 of the classifier pipeline: turn every gemma-labeled document into one
embedding vector (using gte-base-en-v1.5, CPU-only), and save vectors +
labels together so every classifier head we train later reuses the same
embeddings instead of recomputing them.
"""
import json
import time
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

IN_FILE = "/projects/data/datasets/code_data/sai_rupesh/taxonomy/sft_output_sample_mapped.jsonl"
OUT_EMB = "/projects/data/datasets/code_data/sai_rupesh/taxonomy/embeddings/gte_base_embeddings.npy"
OUT_META = "/projects/data/datasets/code_data/sai_rupesh/taxonomy/embeddings/gte_base_meta.jsonl"
HF_HOME = "/projects/data/datasets/code_data/sai_rupesh/taxonomy/.hf_cache"
MAX_CHARS = 3000  # truncate long conversations; keeps CPU encoding fast, keeps the informative opening

import os
os.makedirs(os.path.dirname(OUT_EMB), exist_ok=True)

torch.set_num_threads(os.cpu_count())

def load_rows():
    rows = []
    with open(IN_FILE, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if not r.get("ok"):
                continue
            rows.append(r)
    return rows

def main():
    print("loading model...", flush=True)
    model = SentenceTransformer("Alibaba-NLP/gte-base-en-v1.5", trust_remote_code=True, cache_folder=HF_HOME)

    rows = load_rows()
    print(f"{len(rows):,} labeled docs to embed", flush=True)

    texts = [r["full_text"][:MAX_CHARS] for r in rows]

    t0 = time.time()
    embs = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    dt = time.time() - t0
    print(f"encoded {len(texts):,} docs in {dt/60:.1f} min ({len(texts)/dt:.1f} docs/sec)", flush=True)

    np.save(OUT_EMB, embs.astype(np.float32))

    with open(OUT_META, "w", encoding="utf-8") as f:
        for r in rows:
            res = r["result"]
            f.write(json.dumps({
                "uid": r["uid"],
                "orig_domain": r["orig_domain"],
                "orig_family": r["orig_family"],
                "domain": res.get("domain", []),
                "domain_subdomain": res.get("domain_subdomain", []),
                "task_family": res.get("task_family", []),
                "task_subfamily": res.get("task_subfamily", []),
                "interaction_mode": res.get("interaction_mode"),
                "tool_requirement": res.get("tool_requirement"),
                "complexity": res.get("complexity"),
                "task_composition": res.get("task_composition"),
                "response_behavior": res.get("response_behavior"),
                "input_language": (res.get("input") or {}).get("language"),
            }, ensure_ascii=False) + "\n")

    print(f"saved embeddings: {OUT_EMB}  shape={embs.shape}")
    print(f"saved metadata:   {OUT_META}")

if __name__ == "__main__":
    main()
