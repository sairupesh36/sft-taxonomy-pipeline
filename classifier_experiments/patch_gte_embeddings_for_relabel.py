"""
The validator agent found that 1,330 documents had their `full_text`
corrected (as part of tonight's truncation-bug relabel) between the
original `sft_output_sample_combined_98436.jsonl` (which the big overnight
gte-base-en-v1.5 encoding job was already hours into when the fix landed)
and the new `sft_output_sample_combined_98436_clean.jsonl`.

Restarting the whole multi-hour encode job over the corrected file would
throw away everything already computed just to fix 1,330 out of 98,436 rows
(1.4%). Cheaper, equally correct fix: re-encode ONLY those 1,330 rows using
their corrected text, then patch just those rows into the final embeddings
array before training the classifier head. Run this AFTER
embed_gte_base_98k.py finishes (needs its full output array + meta file).
"""
import os
os.environ["HF_HOME"] = "/projects/data/datasets/code_data/sai_rupesh/taxonomy/.hf_cache"
os.environ["TRANSFORMERS_CACHE"] = os.environ["HF_HOME"]
os.environ["SENTENCE_TRANSFORMERS_HOME"] = os.environ["HF_HOME"]

import json
import re
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

WORK_DIR = "/projects/data/datasets/code_data/sai_rupesh/taxonomy/classifier_experiments"
CLEAN_PATH = f"{WORK_DIR}/sft_output_sample_combined_98436_clean.jsonl"
CHANGED_LIST = f"{WORK_DIR}/gte_relabeled_rows_to_reencode.json"
EMB_IN = f"{WORK_DIR}/gte_base_embeddings_98436.npy"
EMB_OUT = f"{WORK_DIR}/gte_base_embeddings_98436_patched.npy"
MAX_CHARS = 3000

torch.set_num_threads(os.cpu_count())


def clean_text(text):
    text = text.replace("\n", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    with open(CHANGED_LIST) as f:
        changed = json.load(f)
    print(f"{len(changed)} rows need re-encoding with corrected text", flush=True)

    # Pull the corrected full_text for exactly these row indices from the
    # clean file (indices line up 1:1 with the original file -- verified
    # 0 uid mismatches across all 98,436 lines before trusting this).
    idx_set = {c["idx"] for c in changed}
    idx_to_text = {}
    with open(CLEAN_PATH) as f:
        for i, line in enumerate(f):
            if i in idx_set:
                d = json.loads(line)
                assert d["uid"] == next(c["uid"] for c in changed if c["idx"] == i)
                idx_to_text[i] = clean_text(d["full_text"])[:MAX_CHARS]
            if len(idx_to_text) == len(idx_set):
                break

    ordered_idx = [c["idx"] for c in changed]
    texts = [idx_to_text[i] for i in ordered_idx]

    print("loading gte-base-en-v1.5...", flush=True)
    model = SentenceTransformer(
        "Alibaba-NLP/gte-base-en-v1.5",
        trust_remote_code=True,
        cache_folder=os.environ["HF_HOME"],
    )
    print("encoding corrected rows...", flush=True)
    new_embs = model.encode(
        texts, batch_size=64, show_progress_bar=False,
        convert_to_numpy=True, normalize_embeddings=True,
    )

    embs = np.load(EMB_IN)
    for pos, i in enumerate(ordered_idx):
        embs[i] = new_embs[pos]
    np.save(EMB_OUT, embs)
    print(f"patched {len(ordered_idx)} rows, saved -> {EMB_OUT}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
