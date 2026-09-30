import os, sys, json, re
os.environ["HF_HOME"] = "/projects/data/datasets/code_data/sai_rupesh/taxonomy/classifier_experiments/hf_model_cache"
sys.path.insert(0, "/projects/data/datasets/code_data/sai_rupesh/taxonomy/pylibs")
import numpy as np
from sentence_transformers import SentenceTransformer
d = json.load(open("lc_work/mismatch_probe.json")); rand = d["rand"]
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
U = model.encode([x[1] for x in rand], batch_size=64, normalize_embeddings=True, show_progress_bar=False)
A = model.encode([x[2] for x in rand], batch_size=64, normalize_embeddings=True, show_progress_bar=False)
cos = (U * A).sum(1)
# shuffled control: pair each question with a DIFFERENT row's answer -> what a truly mismatched pair scores
rng = np.random.default_rng(0); perm = rng.permutation(len(rand)); ctrl = (U * A[perm]).sum(1)
print(f"n={len(rand)}  related pairs cosine: median {np.median(cos):.2f}  p5 {np.percentile(cos,5):.2f}  p1 {np.percentile(cos,1):.2f}")
print(f"random-mismatch control cosine: median {np.median(ctrl):.2f}  p95 {np.percentile(ctrl,95):.2f}")
thr = float(np.percentile(ctrl, 95))
low = np.argsort(cos)[:12]; print(f"\npairs with cosine below the mismatch-control p95 ({thr:.2f}): {int((cos < thr).sum())} of {len(rand)}  ({100*(cos<thr).mean():.1f}%)")
print("\n12 lowest-scoring pairs (question || answer start):")
for i in low: print(f"  cos={cos[i]:.2f} | Q: {rand[i][1][:110]!r}\n            A: {rand[i][2][:150]!r}")
json.dump(dict(cos=cos.tolist(), thr=thr), open("lc_work/mismatch_embed.json", "w"))
