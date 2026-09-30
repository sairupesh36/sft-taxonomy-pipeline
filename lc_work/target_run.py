import sys, os, json, random, re, collections
sys.path.insert(0, "/projects/data/datasets/code_data/sai_rupesh/taxonomy/pylibs")
sys.path.insert(0, "/projects/data/datasets/code_data/sai_rupesh/taxonomy")
sys.path.insert(0, "/projects/data/datasets/code_data/sai_rupesh/taxonomy/classifier_experiments")
import numpy as np
_o = np.array
def _c(*a, **k):
    if k.get("copy") is False: k["copy"] = None
    return _o(*a, **k)
np.array = _c
import fasttext
from map_batch2 import to_full_text            # the FIXED truncation logic used for the training labels
CE = "/projects/data/datasets/code_data/sai_rupesh/taxonomy/classifier_experiments"
EN = "/projects/data/datasets/code_data/sai_rupesh/taxonomy/sft_43_language_wise/en"
THR = 0.8
PER_SHARD = 200
random.seed(2026)

def clean(t): return re.sub(r"\s+", " ", t.replace("\n", " ").replace("\t", " ")).strip()

rows = []
for fn in sorted(os.listdir(EN)):
    path = f"{EN}/{fn}"; size = os.path.getsize(path); got = 0; tries = 0
    with open(path, "rb") as f:
        while got < PER_SHARD and tries < PER_SHARD * 5:
            tries += 1
            f.seek(random.randint(0, size - 1)); f.readline(); line = f.readline()
            if not line: continue
            try: d = json.loads(line)
            except Exception: continue
            msgs = d.get("messages")
            if not msgs: continue
            rows.append(dict(shard=fn.split("_openai")[0], messages=msgs)); got += 1
print("sampled", len(rows), flush=True)

md = fasttext.load_model(f"{CE}/domain_fasttext.bin"); mt = fasttext.load_model(f"{CE}/task_family_fasttext.bin")
def predict(model, text):
    labs, probs = model.predict(text, k=-1, threshold=THR)
    acc = {l.replace("__label__", ""): round(float(p), 3) for l, p in zip(labs, probs)}
    tl, tp = model.predict(text, k=1)
    return acc, tl[0].replace("__label__", ""), round(float(tp[0]), 3)

out = []
for r in rows:
    full = to_full_text(r["messages"]); text = clean(full)
    da, dtop, dconf = predict(md, text); ta, ttop, tconf = predict(mt, text)
    out.append(dict(shard=r["shard"], full_text=full, domain_accepted=da, domain_top1=dtop, domain_conf=dconf,
                    task_accepted=ta, task_top1=ttop, task_conf=tconf))
json.dump(out, open("/projects/data/datasets/code_data/sai_rupesh/taxonomy/lc_work/target_run_80.json", "w"))
print("saved", len(out))
