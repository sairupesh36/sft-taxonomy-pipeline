"""Learning curve + 'add weak_categories_v2' test. One process = one (axis, config).
Same split logic as classifier_experiments/fasttext_baseline.py (clean file, ok rows, seed 42, 15% val),
same FastText hyperparameters (epoch25 lr0.5 wordNgrams2 dim100 ova). Metrics only; models are not saved."""
import sys, os, json, random, re
sys.path.insert(0, "/projects/data/datasets/code_data/sai_rupesh/taxonomy/pylibs")
sys.path.insert(0, "/projects/data/datasets/code_data/sai_rupesh/taxonomy/classifier_experiments")
import fasttext_baseline as B          # reuses clean_text, make_fasttext_line + numpy/fasttext patch
from fasttext_predict_utils import predict_with_fallback
import fasttext
CE = "/projects/data/datasets/code_data/sai_rupesh/taxonomy/classifier_experiments"
axis, config = sys.argv[1], sys.argv[2]           # config: frac_0.125 ... frac_1.0 | plus_weak
WORK = "/projects/data/datasets/code_data/sai_rupesh/taxonomy/lc_work"

def load(path):
    out = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            if not d.get("ok"): continue
            labels = d["result"].get(axis, [])
            if labels: out.append((labels, B.clean_text(d["full_text"])))
    return out

ex = load(f"{CE}/sft_output_sample_combined_98436_clean.jsonl")
random.Random(42).shuffle(ex)
nval = max(1, int(len(ex) * 0.15)); val, train = ex[:nval], ex[nval:]
weak = load(f"{CE}/sft_output_sample_weak_categories_v2_mapped.jsonl")
random.Random(42).shuffle(weak)
nw = max(1, int(len(weak) * 0.15)); weak_val, weak_train = weak[:nw], weak[nw:]

if config.startswith("frac_"):
    frac = float(config.split("_")[1]); tr = train[: int(len(train) * frac)]
elif config == "plus_weak":
    tr = train + weak_train
else:
    raise SystemExit("bad config")

tp = f"{WORK}/tmp/{axis}_{config}.train.txt"
with open(tp, "w") as f:
    for labels, text in tr: f.write(B.make_fasttext_line(labels, text))
model = fasttext.train_supervised(input=tp, epoch=25, lr=0.5, wordNgrams=2, dim=100, loss="ova", thread=8, verbose=0)
os.remove(tp)

def score(valset):
    tp_ = fp = fn = ex_ = 0; kept = ktp = kfp = kfn = 0
    for labels, text in valset:
        pred = predict_with_fallback(model, text, 0.5); true = set(labels)
        tp_ += len(pred & true); fp += len(pred - true); fn += len(true - pred); ex_ += (pred == true)
        top = model.predict(text, k=1)[1][0]
        if top >= 0.8:
            kept += 1; ktp += len(pred & true); kfp += len(pred - true); kfn += len(true - pred)
    f = lambda a, b, c: (2 * a / (2 * a + b + c)) if (2 * a + b + c) else 0
    return dict(n=len(valset), f1=f(tp_, fp, fn), exact=ex_ / len(valset),
                gate80_coverage=kept / len(valset), gate80_f1=f(ktp, kfp, kfn))

res = dict(axis=axis, config=config, train_n=len(tr), clean_val=score(val), weak_val=score(weak_val))
json.dump(res, open(f"{WORK}/results/{axis}_{config}.json", "w"), indent=1)
print("done", axis, config, res["clean_val"]["f1"], res["weak_val"]["f1"])
