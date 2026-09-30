"""Train one (axis, config) and score it on: clean val, weak val, and the UNBIASED Gemma-labelled target test set T.
config: base | A (base + uncertain-selected rows) | B (base + equally many random rows) | AB (both)"""
import sys, os, json, random
sys.path.insert(0, "/projects/data/datasets/code_data/sai_rupesh/taxonomy/pylibs")
sys.path.insert(0, "/projects/data/datasets/code_data/sai_rupesh/taxonomy/classifier_experiments")
import fasttext_baseline as B
from fasttext_predict_utils import predict_with_fallback
import fasttext
CE = "/projects/data/datasets/code_data/sai_rupesh/taxonomy/classifier_experiments"; W = "/projects/data/datasets/code_data/sai_rupesh/taxonomy/lc_work"
axis, config = sys.argv[1], sys.argv[2]
def load(path):
    out = []
    for line in open(path):
        d = json.loads(line)
        if d.get("ok") and d["result"].get(axis): out.append((d["result"][axis], B.clean_text(d["full_text"])))
    return out
ex = load(f"{CE}/sft_output_sample_combined_98436_clean.jsonl"); random.Random(42).shuffle(ex)
nv = int(len(ex) * 0.15); val, train = ex[:nv], ex[nv:]
weak = load(f"{CE}/sft_output_sample_weak_categories_v2_mapped.jsonl"); random.Random(42).shuffle(weak)
nw = int(len(weak) * 0.15); weak_val, weak_train = weak[:nw], weak[nw:]
tr = train + weak_train
if "A" in config: tr += load(f"{W}/labeled_A.jsonl")
if "B" in config: tr += load(f"{W}/labeled_B.jsonl")
test = load(f"{W}/labeled_T.jsonl")
target_texts = [B.clean_text(r["full_text"]) for r in json.load(open(f"{W}/target_run_80.json"))]
tp_ = f"{W}/tmp/{axis}_arm_{config}.txt"
with open(tp_, "w") as f:
    for labels, text in tr: f.write(B.make_fasttext_line(labels, text))
model = fasttext.train_supervised(input=tp_, epoch=25, lr=0.5, wordNgrams=2, dim=100, loss="ova", thread=16, verbose=0); os.remove(tp_)
if os.environ.get("SAVE_MODEL"): model.save_model(os.environ["SAVE_MODEL"])
def score(vs):
    tp = fp = fn = ex_ = kept = ktp = kfp = kfn = 0
    for labels, text in vs:
        pred = predict_with_fallback(model, text, 0.5); true = set(labels)
        tp += len(pred & true); fp += len(pred - true); fn += len(true - pred); ex_ += pred == true
        if model.predict(text, k=1)[1][0] >= 0.8: kept += 1; ktp += len(pred & true); kfp += len(pred - true); kfn += len(true - pred)
    f = lambda a, b, c: 2 * a / max(1, 2 * a + b + c)
    return dict(n=len(vs), f1=round(f(tp, fp, fn), 4), exact=round(ex_ / len(vs), 4), keep80=round(kept / len(vs), 4), f1_keep80=round(f(ktp, kfp, kfn), 4))
acc = sum(1 for t in target_texts if model.predict(t, k=1)[1][0] >= 0.8) / len(target_texts)
res = dict(axis=axis, config=config, train_n=len(tr), clean_val=score(val), weak_val=score(weak_val), target_test=score(test), target_10k_accept80=round(acc, 4))
json.dump(res, open(f"{W}/results/arm_{axis}_{config}.json", "w"), indent=1); print("done", axis, config)
