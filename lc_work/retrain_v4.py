"""Full retrain using every real labeled source now available, including the new 21,406-doc
weak-class batch. Same oversampling recipe as retrain_balanced.py (proven to give a real, if
partial, macro-F1 gain), same fastText hyperparams throughout this project, so any improvement
seen is attributable to the new DATA, not a changed recipe.

Eval sets:
  clean_val, weak_val       -- same 15% holdouts as before (existing sources)
  weak_candidates_val       -- NEW 15% holdout carved from the new weak-class batch itself. This
                               exists because target_test has only 1000 docs and near-zero real
                               support for many weak classes -- this new val set is built FROM the
                               weak-class search, so it has real statistical power for exactly the
                               classes we're trying to fix.
  target_test (T)           -- the untouched, unbiased 1000-doc real sample. Never trained on,
                               used here only to report the final honest number.
"""
import sys, os, json, random, collections, math, time
sys.path.insert(0, "/projects/data/datasets/code_data/sai_rupesh/taxonomy/pylibs")
sys.path.insert(0, "/projects/data/datasets/code_data/sai_rupesh/taxonomy/classifier_experiments")
import numpy as np
_orig = np.array
def _c(*a, **k):
    if k.get("copy") is False: k["copy"] = None
    return _orig(*a, **k)
np.array = _c
import fasttext
import fasttext_baseline as B
from fasttext_predict_utils import predict_capped

CE = "/projects/data/datasets/code_data/sai_rupesh/taxonomy/classifier_experiments"
W = "/projects/data/datasets/code_data/sai_rupesh/taxonomy/lc_work"
FLOOR = 400; MAX_DUP = 25

def load(path, axis):
    out = []
    for line in open(path):
        d = json.loads(line)
        if d.get("ok") and d["result"].get(axis):
            out.append((d["result"][axis], B.clean_text(d["full_text"])))
    return out

def macro_micro(model, examples, threshold=0.8):
    per_class = collections.defaultdict(lambda: [0, 0, 0]); tp = fp = fn = 0
    for labels, text in examples:
        pred = set(predict_capped(model, text, threshold)); true = set(labels)
        tp += len(pred & true); fp += len(pred - true); fn += len(true - pred)
        for c in true | pred:
            e = per_class[c]
            if c in true and c in pred: e[0] += 1
            elif c in pred: e[1] += 1
            elif c in true: e[2] += 1
    f = lambda a, b, c: 2 * a / max(1, 2 * a + b + c)
    rows = sorted(((c, a + cc, f(a, b, cc)) for c, (a, b, cc) in per_class.items()), key=lambda r: r[2])
    macro = sum(r[2] for r in rows) / len(rows) if rows else 0.0
    return f(tp, fp, fn), macro, rows

for axis in ("domain", "task_family"):
    print(f"\n########## {axis}", flush=True)
    ex = load(f"{CE}/sft_output_sample_combined_98436_clean.jsonl", axis); random.Random(42).shuffle(ex)
    nv = int(len(ex) * 0.15); clean_val, train = ex[:nv], ex[nv:]
    weak = load(f"{CE}/sft_output_sample_weak_categories_v2_mapped.jsonl", axis); random.Random(42).shuffle(weak)
    nw = int(len(weak) * 0.15); weak_val, weak_train = weak[:nw], weak[nw:]
    wc = load(f"{W}/labeled_weak_candidates.jsonl", axis); random.Random(42).shuffle(wc)
    nc = int(len(wc) * 0.15); weak_candidates_val, weak_candidates_train = wc[:nc], wc[nc:]
    A = load(f"{W}/labeled_A.jsonl", axis); Bx = load(f"{W}/labeled_B.jsonl", axis)
    target_test = load(f"{W}/labeled_T.jsonl", axis)

    tr = train + weak_train + weak_candidates_train + A + Bx
    print(f"  train pool: base={len(train)} weak={len(weak_train)} NEW_weak_candidates={len(weak_candidates_train)} A={len(A)} B={len(Bx)} -> total {len(tr):,}")
    print(f"  eval sets: clean_val={len(clean_val)} weak_val={len(weak_val)} weak_candidates_val={len(weak_candidates_val)} target_test={len(target_test)}")

    counts = collections.Counter(l for labels, _ in tr for l in labels)
    dup = []
    for labels, text in tr:
        d_ = min(MAX_DUP, max(1, math.ceil(FLOOR / min(counts[l] for l in labels))))
        dup += [(labels, text)] * d_
    random.Random(7).shuffle(dup)
    print(f"  oversampled: {len(tr):,} -> {len(dup):,} rows (rarest real class: {counts.most_common()[-1]})")

    tmp = f"{W}/tmp_{axis}_v4.txt"
    with open(tmp, "w") as f:
        for labels, text in dup: f.write(B.make_fasttext_line(labels, text))
    t0 = time.time()
    model = fasttext.train_supervised(input=tmp, epoch=25, lr=0.5, wordNgrams=2, dim=100, loss="ova", thread=32, verbose=0)
    os.remove(tmp)
    print(f"  trained in {time.time()-t0:.0f}s")
    out_path = f"{W}/models/{axis}_fasttext_v4.bin"; model.save_model(out_path); print(f"  saved -> {out_path}")

    old_model = fasttext.load_model(f"{W}/models/{axis}_fasttext_v2_AB.bin")
    for name, s in (("clean_val", clean_val), ("weak_val", weak_val), ("weak_candidates_val (NEW)", weak_candidates_val), ("target_test", target_test)):
        old_mi, old_ma, _ = macro_micro(old_model, s)
        new_mi, new_ma, new_rows = macro_micro(model, s)
        print(f"  {name:28s} n={len(s):5d} | v2_AB micro={old_mi:.4f} macro={old_ma:.4f}  ->  v4 micro={new_mi:.4f} macro={new_ma:.4f}  (delta macro {new_ma-old_ma:+.4f})")
        if "weak_candidates" in name or name == "target_test":
            zero = sum(1 for c, sup, f1 in new_rows if f1 == 0.0)
            print(f"      classes scoring ZERO F1 on {name}: {zero}/{len(new_rows)}")
            print(f"      worst 10: " + ", ".join(f"{c}(sup={sup},F1={f1:.2f})" for c, sup, f1 in new_rows[:10]))
