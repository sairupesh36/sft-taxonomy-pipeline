"""Rebuild domain/task_family with rare-class oversampling, aimed at macro-F1 (not micro-F1).
Same 4 data sources + same fastText hyperparams as the current v2_AB model (arms.py config "AB"),
the ONLY change is: examples touching an under-represented label get duplicated in the training
file so every class has at least a floor count of occurrences. This does not add any new labeled
data -- it just stops the trainer from being dominated by mathematics/calculation. Classes with
only a handful of REAL examples (see the training-count audit) can't be fixed by this alone;
duplication there just means the model can memorize that one example's wording, not generalize --
flagged separately in the printed report, not hidden."""
import sys, os, json, random, collections, math
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
from fasttext_predict_utils import predict_with_fallback

CE = "/projects/data/datasets/code_data/sai_rupesh/taxonomy/classifier_experiments"
W = "/projects/data/datasets/code_data/sai_rupesh/taxonomy/lc_work"
FLOOR = 400        # target minimum effective occurrences per class after oversampling
MAX_DUP = 25       # cap on how many times any single example gets duplicated (avoid pure memorization of one line dominating)

def load(path, axis):
    out = []
    for line in open(path):
        d = json.loads(line)
        if d.get("ok") and d["result"].get(axis):
            out.append((d["result"][axis], B.clean_text(d["full_text"])))
    return out

def macro_and_micro(model, examples, threshold=0.5):
    per_class = collections.defaultdict(lambda: [0, 0, 0]); tp = fp = fn = 0
    for labels, text in examples:
        pred = predict_with_fallback(model, text, threshold); true = set(labels)
        tp += len(pred & true); fp += len(pred - true); fn += len(true - pred)
        for c in true | pred:
            e = per_class[c]
            if c in true and c in pred: e[0] += 1
            elif c in pred: e[1] += 1
            elif c in true: e[2] += 1
    f = lambda a, b, c: 2 * a / max(1, 2 * a + b + c)
    rows = sorted(((c, a + cc, f(a, b, cc)) for c, (a, b, cc) in per_class.items()), key=lambda r: r[2])
    return f(tp, fp, fn), (sum(r[2] for r in rows) / len(rows) if rows else 0.0), rows

for axis in ("domain", "task_family"):
    print(f"\n########## {axis}")
    ex = load(f"{CE}/sft_output_sample_combined_98436_clean.jsonl", axis); random.Random(42).shuffle(ex)
    nv = int(len(ex) * 0.15); clean_val, train = ex[:nv], ex[nv:]
    weak = load(f"{CE}/sft_output_sample_weak_categories_v2_mapped.jsonl", axis); random.Random(42).shuffle(weak)
    nw = int(len(weak) * 0.15); weak_val, weak_train = weak[:nw], weak[nw:]
    tr = train + weak_train + load(f"{W}/labeled_A.jsonl", axis) + load(f"{W}/labeled_B.jsonl", axis)
    target_test = load(f"{W}/labeled_T.jsonl", axis)

    counts = collections.Counter(l for labels, _ in tr for l in labels)
    dup = []
    for labels, text in tr:
        d_ = min(MAX_DUP, max(1, math.ceil(FLOOR / min(counts[l] for l in labels))))
        dup += [(labels, text)] * d_
    random.Random(7).shuffle(dup)
    print(f"  base train n={len(tr):,} -> oversampled n={len(dup):,}  (rarest class before: {counts.most_common()[-1]})")

    tmp = f"{W}/tmp_{axis}_balanced.txt"
    with open(tmp, "w") as f:
        for labels, text in dup: f.write(B.make_fasttext_line(labels, text))
    model = fasttext.train_supervised(input=tmp, epoch=25, lr=0.5, wordNgrams=2, dim=100, loss="ova", thread=32, verbose=0)
    os.remove(tmp)
    out_path = f"{W}/models/{axis}_fasttext_v3_balanced.bin"
    model.save_model(out_path); print(f"  saved -> {out_path}")

    old_model = fasttext.load_model(f"{W}/models/{axis}_fasttext_v2_AB.bin")
    for name, s in (("clean_val", clean_val), ("weak_val", weak_val), ("target_test", target_test)):
        old_mi, old_ma, _ = macro_and_micro(old_model, s)
        new_mi, new_ma, new_rows = macro_and_micro(model, s)
        print(f"  {name:14s} n={len(s):5d} | v2_AB micro={old_mi:.4f} macro={old_ma:.4f}  ->  v3_balanced micro={new_mi:.4f} macro={new_ma:.4f}")
        if name == "target_test":
            zero_before = sum(1 for c, sup, f1 in macro_and_micro(old_model, s)[2] if f1 == 0.0)
            zero_after = sum(1 for c, sup, f1 in new_rows if f1 == 0.0)
            print(f"    classes scoring ZERO F1 on target_test: {zero_before} -> {zero_after}")
            print("    still-worst classes after balancing:")
            for c, sup, f1 in new_rows[:10]: print(f"      {c:42s} support={sup:>4d}  F1={f1:.3f}  (train_count={counts.get(c,0)})")
