import sys, json, random, re, collections
sys.path.insert(0, "/projects/data/datasets/code_data/sai_rupesh/taxonomy/pylibs")
sys.path.insert(0, "/projects/data/datasets/code_data/sai_rupesh/taxonomy/classifier_experiments")
import fasttext_baseline as B
from fasttext_predict_utils import predict_with_fallback
import fasttext
CE = "/projects/data/datasets/code_data/sai_rupesh/taxonomy/classifier_experiments"
rows = []
with open(f"{CE}/sft_output_sample_combined_98436_clean.jsonl") as f:
    for line in f:
        d = json.loads(line)
        if d.get("ok"): rows.append(d)
report = {}
for axis in ("domain", "task_family"):
    ex = [(d["result"][axis], B.clean_text(d["full_text"]), d["orig_source"], d["orig_domain"], d["full_text"]) for d in rows if d["result"].get(axis)]
    random.Random(42).shuffle(ex); val = ex[: int(len(ex) * 0.15)]
    model = fasttext.load_model(f"{CE}/{axis}_fasttext.bin")
    tp = collections.Counter(); fp = collections.Counter(); fn = collections.Counter(); conf = collections.Counter()
    src = collections.defaultdict(lambda: [0, 0]); dom = collections.defaultdict(lambda: [0, 0]); errs = []
    for labels, text, source, odom, raw in val:
        pred = predict_with_fallback(model, text, 0.5); true = set(labels)
        for l in true & pred: tp[l] += 1
        for l in pred - true: fp[l] += 1
        for l in true - pred: fn[l] += 1
        if len(true) == 1 and len(pred) == 1 and true != pred: conf[(next(iter(true)), next(iter(pred)))] += 1
        okd = pred == true; src[source][0] += 1; src[source][1] += (not okd); dom[odom][0] += 1; dom[odom][1] += (not okd)
        if not okd: errs.append((source, sorted(true), sorted(pred), re.sub(r"\s+", " ", raw)[:160]))
    classes = sorted(set(tp) | set(fp) | set(fn))
    f1 = lambda c: 2 * tp[c] / max(1, 2 * tp[c] + fp[c] + fn[c])
    print(f"\n================ {axis}  (val {len(val)}) ================")
    weak = sorted([c for c in classes if tp[c] + fn[c] >= 40], key=f1)[:12]
    print("Weakest classes with >=40 val examples (class: F1, support, recall, precision):")
    for c in weak:
        sup = tp[c] + fn[c]; print(f"  {c:38s} F1 {f1(c):.2f}  n={sup:4d}  rec {tp[c]/max(1,sup):.2f}  prec {tp[c]/max(1,tp[c]+fp[c]):.2f}")
    print(f"Classes with <40 val examples: {sum(1 for c in classes if tp[c]+fn[c] < 40)} of {len(classes)}; they hold {sum(tp[c]+fn[c] for c in classes if tp[c]+fn[c] < 40)} of {sum(tp[c]+fn[c] for c in classes)} true labels")
    print("Top confusions (true -> predicted, count):")
    for (t, p), n in conf.most_common(10): print(f"  {t:34s} -> {p:34s} {n}")
    print("Worst source datasets by wrong-rate (n>=60):")
    for s, (n, w) in sorted([(s, v) for s, v in src.items() if v[0] >= 60], key=lambda kv: -kv[1][1] / kv[1][0])[:10]:
        print(f"  {s[:62]:62s} n={n:4d}  wrong {100*w/n:.0f}%")
    print("Wrong-rate by source folder:", {d: f"{100*w/n:.0f}% of {n}" for d, (n, w) in sorted(dom.items(), key=lambda kv: -kv[1][1] / kv[1][0]) if n >= 100})
