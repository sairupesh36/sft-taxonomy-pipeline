"""Risk-coverage curve: if we're willing to leave a document with NO label at all (abstain) unless
the model's own best guess for it clears some confidence bar, how much do micro-F1 AND macro-F1
improve on the documents we DO keep ("kept docs"), and what fraction of documents do we lose?
Real numbers, parametrized model, on 4 real eval sets. Uses predict_capped's own top-2-cap logic
(top-2 by score, membership threshold applied per label) so this matches production behavior.
Usage: python3 coverage_vs_macro_f1.py --domain-model PATH --task-model PATH [--label RUN_NAME]
"""
import sys, os, json, random, argparse, collections
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
W = "/projects/data/datasets/code_data/sai_rupesh/taxonomy/lc_work"
CE = "/projects/data/datasets/code_data/sai_rupesh/taxonomy/classifier_experiments"

def load(path, axis):
    out = []
    for line in open(path):
        d = json.loads(line)
        if d.get("ok") and d["result"].get(axis): out.append((d["result"][axis], B.clean_text(d["full_text"])))
    return out

GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

def curve(model, examples, thresh_for_membership=0.5, max_labels=2):
    scored = []
    for labels, text in examples:
        labs, scores = model.predict(text, k=-1, threshold=0.0)
        order = sorted(zip(labs, scores), key=lambda x: -x[1])[:max_labels]  # fastText already sorts desc, but be explicit
        sd = {l.replace("__label__", ""): float(s) for l, s in order}
        top = max(sd.values()) if sd else 0.0
        scored.append((set(labels), sd, top))
    rows = []
    for abstain_th in GRID:
        kept = [(t, sd) for t, sd, top in scored if top >= abstain_th]
        coverage = len(kept) / len(scored)
        per_class = collections.defaultdict(lambda: [0, 0, 0]); tp = fp = fn = 0
        for true, sd in kept:
            pred = {l for l, s in sd.items() if s >= thresh_for_membership}
            tp += len(pred & true); fp += len(pred - true); fn += len(true - pred)
            for c in true | pred:
                e = per_class[c]
                if c in true and c in pred: e[0] += 1
                elif c in pred: e[1] += 1
                elif c in true: e[2] += 1
        f = lambda a, b, c: 2 * a / max(1, 2 * a + b + c)
        macro = sum(f(a, b, c) for a, b, c in per_class.values()) / max(1, len(per_class))
        micro = f(tp, fp, fn)
        rows.append((abstain_th, coverage, micro, macro, len(kept), len(per_class)))
    return rows

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain-model", required=True); ap.add_argument("--task-model", required=True)
    ap.add_argument("--label", default="run")
    a = ap.parse_args()
    models = {"domain": fasttext.load_model(a.domain_model), "task_family": fasttext.load_model(a.task_model)}
    results = {}
    for axis in ("domain", "task_family"):
        model = models[axis]
        ex = load(f"{CE}/sft_output_sample_combined_98436_clean.jsonl", axis); random.Random(42).shuffle(ex)
        clean_val = ex[:int(len(ex) * 0.15)]
        wc = load(f"{W}/labeled_weak_candidates.jsonl", axis); random.Random(42).shuffle(wc)
        weak_candidates_val = wc[:int(len(wc) * 0.15)]
        test = load(f"{W}/labeled_T.jsonl", axis)
        print(f"\n########## {axis}  ({a.label})")
        results[axis] = {}
        for name, s in (("target_test (real, unbiased)", test), ("clean_val (bigger sample)", clean_val),
                         ("weak_candidates_val (weak-class focus)", weak_candidates_val)):
            print(f"  -- {name}, n={len(s)}")
            results[axis][name] = []
            for th, cov, micro, macro, n_kept, n_cls in curve(model, s):
                print(f"     abstain-below={th:<5} coverage={cov*100:5.1f}% (kept {n_kept:>5}/{len(s)})  "
                      f"KEPT-DOCS micro_F1={micro:.4f}  macro_F1={macro:.4f}  classes_still_scored={n_cls}")
                results[axis][name].append(dict(abstain_th=th, coverage=cov, kept_micro_f1=micro, kept_macro_f1=macro, n_kept=n_kept, n_classes=n_cls))
    json.dump(results, open(f"{W}/coverage_result_{a.label}.json", "w"), indent=1)
    print(f"\nsaved -> lc_work/coverage_result_{a.label}.json")
