"""Optuna hyperparameter search re-aimed at MACRO-F1 (the existing fasttext_optuna.py only ever
optimized micro-F1 -- see CLAUDE.md's "Classifier distillation" section). Trains on the v4 combined
pool (98,436 + weak_categories + labeled_A/B + the new 21,406 weak-class batch, oversampled the
same way as retrain_v4.py), scores each trial by macro-F1 on a COMBINED validation set (clean_val +
weak_val + weak_candidates_val -- pooling all three gives much better per-class statistical power
than any one alone; target_test is deliberately NEVER used here, kept clean for final reporting
only, so the search can't overfit to the one truly unbiased sample). Also jointly tunes the
confidence threshold, same as the original script -- that part of its design was sound.
"""
import sys, os, json, random, math, collections, time
sys.path.insert(0, "/projects/data/datasets/code_data/sai_rupesh/taxonomy/pylibs")
sys.path.insert(0, "/projects/data/datasets/code_data/sai_rupesh/taxonomy/classifier_experiments")
import numpy as np
_orig = np.array
def _c(*a, **k):
    if k.get("copy") is False: k["copy"] = None
    return _orig(*a, **k)
np.array = _c
import fasttext, optuna
import fasttext_baseline as B
from fasttext_predict_utils import predict_capped

CE = "/projects/data/datasets/code_data/sai_rupesh/taxonomy/classifier_experiments"
W = "/projects/data/datasets/code_data/sai_rupesh/taxonomy/lc_work"
FLOOR = 400; MAX_DUP = 25
N_TRIALS = 20

def load(path, axis):
    out = []
    for line in open(path):
        d = json.loads(line)
        if d.get("ok") and d["result"].get(axis): out.append((d["result"][axis], B.clean_text(d["full_text"])))
    return out

def macro_f1(model, examples, threshold):
    per_class = collections.defaultdict(lambda: [0, 0, 0])
    for labels, text in examples:
        pred = set(predict_capped(model, text, threshold)); true = set(labels)
        for c in true | pred:
            e = per_class[c]
            if c in true and c in pred: e[0] += 1
            elif c in pred: e[1] += 1
            elif c in true: e[2] += 1
    f = lambda a, b, c: 2 * a / max(1, 2 * a + b + c)
    rows = [f(a, b, c) for a, b, c in per_class.values()]
    return sum(rows) / len(rows) if rows else 0.0

for axis in ("domain", "task_family"):
    print(f"\n########## {axis}", flush=True)
    ex = load(f"{CE}/sft_output_sample_combined_98436_clean.jsonl", axis); random.Random(42).shuffle(ex)
    nv = int(len(ex) * 0.15); clean_val, train = ex[:nv], ex[nv:]
    weak = load(f"{CE}/sft_output_sample_weak_categories_v2_mapped.jsonl", axis); random.Random(42).shuffle(weak)
    nw = int(len(weak) * 0.15); weak_val, weak_train = weak[:nw], weak[nw:]
    wc = load(f"{W}/labeled_weak_candidates.jsonl", axis); random.Random(42).shuffle(wc)
    nc = int(len(wc) * 0.15); weak_candidates_val, weak_candidates_train = wc[:nc], wc[nc:]
    A = load(f"{W}/labeled_A.jsonl", axis); Bx = load(f"{W}/labeled_B.jsonl", axis)
    target_test = load(f"{W}/labeled_T.jsonl", axis)                      # held out, reporting only

    combined_val = clean_val + weak_val + weak_candidates_val
    tr = train + weak_train + weak_candidates_train + A + Bx
    counts = collections.Counter(l for labels, _ in tr for l in labels)
    dup = []
    for labels, text in tr:
        d_ = min(MAX_DUP, max(1, math.ceil(FLOOR / min(counts[l] for l in labels))))
        dup += [(labels, text)] * d_
    random.Random(7).shuffle(dup)
    train_path = f"{W}/optuna_{axis}_train.txt"
    with open(train_path, "w") as f:
        for labels, text in dup: f.write(B.make_fasttext_line(labels, text))
    print(f"  train (oversampled) = {len(dup):,} rows | combined_val (macro-F1 objective) = {len(combined_val):,} | target_test (report only) = {len(target_test)}", flush=True)

    def objective(trial):
        params = dict(
            input=train_path,
            lr=trial.suggest_float("lr", 0.05, 1.5, log=True),
            epoch=trial.suggest_int("epoch", 10, 60),
            wordNgrams=trial.suggest_int("wordNgrams", 1, 3),
            dim=trial.suggest_int("dim", 50, 300),
            minCount=trial.suggest_int("minCount", 1, 5),
            minn=trial.suggest_int("minn", 0, 4),
            maxn=trial.suggest_int("maxn", 0, 6),
            bucket=trial.suggest_int("bucket", 200_000, 2_000_000, log=True),
            loss="ova", thread=32, verbose=0,
        )
        if params["maxn"] < params["minn"]: params["maxn"] = params["minn"]
        model = fasttext.train_supervised(**params)
        threshold = trial.suggest_float("threshold", 0.1, 0.9)
        return macro_f1(model, combined_val, threshold)

    t0 = [time.time()]
    def log_trial(study, trial):
        now = time.time(); print(f"  trial {trial.number}: macro_F1={trial.value:.4f}  ({now - t0[0]:.1f}s)", flush=True); t0[0] = now
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False, callbacks=[log_trial])

    best = study.best_trial
    print(f"  BEST trial {best.number}: macro_F1={best.value:.4f}  params={best.params}", flush=True)
    best_params = {k: v for k, v in best.params.items() if k != "threshold"}
    best_params.update(dict(input=train_path, loss="ova", thread=32, verbose=0))
    if best_params["maxn"] < best_params["minn"]: best_params["maxn"] = best_params["minn"]
    final_model = fasttext.train_supervised(**best_params)
    os.remove(train_path)
    model_path = f"{W}/models/{axis}_fasttext_optuna_macro.bin"
    final_model.save_model(model_path)

    th = best.params["threshold"]
    for name, s in (("combined_val (tuned on this)", combined_val), ("target_test (held out, real check)", target_test)):
        mf1 = macro_f1(final_model, s, th)
        print(f"  final model @ threshold={th:.2f} on {name}: macro_F1={mf1:.4f}")
    old_model = fasttext.load_model(f"{W}/models/{axis}_fasttext_v4.bin")
    old_tt = macro_f1(old_model, target_test, 0.8)
    new_tt = macro_f1(final_model, target_test, th)
    print(f"  target_test macro_F1: v4(threshold=0.8)={old_tt:.4f}  ->  optuna_macro(threshold={th:.2f})={new_tt:.4f}  (delta {new_tt-old_tt:+.4f})")
    json.dump({"axis": axis, "n_trials": N_TRIALS, "best_params": best.params, "combined_val_macro_f1": best.value,
               "target_test_macro_f1": new_tt, "target_test_macro_f1_v4_baseline": old_tt}, open(f"{W}/optuna_macro_{axis}_result.json", "w"), indent=1)
