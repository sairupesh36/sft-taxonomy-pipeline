"""
Optuna hyperparameter search for the FastText taxonomy classifiers, one
study per axis (domain, task_family). Reuses the EXACT train/val split files
fasttext_baseline.py already wrote (same seed=42), so results are directly
comparable to the baseline numbers in fasttext_baseline_results.json.

Tunes both the fasttext training hyperparameters AND the prediction
threshold (fasttext's own default of 0.5 is just a guess -- the right value
depends on the trained model, so it must be tuned jointly with the training
params, not fixed).

loss is kept fixed at "ova" (one-vs-all) because this is a multi-label
problem (a doc can have 2 labels) -- "softmax" forces mutually-exclusive
single-label predictions and would be the wrong tool here, not a valid
comparison point.
"""
import json
import time
import numpy as np
import fasttext
import optuna

from fasttext_predict_utils import predict_with_fallback

# The installed fasttext package calls np.array(probs, copy=False) inside
# predict(), which numpy>=2.0 now raises on instead of silently copying (a
# real upstream fasttext/numpy version mismatch, not our bug). Patch
# np.array in this process only so copy=False falls back to "copy if
# needed" (numpy's old behavior), since fasttext can't be fixed upstream.
_orig_np_array = np.array
def _np_array_compat(*args, **kwargs):
    if kwargs.get("copy") is False:
        kwargs["copy"] = None
    return _orig_np_array(*args, **kwargs)
np.array = _np_array_compat

WORK_DIR = "/projects/data/datasets/code_data/sai_rupesh/taxonomy/classifier_experiments"
AXES = ["domain", "task_family"]
N_TRIALS = 5

optuna.logging.set_verbosity(optuna.logging.WARNING)
fasttext.FastText.eprint = lambda *a, **k: None  # silence fasttext's own per-call stderr spam during the search


def load_val(axis):
    val = []
    with open(f"{WORK_DIR}/{axis}_val.txt") as f:
        for line in f:
            parts = line.strip().split(" ")
            labels = [p.replace("__label__", "") for p in parts if p.startswith("__label__")]
            text = " ".join(p for p in parts if not p.startswith("__label__"))
            val.append((labels, text))
    return val


def score(model, val, threshold):
    tp = fp = fn = 0
    exact = 0
    for labels, text in val:
        pred_set = predict_with_fallback(model, text, threshold)
        true_set = set(labels)
        tp += len(pred_set & true_set)
        fp += len(pred_set - true_set)
        fn += len(true_set - pred_set)
        if pred_set == true_set:
            exact += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1, exact / len(val)


def make_objective(axis, val):
    train_path = f"{WORK_DIR}/{axis}_train.txt"

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
            loss="ova",
            thread=8,
            verbose=0,
        )
        if params["maxn"] < params["minn"]:
            params["maxn"] = params["minn"]

        model = fasttext.train_supervised(**params)

        threshold = trial.suggest_float("threshold", 0.1, 0.7)
        _, _, f1, _ = score(model, val, threshold)

        trial.set_user_attr("model_path", None)  # filled in only for the best trial, after the study
        return f1

    return objective


def main():
    all_results = {}
    for axis in AXES:
        print(f"\n=== tuning axis: {axis} ({N_TRIALS} trials) ===", flush=True)
        val = load_val(axis)
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))

        t_last = time.time()
        def log_trial(study, trial, _t_last=[t_last]):
            now = time.time()
            print(f"  trial {trial.number}: f1={trial.value:.4f}  ({now - _t_last[0]:.1f}s)", flush=True)
            _t_last[0] = now

        study.optimize(make_objective(axis, val), n_trials=N_TRIALS, show_progress_bar=False, callbacks=[log_trial])

        best = study.best_trial
        print(f"best F1 for {axis}: {best.value:.4f}")
        print(f"best params: {best.params}")

        # Retrain once cleanly with the winning params so we have a saved model + final metrics.
        best_params = {k: v for k, v in best.params.items() if k != "threshold"}
        best_params.update(dict(input=f"{WORK_DIR}/{axis}_train.txt", loss="ova", thread=8, verbose=0))
        if best_params["maxn"] < best_params["minn"]:
            best_params["maxn"] = best_params["minn"]
        final_model = fasttext.train_supervised(**best_params)
        precision, recall, f1, exact_match = score(final_model, val, best.params["threshold"])

        model_path = f"{WORK_DIR}/{axis}_fasttext_optuna_best.bin"
        final_model.save_model(model_path)

        result = {
            "axis": axis,
            "n_trials": N_TRIALS,
            "best_params": best.params,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "exact_match_rate": exact_match,
            "val_n": len(val),
            "model_path": model_path,
        }
        all_results[axis] = result
        print(f"confirmed on clean retrain: precision={precision:.3f} recall={recall:.3f} f1={f1:.3f} exact={exact_match:.3f}", flush=True)

        with open(f"{WORK_DIR}/fasttext_optuna_{axis}_trials.json", "w") as f:
            json.dump([{"number": t.number, "value": t.value, "params": t.params} for t in study.trials], f, indent=2)

    with open(f"{WORK_DIR}/fasttext_optuna_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n=== summary: baseline vs optuna-tuned ===")
    with open(f"{WORK_DIR}/fasttext_baseline_results.json") as f:
        baseline = {r["axis"]: r for r in json.load(f)}
    for axis in AXES:
        b = baseline[axis]["f1"]
        o = all_results[axis]["f1"]
        print(f"{axis:15s} baseline F1={b:.3f}  ->  optuna F1={o:.3f}  (delta={o-b:+.3f})")


if __name__ == "__main__":
    main()
