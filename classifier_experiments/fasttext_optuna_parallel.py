"""
Same search as fasttext_optuna.py, but runs many trials AT ONCE instead of
one after another, to actually use the machine's spare CPU capacity (112
threads available, each single trial only used 8 -- so the sequential
version left ~90% of the machine idle the whole time).

Design: N_WORKERS separate OS processes, each running its own independent
Optuna study (in-memory, no shared database -- avoids SQLite lock
contention between many concurrent writers, which is the real risk with
a shared-storage approach under time pressure). Each worker explores
its own share of the trial budget with a different random seed; the best
trial across ALL workers, pooled together, is the final answer. This
loses a small amount of the "smart search" benefit (workers can't learn
from each other's trials mid-search), which is an acceptable tradeoff for
a several-times wall-clock speedup on a modest trial budget.

Reuses load_val/score/make_objective from fasttext_optuna.py directly, so
the actual training/eval logic (and the numpy/fasttext compatibility patch
applied at import time) is identical to the sequential version -- only the
scheduling changes.
"""
import json
import os
import time
from multiprocessing import Process

import fasttext
import optuna

from fasttext_optuna import load_val, score, make_objective, WORK_DIR, AXES

N_TRIALS_PER_AXIS = 50
N_WORKERS = 10  # 10 workers x thread=8 per trial = 80 of 112 threads, leaving headroom

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _worker(axis, n_trials, seed, out_path):
    val = load_val(axis)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(make_objective(axis, val), n_trials=n_trials, show_progress_bar=False)
    trials = [{"number": t.number, "value": t.value, "params": t.params} for t in study.trials]
    with open(out_path, "w") as f:
        json.dump(trials, f)


def run_axis(axis):
    print(f"\n=== parallel tuning axis: {axis}  ({N_TRIALS_PER_AXIS} trials across {N_WORKERS} workers) ===", flush=True)
    per_worker = [N_TRIALS_PER_AXIS // N_WORKERS + (1 if i < N_TRIALS_PER_AXIS % N_WORKERS else 0) for i in range(N_WORKERS)]
    out_paths = [f"{WORK_DIR}/.optuna_worker_{axis}_{i}.json" for i in range(N_WORKERS)]

    t0 = time.time()
    procs = []
    for i, n in enumerate(per_worker):
        if n == 0:
            continue
        p = Process(target=_worker, args=(axis, n, 1000 + i, out_paths[i]))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    dt = time.time() - t0
    print(f"{axis}: all {len(procs)} workers done in {dt:.1f}s", flush=True)

    all_trials = []
    for path in out_paths:
        if os.path.exists(path):
            with open(path) as f:
                all_trials.extend(json.load(f))
            os.remove(path)

    best = max(all_trials, key=lambda t: t["value"])
    print(f"best F1 for {axis}: {best['value']:.4f}")
    print(f"best params: {best['params']}")

    val = load_val(axis)
    best_params = {k: v for k, v in best["params"].items() if k != "threshold"}
    best_params.update(dict(input=f"{WORK_DIR}/{axis}_train.txt", loss="ova", thread=8, verbose=0))
    if best_params["maxn"] < best_params["minn"]:
        best_params["maxn"] = best_params["minn"]
    final_model = fasttext.train_supervised(**best_params)
    precision, recall, f1, exact_match = score(final_model, val, best["params"]["threshold"])

    model_path = f"{WORK_DIR}/{axis}_fasttext_optuna_best.bin"
    final_model.save_model(model_path)
    print(f"confirmed on clean retrain: precision={precision:.3f} recall={recall:.3f} f1={f1:.3f} exact={exact_match:.3f}", flush=True)

    with open(f"{WORK_DIR}/fasttext_optuna_{axis}_trials.json", "w") as f:
        json.dump(all_trials, f, indent=2)

    return {
        "axis": axis,
        "n_trials": len(all_trials),
        "n_workers": N_WORKERS,
        "search_wall_time_sec": dt,
        "best_params": best["params"],
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_match_rate": exact_match,
        "val_n": len(val),
        "model_path": model_path,
    }


def main():
    all_results = {}
    for axis in AXES:
        all_results[axis] = run_axis(axis)

    with open(f"{WORK_DIR}/fasttext_optuna_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n=== summary: baseline vs optuna-tuned (parallel search) ===")
    with open(f"{WORK_DIR}/fasttext_baseline_results.json") as f:
        baseline = {r["axis"]: r for r in json.load(f)}
    for axis in AXES:
        b = baseline[axis]["f1"]
        o = all_results[axis]["f1"]
        print(f"{axis:15s} baseline F1={b:.3f}  ->  optuna F1={o:.3f}  (delta={o-b:+.3f})")


if __name__ == "__main__":
    main()
