"""
Measures the REAL inference speed of the trained FastText classifiers --
not Gemma's labeling speed (already measured extensively elsewhere in this
project), but how fast the trained .bin model itself predicts labels on new
text, on this machine's CPU. Direct measurement, not an assumption.

Two numbers reported per axis:
  1. Single-process throughput (docs/sec), predicting one real document at a
     time in a plain loop -- the true per-core rate.
  2. Multi-process throughput using this machine's actual available cores
     (each worker process loads its own copy of the model -- fasttext's
     predict() holds the GIL during the C call, so real parallelism needs
     separate processes, not threads).

Then does the 1.2B-document math at both rates so the actual "is this fast
enough" question has a real answer in hours/days, not just docs/sec.
"""
import time
import json
import multiprocessing as mp

import numpy as np
_orig_np_array = np.array
def _np_array_compat(*args, **kwargs):
    if kwargs.get("copy") is False:
        kwargs["copy"] = None
    return _orig_np_array(*args, **kwargs)
np.array = _np_array_compat
import fasttext

WORK_DIR = "/projects/data/datasets/code_data/sai_rupesh/taxonomy/classifier_experiments"
AXES = ["domain", "task_family"]
N_DOCS = 60000         # real documents used for the timed benchmark -- bumped
                       # from 5000 after the first two runs came back noisy
                       # and non-monotonic (e.g. 64 processes looking slower
                       # than 8) even after fixing the model-load-timing bug;
                       # 5000 docs split across 128 workers is only ~39
                       # docs/worker, too small a per-worker workload for a
                       # stable read against real OS scheduling jitter
TOTAL_CORPUS = 1_200_000_000


def load_real_docs(n):
    """Real document texts (not synthetic) straight from the actual training
    file's full_text field, exactly as the classifier would see production
    text -- same clean_text() normalization fasttext_baseline.py applies."""
    import re

    def clean_text(text):
        text = text.replace("\n", " ").replace("\t", " ")
        text = re.sub(r"\s+", " ", text).strip()
        return text.replace("__label__", "")

    docs = []
    with open(f"{WORK_DIR}/sft_output_sample_combined_98436_clean.jsonl") as f:
        for line in f:
            d = json.loads(line)
            if not d.get("ok"):
                continue
            docs.append(clean_text(d["full_text"]))
            if len(docs) >= n:
                break
    return docs


def bench_single_process(model_path, docs):
    model = fasttext.load_model(model_path)
    # warm up (first call pays one-time setup cost)
    model.predict(docs[0], k=-1, threshold=0.5)
    t0 = time.time()
    for d in docs:
        model.predict(d, k=-1, threshold=0.5)
    dt = time.time() - t0
    return len(docs) / dt, dt


_WORKER_MODEL = None


def _mp_init(model_path):
    # Model is loaded ONCE per worker process, at pool-startup time, matching
    # how a real batch job would actually run (load once, then process
    # billions of docs) -- NOT reloaded per benchmark call. Excluding this
    # one-time ~1.3GB model-load cost from the timed section is what makes
    # this a fair docs/sec comparison against the single-process number
    # (a first version of this script loaded the model INSIDE the timed
    # region per worker, which -- with only a few thousand benchmark docs
    # split across many workers -- let one-time load cost dominate and made
    # more processes look SLOWER than one, an artifact of the benchmark, not
    # a real property of fasttext inference; fixed before trusting the
    # numbers).
    global _WORKER_MODEL
    _WORKER_MODEL = fasttext.load_model(model_path)


def _mp_predict_shard(docs_shard):
    model = _WORKER_MODEL
    t0 = time.time()
    for d in docs_shard:
        model.predict(d, k=-1, threshold=0.5)
    return len(docs_shard), time.time() - t0


def _mp_noop(_):
    return True


def bench_multi_process(model_path, docs, n_workers):
    shards = [docs[i::n_workers] for i in range(n_workers)]
    pool = mp.Pool(n_workers, initializer=_mp_init, initargs=(model_path,))
    try:
        # mp.Pool() spawns workers and returns immediately -- it does NOT
        # wait for each worker's initializer (the ~1.3GB model load) to
        # finish first. Without this warm-up round-trip, pool.map() below
        # would start racing against still-in-progress model loads in every
        # worker, and that load time would leak into the "timed" section --
        # which is exactly what made more processes look SLOWER than one in
        # an earlier version of this script (128 workers all loading a
        # 1.3GB file at once, still mid-load when the clock started).
        # A round-trip no-op task per worker blocks here until every
        # worker's initializer has actually completed.
        pool.map(_mp_noop, range(n_workers * 2))

        t0 = time.time()
        results = pool.map(_mp_predict_shard, shards)
        wall = time.time() - t0
    finally:
        pool.close()
        pool.join()
    total_docs = sum(r[0] for r in results)
    return total_docs / wall, wall


def format_duration(seconds):
    days = seconds / 86400
    if days > 365:
        return f"{seconds/86400/365:.1f} years ({days:.0f} days)"
    if days > 1:
        return f"{days:.1f} days"
    return f"{seconds/3600:.2f} hours"


def main():
    n_cores = mp.cpu_count()
    print(f"machine has {n_cores} CPU cores available\n")

    docs = load_real_docs(N_DOCS)
    print(f"benchmarking on {len(docs)} real documents (from sft_output_sample_combined_98436.jsonl)\n")

    report = {"n_cores": n_cores, "n_bench_docs": len(docs), "axes": {}}

    for axis in AXES:
        model_path = f"{WORK_DIR}/{axis}_fasttext.bin"
        print(f"=== axis: {axis} ({model_path}) ===")

        rate_1, dt_1 = bench_single_process(model_path, docs)
        print(f"  single-process: {rate_1:.1f} docs/sec  ({dt_1:.2f}s for {len(docs)} docs)")

        mp_results = {}
        for nw in [8, 16, 32, 64, n_cores]:
            if nw > n_cores:
                continue
            rate_mp, dt_mp = bench_multi_process(model_path, docs, nw)
            speedup = rate_mp / rate_1
            print(f"  {nw:3d} processes: {rate_mp:8.1f} docs/sec  (speedup {speedup:.1f}x over single-process)")
            mp_results[nw] = rate_mp

        best_rate = max(mp_results.values())
        best_nw = max(mp_results, key=mp_results.get)

        t_single = TOTAL_CORPUS / rate_1
        t_best = TOTAL_CORPUS / best_rate

        print(f"\n  --- 1.2B-document math for axis={axis} ---")
        print(f"  single-threaded: {format_duration(t_single)}")
        print(f"  {best_nw} processes (this machine, all cores): {format_duration(t_best)}")
        print()

        report["axes"][axis] = {
            "single_process_docs_per_sec": rate_1,
            "multi_process_docs_per_sec": mp_results,
            "best_docs_per_sec": best_rate,
            "best_n_workers": best_nw,
            "seconds_for_1.2B_single_process": t_single,
            "seconds_for_1.2B_best_multiprocess": t_best,
        }

    with open(f"{WORK_DIR}/fasttext_inference_speed_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {WORK_DIR}/fasttext_inference_speed_report.json")


if __name__ == "__main__":
    main()
