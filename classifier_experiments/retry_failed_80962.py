"""
Retries the ok=false rows from sft_output_sample_combined_80962.jsonl.
These already have full_text saved from the first attempt (Gemma failed on
the API call, not on sampling), so this re-calls map_document directly on
the stored text -- no re-sampling from OUTPUT needed. Most large-batch API
failures are transient (timeout/network), so most of these should succeed
on retry.
"""
import sys
import json
import time
import asyncio
import aiohttp

sys.path.insert(0, "/projects/data/datasets/code_data/sai_rupesh/taxonomy")
from taxonomy_map_document import map_document

IN_FILE = "/projects/data/datasets/code_data/sai_rupesh/taxonomy/classifier_experiments/sft_output_sample_combined_80962.jsonl"
OUT_FILE = "/projects/data/datasets/code_data/sai_rupesh/taxonomy/classifier_experiments/retried_failed_80962_mapped.jsonl"
CONCURRENCY = 192


async def worker(worker_id, queue, sem, session, out_f, counter, lock):
    while True:
        try:
            row = queue.get_nowait()
        except asyncio.QueueEmpty:
            return
        text = row["full_text"]
        result = None
        if text and text.strip():
            result = await map_document(session, sem, text)
        rec = dict(row)
        rec["ok"] = result is not None
        if result is not None:
            rec["result"] = result
        async with lock:
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_f.flush()
            counter[0] += 1
            if result is not None:
                counter[2] += 1
            n = counter[0]
        if n <= 10 or n % 100 == 0:
            el = time.time() - counter[1]
            print(f"  [{n}] recovered so far={counter[2]}  {el/60:.1f} min elapsed, rate {n/el:.2f} docs/sec", flush=True)
        queue.task_done()


async def main():
    rows = []
    with open(IN_FILE, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if not d.get("ok"):
                rows.append(d)
    print(f"loaded {len(rows)} previously-failed rows to retry", flush=True)

    queue = asyncio.Queue()
    for r in rows:
        queue.put_nowait(r)

    sem = asyncio.Semaphore(CONCURRENCY)
    counter = [0, time.time(), 0]
    lock = asyncio.Lock()
    with open(OUT_FILE, "w", encoding="utf-8") as out_f:
        async with aiohttp.ClientSession() as session:
            workers = [asyncio.create_task(worker(i, queue, sem, session, out_f, counter, lock))
                       for i in range(CONCURRENCY)]
            print(f"spawned {len(workers)} workers", flush=True)
            await asyncio.gather(*workers)

    dt = time.time() - counter[1]
    print(f"DONE: {counter[0]} retried, {counter[2]} recovered ({counter[2]/counter[0]*100:.1f}%) in {dt/60:.1f} min -> {OUT_FILE}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
