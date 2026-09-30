"""
Retries the 184 rows from relabel_severe_98436.py that got recovered raw
text but failed the Gemma API call (transient network/timeout, same
pattern retry_failed_80962.py already found: ~94% of failures like this
succeed on a second try). Reads the already-saved new_full_text straight
out of sft_output_sample_combined_98436_fixed.jsonl for the still-failed
uids (no need to re-recover raw messages), re-calls map_document, and
patches the fixed file in place with whatever succeeds.
"""
import sys
import json
import time
import asyncio
import aiohttp

sys.path.insert(0, "/projects/data/datasets/code_data/sai_rupesh/taxonomy")
from taxonomy_map_document import map_document

FIXED_FILE = "sft_output_sample_combined_98436_fixed.jsonl"
REPORT_FILE = "relabel_severe_98436_report.json"
CONCURRENCY = 96


async def worker(queue, sem, session, results, counter, lock):
    while True:
        try:
            uid, text = queue.get_nowait()
        except asyncio.QueueEmpty:
            return
        result = await map_document(session, sem, text)
        async with lock:
            results[uid] = result
            counter[0] += 1
            if result is not None:
                counter[1] += 1
        queue.task_done()


async def main():
    with open(REPORT_FILE) as f:
        report = json.load(f)
    failed_uids = set(report["detail"]["recovered_but_api_failed"])
    print(f"retrying {len(failed_uids)} previously API-failed rows", flush=True)

    rows = []
    with open(FIXED_FILE) as f:
        for line in f:
            d = json.loads(line)
            rows.append(d)

    to_retry = [(d["uid"], d["full_text"]) for d in rows if d["uid"] in failed_uids]
    print(f"found {len(to_retry)} matching rows with saved text", flush=True)

    queue = asyncio.Queue()
    for item in to_retry:
        queue.put_nowait(item)

    results = {}
    sem = asyncio.Semaphore(CONCURRENCY)
    counter = [0, 0]
    lock = asyncio.Lock()
    t0 = time.time()
    async with aiohttp.ClientSession() as session:
        workers = [asyncio.create_task(worker(queue, sem, session, results, counter, lock))
                   for _ in range(CONCURRENCY)]
        await asyncio.gather(*workers)
    print(f"retry done in {time.time()-t0:.1f}s: {counter[1]}/{counter[0]} succeeded", flush=True)

    # Patch the fixed file in place with any newly-successful results.
    n_patched = 0
    for d in rows:
        uid = d["uid"]
        if uid in results and results[uid] is not None:
            d["ok"] = True
            d["result"] = results[uid]
            n_patched += 1

    with open(FIXED_FILE, "w", encoding="utf-8") as f:
        for d in rows:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    still_failed = len(failed_uids) - n_patched
    print(f"DONE: patched {n_patched} rows to ok=True in {FIXED_FILE}; {still_failed} still failing after retry (left excluded)", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
