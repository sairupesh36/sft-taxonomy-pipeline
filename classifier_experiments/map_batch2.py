"""
Runs the real Gemma per-document mapper (taxonomy_map_document.map_document,
imported directly, not reimplemented) over the new rows pulled from
/projects/data/datasets/translation_data/SFT/OUTPUT by sample_from_sft_output.py.

Output schema matches sft_output_sample_mapped.jsonl exactly, so the two
files can be concatenated directly to build the classifier's expanded
training set.

Accepts an optional row-limit argument for a quick test run before
committing to the full batch.
"""
import sys
import time
import json
import asyncio
import aiohttp

sys.path.insert(0, "/projects/data/datasets/code_data/sai_rupesh/taxonomy")
from taxonomy_map_document import map_document

IN_FILE = "/projects/data/datasets/code_data/sai_rupesh/taxonomy/classifier_experiments/sft_output_sample_weak_categories_v2.jsonl"
OUT_FILE = "/projects/data/datasets/code_data/sai_rupesh/taxonomy/classifier_experiments/sft_output_sample_weak_categories_v2_mapped.jsonl"
MAX_CHARS = 3500
CONCURRENCY = 192  # bumped from 32 -- real-measured safe ceiling for a single process
                    # (192-320 plateaus around ~11-12 docs/sec; 8 processes at this level
                    # crashed the gemma-4-31b backend, so staying single-process here)


def to_full_text(messages, max_chars=MAX_CHARS):
    # Real conversation content ([USER]/[ASSISTANT]/[TOOL]) is always kept in
    # full first; the system prompt is truncated or dropped to make room,
    # not the other way around. The old raw-order version could burn the
    # whole budget on a long system prompt (some are 3900+ chars alone) and
    # never reach the actual user question -- confirmed on real data: 1.34%
    # of docs (1,778/132,610) had zero [USER]/[ASSISTANT] content visible to
    # Gemma under the old logic. Verified via verify_truncation_fix.py that
    # this version brings that to 0/132,610.
    system_parts = [f"[{m['role'].upper()}]: {m['content']}" for m in messages if m["role"] == "system"]
    other_parts = [f"[{m['role'].upper()}]: {m['content']}" for m in messages if m["role"] != "system"]

    other_text = "\n".join(other_parts)
    system_text = "\n".join(system_parts)

    if not other_text:
        # no real conversation content at all (rare) -- fall back to system text
        if len(system_text) > max_chars:
            return system_text[:max_chars] + " ...[truncated]"
        return system_text

    if len(other_text) > max_chars:
        # conversation itself doesn't fit even with zero system prompt
        return other_text[:max_chars] + " ...[truncated]"

    remaining = max_chars - len(other_text) - 1  # -1 for the joining newline
    if system_text and remaining > 0:
        if len(system_text) > remaining:
            system_text = system_text[:remaining] + " ...[truncated]"
        return system_text + "\n" + other_text
    return other_text


async def worker(worker_id, queue, sem, session, out_f, counter, lock):
    while True:
        try:
            row = queue.get_nowait()
        except asyncio.QueueEmpty:
            return
        text = to_full_text(row["messages"])
        if text.strip():
            t0 = time.time()
            result = await map_document(session, sem, text)
            dt = time.time() - t0
            ok = result is not None
            rec = {
                "uid": row["uuid"],
                "src_file": row["file_name"],
                "orig_domain": row["domain"],
                "orig_family": row["source"],
                "orig_source": row["source"],
                "ok": ok,
                "result": result or {},
                "full_text": text,
            }
            async with lock:
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out_f.flush()
                counter[0] += 1
                n = counter[0]
            if n <= 10 or n % 50 == 0:
                el = time.time() - counter[1]
                print(f"  [{n}] worker{worker_id} took {dt:.1f}s  |  {el/60:.1f} min elapsed, rate {n/el:.2f} docs/sec", flush=True)
        queue.task_done()


async def main(limit=None):
    rows = []
    with open(IN_FILE, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    if limit:
        rows = rows[:limit]
    print(f"mapping {len(rows)} rows, {CONCURRENCY} workers", flush=True)

    queue = asyncio.Queue()
    for r in rows:
        queue.put_nowait(r)

    sem = asyncio.Semaphore(CONCURRENCY)
    counter = [0, time.time()]
    lock = asyncio.Lock()
    with open(OUT_FILE, "w", encoding="utf-8") as out_f:
        async with aiohttp.ClientSession() as session:
            workers = [asyncio.create_task(worker(i, queue, sem, session, out_f, counter, lock))
                       for i in range(CONCURRENCY)]
            print(f"spawned {len(workers)} workers", flush=True)
            await asyncio.gather(*workers)

    dt = time.time() - counter[1]
    print(f"DONE: {counter[0]} docs mapped in {dt/60:.1f} min ({counter[0]/dt:.2f} docs/sec) -> {OUT_FILE}", flush=True)


if __name__ == "__main__":
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else None
    asyncio.run(main(lim))
