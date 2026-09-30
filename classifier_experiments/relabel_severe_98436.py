"""
Fixes the 1,341 "severe truncation" rows in sft_output_sample_combined_98436.jsonl
(rows where neither [USER]: nor [ASSISTANT]: survived the old truncation logic,
so Gemma's label for them is a pure guess with zero real signal).

Approach (per the overnight task instructions -- prefer re-labeling over
exclusion when the original raw messages are still available):
  1. Find all "severe" rows in the combined 98,436 file.
  2. Recover each row's ORIGINAL raw messages (same uid-recovery logic as
     remediate_truncation_dryrun.py: uuid lookup in batch2/3/4 raw files, or
     "{src_file}:{line_number}" lookup in ../sft_output_sample for the
     original 15,722-doc round).
  3. Re-render full_text using the FIXED to_full_text() from map_batch2.py
     (keeps [USER]/[ASSISTANT]/[TOOL] content first, truncates/drops the
     system prompt first if space is tight).
  4. Re-run the real Gemma mapper (map_document, imported directly) on the
     fixed text.
  5. Rows that can't be recovered (raw messages genuinely gone) are marked
     ok=False and excluded from training -- their old label was a guess and
     there's nothing to re-label from.

Writes a NEW file (sft_output_sample_combined_98436_fixed.jsonl) rather than
mutating the original in place, and a small JSON report of what happened to
each affected uid.
"""
import sys
import json
import time
import asyncio
import aiohttp

sys.path.insert(0, "/projects/data/datasets/code_data/sai_rupesh/taxonomy")
from taxonomy_map_document import map_document
from map_batch2 import to_full_text

COMBINED_FILE = "sft_output_sample_combined_98436.jsonl"
FIXED_FILE = "sft_output_sample_combined_98436_fixed.jsonl"
REPORT_FILE = "relabel_severe_98436_report.json"
ORIGINAL_SAMPLE_DIR = "../sft_output_sample"
RAW_BATCH_FILES = ["sft_output_sample_batch2.jsonl", "sft_output_sample_batch3.jsonl", "sft_output_sample_batch4.jsonl"]
CONCURRENCY = 192


def is_severe(full_text):
    return "[USER]:" not in full_text and "[ASSISTANT]:" not in full_text


def build_uuid_index():
    index = {}
    for fn in RAW_BATCH_FILES:
        with open(fn) as f:
            for line in f:
                d = json.loads(line)
                index[d["uuid"]] = d["messages"]
    return index


def recover_messages(uid, uuid_index):
    if ":" in uid and uid.split(":")[0].endswith(".jsonl"):
        src_file, line_num = uid.rsplit(":", 1)
        line_num = int(line_num)
        path = f"{ORIGINAL_SAMPLE_DIR}/{src_file}"
        try:
            with open(path) as f:
                for i, line in enumerate(f):
                    if i == line_num:
                        return json.loads(line)["messages"]
        except FileNotFoundError:
            return None
        return None
    else:
        return uuid_index.get(uid)


async def relabel_worker(worker_id, queue, sem, session, results, counter, lock):
    while True:
        try:
            uid, messages = queue.get_nowait()
        except asyncio.QueueEmpty:
            return
        new_text = to_full_text(messages)
        result = None
        if new_text and new_text.strip():
            result = await map_document(session, sem, new_text)
        async with lock:
            results[uid] = {"new_full_text": new_text, "result": result, "ok": result is not None}
            counter[0] += 1
            n = counter[0]
        if n <= 5 or n % 100 == 0:
            el = time.time() - counter[1]
            print(f"  [{n}] relabeled  {el/60:.1f} min elapsed, rate {n/el:.2f} docs/sec", flush=True)
        queue.task_done()


async def main():
    print("loading combined file and finding severe rows...", flush=True)
    all_rows = []
    severe_uids = []
    with open(COMBINED_FILE) as f:
        for line in f:
            d = json.loads(line)
            all_rows.append(d)
            if is_severe(d.get("full_text", "")):
                severe_uids.append(d["uid"])
    print(f"total rows: {len(all_rows)}, severe rows: {len(severe_uids)}", flush=True)

    uuid_index = build_uuid_index()
    print(f"uuid index built: {len(uuid_index)} entries", flush=True)

    recovered = {}
    unrecovered = []
    for uid in severe_uids:
        msgs = recover_messages(uid, uuid_index)
        if msgs is not None:
            recovered[uid] = msgs
        else:
            unrecovered.append(uid)
    print(f"recovered raw messages: {len(recovered)}/{len(severe_uids)}  (unrecovered: {len(unrecovered)})", flush=True)

    # Relabel all recovered docs with the real Gemma pipeline via the FIXED to_full_text.
    queue = asyncio.Queue()
    for uid, msgs in recovered.items():
        queue.put_nowait((uid, msgs))

    results = {}
    sem = asyncio.Semaphore(CONCURRENCY)
    counter = [0, time.time()]
    lock = asyncio.Lock()
    async with aiohttp.ClientSession() as session:
        workers = [asyncio.create_task(relabel_worker(i, queue, sem, session, results, counter, lock))
                   for i in range(CONCURRENCY)]
        await asyncio.gather(*workers)

    n_ok = sum(1 for r in results.values() if r["ok"])
    n_fail = sum(1 for r in results.values() if not r["ok"])
    print(f"relabel done: {n_ok} ok, {n_fail} still failed (API), {len(unrecovered)} unrecoverable (excluded)", flush=True)

    # Build the fixed combined file: same rows, but severe ones get updated
    # full_text/result/ok, or ok=False (excluded) if unrecoverable.
    report = {"recovered_and_relabeled_ok": [], "recovered_but_api_failed": [], "unrecoverable_excluded": unrecovered}
    out_rows = []
    for d in all_rows:
        uid = d["uid"]
        if uid in results:
            r = results[uid]
            new_d = dict(d)
            new_d["full_text"] = r["new_full_text"]
            new_d["ok"] = r["ok"]
            if r["ok"]:
                new_d["result"] = r["result"]
                report["recovered_and_relabeled_ok"].append(uid)
            else:
                new_d["result"] = {}
                report["recovered_but_api_failed"].append(uid)
            out_rows.append(new_d)
        elif uid in unrecovered:
            new_d = dict(d)
            new_d["ok"] = False  # exclude from training -- old label was a blind guess, no raw text to re-label from
            out_rows.append(new_d)
        else:
            out_rows.append(d)

    with open(FIXED_FILE, "w", encoding="utf-8") as f:
        for d in out_rows:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    with open(REPORT_FILE, "w") as f:
        json.dump({
            "total_rows": len(all_rows),
            "severe_rows_found": len(severe_uids),
            "recovered_raw_messages": len(recovered),
            "unrecoverable": len(unrecovered),
            "relabeled_ok": n_ok,
            "relabeled_api_failed": n_fail,
            "detail": report,
        }, f, indent=2)

    print(f"DONE: wrote {FIXED_FILE} ({len(out_rows)} rows) and {REPORT_FILE}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
