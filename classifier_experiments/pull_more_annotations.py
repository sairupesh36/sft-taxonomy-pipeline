"""
Pulls a fresh batch of Gemma-mapped documents from the project's main corpus
(the 19 tulu3 + 17 agentic sources under data/ and agentic_data/) to expand
the classifier's training pool beyond the 15,722 docs already used
(sft_output_sample/ -- a separate, now fully-consumed dataset).

Reuses the real, vetted per-document mapper (sample_docs_stratified,
extract_text_from_row, map_document) from taxonomy_map_document.py directly
-- same 2-call Gemma pipeline, same validation/deterministic_fixes, same
closed-set enforcement -- so these new annotations are produced exactly the
same way as every other document in this project, not a reimplementation.

Output schema matches sft_output_sample_mapped.jsonl (uid, src_file,
orig_domain, orig_family, orig_source, ok, result, full_text) so the two
files can be concatenated directly for classifier retraining.

Writes incrementally (one line per completed doc, flushed immediately) so a
kill/restart mid-run loses at most the in-flight batch, not prior progress.
"""
import sys
import time
sys.path.insert(0, "/projects/data/datasets/code_data/sai_rupesh/taxonomy")

import asyncio
import aiohttp
import json

from taxonomy_map_document import sample_docs_stratified, map_document, extract_text_from_row

OUT_FILE = "/projects/data/datasets/code_data/sai_rupesh/taxonomy/classifier_experiments/gemma_annotations_batch2.jsonl"
N_TULU3 = 5000
N_AGENTIC = 5000
CONCURRENCY = 32


async def process_source(source_type, n, session, sem, out_f, counter):
    rows = sample_docs_stratified(n, source_type)
    print(f"[{source_type}] sampled {len(rows)} rows, starting mapping...", flush=True)

    async def handle(i, row):
        text = extract_text_from_row(row)
        if not text.strip():
            return
        result = await map_document(session, sem, text)
        ok = result is not None
        rec = {
            "uid": f"{source_type}:{row.get('id', i)}",
            "src_file": source_type,
            "orig_domain": source_type,
            "orig_family": str(row.get("source", source_type)),
            "orig_source": str(row.get("source", source_type)),
            "ok": ok,
            "result": result or {},
            "full_text": text,
        }
        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out_f.flush()
        counter[0] += 1
        if counter[0] % 100 == 0:
            elapsed = time.time() - counter[1]
            rate = counter[0] / elapsed if elapsed > 0 else 0
            print(f"  [{counter[0]}] done, elapsed {elapsed/60:.1f} min, rate {rate:.2f} docs/sec", flush=True)

    await asyncio.gather(*[handle(i, row) for i, row in enumerate(rows)])


async def main():
    sem = asyncio.Semaphore(CONCURRENCY)
    counter = [0, time.time()]
    with open(OUT_FILE, "w", encoding="utf-8") as out_f:
        async with aiohttp.ClientSession() as session:
            await process_source("tulu3", N_TULU3, session, sem, out_f, counter)
            await process_source("agentic", N_AGENTIC, session, sem, out_f, counter)
    dt = time.time() - counter[1]
    print(f"DONE: {counter[0]} docs mapped in {dt/60:.1f} min -> {OUT_FILE}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
