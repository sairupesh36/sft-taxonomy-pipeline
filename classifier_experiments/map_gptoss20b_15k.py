"""
Runs the SAME taxonomy prompts (SYSTEM_A, build_system_b, validate_result,
deterministic_fixes -- all imported directly from taxonomy_map_document.py,
never copied/reimplemented) on the same 15,722 documents that Gemma already
labeled, but pointed at the newly-deployed gpt-oss-20b model instead.

Only two things differ from the real map_document(), both because gpt-oss
models do hidden "reasoning" before writing their answer (gemma does not --
its reasoning_tokens is always 0), confirmed by testing directly against the
router before writing this:
  1. max_tokens is raised well above gemma's 400/500, so the reasoning phase
     doesn't eat the whole budget and leave nothing for the actual JSON answer.
  2. "reasoning_effort": "low" is sent -- confirmed via direct API test that
     this is honored (cut reasoning_tokens roughly in half on a trivial call)
     and still produces a correct answer. This also serves the "squeeze more
     throughput out of it" ask, since less reasoning = faster + cheaper per doc.

Never touches taxonomy_map_document.py itself -- this is an explicit,
separate test copy, same pattern as taxonomy_map_document_fast.py.

Outputs gptoss20b_vs_gemma_15k.jsonl (gpt-oss's own result per doc) and then
prints a field-by-field agreement report against Gemma's already-saved labels.
"""
import sys
import json
import time
import asyncio
import aiohttp
import collections

sys.path.insert(0, "/projects/data/datasets/code_data/sai_rupesh/taxonomy")
import taxonomy_map_document as tmd

IN_FILE = "/projects/data/datasets/code_data/sai_rupesh/taxonomy/sft_output_sample_mapped.jsonl"
OUT_FILE = "/projects/data/datasets/code_data/sai_rupesh/taxonomy/classifier_experiments/gptoss20b_vs_gemma_15k.jsonl"
REPORT_FILE = "/projects/data/datasets/code_data/sai_rupesh/taxonomy/classifier_experiments/gptoss20b_vs_gemma_report.json"

MODEL = "openai/gpt-oss-20b"
MAX_TOKENS_A = 900
MAX_TOKENS_B = 1000
CONCURRENCY = 256  # more GPU instances deployed for this model -- start high, benchmark confirms this below


async def call_llm_oss(session, sem, system_prompt, user_prompt, max_tokens):
    payload = {
        "model": MODEL,
        "messages": [{"role": "system", "content": system_prompt},
                     {"role": "user", "content": user_prompt}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "reasoning_effort": "low",
    }
    headers = {"Authorization": f"Bearer {tmd.API_KEY}", "Content-Type": "application/json"}
    async with sem:
        for attempt in range(tmd.RETRIES):
            try:
                async with session.post(tmd.ROUTER_URL, json=payload, headers=headers, timeout=tmd.TIMEOUT_S) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        raw = data["choices"][0]["message"].get("content", "") or ""
                        s, e = raw.find("{"), raw.rfind("}")
                        if s != -1 and e != -1 and e > s:
                            return json.loads(raw[s:e + 1])
            except Exception:
                pass
            await asyncio.sleep(0.5 * (attempt + 1))
    return None


async def map_document_oss(session, sem, text):
    a = await call_llm_oss(session, sem, tmd.SYSTEM_A, text, MAX_TOKENS_A)
    if not a:
        return None
    domains = tmd._valid_labels(a.get("domain"), "domain")
    task_families = tmd._valid_labels(a.get("task_family"), "task_family")
    if not domains or not task_families:
        a_retry = await call_llm_oss(session, sem, tmd.SYSTEM_A, text, MAX_TOKENS_A)
        if a_retry:
            a = a_retry
            domains = tmd._valid_labels(a.get("domain"), "domain") or domains
            task_families = tmd._valid_labels(a.get("task_family"), "task_family") or task_families
    domains = domains or ["general"]
    task_families = task_families or ["question_answering"]
    a["domain"] = domains
    a["task_family"] = task_families
    system_b = tmd.build_system_b(domains, task_families)
    b = await call_llm_oss(session, sem, system_b, text, MAX_TOKENS_B)
    if not b:
        b = {}
    return tmd.deterministic_fixes(tmd.validate_result({**a, **b}), text)


async def worker(queue, sem, session, out_f, counter, lock):
    while True:
        try:
            row = queue.get_nowait()
        except asyncio.QueueEmpty:
            return
        result = await map_document_oss(session, sem, row["full_text"])
        rec = {"uid": row["uid"], "gemma_result": row["result"], "oss_result": result or {}, "ok": result is not None}
        async with lock:
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_f.flush()
            counter[0] += 1
            n = counter[0]
        if n <= 10 or n % 200 == 0:
            el = time.time() - counter[1]
            print(f"  [{n}] {el/60:.1f} min elapsed, rate {n/el:.2f} docs/sec", flush=True)
        queue.task_done()


async def run_batch(rows, concurrency):
    queue = asyncio.Queue()
    for r in rows:
        queue.put_nowait(r)
    sem = asyncio.Semaphore(concurrency)
    counter = [0, time.time()]
    lock = asyncio.Lock()
    with open(OUT_FILE, "w", encoding="utf-8") as out_f:
        async with aiohttp.ClientSession() as session:
            workers = [asyncio.create_task(worker(queue, sem, session, out_f, counter, lock))
                       for _ in range(concurrency)]
            print(f"spawned {len(workers)} workers, concurrency={concurrency}", flush=True)
            await asyncio.gather(*workers)
    dt = time.time() - counter[1]
    print(f"DONE: {counter[0]} docs in {dt/60:.1f} min ({counter[0]/dt:.2f} docs/sec)", flush=True)


def field_agreement(gemma, oss, field, nested=None):
    def get(d, f, n):
        v = d.get(f)
        if n and isinstance(v, dict):
            v = v.get(n)
        return v

    g_all, o_all = [], []
    for row in ROWS_FOR_REPORT:
        g = get(row["gemma_result"], field, nested)
        o = get(row["oss_result"], field, nested)
        g_all.append(g)
        o_all.append(o)
    tp = fp = fn = exact = 0
    total = len(g_all)
    for g, o in zip(g_all, o_all):
        if isinstance(g, list) or isinstance(o, list):
            gs, os_ = set(g or []), set(o or [])
            tp += len(gs & os_)
            fp += len(os_ - gs)
            fn += len(gs - os_)
            if gs == os_:
                exact += 1
        else:
            if g == o:
                exact += 1
                tp += 1
            else:
                fp += 1
                fn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"exact_match_rate": exact / total if total else 0.0, "precision": precision, "recall": recall, "f1": f1, "n": total}


ROWS_FOR_REPORT = []


def build_report():
    global ROWS_FOR_REPORT
    ROWS_FOR_REPORT = []
    with open(OUT_FILE) as f:
        for line in f:
            d = json.loads(line)
            if d.get("ok"):
                ROWS_FOR_REPORT.append(d)
    print(f"\n{len(ROWS_FOR_REPORT)} docs got a valid gpt-oss-20b result out of {sum(1 for _ in open(OUT_FILE))} attempted")

    fields = [
        ("domain", None), ("task_family", None), ("domain_subdomain", None), ("task_subfamily", None),
        ("interaction_mode", None), ("tool_requirement", None), ("tool_category", None),
        ("constraints", None), ("complexity", None), ("task_composition", None),
        ("input", "type"), ("input", "format"), ("input", "language"),
        ("output", "type"), ("output", "format"), ("output", "language"),
        ("response_behavior", None),
    ]
    report = {}
    for field, nested in fields:
        key = f"{field}.{nested}" if nested else field
        report[key] = field_agreement(None, None, field, nested)
        r = report[key]
        print(f"  {key:30s} exact={r['exact_match_rate']:.3f}  F1={r['f1']:.3f}  (n={r['n']})")

    with open(REPORT_FILE, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nreport saved -> {REPORT_FILE}")
    return report


async def bench_concurrency(rows, levels):
    print("=== quick concurrency benchmark (gpt-oss-20b) ===", flush=True)
    results = {}
    for c in levels:
        sample = rows[:min(len(rows), max(60, c // 2))]
        t0 = time.time()
        sem = asyncio.Semaphore(c)
        async with aiohttp.ClientSession() as session:
            tasks = [map_document_oss(session, sem, r["full_text"]) for r in sample]
            done = await asyncio.gather(*tasks, return_exceptions=True)
        ok = sum(1 for d in done if isinstance(d, dict))
        dt = time.time() - t0
        rate = len(sample) / dt
        results[c] = rate
        print(f"  concurrency={c:4d}  n={len(sample):3d}  ok={ok:3d}  {dt:.1f}s  {rate:.2f} docs/sec", flush=True)
    return results


async def main():
    rows = []
    with open(IN_FILE, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d.get("ok"):
                rows.append(d)
    print(f"loaded {len(rows)} gemma-labeled docs to re-run through gpt-oss-20b", flush=True)

    bench = await bench_concurrency(rows, [64, 128, 256, 384])
    best_c = max(bench, key=bench.get)
    print(f"\nbest concurrency from benchmark: {best_c} ({bench[best_c]:.2f} docs/sec) -- using it for the full run\n", flush=True)

    await run_batch(rows, best_c)
    build_report()


if __name__ == "__main__":
    asyncio.run(main())
