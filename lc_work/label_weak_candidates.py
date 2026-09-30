"""Send the 21,407 real corpus-scan candidates (lc_work/weak_candidates.json) to Gemma for real,
verified labels. Resumable (skips uids already in the output file). Prints a live per-class
"confirmed" counter every 100 docs so progress against the target is visible while it runs -- a
candidate matched a KEYWORD, but only counts toward a class once GEMMA independently confirms it."""
import sys, os, json, asyncio, aiohttp, collections, time
sys.path.insert(0, "/projects/data/datasets/code_data/sai_rupesh/taxonomy")
sys.path.insert(0, "/projects/data/datasets/code_data/sai_rupesh/taxonomy/classifier_experiments")
import dns_override  # noqa -- process-local router DNS fix, see classifier_experiments/dns_override.py
from taxonomy_map_document import map_document

W = "/projects/data/datasets/code_data/sai_rupesh/taxonomy/lc_work"
OUT = f"{W}/labeled_weak_candidates.jsonl"
cand = json.load(open(f"{W}/weak_candidates.json"))
print(f"{len(cand)} candidates loaded", flush=True)

done_uids = set()
if os.path.exists(OUT):
    for line in open(OUT):
        try: done_uids.add(json.loads(line)["uid"])
        except Exception: pass
    print(f"resuming: {len(done_uids)} already labeled", flush=True)
queue = [c for c in cand if f"weak:{c['h']}" not in done_uids]
print(f"{len(queue)} left to label", flush=True)

async def main():
    sem = asyncio.Semaphore(96); lock = asyncio.Lock(); q = asyncio.Queue()
    for c in queue: q.put_nowait(c)
    fout = open(OUT, "a")
    confirmed = collections.Counter(); cnt = [0, 0]; t0 = time.time()
    async with aiohttp.ClientSession() as s:
        async def worker():
            while True:
                try: c = q.get_nowait()
                except asyncio.QueueEmpty: return
                res = await map_document(s, sem, c["full_text"])
                rec = {"uid": f"weak:{c['h']}", "src_file": c.get("src_file"), "hit_labels": c.get("hit_labels"),
                       "ok": res is not None, "result": res or {}, "full_text": c["full_text"]}
                async with lock:
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n"); fout.flush()
                    cnt[0] += 1; cnt[1] += res is not None
                    if res:
                        for ax in ("domain", "task_family"):
                            for lab in res.get(ax, []):
                                confirmed[f"{ax}:{lab}"] += 1
                    if cnt[0] % 100 == 0:
                        el = time.time() - t0
                        below = sum(1 for c2 in c["hit_labels"] for _ in [0]) if False else 0
                        print(f"  labeled {cnt[0]}/{len(queue)}  ok {cnt[1]}  {cnt[0]/el:.2f}/s  elapsed {el/60:.1f}min", flush=True)
        await asyncio.gather(*[worker() for _ in range(96)])
    print("DONE", cnt, flush=True)
    print("\nconfirmed counts for the original weak/target labels:")
    weak = json.load(open(f"{W}/weak_classes.json"))
    for ax, key in (("domain", "domain_master"), ("task_family", "task_family_master")):
        for lab in sorted(weak[key]):
            print(f"  {ax}:{lab:40s} confirmed={confirmed.get(f'{ax}:{lab}', 0)}")
asyncio.run(main())
