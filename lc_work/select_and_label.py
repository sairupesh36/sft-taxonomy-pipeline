"""Pick three sets from the scored pool and have Gemma label them:
   T = 1000 random rows (unbiased TEST set, never used for training)
   B = 3000 random rows (random-sampling arm)
   A = 3000 rows the classifiers are unsure about (min top-1 confidence < 0.8), spread across instruction templates
Labels come from the real production mapper (taxonomy_map_document.map_document), text = the fixed to_full_text()."""
import sys, os, json, random, re, asyncio, aiohttp, collections, time
sys.path.insert(0, "/projects/data/datasets/code_data/sai_rupesh/taxonomy")
sys.path.insert(0, "/projects/data/datasets/code_data/sai_rupesh/taxonomy/classifier_experiments")
import dns_override  # noqa
from taxonomy_map_document import map_document
W = "/projects/data/datasets/code_data/sai_rupesh/taxonomy/lc_work"
random.seed(31)
pool = [json.loads(l) for l in open(f"{W}/pool_scored.jsonl")]
random.shuffle(pool)
def tmpl(t):
    m = re.search(r"\[USER\]:\s*(.*)", t, re.S); s = re.sub(r"\s+", " ", (m.group(1) if m else t))[:70]
    return " ".join(re.sub(r"[\d\$\\]+", "#", s).split()[:4])
T = pool[:1000]; B = pool[1000:4000]; rest = pool[4000:]
groups = collections.defaultdict(list)
for r in rest:
    if min(r["dc1"], r["tc1"]) < 0.8: groups[tmpl(r["full_text"])].append(r)
A = []; cap = 60; keys = list(groups); random.shuffle(keys); rnd = 0
while len(A) < 3000 and rnd < cap:
    for k in keys:
        if rnd < len(groups[k]) and len(A) < 3000: A.append(groups[k][rnd])
    rnd += 1
print(f"pool {len(pool)} | uncertain templates {len(groups)} | A {len(A)} B {len(B)} T {len(T)}", flush=True)
queue = [(x, "T") for x in T] + [(x, "B") for x in B] + [(x, "A") for x in A]; random.shuffle(queue)
done = set()
for arm in "TBA":
    p = f"{W}/labeled_{arm}.jsonl"
    if os.path.exists(p): done |= {json.loads(l)["uid"] for l in open(p)}
queue = [(x, a) for x, a in queue if f"pool:{x['h']}" not in done]; print("to label", len(queue), flush=True)
async def main():
    sem = asyncio.Semaphore(96); lock = asyncio.Lock(); q = asyncio.Queue(); [q.put_nowait(i) for i in queue]
    files = {a: open(f"{W}/labeled_{a}.jsonl", "a") for a in "TBA"}; cnt = [0, 0]; t0 = time.time()
    async with aiohttp.ClientSession() as s:
        async def worker():
            while True:
                try: x, arm = q.get_nowait()
                except asyncio.QueueEmpty: return
                res = await map_document(s, sem, x["full_text"])
                rec = dict(uid=f"pool:{x['h']}", src_file=x["shard"], orig_domain="target_en", orig_family="", orig_source="target_en", ok=res is not None, result=res or {}, full_text=x["full_text"], arm=arm)
                async with lock:
                    files[arm].write(json.dumps(rec, ensure_ascii=False) + "\n"); files[arm].flush(); cnt[0] += 1; cnt[1] += res is not None
                    if cnt[0] % 250 == 0: print(f"  labelled {cnt[0]}  ok {cnt[1]}  {cnt[0]/(time.time()-t0):.2f}/s", flush=True)
        await asyncio.gather(*[worker() for _ in range(96)])
    print("DONE", cnt, flush=True)
asyncio.run(main())
