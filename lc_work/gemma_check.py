import sys, json, random, asyncio, aiohttp
sys.path.insert(0, "/projects/data/datasets/code_data/sai_rupesh/taxonomy")
sys.path.insert(0, "/projects/data/datasets/code_data/sai_rupesh/taxonomy/classifier_experiments")
import dns_override  # noqa  (router hostname doesn't resolve from this shell; process-local redirect)
from taxonomy_map_document import map_document
rows = json.load(open("lc_work/target_run_80.json"))
random.seed(5); idx = random.sample(range(len(rows)), 400)
async def main():
    sem = asyncio.Semaphore(32); out = {}
    async with aiohttp.ClientSession() as s:
        async def one(i):
            r = await map_document(s, sem, rows[i]["full_text"])
            if r: out[i] = dict(domain=r.get("domain"), task_family=r.get("task_family"))
        await asyncio.gather(*[one(i) for i in idx])
    json.dump(out, open("lc_work/gemma_check_400.json", "w")); print("gemma labelled", len(out), "of", len(idx))
asyncio.run(main())
