import os, re, sys, json, random
sys.path.insert(0, "/projects/data/datasets/code_data/sai_rupesh/taxonomy/pylibs")
from multiprocessing import Pool
ROOT = "/projects/data/datasets/code_data/sai_rupesh/taxonomy/sft_43_language_wise/en"
THINK = re.compile(r"<think>(.*?)</think>", re.S | re.I); ANS = re.compile(r"</?answer>", re.I); CODE = re.compile(r"```.*?```", re.S)
def parts(r):
    m = r.get("messages") or []
    u = next((x["content"] for x in m if x.get("role") == "user" and isinstance(x.get("content"), str)), "")
    a = next((x["content"] for x in m if x.get("role") == "assistant" and isinstance(x.get("content"), str)), "")
    return u, a
def worker(args):
    fn, seed, n = args; rnd = random.Random(seed); p = f"{ROOT}/{fn}"; size = os.path.getsize(p); dots, rand, seen = [], [], 0
    with open(p, "rb") as f:
        for _ in range(n):
            f.seek(rnd.randint(0, size - 1)); f.readline(); l = f.readline()
            if not l: continue
            try: r = json.loads(l)
            except Exception: continue
            seen += 1; u, a = parts(r)
            tb = THINK.findall(a); junk = any(len(t.strip()) > 15 and sum(c.isalnum() for c in t) < 5 for t in tb)
            ans = ANS.sub("", THINK.sub(" ", a)).strip()
            if junk: dots.append((fn, u[:400], ans[:600], a[:120]))
            # random natural-language QA pool: user & answer mostly letters, no code, no latex
            if len(rand) < 25 and not junk and len(u) > 40 and len(ans) > 80 and "```" not in a and "$" not in u and sum(c.isalpha() for c in u) / max(1, len(u)) > 0.7 and sum(c.isalpha() for c in ans) / max(1, len(ans)) > 0.7:
                rand.append((fn, u[:400], ans[:600]))
    return seen, dots, rand
if __name__ == "__main__":
    files = sorted(os.listdir(ROOT)); jobs = [(fn, 500 + i * 7, 1400) for i, fn in enumerate(files)]
    with Pool(26) as pool: res = pool.map(worker, jobs)
    seen = sum(r[0] for r in res); dots = [d for r in res for d in r[1]]; rand = [x for r in res for x in r[2]]
    print(f"rows sampled: {seen:,}  | rows whose think block is placeholder/empty-junk: {len(dots)}  ({100*len(dots)/seen:.3f}%)  | natural-language QA rows kept for comparison: {len(rand)}", flush=True)
    json.dump(dict(dots=dots, rand=rand, seen=seen), open("lc_work/mismatch_probe.json", "w"))
