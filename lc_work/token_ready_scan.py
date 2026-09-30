import os, re, json, random, collections
from multiprocessing import Pool
ROOT = "/projects/data/datasets/code_data/sai_rupesh/taxonomy/sft_43_language_wise"
SPECIAL = re.compile(r"<\|(?:im_start|im_end|im_sep|endoftext|eot_id|eom_id|start_header_id|end_header_id|begin_of_text|end_of_text|system|user|assistant|tool|python_tag|fim_[a-z]+|pad|padding)\|>|\[/?INST\]|<</?SYS>>|</s>|<s>|<\|reserved_special_token_\d+\|>")
def worker(a):
    lang, fn, n, seed = a; rnd = random.Random(seed); p = f"{ROOT}/{lang}/{fn}"; size = os.path.getsize(p); seen = set(); lens = []; sp = collections.Counter(); rows = 0
    with open(p, "rb") as f:
        for _ in range(n * 3):
            if rows >= n: break
            f.seek(rnd.randint(0, max(0, size - 1))); f.readline(); off = f.tell(); l = f.readline()
            if not l or off in seen: continue
            seen.add(off)
            try: m = json.loads(l)["messages"]
            except Exception: continue
            rows += 1; text = "".join(x["content"] for x in m if isinstance(x.get("content"), str)); lens.append(len(text))
            for t in set(SPECIAL.findall(text)): sp[t] += 1
    return lang, rows, lens, sp
if __name__ == "__main__":
    tasks = [("en", fn, 1200, 7 + i) for i, fn in enumerate(sorted(os.listdir(f"{ROOT}/en")))]
    for lang in ("hi", "zh", "es", "ar"): tasks += [(lang, fn, 300, 500 + j) for j, fn in enumerate(sorted(os.listdir(f"{ROOT}/{lang}"))[:6])]
    with Pool(20) as pool: res = pool.map(worker, tasks)
    for grp, sel in (("en", lambda l: l == "en"), ("other", lambda l: l != "en")):
        rows = sum(r for l, r, _, _ in res if sel(l)); lens = sorted(x for l, _, ls, _ in res if sel(l) for x in ls); sp = collections.Counter()
        for l, _, _, c in res:
            if sel(l): sp.update(c)
        pct = lambda q: lens[int(q * (len(lens) - 1))]
        print(f"\n== {grp}: {rows:,} rows | chars per row: p50 {pct(.5):,}  p90 {pct(.9):,}  p99 {pct(.99):,}  max {lens[-1]:,}")
        print(f"   over 32k chars (~8k tokens): {100*sum(x>32000 for x in lens)/len(lens):.2f}%  | over 128k chars (~32k tokens): {100*sum(x>128000 for x in lens)/len(lens):.2f}%")
        tot = sum(sp.values()); print(f"   rows containing literal chat-template special tokens: {sum(1 for _ in [0]) and tot:,} hits in {rows:,} rows -> {100*tot/rows:.2f}%  {dict(sp.most_common(6))}")
