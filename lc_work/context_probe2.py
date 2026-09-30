import glob, json, os, random, re, collections, hashlib
random.seed(11)
ROOT = "/projects/data/datasets/code_data/sai_rupesh/taxonomy/sft_43_language_wise_clean"
# (a) tight placeholder rule: bare tag, not inside backticks
TIGHT = re.compile(r"(?<![`\w/])<(image|img|audio|video|DNA|RNA|protein|smiles|mol|seq)>(?![`\w])", re.I)
files = glob.glob(ROOT + "/*/*__p*.jsonl")
sample = random.sample(files, 120)
tagc = collections.Counter(); ex = collections.defaultdict(list); n = 0
for p in sample:
    size = os.path.getsize(p)
    with open(p, "rb") as f:
        f.seek(random.randint(0, max(0, size - 3_000_000))); f.readline()
        for _ in range(4000):
            line = f.readline()
            if not line: break
            n += 1
            if b"<" not in line: continue
            d = json.loads(line)
            for m in d["messages"]:
                if m["role"] != "user" or not isinstance(m.get("content"), str): continue
                for t in TIGHT.findall(m["content"]):
                    tagc[t.lower()] += 1
                    if len(ex[t.lower()]) < 3: ex[t.lower()].append((os.path.basename(p)[:36], m["content"][:200]))
                    break
print("(a) tight placeholder rule over", n, "rows:", dict(tagc), f"-> {sum(tagc.values())/n:.3%} of rows")
for t, v in ex.items():
    for e in v: print("  ", t, e)
# what is <desk> and INSERT_YOUR_EMAIL?
for needle in (b"<desk>", b"INSERT_YOUR_EMAIL"):
    for p in sample:
        with open(p, "rb") as f:
            for line in f:
                if needle in line:
                    d = json.loads(line); print("\nNEEDLE", needle, os.path.basename(p)[:36]); 
                    for m in d["messages"][:2]: print("   ", m["role"], str(m.get("content"))[:260].replace("\n", " | "))
                    break
            else: continue
            break
