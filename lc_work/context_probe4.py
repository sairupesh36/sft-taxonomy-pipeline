"""How many rows still look 'incomplete' AFTER the placeholder rule? Signal: the assistant itself says the input is missing."""
import glob, json, os, random, re, collections
random.seed(21)
ROOT = "/projects/data/datasets/code_data/sai_rupesh/taxonomy/sft_43_language_wise_clean"
TIGHT = re.compile(r"(?<![`\w/])<(image|DNA|RNA|protein|smiles|mol|seq)>(?![`\w])", re.I)
MISSING = re.compile(r"(you (haven'?t|have not|did not|didn'?t) (provide|provided|include|included|attach|attached|share|shared|paste|pasted)|(no|without any) (text|image|attachment|passage|code|file|table|document|data|context|content) (was|were|has been|have been)? ?(provided|attached|included|shared|given)|i (don'?t|do not|can'?t|cannot|am unable to) see (any|an|the|a) (text|image|attachment|passage|code|file|table|document|picture|photo|figure|link)|(text|image|attachment|passage|code|file|table|document|content|equation|problem|question) (is|are|seems? to be|appears? to be|was|were) (missing|incomplete|cut off|not (included|provided|attached))|(message|question|prompt|input|request) (seems|appears) (to be )?(incomplete|cut off|truncated)|please (provide|share|paste|include|attach) the (text|passage|article|document|code|image|picture|table|equation|full|actual|specific|complete))", re.I)
files = glob.glob(ROOT + "/*/*__p*.jsonl"); pick = random.sample(files, 100)
n = 0; placeholder = 0; miss = 0; ex = []
for p in pick:
    size = os.path.getsize(p)
    with open(p, "rb") as f:
        f.seek(random.randint(0, max(0, size - 3_000_000))); f.readline()
        for _ in range(4000):
            line = f.readline()
            if not line: break
            d = json.loads(line); n += 1
            m = d["messages"]
            if any(x["role"] == "user" and isinstance(x.get("content"), str) and TIGHT.search(x["content"]) for x in m):
                placeholder += 1; continue            # already covered by the placeholder rule
            for x in m:
                if x["role"] == "assistant" and isinstance(x.get("content"), str) and MISSING.search(x["content"][:400]):
                    miss += 1
                    if len(ex) < 400: ex.append((os.path.basename(p)[:34], [ (y["role"], str(y.get("content"))[:220].replace("\n"," | ")) for y in m if y["role"] in ("user","assistant")][:2]))
                    break
print(f"rows scanned {n:,} | caught by placeholder rule {placeholder:,} ({placeholder/n:.2%}) | NOT caught, assistant says input is missing {miss:,} ({miss/n:.2%})")
random.shuffle(ex)
for e in ex[:14]:
    print("-", e[0]); [print(f"     {r}: {c!r}") for r, c in e[1]]
