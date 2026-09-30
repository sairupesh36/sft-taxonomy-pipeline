"""Candidate rule: user message cut off mid-sentence (single short line, no closing punctuation, ends on a function word)."""
import glob, json, os, random, re, collections
random.seed(33)
ROOT = "/projects/data/datasets/code_data/sai_rupesh/taxonomy/sft_43_language_wise_clean"
STOP = set("the a an in of to that and or with by for on at from as is are was were be been which who whom whose than into onto during after before between among within without about over under against exposed induced associated related treated compared using via per if when while what how why where".split())
END_OK = tuple(list(".?!:;)]}\"'”’`") + ["…"])
def truncated(c):
    s = c.strip()
    if not (25 <= len(s) <= 140) or "\n" in s: return False
    if s.endswith(END_OK) or s[-1].isdigit(): return False
    w = re.findall(r"[A-Za-z][A-Za-z'-]*$", s)
    return bool(w) and (w[0].lower() in STOP or len(w[0]) <= 3 and s[-2:].isalpha() and False)
def cutmid(c):   # ends in the middle of a word (very common for fixed-length cuts): last token is not a whole word we know -> heuristic: short last fragment after a space and message has no end punctuation
    return False
files = glob.glob(ROOT + "/*/*__p*.jsonl"); pick = random.sample(files, 100)
n = 0; hit = 0; by_file = collections.Counter(); ex = []
for p in pick:
    size = os.path.getsize(p)
    with open(p, "rb") as f:
        f.seek(random.randint(0, max(0, size - 3_000_000))); f.readline()
        for _ in range(4000):
            line = f.readline()
            if not line: break
            d = json.loads(line); n += 1
            u = next((x for x in d["messages"] if x["role"] == "user" and isinstance(x.get("content"), str)), None)
            if u and truncated(u["content"]):
                hit += 1; by_file[os.path.basename(p).split("__p")[0][-32:]] += 1
                a = next((x for x in d["messages"] if x["role"] == "assistant"), None)
                if len(ex) < 500: ex.append((u["content"], str(a.get("content"))[:110].replace("\n", " | ") if a else ""))
print(f"rows {n:,}; flagged truncated-question {hit:,} ({hit/n:.3%})")
print("concentrated in:", by_file.most_common(6))
random.shuffle(ex)
for u, a in ex[:22]: print(f"  U: {u!r}\n     A: {a!r}")
