"""Rows where the assistant says it lacks enough context/information -- confirm they are KEPT, and check the user's question is complete."""
import glob, json, re, random, collections, time
from multiprocessing import Pool
ROOT = "/projects/data/datasets/code_data/sai_rupesh/taxonomy/sft_43_language_wise_clean"
PAT = re.compile(r"\b(don'?t|do not|doesn'?t|does not) have (enough|sufficient|the necessary)?\s*(context|information|details)\b|"
                  r"\bnot enough (context|information)\b|\black(?:s|ing)? (?:sufficient |enough )?(?:context|information)\b|"
                  r"\bwithout (?:more|additional|further) context\b|\bneed(?:s)? (?:more|additional) (?:context|information|details)\b|"
                  r"\bcould you (?:please )?(?:provide|clarify|specify)\b|\bcan you (?:please )?(?:clarify|provide more|specify)\b", re.I)
def scan(path):
    ex = []
    with open(path, "rb") as f:
        for line in f:
            if not PAT.search(line.decode("utf-8", "replace")): continue
            r = json.loads(line); m = r["messages"]
            a = next((x["content"] for x in m if x["role"] == "assistant" and isinstance(x.get("content"), str)), "")
            if PAT.search(a) and len(ex) < 3: ex.append(m)
    return len(ex), ex
if __name__ == "__main__":
    files = sorted(f for f in glob.glob(ROOT + "/en/*.jsonl"))[:400]; random.seed(9); random.shuffle(files); files = files[:120]
    t0 = time.time(); tot = 0; allex = []
    with Pool(48) as p:
        for n, ex in p.imap_unordered(scan, files, chunksize=1):
            tot += n; allex += ex
    print(f"{time.time()-t0:.0f}s | scanned 120 real 'en' files | KEPT rows where assistant says it lacks context/info: {tot}")
    random.shuffle(allex)
    for m in allex[:10]:
        u = next((x["content"] for x in m if x["role"] == "user"), "")
        a = next((x["content"] for x in m if x["role"] == "assistant"), "")
        print(f"\n  U: {u[:200]!r}\n  A: {a[:200]!r}")
