"""Same as before but print the text AROUND the actual match, and also require the match to be in the assistant's own text (not a copied MC option)."""
import glob, json, re, random, collections, time
from multiprocessing import Pool
ROOT = "/projects/data/datasets/code_data/sai_rupesh/taxonomy/sft_43_language_wise_clean"
PAT = re.compile(r"\b(don'?t|do not|doesn'?t|does not) have (?:enough|sufficient|the necessary)?\s*(?:context|information|details)\b|"
                  r"\bnot enough (?:context|information)\b|\black(?:s|ing)? (?:sufficient |enough )?(?:context|information)\b|"
                  r"\bwithout (?:more|additional|further) context\b|\bneed(?:s)? (?:more|additional) (?:context|information|details) to\b|"
                  r"\bcould you (?:please )?(?:provide|clarify|specify)\b|\bcan you (?:please )?(?:clarify|provide more|specify)\b|"
                  r"\bnot (?:enough|sufficient) (?:context|information) (?:to|is given|was given|provided)\b", re.I)
def scan(path):
    ex = []
    with open(path, "rb") as f:
        for line in f:
            r = json.loads(line); m = r["messages"]
            a = next((x["content"] for x in m if x["role"] == "assistant" and isinstance(x.get("content"), str)), "")
            mm = PAT.search(a)
            if mm and len(ex) < 4: ex.append((m, mm.start()))
    return ex
if __name__ == "__main__":
    files = sorted(f for f in glob.glob(ROOT + "/en/*.jsonl"))[:400]; random.seed(9); random.shuffle(files); files = files[:150]
    t0 = time.time(); allex = []
    with Pool(48) as p:
        for ex in p.imap_unordered(scan, files, chunksize=1): allex += ex
    print(f"{time.time()-t0:.0f}s | scanned 150 real 'en' files | KEPT rows matching, with a real 'not enough context/info' phrase in the assistant's OWN reply: {len(allex)}")
    random.shuffle(allex)
    for m, i in allex[:8]:
        u = next((x["content"] for x in m if x["role"] == "user"), "")
        a = next((x["content"] for x in m if x["role"] == "assistant"), "")
        print(f"\n  U: {u[:180]!r}\n  A (around match): ...{a[max(0,i-90):i+110]!r}...")
