"""Discovery probe: which 'incomplete / no-context' patterns really occur in the cleaned output?
Read-only. Samples random rows from random finished chunk files (all languages), applies candidate rules,
counts hits per rule and saves examples for manual reading."""
import glob, json, os, random, re, collections, sys
random.seed(7)
ROOT = "/projects/data/datasets/code_data/sai_rupesh/taxonomy/sft_43_language_wise_clean"
files = [f for f in glob.glob(ROOT + "/*/*__p*.jsonl")]
N_FILES, N_ROWS = int(sys.argv[1]) if len(sys.argv) > 1 else 80, 4000
pick = random.sample(files, min(N_FILES, len(files)))

MEDIA_TAG = re.compile(r"<(image|img|audio|video|DNA|RNA|protein|smiles|mol|seq|sequence|molecule|pdb|cif|structure|table|file|figure|photo|picture)>", re.I)
BRACKET_MEDIA = re.compile(r"\[(image|img|audio|video|photo|picture|figure|attachment)[^\]]{0,30}\]", re.I)
UNFILLED = re.compile(r"\{\{\s*\w+\s*\}\}|\[INSERT[^\]]{0,40}\]|<insert[^>]{0,40}>|\{(input|text|context|question|document|passage|query|prompt|article|sentence)\}", re.I)
DEICTIC = re.compile(r"\b(following|below|above|attached|given)\s+(text|passage|paragraph|article|document|code|snippet|table|image|picture|photo|figure|graph|chart|diagram|sentence|sentences|email|essay|story|review|excerpt|conversation|dialogue|data|list|file|context|abstract|program|function|script|equation|statement|tweet|comment|poem)\b", re.I)

def user_msgs(m):
    return [x for x in m if x.get("role") == "user" and isinstance(x.get("content"), str)]

def rules(msgs):
    hits = []
    us = user_msgs(msgs)
    for u in us:
        c = u["content"]
        if MEDIA_TAG.search(c): hits.append("media_tag")
        if BRACKET_MEDIA.search(c): hits.append("bracket_media")
        if UNFILLED.search(c): hits.append("unfilled_template")
        s = c.strip()
        if len(s) < 300 and "\n" not in s and s.endswith(":") and DEICTIC.search(s): hits.append("dangling_colon_deictic")
        elif len(s) < 300 and "\n" not in s and s.endswith(":"): hits.append("dangling_colon_other")
        elif len(s) < 200 and DEICTIC.search(s): hits.append("short_deictic_no_colon")
    return sorted(set(hits))

cnt = collections.Counter(); ex = collections.defaultdict(list); n = 0; tag_counter = collections.Counter()
by_lang = collections.Counter(); prompt_seen = collections.defaultdict(set)
TAG = re.compile(r"<([A-Za-z][A-Za-z0-9_]{1,18})>")
for p in pick:
    size = os.path.getsize(p); lang = p.split("/")[-2]
    with open(p, "rb") as f:
        f.seek(random.randint(0, max(0, size - 2_000_000))); f.readline()
        for _ in range(N_ROWS):
            line = f.readline()
            if not line: break
            try: d = json.loads(line)
            except Exception: continue
            n += 1; by_lang[lang] += 1
            msgs = d["messages"]
            for u in user_msgs(msgs):
                for t in TAG.findall(u["content"][:3000]): tag_counter[t] += 1
            h = rules(msgs)
            for r in h:
                cnt[r] += 1
                if len(ex[r]) < 40: ex[r].append({"file": os.path.basename(p), "lang": lang, "messages": [{"role": m["role"], "content": (m.get("content") or "")[:400] if isinstance(m.get("content"), str) else str(m.get("content"))[:200]} for m in msgs[:3]]})
print("rows scanned", n, "from", len(pick), "files; languages:", dict(by_lang.most_common(8)))
for r, c in cnt.most_common(): print(f"{r:26s} {c:>7,}  ({c/n:.3%})")
print("\nmost frequent <tag> tokens in user messages (top 40):", tag_counter.most_common(40))
json.dump(ex, open("/projects/data/datasets/code_data/sai_rupesh/taxonomy/lc_work/context_probe_examples.json", "w"), ensure_ascii=False, indent=1)
