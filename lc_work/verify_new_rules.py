import glob, json, os, random, re, sys, collections
sys.path.insert(0, "/projects/data/datasets/code_data/sai_rupesh/taxonomy")
import sft_clean_filter as F
random.seed(17)
base = dict(think="keep", empty_think="drop", max_chars=200000, max_per_prompt=0, lexical=False, code_tool="python", special="strip", pii="off", sample=0, context=True)
# 1) rows my first post-pass dropped: what does the corrected rule do with them?
files = glob.glob("sft_43_language_wise_clean/_rejects_context/*/*.jsonl"); random.shuffle(files)
res = collections.Counter(); tagres = collections.defaultdict(collections.Counter); kept_ex = []
for p in files[:150]:
    with open(p, "rb") as f:
        for i, line in enumerate(f):
            if i >= 1500: break
            row, reason, _ = F.process_row(line, base, F.new_state())
            m = F.PH_TAG.search(line.decode("utf-8", "replace")); tag = m.group(1).lower() if m else ("seq" if b"<seq>" in line else "other")
            res[reason or "KEPT"] += 1; tagres[tag][reason or "KEPT"] += 1
            if row is not None and len(kept_ex) < 6 and tag != "seq": kept_ex.append(line[:300])
print("== 1) previously dropped rows, sampled:", sum(res.values()))
for t, c in tagres.items(): print(f"   tag <{t}>: {dict(c)}")
print("   examples of rows now KEPT that are not <seq>:", [k.decode('utf-8', 'replace')[:160] for k in kept_ex[:3]])
# 2) PII on real rows
clean = [f for f in glob.glob("sft_43_language_wise_clean/*/*.jsonl") if not f.split("/")[-2].startswith("_")]; random.shuffle(clean)
n = em = ip = rows_changed = 0; ex = []
for p in clean[:90]:
    size = os.path.getsize(p)
    with open(p, "rb") as f:
        f.seek(random.randint(0, max(0, size - 4_000_000))); f.readline()
        for _ in range(3000):
            line = f.readline()
            if not line: break
            n += 1; d = json.loads(line); rep = collections.Counter(); changed = False
            for m in d["messages"]:
                c = m.get("content")
                if isinstance(c, str) and ("@" in c or "." in c):
                    r = collections.Counter(); c2 = F.pii_datatrove(c, r)
                    if c2 != c:
                        changed = True; rep.update(r)
                        if len(ex) < 400:
                            i = next(k for k in range(min(len(c), len(c2))) if c[k] != c2[k]); ex.append((c[max(0, i-45): i+40].replace("\n", " "), c2[max(0, i-45): i+40].replace("\n", " ")))
            em += rep["pii_email_replaced"]; ip += rep["pii_ip_replaced"]; rows_changed += changed
print(f"\n== 2) PII (datatrove-style) on {n:,} real rows: rows changed {rows_changed:,} ({rows_changed/n:.3%}); emails replaced {em:,}; public IPs replaced {ip:,}")
random.shuffle(ex)
for b, a in ex[:26]: print("   BEFORE:", repr(b), "\n   AFTER: ", repr(a))
