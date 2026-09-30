import glob, json, os, random, collections, hashlib
random.seed(5)
ROOT = "/projects/data/datasets/code_data/sai_rupesh/taxonomy/sft_43_language_wise_clean"
files = glob.glob(ROOT + "/*/*__p*.jsonl")
ctrl = [ROOT + "/en/phase1_shard0of16_openai_sft_full__p0000.jsonl"]
sample = ctrl + random.sample([f for f in files if f not in ctrl], 14)
K = 20
tot_rows = tot_flag = 0
for p in sample:
    cnt = collections.Counter(); ans = collections.defaultdict(set); first = {}
    rows = 0
    with open(p, "rb") as f:
        for line in f:
            rows += 1
            d = json.loads(line); m = d["messages"]
            u = next((x for x in m if x["role"] == "user" and isinstance(x.get("content"), str)), None)
            a = next((x for x in m if x["role"] == "assistant" and isinstance(x.get("content"), str)), None)
            if not u or not a: continue
            h = hashlib.blake2b(u["content"].encode(), digest_size=8).digest()
            cnt[h] += 1
            if len(ans[h]) < 5: ans[h].add(a["content"][:60])
            first.setdefault(h, u["content"][:150])
    flagged = {h for h, c in cnt.items() if c >= K and len(ans[h]) >= 3}
    nflag = sum(cnt[h] for h in flagged)
    tot_rows += rows; tot_flag += nflag
    print(f"\n{os.path.basename(p)[:52]:52s} lang={p.split('/')[-2]} rows={rows:,} flagged_rows={nflag:,} ({nflag/rows:.1%}) distinct_prompts_flagged={len(flagged)}")
    for h in sorted(flagged, key=lambda h: -cnt[h])[:3]:
        print(f"    x{cnt[h]:>7,}  {first[h]!r}")
print(f"\nTOTAL rows {tot_rows:,} flagged {tot_flag:,} ({tot_flag/tot_rows:.2%})")
