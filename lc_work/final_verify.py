"""Full scan of EVERY output row: strict OpenAI format (validate_openai.check) + leftover-pattern checks + repeated-prompt suspects."""
import glob, hashlib, json, os, sys, collections, time
from multiprocessing import Pool
sys.path.insert(0, "/projects/data/datasets/code_data/sai_rupesh/taxonomy"); sys.path.insert(0, "/projects/data/datasets/code_data/sai_rupesh/taxonomy/lc_work")
import sft_clean_filter as F
from validate_openai import check
ROOT = "/projects/data/datasets/code_data/sai_rupesh/taxonomy/sft_43_language_wise_clean"
K = 300
def scan(path):
    viol = collections.Counter(); kinds = collections.Counter(); n = 0
    cnt = collections.Counter(); ans = collections.defaultdict(set); first = {}
    with open(path, "rb") as f:
        for line in f:
            n += 1
            try: row = json.loads(line)
            except Exception: viol["json_parse_error"] += 1; continue
            v = check(row); viol.update(set(v))
            m = row["messages"]
            if any(x.get("role") == "user" and isinstance(x.get("content"), str) and F.PH_TAG.search(x["content"]) and F.missing_context_tag(x["content"]) for x in m): viol["LEFTOVER_missing_context_placeholder"] += 1
            if F.missing_context_cutoff(m): viol["LEFTOVER_missing_context_cutoff"] += 1
            for x in m:
                c = x.get("content")
                if isinstance(c, str) and F.SPECIAL_TOK.search(c): viol["LEFTOVER_special_token_in_text"] += 1; break
            kinds["rows"] += 1; kinds["has_tool_calls"] += any(x.get("tool_calls") for x in m); kinds["has_tools_field"] += "tools" in row; kinds["has_system"] += m[0]["role"] == "system"
            kinds["multi_turn"] += sum(x["role"] == "user" for x in m) > 1
            u = next((x["content"] for x in m if x["role"] == "user" and isinstance(x.get("content"), str)), None); a = next((x["content"] for x in m if x["role"] == "assistant" and isinstance(x.get("content"), str)), None)
            if u and a:
                h = hashlib.blake2b(u.encode(), digest_size=8).digest(); cnt[h] += 1
                if len(ans[h]) < 3: ans[h].add(a[:60])
                first.setdefault(h, u[:140])
    sus = [(c, first[h]) for h, c in cnt.items() if c >= K and len(ans[h]) >= 3]
    return path, n, dict(viol), dict(kinds), sus
if __name__ == "__main__":
    files = sorted(f for f in glob.glob(ROOT + "/*/*.jsonl") if not f.split("/")[-2].startswith("_")); t0 = time.time()
    viol = collections.Counter(); kinds = collections.Counter(); rows = 0; by_f = collections.defaultdict(lambda: [0, 0, collections.Counter()]); sus_by = collections.defaultdict(lambda: [0, 0, ""])
    with Pool(64) as p:
        for i, (path, n, v, k, sus) in enumerate(p.imap_unordered(scan, files, chunksize=1), 1):
            rows += n; viol.update(v); kinds.update(k); lang = path.split("/")[-2]
            b = by_f[lang]; b[0] += n; b[1] += sum(v.values()); b[2].update(v)
            for c, txt in sus:
                key = txt[:70]; e = sus_by[key]; e[0] += c; e[1] += 1; e[2] = txt
            if i % 500 == 0: print(f"[{i}/{len(files)}] {time.time()-t0:.0f}s rows {rows:,} violations so far {sum(viol.values()):,}", flush=True)
    print(f"\nFULL SCAN: {rows:,} rows in {len(files):,} files, {time.time()-t0:.0f}s")
    print("violations:", dict(viol) if viol else "NONE")
    print("composition:", dict(kinds))
    print("folders with violations:", {k: dict(v[2]) for k, v in by_f.items() if v[1]})
    top = sorted(sus_by.values(), key=lambda e: -e[0])[:15]
    print(f"\nREPEATED-PROMPT SUSPECTS (same first user message >= {K}x inside one chunk, >=3 different answers): {len(sus_by):,} distinct prompts; top 15 by rows")
    for c, nf, txt in top: print(f"  {c:>9,} rows in {nf:>3} chunks | {txt!r}")
    json.dump({"rows": rows, "violations": dict(viol), "composition": dict(kinds), "suspects": sorted(sus_by.values(), key=lambda e: -e[0])[:200]}, open("/projects/data/datasets/code_data/sai_rupesh/taxonomy/lc_work/final_verify.json", "w"), indent=1)
