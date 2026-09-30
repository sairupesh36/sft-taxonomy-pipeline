import sys, os, json, random, re, collections
sys.path.insert(0, "/projects/data/datasets/code_data/sai_rupesh/taxonomy")
import sft_clean_filter as F
random.seed(5)
ROOT = "/projects/data/datasets/code_data/sai_rupesh/taxonomy/sft_43_language_wise/" + sys.argv[1]
want = sys.argv[2].split(","); per = int(sys.argv[3]) if len(sys.argv) > 3 else 100
cfg = dict(think="keep", empty_think="drop", max_chars=200000, max_per_prompt=0, lexical=True)
got = collections.defaultdict(list); seen = collections.Counter()
for fn in sorted(os.listdir(ROOT)):
    p = f"{ROOT}/{fn}"; size = os.path.getsize(p)
    with open(p, "rb") as f:
        for _ in range(per):
            f.seek(random.randint(0, size - 1)); f.readline(); l = f.readline()
            if not l: continue
            row, why, rep = F.process_row(l, cfg, F.new_state())
            if why in want and len(got[why]) < 3: got[why].append(json.loads(l))
            seen[why] += 1
def snip(t, n=230): return re.sub(r"\s+", " ", t)[:n]
for why in want:
    print(f"\n########## {why}  (seen {seen[why]})")
    for r in got[why]:
        m = r["messages"]; roles = "".join(x["role"][0] for x in m)
        print(f"  roles={roles[:40]}")
        if why == "tool_mismatch":
            for i, x in enumerate(m):
                if x["role"] == "assistant" and "<tool_call>" in x["content"]:
                    blocks = F.HERMES_CALL.findall(x["content"]); nxt = []
                    j = i + 1
                    while j < len(m) and m[j]["role"] == "tool": nxt.append(m[j]["content"]); j += 1
                    resp = sum(len(F.HERMES_RESP.findall(t)) or 1 for t in nxt)
                    bad = []
                    for b in blocks:
                        try: json.loads(b)
                        except Exception as e: bad.append(b[:160])
                    if bad or resp != len(blocks):
                        print(f"   msg#{i}: calls={len(blocks)} responses={resp} unparseable_call={bool(bad)}  {('BAD: ' + repr(bad[0])) if bad else ''}"); break
        elif why == "think_unbalanced":
            for i, x in enumerate(m):
                c = x["content"] if isinstance(x["content"], str) else ""
                o, cl = c.lower().count("<think>"), c.lower().count("</think>")
                if o != cl: print(f"   msg#{i} role={x['role']} open={o} close={cl} start={snip(c,100)!r} ... end={snip(c[-120:],120)!r}"); break
        elif why == "think_in_non_assistant":
            for i, x in enumerate(m):
                c = x["content"] if isinstance(x["content"], str) else ""
                if x["role"] != "assistant" and re.search(r"</?think", c, re.I): print(f"   msg#{i} role={x['role']} {snip(c, 260)!r}"); break
        elif why == "repr_unparseable":
            for i, x in enumerate(m):
                c = x["content"] if isinstance(x["content"], str) else ""
                if c.lstrip()[:2] in ("[{", "[ "): print(f"   msg#{i} role={x['role']} {snip(c, 300)!r}"); break
        elif why == "repetition_loop":
            a = "".join(x["content"] for x in m if x["role"] == "assistant" and isinstance(x["content"], str)); w = F.CODE_FENCE.sub(" ", a).split()
            g = collections.Counter(" ".join(w[k:k+10]) for k in range(0, len(w)-9)); top = g.most_common(1)[0] if g else ("", 0)
            print(f"   words={len(w)} top 10-gram x{top[1]}: {top[0][:150]!r}")
        elif why == "think_no_answer":
            a = [x["content"] for x in m if x["role"] == "assistant"][-1]; print(f"   ...{snip(a[-200:],200)!r}")
        elif why == "qa_no_word_overlap":
            print(f"   Q: {snip(m[0]['content'],200)!r}\n   A: {snip(m[-1]['content'],220)!r}")
        else:
            print(f"   {snip(json.dumps(m[-1]), 200)!r}")
