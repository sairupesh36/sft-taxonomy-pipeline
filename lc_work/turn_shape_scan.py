"""How are conversations shaped: how many user asks (real turns) vs how many total messages (assistant/tool back-and-forth)?"""
import glob, json, collections, time
from multiprocessing import Pool
ROOT = "/projects/data/datasets/code_data/sai_rupesh/taxonomy/sft_43_language_wise_clean"
def scan(path):
    c = collections.Counter(); msg_ct = collections.Counter(); user_ct = collections.Counter(); ex = collections.defaultdict(list)
    with open(path, "rb") as f:
        for line in f:
            r = json.loads(line); m = r["messages"]
            nu = sum(x["role"] == "user" for x in m); nmsg = len(m); has_tool = any(x.get("tool_calls") for x in m)
            user_ct[min(nu, 5)] += 1; msg_ct[min(nmsg, 12)] += 1
            if nu == 1 and nmsg == 2: c["single_ask__plain_QA"] += 1                    # user, assistant (+ maybe system before)
            elif nu == 1 and nmsg > 2 and has_tool: c["single_ask__tool_trajectory"] += 1  # 1 real user request, many assistant/tool turns
            elif nu == 1 and nmsg > 2 and not has_tool: c["single_ask__long_assistant_no_tool"] += 1
            elif nu >= 2: c["multi_ask__real_dialogue"] += 1                              # 2+ separate user turns
            else: c["other"] += 1
            k = "multi_ask__real_dialogue" if nu >= 2 else ("single_ask__tool_trajectory" if (nu == 1 and has_tool and nmsg > 2) else None)
            if k and len(ex[k]) < 2: ex[k].append([x["role"] for x in m])
    return c, msg_ct, user_ct, ex
if __name__ == "__main__":
    files = sorted(f for f in glob.glob(ROOT + "/*/*.jsonl") if not f.split("/")[-2].startswith("_")); t0 = time.time()
    tot = collections.Counter(); mc = collections.Counter(); uc = collections.Counter(); exx = collections.defaultdict(list)
    with Pool(64) as p:
        for c, m, u, e in p.imap_unordered(scan, files, chunksize=1):
            tot.update(c); mc.update(m); uc.update(u)
            for k, v in e.items():
                if len(exx[k]) < 2: exx[k] += v
    n = sum(tot.values()); print(f"{time.time()-t0:.0f}s | total rows {n:,}")
    print("\nby shape:"); [print(f"  {k:32s} {v:>12,} ({100*v/n:.2f}%)") for k, v in tot.most_common()]
    print("\nnumber of USER turns per row:"); [print(f"  {k}{'+' if k==5 else ''} user turn(s): {v:,} ({100*v/n:.2f}%)") for k, v in sorted(uc.items())]
    print("\ntotal messages per row (system+user+assistant+tool):"); [print(f"  {k}{'+' if k==12 else ''} messages: {v:,} ({100*v/n:.2f}%)") for k, v in sorted(mc.items())]
    print("\nexample role sequences:"); [print(f"  [{k}] {v}") for k, vv in exx.items() for v in vv]
