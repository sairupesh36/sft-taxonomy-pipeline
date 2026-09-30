"""Two-signal cut-off rule: (1) user message ends mid-sentence AND (2) the assistant itself says the question is incomplete/cut off.
Also measures a separate 'no-context' variant (assistant says a study/passage is needed) for reporting only."""
import glob, json, os, random, re, collections
random.seed(44)
ROOT = "/projects/data/datasets/code_data/sai_rupesh/taxonomy/sft_43_language_wise_clean"
END_OK = tuple(list(".?!:;)]}\"'”’`>") + ["…", "。", "？", "！"])
CUT = re.compile(r"(question|sentence|prompt|query|message|text|input|request|statement|title)\s+(seems|appears|looks|is|was|got|has been)\s+(to be\s+|like it('s| is| was)\s+)?(incomplete|cut off|cut-off|truncated|cut short|unfinished|abruptly)|(seems|appears|looks)\s+(to be\s+)?(incomplete|cut off|cut-off|truncated|unfinished)|(incomplete|truncated|cut[- ]off)\s+(question|sentence|prompt|query|message|text|input)|ends abruptly|trails off|cut off mid", re.I)
NOCTX = re.compile(r"no (context|passage|study|text|information) (is |was )?(given|provided)|without (more |additional |any )?context|(need|require)s? (more |additional |the )?context|which (study|paper|article|trial|experiment)|refer(s|ring)? to a specific (study|paper|article|trial|experiment|text|passage)", re.I)
def user_cut(c):
    s = c.strip()
    return 15 <= len(s) <= 220 and "\n" not in s and not s.endswith(END_OK)
files = [f for f in glob.glob(ROOT + "/*/*__p*.jsonl")]; pick = random.sample(files, 120)
n = 0; cut = 0; noctx = 0; exc = []; exn = []; cutfile = collections.Counter()
for p in pick:
    size = os.path.getsize(p)
    with open(p, "rb") as f:
        f.seek(random.randint(0, max(0, size - 3_000_000))); f.readline()
        for _ in range(4000):
            line = f.readline()
            if not line: break
            n += 1
            if b"<think>" not in line and b"ncomplete" not in line and b"truncated" not in line and b"cut off" not in line and b"context" not in line: continue
            d = json.loads(line); m = d["messages"]
            u = next((x for x in m if x["role"] == "user" and isinstance(x.get("content"), str)), None)
            a = next((x for x in m if x["role"] == "assistant" and isinstance(x.get("content"), str)), None)
            if not u or not a: continue
            head = a["content"][:700]
            if user_cut(u["content"]) and CUT.search(head):
                cut += 1; cutfile[os.path.basename(p).split("__p")[0][-30:]] += 1
                if len(exc) < 300: exc.append((u["content"], head[:150].replace("\n", " | ")))
            elif len(u["content"].strip()) < 200 and NOCTX.search(head[:400]):
                noctx += 1
                if len(exn) < 300: exn.append((u["content"], head[:150].replace("\n", " | ")))
print(f"rows {n:,} | two-signal cut-off hits {cut:,} ({cut/n:.3%}) | no-context variant {noctx:,} ({noctx/n:.3%})")
print("cut-off hits concentrated in:", cutfile.most_common(5))
random.shuffle(exc); random.shuffle(exn)
print("\n=== CUT-OFF (two signals) - 25 random hits")
for u, a in exc[:25]: print(f"  U: {u!r}\n     A: {a!r}")
print("\n=== NO-CONTEXT variant (report only) - 10 random hits")
for u, a in exn[:10]: print(f"  U: {u!r}\n     A: {a!r}")
