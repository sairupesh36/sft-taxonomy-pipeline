import os, sys, json, glob, random, re, collections
os.environ["HF_HOME"] = "/projects/data/datasets/code_data/sai_rupesh/taxonomy/classifier_experiments/hf_model_cache"
sys.path.insert(0, "/projects/data/datasets/code_data/sai_rupesh/taxonomy/pylibs")
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download
random.seed(8)
OUT = "/projects/data/datasets/code_data/sai_rupesh/taxonomy/sft_43_language_wise_clean"
toks = {}
t = AutoTokenizer.from_pretrained("zai-org/GLM-4.5"); t.chat_template = open(hf_hub_download("zai-org/GLM-4.5", "chat_template.jinja")).read(); toks["GLM-4.5"] = t
t2 = AutoTokenizer.from_pretrained("zai-org/GLM-4.6")
try: t2.chat_template = open(hf_hub_download("zai-org/GLM-4.6", "chat_template.jinja")).read(); toks["GLM-4.6"] = t2
except Exception as e: print("GLM-4.6 template file:", str(e)[:80])
toks["GLM-4-9b-chat-hf"] = AutoTokenizer.from_pretrained("THUDM/glm-4-9b-chat-hf")
folders = sorted(d for d in os.listdir(OUT) if not d.startswith("_") and os.path.isdir(f"{OUT}/{d}"))
gen, tool = [], []
for fo in folders:
    fs = [f for f in glob.glob(f"{OUT}/{fo}/*.jsonl") if os.path.getsize(f) > 100_000]; random.shuffle(fs); g = tl = 0
    for fn in fs[:5]:
        size = os.path.getsize(fn)
        with open(fn, "rb") as f:
            f.seek(random.randint(0, max(0, size - 3_000_000))); f.readline()
            for _ in range(500):
                l = f.readline()
                if not l: break
                if b'"tool_calls"' in l:
                    if tl < 10: tool.append(json.loads(l)); tl += 1
                elif g < 60 and random.random() < 0.1: gen.append(json.loads(l)); g += 1
print(f"sampled {len(gen)} general + {len(tool)} tool rows over {len(folders)} folders", flush=True)
def dict_args(row):
    r = json.loads(json.dumps(row))
    for m in r["messages"]:
        for tc in m.get("tool_calls") or []:
            try: tc["function"]["arguments"] = json.loads(tc["function"]["arguments"])
            except Exception: pass
    return r
def render(tok, msgs, tools, gp=False): return tok.apply_chat_template(msgs, tools=tools, tokenize=False, add_generation_prompt=gp)
def run(tok, rows, tag, st, ex):
    for r in rows:
        try: full = render(tok, r["messages"], r.get("tools")); st[tag + ":ok"] += 1
        except Exception as e:
            st[tag + ":render_error"] += 1; k = type(e).__name__ + ": " + str(e)[:90]; st["err:" + k] += 1; ex.setdefault(tag + "err", (k, r["messages"][:1])); continue
        try:
            pre = render(tok, r["messages"][:-1], r.get("tools"), True)
            if full.startswith(pre) and len(full) > len(pre): st[tag + ":prefix_ok"] += 1
            else:
                st[tag + ":prefix_MISMATCH"] += 1
                i = next((j for j in range(min(len(full), len(pre))) if full[j] != pre[j]), min(len(full), len(pre))); ex.setdefault(tag + "mm", (full[max(0, i - 50): i + 90], pre[max(0, i - 50): i + 90]))
        except Exception: st[tag + ":prefix_error"] += 1
SPECIAL_STRS = ["<|system|>", "<|user|>", "<|assistant|>", "<|observation|>", "<|endoftext|>", "[gMASK]", "<sop>", "<|begin_of_image|>", "<|end_of_image|>"]
for name, tok in toks.items():
    st = collections.Counter(); ex = {}
    run(tok, gen, "general", st, ex); run(tok, tool, "tool[args=str]", st, ex); run(tok, [dict_args(r) for r in tool], "tool[args=dict]", st, ex)
    print(f"\n=========== {name}")
    for tag, n in (("general", len(gen)), ("tool[args=str]", len(tool)), ("tool[args=dict]", len(tool))):
        print(f"  {tag:16s} n={n:5d} render ok {100*st[tag+':ok']/n:5.1f}% | assistant part splittable {100*st[tag+':prefix_ok']/n:5.1f}%")
    for k, v in sorted(((k[4:], v) for k, v in st.items() if k.startswith("err:")), key=lambda kv: -kv[1])[:3]: print(f"     error x{v}: {k}")
    for k in ("generalmm", "tool[args=dict]mm", "generalerr"):
        if k in ex: print(f"     [{k}] {ex[k][0]!r}\n        vs {ex[k][1]!r}" if isinstance(ex[k][0], str) else f"     [{k}] {ex[k]}")
# do GLM control strings appear literally inside the data? (they would turn into real special tokens when tokenized)
t0 = toks["GLM-4.5"]; ids = {s: t0.convert_tokens_to_ids(s) for s in SPECIAL_STRS}; print("\nGLM-4.5 control strings that are real special tokens:", {s: i for s, i in ids.items() if i is not None and i != t0.unk_token_id})
hits = collections.Counter(); n = 0
for r in gen + tool:
    n += 1
    txt = "\n".join(m["content"] for m in r["messages"] if isinstance(m.get("content"), str))
    for s in SPECIAL_STRS:
        if s in txt: hits[s] += 1
print(f"literal GLM control strings inside {n} sampled rows:", dict(hits) if hits else "none")
r = tool[0]; print("\nGLM-4.5 render of a tool row, args as dict (first 700 chars):\n", render(toks["GLM-4.5"], dict_args(r)["messages"], r.get("tools"))[:700])
g = next(x for x in gen if any("<think>" in (m.get("content") or "") for m in x["messages"])); print("\nGLM-4.5 render of a row with <think> (last 500 chars):\n", render(toks["GLM-4.5"], g["messages"], g.get("tools"))[-500:])
