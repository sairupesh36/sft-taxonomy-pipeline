import json, os, random, re, collections, sys
os.environ["HF_HOME"] = "/projects/data/datasets/code_data/sai_rupesh/taxonomy/classifier_experiments/hf_model_cache"
sys.path.insert(0, "/projects/data/datasets/code_data/sai_rupesh/taxonomy/pylibs")
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download
random.seed(31)
R = "/projects/data/datasets/code_data/sai_rupesh/taxonomy/sft_43_language_wise_clean/en/"
FILES = ["phase2_shard21of32_openai_sft_full__p0029", "phase2_shard21of32_openai_sft_full__p0015", "phase2_shard20of32_openai_sft_full__p0094", "phase1_shard2of16_openai_sft_full__p0038", "phase2_shard21of32_openai_sft_full__p0075", "phase2_shard20of32_openai_sft_full__p0080"]
rows = []
for fb in FILES:
    with open(R + fb + ".jsonl", "rb") as f:
        got = [l for l in f if b'"tool_calls"' in l]
    random.shuffle(got); rows += [json.loads(l) for l in got[:100]]
no_tools = [r for r in rows if "tools" not in r]; with_tools = [r for r in rows if "tools" in r]
print(f"tool-call rows sampled {len(rows)} from 6 tool-rich files: WITH tools field {len(with_tools)}, WITHOUT {len(no_tools)}", flush=True)
def called(r): return {tc["function"]["name"] for m in r["messages"] for tc in (m.get("tool_calls") or [])}
kinds = collections.Counter(); inline = 0
for r in no_tools:
    sysm = next((m["content"] for m in r["messages"] if m["role"] == "system"), "")
    n = called(r)
    inline += bool(n) and all(x in sysm for x in n)
    kinds["<tools> block text in system" if "<tools>" in sysm else ("system mentions functions" if re.search(r"function|tool", sysm, re.I) else "no tool info in system")] += 1
print(f"WITHOUT tools field: system text already names every called function in {inline}/{len(no_tools)}; how they describe tools: {dict(kinds)}")
def dict_args(row):
    r = json.loads(json.dumps(row))
    for m in r["messages"]:
        for tc in m.get("tool_calls") or []:
            tc["function"]["arguments"] = json.loads(tc["function"]["arguments"])
    return r
def render(tok, msgs, tools, gp=False): return tok.apply_chat_template(msgs, tools=tools, tokenize=False, add_generation_prompt=gp)
toks = {}
for nm in ("zai-org/GLM-4.5", "zai-org/GLM-4.6"):
    t = AutoTokenizer.from_pretrained(nm); t.chat_template = open(hf_hub_download(nm, "chat_template.jinja")).read(); toks[nm.split("/")[1]] = t
toks["GLM-4-9b-chat-hf"] = AutoTokenizer.from_pretrained("THUDM/glm-4-9b-chat-hf")
toks["Qwen2.5"] = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct"); toks["Qwen3"] = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
multi = [r for r in rows if sum(len(m.get("tool_calls") or []) for m in r["messages"]) > 1 and any(len(m.get("tool_calls") or []) > 1 for m in r["messages"])]
print(f"rows with several tool calls in ONE assistant turn: {len(multi)}\n")
for nm, tok in toks.items():
    out = collections.Counter(); errs = collections.Counter()
    for tag, rs in (("args=JSON string", rows), ("args=dict", [dict_args(r) for r in rows])):
        ok = split = 0
        for r in rs:
            try: full = render(tok, r["messages"], r.get("tools")); ok += 1
            except Exception as e: errs[tag + ": " + type(e).__name__ + ": " + str(e)[:60]] += 1; continue
            try:
                pre = render(tok, r["messages"][:-1], r.get("tools"), True); split += full.startswith(pre) and len(full) > len(pre)
            except Exception: pass
        out[tag] = (ok, split)
    print(f"{nm:18s} " + " | ".join(f"{k}: render {v[0]}/{len(rows)}, splittable {v[1]}/{len(rows)}" for k, v in out.items()))
    for k, v in errs.most_common(2): print(f"      x{v} {k}")
