"""Real check: render finished OUTPUT rows through real HF chat templates + tokenizers."""
import os, sys, glob, json, random, collections, re, time
os.environ["HF_HOME"] = "/projects/data/datasets/code_data/sai_rupesh/taxonomy/classifier_experiments/hf_model_cache"
sys.path.insert(0, "/projects/data/datasets/code_data/sai_rupesh/taxonomy/pylibs")
from transformers import AutoTokenizer
random.seed(5)
OUT = "/projects/data/datasets/code_data/sai_rupesh/taxonomy/sft_43_language_wise_clean"
MODELS = {"qwen2.5": "Qwen/Qwen2.5-7B-Instruct", "qwen3": "Qwen/Qwen3-8B", "llama3.1": "unsloth/Llama-3.1-8B-Instruct"}
toks = {}
for k, name in MODELS.items():
    try: toks[k] = AutoTokenizer.from_pretrained(name); print("loaded", k, name, "| has chat_template:", bool(toks[k].chat_template), flush=True)
    except Exception as e: print("FAILED to load", k, name, str(e)[:100], flush=True)
files = [f for f in glob.glob(f"{OUT}/*/*.jsonl")]; print("finished output files:", len(files), "in folders:", sorted({f.split('/')[-2] for f in files})[:12], flush=True)
random.shuffle(files)
gen, tool = [], []
for fn in files:
    if len(gen) >= 3000 and len(tool) >= 900: break
    with open(fn, encoding="utf-8") as f:
        for i, l in enumerate(f):
            if i > 400: break
            if random.random() > 0.08 and '"tool_calls"' not in l: continue
            r = json.loads(l)
            (tool if '"tool_calls"' in l else gen).append(r)
gen, tool = gen[:3000], tool[:900]; print(f"sampled: {len(gen)} general rows, {len(tool)} tool-calling rows", flush=True)
def with_dict_args(row):
    r = json.loads(json.dumps(row))
    for m in r["messages"]:
        for tc in m.get("tool_calls") or []:
            try: tc["function"]["arguments"] = json.loads(tc["function"]["arguments"])
            except Exception: pass
    return r
def render(tok, msgs, tools, gen_prompt=False):
    return tok.apply_chat_template(msgs, tools=tools, tokenize=False, add_generation_prompt=gen_prompt)
def check(tok, row, stats, tag, ex):
    msgs, tools = row["messages"], row.get("tools")
    try: full = render(tok, msgs, tools)
    except Exception as e:
        stats[tag + ":render_error"] += 1; k = type(e).__name__ + ": " + str(e)[:90]; stats["err:" + k] += 1; ex.setdefault(tag + ":render_error", (k, msgs[:2])); return
    stats[tag + ":ok"] += 1
    try:
        prefix = render(tok, msgs[:-1], tools, True)
        if full.startswith(prefix) and len(full) > len(prefix): stats[tag + ":prefix_ok"] += 1
        else:
            stats[tag + ":prefix_MISMATCH"] += 1
            i = next((j for j in range(min(len(full), len(prefix))) if full[j] != prefix[j]), min(len(full), len(prefix))); ex.setdefault(tag + ":prefix_MISMATCH", (full[max(0, i - 60): i + 80], prefix[max(0, i - 60): i + 80]))
    except Exception as e: stats[tag + ":prefix_error"] += 1
    sid = CONTROL[id(tok)]
    leak = False
    for m in msgs:
        c = m.get("content")
        if isinstance(c, str) and c:
            ids = tok(c, add_special_tokens=False)["input_ids"]
            if any(x in sid for x in ids): leak = True; break
    stats[tag + ":content_becomes_special_token"] += leak
    if tag == "general":
        lens.append(len(tok(full, add_special_tokens=False)["input_ids"]))
    if tag.startswith("tool"):
        body = full[full.rfind("<|im_start|>system"):] if "<|im_start|>" in full else full
        for nm, first in CALL.findall(full[full.find("<|im_start|>user"):] if "<|im_start|>user" in full else full):
            stats[tag + (":call_args_are_object" if first == "{" else ":call_args_are_STRING(double-encoded)")] += 1
    stats["_lens_" + tag].append(len(tok(full, add_special_tokens=False)["input_ids"])) if False else None
CONTROL = {}
for _n, _t in toks.items():
    CONTROL[id(_t)] = set(_t.all_special_ids) | {i for s_, i in _t.get_added_vocab().items() if re.match(r"^<\|.*\|>$", s_)}
CALL = re.compile(r'<tool_call>\s*\{"name": "([^"]+)", "arguments": (.)')
for name, tok in toks.items():
    stats = collections.Counter(); ex = {}; t0 = time.time(); lens = []
    for r in gen: check(tok, r, stats, "general", ex)
    for r in tool:
        check(tok, r, stats, "tool[args=JSON string]", ex)
        check(tok, with_dict_args(r), stats, "tool[args=dict]", ex)
    print(f"\n================ {name}  ({time.time()-t0:.0f}s)")
    for tag, n in (("general", len(gen)), ("tool[args=JSON string]", len(tool)), ("tool[args=dict]", len(tool))):
        print(f"  {tag:24s} n={n:5d} | render ok {100*stats[tag+':ok']/n:5.1f}% | assistant-part splittable {100*stats[tag+':prefix_ok']/n:5.1f}% | content->special token {100*stats[tag+':content_becomes_special_token']/n:5.2f}%")
    if lens:
        lens.sort(); q = lambda x: lens[int(x * (len(lens) - 1))]
        print(f"  token length of rendered general rows: p50 {q(.5):,}  p90 {q(.9):,}  p99 {q(.99):,}  | over 4k {100*sum(x>4096 for x in lens)/len(lens):.1f}%  over 8k {100*sum(x>8192 for x in lens)/len(lens):.1f}%  over 32k {100*sum(x>32768 for x in lens)/len(lens):.1f}%")
    for tag in ("tool[args=JSON string]", "tool[args=dict]"):
        o = stats[tag + ":call_args_are_object"]; st_ = stats[tag + ":call_args_are_STRING(double-encoded)"]
        if o + st_: print(f"  {tag:24s} rendered tool calls: arguments as JSON object {o}  |  as a quoted STRING (double-encoded) {st_}")
    errs = [(k[4:], v) for k, v in stats.items() if k.startswith("err:")]
    for k, v in sorted(errs, key=lambda kv: -kv[1])[:3]: print(f"     top error x{v}: {k}")
    for tag in ("general:prefix_MISMATCH", "tool[args=dict]:prefix_MISMATCH"):
        if tag in ex: print(f"     [{tag}] first difference:\n        full  ...{ex[tag][0]!r}\n        prefix...{ex[tag][1]!r}")
r = next((x for x in tool if x.get("tools")), tool[0] if tool else None)
if r and "qwen2.5" in toks:
    for label, row in (("arguments = JSON string", r), ("arguments = dict", with_dict_args(r))):
        t = render(toks["qwen2.5"], row["messages"], row.get("tools")); j = t.find("<|im_start|>assistant"); k = t.find("<tool_call>", j)
        print(f"\nQwen2.5, {label}: has tools field={bool(row.get('tools'))}\n   {t[k:k+200]!r}")
print("\nDONE", flush=True)
