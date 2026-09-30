"""Sampled SFT stats that need a model per row: REAL GLM-5.2 token counts and
language. Read-only. (No domain/task classifier -- user decision 2026-09-24.)

    python3 sft_stats_sample.py [--rate 0.01] [DATASET ...]

Takes every (1/rate)-th row of every file (deterministic), so totals are
estimated as sample_total / rate. --rate 1.0 = every row (slow).
  * tokens (GLM-5.2 tokenizer = the model we train):
      text_tokens      = tokens of all message text (+ tool-call arguments)
      training_tokens  = the whole row rendered by the GLM chat template
                         (special tokens, tools list included) -- what the
                         model actually reads; use THIS for max length
      assistant_tokens = tokens of assistant turns only (what SFT learns)
  * language: fastText lid.176 on the first user message
Output: reports/sft_stats_sample_<timestamp>.xlsx
"""
import gzip, json, os, sys, time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd

T = "/projects/data/datasets/code_data/sai_rupesh/taxonomy"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sft_audit import DATASETS  # noqa: E402

TOK_PATH = "/projects/data/hf_cache/hub/models--zai-org--GLM-5.2-FP8"
LID = f"{T}/lang_id_models/lid.176.bin"
LEN_BUCKETS = [(0, 2048), (2049, 4096), (4097, 8192), (8193, 16384), (16385, 32768),
               (32769, 65536), (65537, 131072), (131073, None)]
W = {}


def init():
    import warnings; warnings.filterwarnings("ignore")
    from transformers import AutoTokenizer  # system transformers (tokenizer only, no torch needed)
    W["tok"] = AutoTokenizer.from_pretrained(TOK_PATH, trust_remote_code=True)
    # fasttext only exists in pylibs; pylibs also holds an OLD transformers/torch that must not
    # shadow the system ones, so it is on sys.path only while fasttext is imported
    sys.path.insert(0, f"{T}/pylibs")
    import fasttext
    sys.path.remove(f"{T}/pylibs")
    fasttext.FastText.eprint = lambda *a, **k: None
    W["lid"] = fasttext.load_model(LID)


def glm_view(msgs):
    out = []
    for m in msgs:
        if m.get("tool_calls"):
            m = dict(m)
            m["tool_calls"] = [{**tc, "function": {**tc["function"], "arguments":
                                json.loads(tc["function"]["arguments"])}} for tc in m["tool_calls"]]
        out.append(m)
    return out


def ntok(s):
    return len(W["tok"](s, add_special_tokens=False)["input_ids"]) if s else 0


def lbucket(v):
    for lo, hi in LEN_BUCKETS:
        if v >= lo and (hi is None or v <= hi):
            return f"{lo}+" if hi is None else f"{lo}-{hi}"


def one(args):
    ds, root, rel, step = args
    p = os.path.join(root, rel)
    C = Counter()
    with (gzip.open(p, "rt") if p.endswith(".gz") else open(p)) as f:
        for i, line in enumerate(f):
            if i % step or not line.strip():
                continue
            d = json.loads(line)
            msgs = d["messages"]
            text = sum(ntok(m.get("content") or "") for m in msgs) + sum(
                ntok(tc["function"]["arguments"]) for m in msgs for tc in (m.get("tool_calls") or []))
            asst = sum(ntok(m.get("content") or "") + sum(ntok(tc["function"]["arguments"]) for tc in (m.get("tool_calls") or []))
                       for m in msgs if m["role"] == "assistant")
            try:
                rendered = W["tok"].apply_chat_template(glm_view(msgs), tools=d.get("tools"), tokenize=False)
                train = ntok(rendered)
            except Exception:
                train = text
                C["template_render_error"] += 1
            u = next((m["content"] for m in msgs if m["role"] == "user"), "")[:1000].replace("\n", " ")
            lang = W["lid"].predict(u or " ")[0][0].replace("__label__", "")
            C["rows"] += 1; C["text"] += text; C["train"] += train; C["asst"] += asst
            C[("len", lbucket(train), "rows")] += 1; C[("len", lbucket(train), "tok")] += train
            for k, v in (("lang", lang),):
                C[(k, v, "rows")] += 1; C[(k, v, "tok")] += train
    return ds, C


def main():
    a = sys.argv[1:]
    rate = 0.01
    if "--rate" in a:
        i = a.index("--rate"); rate = float(a[i + 1]); a = a[:i] + a[i + 2:]
    step = max(1, round(1 / rate))
    names = a or [d for d in DATASETS if "posttraining" not in d]
    jobs = []
    for ds in names:
        root = DATASETS[ds]
        for r, dirs, fs in os.walk(root):
            dirs[:] = [x for x in dirs if not x.startswith("_")]
            jobs += [(ds, root, os.path.relpath(os.path.join(r, f), root), step) for f in fs
                     if f.endswith((".jsonl", ".jsonl.gz"))]
    jobs.sort(key=lambda j: -os.path.getsize(os.path.join(j[1], j[2])))
    per = defaultdict(Counter)
    with ProcessPoolExecutor(max_workers=int(os.environ.get("WORKERS", 48)), initializer=init) as ex:
        for i, fu in enumerate(as_completed([ex.submit(one, j) for j in jobs]), 1):
            ds, C = fu.result(); per[ds].update(C)
            if i % 200 == 0 or i == len(jobs):
                print(f"[{i}/{len(jobs)}] files", flush=True)
    B = 1e9
    tok = [{"dataset": ds, "sampled_rows": C["rows"], "sample_rate": rate,
            "est_rows": round(C["rows"] / rate),
            "est_text_tokens_B": round(C["text"] / rate / B, 2),
            "est_training_tokens_B": round(C["train"] / rate / B, 2),
            "est_assistant_tokens_B": round(C["asst"] / rate / B, 2),
            "assistant_share_%": round(100 * C["asst"] / max(C["text"], 1), 1),
            "avg_training_tokens_per_row": round(C["train"] / max(C["rows"], 1)),
            "template_render_errors": C["template_render_error"]} for ds, C in per.items()]

    def dist(kind, top=None):
        out = []
        for ds, C in per.items():
            items = sorted(((k[1], C[k]) for k in C if isinstance(k, tuple) and k[0] == kind and k[2] == "rows"),
                           key=lambda x: -x[1])
            if kind == "len":
                order = [f"{lo}+" if hi is None else f"{lo}-{hi}" for lo, hi in LEN_BUCKETS]
                items = sorted(items, key=lambda x: order.index(x[0]))
            rest_r = rest_t = 0
            for j, (lab, r) in enumerate(items):
                t = C[(kind, lab, "tok")]
                if top and j >= top:
                    rest_r += r; rest_t += t; continue
                out.append({"dataset": ds, kind: lab, "est_rows": round(r / rate), "pct_rows": round(100 * r / C["rows"], 2),
                            "est_training_tokens_B": round(t / rate / B, 3), "pct_tokens": round(100 * t / max(C["train"], 1), 2)})
            if rest_r:
                out.append({"dataset": ds, kind: "(all others)", "est_rows": round(rest_r / rate),
                            "pct_rows": round(100 * rest_r / C["rows"], 2), "est_training_tokens_B": round(rest_t / rate / B, 3),
                            "pct_tokens": round(100 * rest_t / max(C["train"], 1), 2)})
        return pd.DataFrame(out)

    notes = pd.DataFrame({"note": [
        f"Sampled every {step}th row of every file (rate {rate}); est_* = sample value / rate.",
        "training_tokens = the full row rendered by the GLM-5.2 chat template (incl. special tokens and the tools list) -- use it for the max-sequence-length decision.",
        "assistant_tokens = tokens in assistant turns (text + tool-call arguments) = what SFT trains on; assistant_share_% = assistant / text tokens.",
        "language = fastText lid.176 on the first user message (first 1000 chars).",
    ]})
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = os.path.join(HERE, "reports", f"sft_stats_sample_{stamp}.xlsx")
    with pd.ExcelWriter(out, engine="openpyxl") as xw:
        pd.DataFrame(tok).to_excel(xw, sheet_name="Tokens (GLM)", index=False)
        dist("len").to_excel(xw, sheet_name="Token length (GLM)", index=False)
        dist("lang", 25).to_excel(xw, sheet_name="Language", index=False)
        notes.to_excel(xw, sheet_name="Notes", index=False)
        for ws in xw.book.worksheets:
            for col in ws.columns:
                w = max((len(str(c.value)) for c in col[:300] if c.value is not None), default=8)
                ws.column_dimensions[col[0].column_letter].width = min(max(10, w + 2), 70)
            ws.freeze_panes = "A2"
    print(pd.DataFrame(tok).to_string(index=False)); print("written", out)


if __name__ == "__main__":
    main()
