"""Stage 5 -- render a sample of every output file through the real GLM-5.2
chat template, WITH the row's "tools", and check nothing was lost.

v1 only checked "renders without an exception" -- which passed 100% on
data where whole messages were being silently dropped (the template has no
branch for an unknown role). So besides rendering, this checks that every
piece that should be visible actually is:
  * each message's visible content (text after any </think>; the template
    intentionally hides reasoning of earlier turns, so reasoning is not
    checked) -- first 80 chars, whitespace-normalised;
  * each tool call as "<tool_call>NAME";
  * each tool definition's name.
Samples the first 10 and last 10 rows of each file.
"""

import json
import os
import re
import sys
from collections import Counter, defaultdict

from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from common import OUT_DIR, REPORT_DIR  # noqa: E402

TOK_PATH = "/projects/data/hf_cache/hub/models--zai-org--GLM-5.2-FP8"
N = 10
WS = re.compile(r"\s+")


def sample_lines(path, n):
    head = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            head.append(line)
    if len(head) < n:
        return head
    with open(path, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        chunk = min(size, 8_000_000)
        f.seek(size - chunk)
        parts = f.read().decode("utf-8", errors="ignore").split("\n")
    if size > chunk:
        parts = parts[1:]
    return head + [p for p in parts if p.strip()][-n:]


def glm_view(msgs):
    """In-memory adapter only: GLM-5.2's template calls arguments.items(),
    so hand it a COPY with arguments parsed to dicts. Stored data stays in
    OpenAI format (arguments = JSON string)."""
    out = []
    for m in msgs:
        if m.get("tool_calls"):
            m = dict(m)
            m["tool_calls"] = [{**tc, "function": {**tc["function"], "arguments":
                                json.loads(tc["function"]["arguments"]) if isinstance(tc["function"]["arguments"], str)
                                else tc["function"]["arguments"]}} for tc in m["tool_calls"]]
        out.append(m)
    return out


def missing_pieces(row, text):
    flat = WS.sub(" ", text)
    miss = []
    for m in row["messages"]:
        c = m.get("content")
        if isinstance(c, str):
            if m["role"] == "assistant":
                c = c.split("</think>")[-1]
            c = WS.sub(" ", c).strip()[:80].strip()
            if c and c not in flat:
                miss.append(f"content:{m['role']}")
        for tc in m.get("tool_calls") or []:
            if "<tool_call>" + tc["function"]["name"] not in text:
                miss.append("tool_call")
    for t in row.get("tools") or []:
        if t["function"]["name"] not in text:
            miss.append("tool_def")
    return miss


def main():
    tok = AutoTokenizer.from_pretrained(TOK_PATH, trust_remote_code=True)
    files = sorted(f for f in os.listdir(OUT_DIR) if f.endswith(".jsonl"))
    total = ok = 0
    errs = Counter()
    examples = defaultdict(list)
    failing = Counter()
    for fname in files:
        for line in sample_lines(os.path.join(OUT_DIR, fname), N):
            total += 1
            row = json.loads(line)
            try:
                text = tok.apply_chat_template(glm_view(row["messages"]), tools=row.get("tools"), tokenize=False)
            except Exception as e:
                k = f"render_error: {type(e).__name__}: {str(e)[:100]}"
                errs[k] += 1
                failing[fname] += 1
                if len(examples[k]) < 3:
                    examples[k].append(fname)
                continue
            miss = missing_pieces(row, text)
            if miss:
                for k in set(miss):
                    errs["missing_" + k] += 1
                    if len(examples["missing_" + k]) < 5:
                        examples["missing_" + k].append(fname)
                failing[fname] += 1
            else:
                ok += 1
        print(f"{fname}: failing so far {failing[fname]}", flush=True)
    rep = {"total_sampled": total, "ok": ok, "error_counts": dict(errs),
           "failing_files": dict(failing), "examples": examples}
    json.dump(rep, open(f"{REPORT_DIR}/stage5_glm_validation_report.json", "w"), indent=1)
    print(f"\nsampled {total}, fully OK {ok}", dict(errs))


if __name__ == "__main__":
    main()
