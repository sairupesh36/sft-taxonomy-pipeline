"""Fixes for what the first run of sft_audit.py found (2026-09-24). Applies
to any OpenAI-format SFT folder; each rule only fires on its exact pattern.

1. tool_def_bad_shape: a tools[] entry stored flat as {name, description,
   parameters} (521M data, 249,371 rows) -> wrapped as the OpenAI shape
   {"type": "function", "function": {...}}.
2. Kimi tool-declare system message: a system message that is only
   "<|im_system|>tool_declare<|im_middle|>[...tools...]<|im_end|>" (Kimi's
   own chat-template text; Toucan, 159K rows) -- the same tools are already
   in the row's "tools" list, so the message is removed.
3. "null" answers: an assistant turn whose visible text after </think> is
   just "null" while <think> holds the real reply (521M terminal-agent rows:
   the JSON command response ended up inside <think>) -> the <think> text
   becomes the answer.
4. countdown answers inside <think>: "<think>...<answer>X</answer></think>"
   with nothing after </think> -> the last <answer>...</answer> is moved
   after </think>.
5. content that is not a string (image content parts; text-only data) and
   rows whose final assistant turn still shows nothing -> row dropped.

    python3 fix_20260924_shapes.py FOLDER --backup-dir DIR
Recursive (skips "_" dirs), streaming, atomic per file, backs up changed files.
"""

import json
import os
import re
import shutil
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

KIMI_DECL = re.compile(r"^\s*<\|im_system\|>tool_declare<\|im_middle\|>(.*)<\|im_end\|>\s*$", re.S)
ANSWER_TAG = re.compile(r"<answer>.*?</answer>", re.S)


def after_think(c):
    return c.split("</think>")[-1] if "</think>" in c else c


def fix_row(d, c):
    changed = False
    # 1. flat tool definitions
    tools = d.get("tools")
    if tools:
        new = []
        for t in tools:
            if isinstance(t, dict) and "function" not in t and "name" in t:
                t = {"type": "function", "function": {k: v for k, v in t.items() if k != "type"}}
                c["tool_def_wrapped"] += 1
                changed = True
            if isinstance(t, dict) and isinstance(t.get("function"), dict) and "parameters" not in t["function"]:
                t["function"]["parameters"] = {"type": "object", "properties": {}}
                changed = True
            new.append(t)
        d["tools"] = new
    msgs = []
    for m in d["messages"]:
        cont = m.get("content")
        # 5a. non-string content -> drop row
        if not isinstance(cont, str):
            c["row_dropped_non_string_content"] += 1
            return None, True
        # 2. Kimi tool-declare system message
        if m.get("role") == "system" and KIMI_DECL.match(cont):
            c["kimi_tool_declare_removed"] += 1
            changed = True
            continue
        if m.get("role") == "assistant" and "</think>" in cont and not m.get("tool_calls"):
            vis = after_think(cont).strip()
            think = cont.split("</think>")[0].replace("<think>", "", 1).strip()
            # 3. "null" visible answer, real reply inside <think>
            if vis == "null" and think:
                m = dict(m, content=think)
                c["null_answer_unwrapped"] += 1
                changed = True
            # 4. <answer> tag trapped inside <think>
            elif not vis and ANSWER_TAG.search(think):
                tag = ANSWER_TAG.findall(think)[-1]
                i = think.rfind(tag)
                think = (think[:i] + think[i + len(tag):]).rstrip()
                m = dict(m, content=f"<think>{think}</think>\n{tag}")
                c["answer_tag_moved_out_of_think"] += 1
                changed = True
        msgs.append(m)
    d["messages"] = msgs
    # 5b. final assistant turn still shows nothing -> drop
    last = msgs[-1] if msgs else {}
    if last.get("role") == "assistant" and not last.get("tool_calls") and not after_think(last.get("content", "")).strip():
        c["row_dropped_final_answer_empty"] += 1
        return None, True
    return d, changed


def fix_file(args):
    path, backup = args
    c = Counter()
    tmp = path + ".shapetmp"
    rows_changed = 0
    with open(path) as fi, open(tmp, "w") as fo:
        for line in fi:
            if not any(s in line for s in ('"tools"', "</think>", "<|im_system|>", '"content": [', '"content": null')):
                fo.write(line)
                continue
            d, changed = fix_row(json.loads(line), c)
            if d is None:
                rows_changed += 1
                continue
            if changed:
                rows_changed += 1
                fo.write(json.dumps(d, ensure_ascii=False) + "\n")
            else:
                fo.write(line)
    if rows_changed:
        if backup:
            os.makedirs(os.path.dirname(backup), exist_ok=True)
            shutil.copy2(path, backup)
        os.replace(tmp, path)
        c["rows_changed"] = rows_changed
    else:
        os.remove(tmp)
    return path, dict(c)


def main():
    top = sys.argv[1]
    bdir = sys.argv[sys.argv.index("--backup-dir") + 1] if "--backup-dir" in sys.argv else None
    jobs = []
    for root, dirs, fs in os.walk(top):
        dirs[:] = [x for x in dirs if not x.startswith("_")]
        for f in fs:
            if f.endswith(".jsonl"):
                p = os.path.join(root, f)
                jobs.append((p, os.path.join(bdir, os.path.relpath(p, top)) if bdir else None))
    tot = Counter()
    with ProcessPoolExecutor(max_workers=int(os.environ.get("WORKERS", 48))) as ex:
        futs = [ex.submit(fix_file, j) for j in sorted(jobs)]
        for i, fu in enumerate(as_completed(futs), 1):
            p, c = fu.result()
            tot.update(c)
            if c:
                print(f"[{i}/{len(jobs)}] {p} {c}", flush=True)
    print("TOTAL", dict(tot), flush=True)


if __name__ == "__main__":
    main()
