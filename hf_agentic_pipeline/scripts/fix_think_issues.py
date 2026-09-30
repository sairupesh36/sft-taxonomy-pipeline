"""Fix three reasoning/answer problems found by the audit's empty-answer
checks (2026-09-23), in any folder of OpenAI-format SFT JSONL:

1. SPLIT TURN: an assistant message that is ONLY "<think>...</think>" (no
   tool calls, nothing after </think>) immediately followed by another
   assistant message = one reply the source logged as two messages
   (Toucan: reasoning and answer as separate records). Merged into one.
2. PROBLEM-CREATION REASONING: the <think> block describes CREATING a new
   problem ("The user wants me to create an original algebra problem...")
   while the user asked to SOLVE one -- the generator's own trace from
   writing the question, attached to the answer by mistake (math team's
   generated_* data). The wrong <think> block is removed; the solution
   after </think> is kept (user decision 2026-09-23).
3a. ANSWER INSIDE <think>: if a row's only assistant turn is all <think> but
   ends in a real final answer (\\boxed{}, "Final Answer", "the answer is"),
   the tags are removed so it reads as a normal answer -- NOT dropped.
   (First version dropped these: 18,829 good rows in shaafsalman/hoanganhpham,
   restored from backup and re-fixed.)
3. EMPTY FINAL ANSWER: after 1-2, a LAST assistant turn that is only
   <think>...</think> gives the user no answer -- treated like an empty
   assistant turn (existing cleanup rule): removed, the row trimmed back to
   its last real assistant turn, or dropped if none is left.

    python3 fix_think_issues.py FOLDER [--backup-dir DIR]
Recursive (skips "_" dirs), streaming, atomic per file, only rewrites files
that change; with --backup-dir, each changed file is copied there first.
"""

import json
import os
import re
import shutil
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

CREATION = re.compile(
    r"^<think>\s*The user wants (me to (create|generate|write|design)|an? (new|original|completely original)"
    r"|a completely original)[^.]{0,200}?\b(problem|question)", re.I)
USER_ASKS_CREATE = re.compile(
    r"\b(create|generate|write|design|compose|make up|come up with)\b.{0,80}\b(problem|question|exercise|puzzle)",
    re.I | re.S)


FINAL_MARK = re.compile(r"\\boxed\{|final answer|the answer is|answer:\s*\(?[A-E]\b", re.I)


def after_think(c):
    return c.split("</think>")[-1].strip() if "</think>" in c else c.strip()


def think_only(m):
    c = m.get("content")
    return (m.get("role") == "assistant" and not m.get("tool_calls") and isinstance(c, str)
            and "</think>" in c and not after_think(c))


def fix_row(msgs, c):
    changed = False
    # 1. merge split turns
    out = []
    for m in msgs:
        if out and m.get("role") == "assistant" and think_only(out[-1]):
            prev = out.pop()
            m = dict(m)
            m["content"] = prev["content"].rstrip() + "\n\n" + (m.get("content") or "")
            c["split_turn_merged"] += 1
            changed = True
        out.append(m)
    msgs = out
    # 2. problem-creation reasoning on a solve request
    last_user = ""
    for i, m in enumerate(msgs):
        if m.get("role") == "user" and isinstance(m.get("content"), str):
            last_user = m["content"]
        elif m.get("role") == "assistant" and isinstance(m.get("content"), str) \
                and CREATION.search(m["content"][:600]) and "</think>" in m["content"] \
                and not USER_ASKS_CREATE.search(last_user[:800]):
            msgs[i] = dict(m, content=after_think(m["content"]))
            c["creation_think_removed"] += 1
            changed = True
    # 3a. the ONLY answer sits inside <think> but does contain a final answer
    # (shaafsalman / hoanganhpham put "</think>" after their final answer):
    # unwrap it into a normal visible answer instead of dropping the row.
    asst = [i for i, m in enumerate(msgs) if m.get("role") == "assistant"]
    if len(asst) == 1 and asst[0] == len(msgs) - 1 and think_only(msgs[-1]) \
            and FINAL_MARK.search(msgs[-1]["content"][-3000:]):
        inner = msgs[-1]["content"].replace("<think>", "").replace("</think>", "").strip()
        msgs[-1] = dict(msgs[-1], content=inner)
        c["answer_inside_think_unwrapped"] += 1
        changed = True
    # 3. empty final answer -> trim / drop
    while msgs and msgs[-1].get("role") == "assistant" and think_only(msgs[-1]):
        msgs = msgs[:-1]
        last = max((i for i, m in enumerate(msgs) if m.get("role") == "assistant"), default=-1)
        msgs = msgs[: last + 1]
        c["think_only_final_removed"] += 1
        changed = True
    if not any(m.get("role") == "assistant" for m in msgs):
        return None, True
    return msgs, changed


def fix_file(args):
    path, backup = args
    c = Counter()
    tmp = path + ".thinktmp"
    rows_changed = 0
    with open(path) as fi, open(tmp, "w") as fo:
        for line in fi:
            if "<think>" not in line:
                fo.write(line)
                continue
            d = json.loads(line)
            msgs, changed = fix_row(d["messages"], c)
            if msgs is None:
                c["rows_dropped_no_assistant_left"] += 1
                rows_changed += 1
                continue
            if changed:
                rows_changed += 1
                d["messages"] = msgs
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
        dirs[:] = [d for d in dirs if not d.startswith("_")]
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
