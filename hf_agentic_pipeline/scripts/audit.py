"""Stage 3 -- structural audit of every output row. Read-only.

Same two role rules as v1's check_role_structure.py (ends on assistant;
system followed by user), plus checks for the bug classes seen in v1 and
while building this dataset -- things the chat template renders WITHOUT
an error while silently losing content:

  unknown_role                  role outside system/user/assistant/tool
  empty_assistant               assistant with no content, no tool_calls,
                                no reasoning (the legacy function_call bug)
  tool_without_call             tool reply with no preceding assistant
                                tool_calls in the same turn
  args_not_json_string          tool_calls[].function.arguments not a JSON
                                string (OpenAI format; standing user rule)
  args_invalid_json             arguments string that doesn't parse
  null_in_struct_args           a null argument value (possible leftover of
                                the parquet struct-union bug; informational)
  system_then_assistant         system immediately followed by assistant
  system_not_first              a system message after position 0
  non_openai_msg_key / non_openai_top_key / content_not_string
                                anything outside plain OpenAI chat format
  tool_without_id               tool reply with no tool_call_id
  answer_empty_after_think      assistant (no tool calls) whose content is only
                                <think>...</think> with nothing after it
  final_answer_empty            the LAST assistant turn has no visible answer
  think_is_problem_creation     <think> is about creating a problem while the
                                user asked to solve one (wrong reasoning)
  think_answer_boxed_mismatch   last \\boxed{} in <think> differs from the one
                                in the answer -- a REVIEW flag, can be notation
"""

import json
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(__file__))
from common import OUT_DIR, REPORT_DIR  # noqa: E402

# optional: audit another folder, e.g. traces_v1_final (read-only either way)
if len(sys.argv) > 1:
    OUT_DIR = sys.argv[1]
REPORT_NAME = sys.argv[2] if len(sys.argv) > 2 else "stage3_audit_report.json"

CORE = {"system", "user", "assistant", "tool"}

import re  # noqa: E402

# --- reasoning-vs-answer checks (user request 2026-09-23) ---
CREATION = re.compile(
    r"^<think>\s*The user wants (me to (create|generate|write|design)|an? (new|original|completely original)"
    r"|a completely original)[^.]{0,200}?\b(problem|question)", re.I)
USER_ASKS_CREATE = re.compile(
    r"\b(create|generate|write|design|compose|make up|come up with)\b.{0,80}\b(problem|question|exercise|puzzle)",
    re.I | re.S)


def last_boxed(s):
    i = s.rfind("\\boxed{")
    if i < 0:
        return None
    j, depth = i + 7, 1
    while j < len(s) and depth:
        depth += {"{": 1, "}": -1}.get(s[j], 0)
        j += 1
    v = s[i + 7:j - 1]
    for a, b in (("\\left", ""), ("\\right", ""), ("\\dfrac", "\\frac"), ("\\tfrac", "\\frac"),
                 ("\\!", ""), ("\\,", ""), ("$", ""), (" ", "")):
        v = v.replace(a, b)
    return v.rstrip(".")
# plain OpenAI chat format (standing user rule): only these keys may appear
OPENAI_KEYS = {
    "system": {"role", "content", "name"},
    "user": {"role", "content", "name"},
    "assistant": {"role", "content", "name", "tool_calls"},
    "tool": {"role", "content", "tool_call_id"},
}


def check(row):
    msgs = row.get("messages") or []
    bad = []
    if not msgs:
        return ["empty_messages"]
    if set(row) - {"messages", "tools"}:
        bad.append("non_openai_top_key")
    for m in msgs:
        if set(m) - OPENAI_KEYS.get(m.get("role"), set(m)):
            bad.append("non_openai_msg_key")
        if not isinstance(m.get("content"), str):
            bad.append("content_not_string")
        if m.get("role") == "tool" and not m.get("tool_call_id"):
            bad.append("tool_without_id")
    roles = [m.get("role") for m in msgs]
    if any(r not in CORE for r in roles):
        bad.append("unknown_role")
    if roles[-1] != "assistant":
        bad.append("does_not_end_with_assistant")
    else:
        last = msgs[-1]
        c = last.get("content") if isinstance(last.get("content"), str) else ""
        if not last.get("tool_calls") and not c.split("</think>")[-1].strip():
            bad.append("final_answer_empty")  # last turn says nothing to the user
    if roles[0] == "system" and len(roles) > 1 and roles[1] == "assistant":
        bad.append("system_then_assistant")
    elif roles[0] == "system" and (len(roles) < 2 or roles[1] != "user"):
        bad.append("system_not_followed_by_user")
    if "system" in roles[1:]:
        bad.append("system_not_first")
    last_user = ""
    for m in msgs:
        c = m.get("content") if isinstance(m.get("content"), str) else ""
        if m.get("role") == "user":
            last_user = c
        elif m.get("role") == "assistant" and "</think>" in c:
            think, ans = c.split("</think>")[0], c.split("</think>")[-1]
            if CREATION.search(c[:600]) and not USER_ASKS_CREATE.search(last_user[:800]):
                bad.append("think_is_problem_creation")   # reasoning is about a different task
            bt, ba = last_boxed(think), last_boxed(ans)
            if bt is not None and ba is not None and bt != ba:
                bad.append("think_answer_boxed_mismatch")  # REVIEW flag: may be notation only
    open_calls = 0
    for m in msgs:
        r = m.get("role")
        if r == "assistant":
            tcs = m.get("tool_calls") or []
            if not tcs and not (m.get("content") or "").strip() and not m.get("reasoning_content"):
                bad.append("empty_assistant")
            elif not tcs and isinstance(m.get("content"), str) and "</think>" in m["content"] \
                    and not m["content"].split("</think>")[-1].strip():
                # only <think>...</think>, no actual answer after it
                bad.append("answer_empty_after_think")
            for tc in tcs:
                a = (tc.get("function") or {}).get("arguments")
                if not isinstance(a, str):
                    bad.append("args_not_json_string")  # OpenAI format wants a string
                else:
                    try:
                        a = json.loads(a)
                    except Exception:
                        bad.append("args_invalid_json")
                        a = None
                    if isinstance(a, dict) and any(v is None for v in a.values()):
                        bad.append("null_in_args")
            open_calls = len(tcs)
        elif r == "tool":
            if open_calls <= 0:
                bad.append("tool_without_call")
            open_calls -= 1
        else:
            open_calls = 0
    return sorted(set(bad))


def audit_file(fname):
    c = Counter()
    ex = {}
    rows = 0
    with open(os.path.join(OUT_DIR, fname)) as f:
        for ln, line in enumerate(f):
            rows += 1
            for b in check(json.loads(line)):
                c[b] += 1
                ex.setdefault(b, ln)
    return fname, rows, dict(c), ex


def main():
    # recursive, so language-split folders work too; "_"-prefixed dirs
    # (_meta, _rejects_*) hold reports/rejects, not data
    files = []
    for root, dirs, fs in os.walk(OUT_DIR):
        dirs[:] = [d for d in dirs if not d.startswith("_")]
        files += [os.path.relpath(os.path.join(root, f), OUT_DIR) for f in fs if f.endswith(".jsonl")]
    files.sort()
    total = Counter()
    per_file = {}
    rows = 0
    with ProcessPoolExecutor(max_workers=int(os.environ.get('AUDIT_WORKERS', 32))) as ex:
        futs = [ex.submit(audit_file, f) for f in files]
        for i, fu in enumerate(as_completed(futs), 1):
            name, n, c, e = fu.result()
            rows += n
            total.update(c)
            if c:
                per_file[name] = {"rows": n, "violations": c, "first_line": e}
            print(f"[{i}/{len(files)}] {name} rows={n:,} {c}", flush=True)
    rep = {"total_files": len(files), "total_rows": rows, "violation_counts": dict(total),
           "files_with_violations": len(per_file), "per_file": per_file}
    json.dump(rep, open(f"{REPORT_DIR}/{REPORT_NAME}", "w"), indent=1)
    print("\nTOTAL rows", f"{rows:,}", dict(total))


if __name__ == "__main__":
    main()
