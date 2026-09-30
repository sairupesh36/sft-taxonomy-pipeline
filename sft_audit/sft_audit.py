"""THE audit / validation for all OpenAI-format SFT data. Read-only.

Standing user rule (2026-09-24): ONE audit script. Every newly discovered
problem type is added here as a check, and every run checks ALL datasets in
DATASETS. To add a check: add one entry to CHECKS (name -> what it means)
and the logic in check_row().

    python3 sft_audit.py                  # all datasets
    python3 sft_audit.py hf-agentic-data  # one dataset (by DATASETS key)

Output: reports/audit_<timestamp>.json + reports/audit_<timestamp>.md
(and reports/latest.md). Each check reports how many rows break it, per
dataset and per file, with the first line number as an example.
"""

import gzip
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from multiprocessing import Pool

T = "/projects/data/datasets/code_data/sai_rupesh/taxonomy"
DATASETS = {
    # renamed 2026-09-24 (were sft_43_language_wise_clean / traces_v1_final / hf-agentic-data)
    "sft_hf_domain_data_openai_ready": f"{T}/sft_hf_domain_data_openai_ready",
    "sft_hf_traces_data_openai_ready": f"{T}/sft_hf_traces_data_openai_ready",
    "sft_hf_agentic_data_openai_ready": f"{T}/sft_hf_agentic_data_openai_ready",
    "sft_code_math_text_openai_ready": f"{T}/sft_code_math_text_openai_ready",
    "sft_rl_reference_data_openai_ready": f"{T}/sft_rl_reference_data_openai_ready",
    "sft_posttraining_b1_marker_mapped_openai_ready": f"{T}/sft_posttraining_b1_marker_mapped_openai_ready",
    "sft_posttraining_a_native_openai_ready": f"{T}/sft_posttraining_a_native_openai_ready",
    "sft_posttraining_b2_heuristic_mapped_openai_ready": f"{T}/sft_posttraining_b2_heuristic_mapped_openai_ready",
    # 2026-09-25: user said the FINAL paths for these two are the traces_team folders (datasets_sft was
    # deduped there at 2026-09-24 22:55, after our copy). Old taxonomy copies are no longer used.
    "sft_datasets_sft_openai_ready": "/projects/data/datasets/traces_team/datasets_sft_openai_sft/by_dataset",
    "sft_synthetic_data_openai_ready": "/projects/data/datasets/traces_team/synthetic_data_openai_sft",
}
HERE = os.path.dirname(os.path.abspath(__file__))
REPORTS = f"{HERE}/reports"

# name -> (group, meaning). Order = report order.
CHECKS = {
    # A. the user's two core rules
    "does_not_end_with_assistant": ("A rules", "last message is not from the assistant"),
    "system_then_assistant": ("A rules", "assistant message directly after the system prompt"),
    # B. plain OpenAI chat format
    "empty_messages": ("B format", "row has no messages"),
    "non_openai_top_key": ("B format", "row has a key other than messages / tools"),
    "non_openai_msg_key": ("B format", "message has a key OpenAI chat format doesn't have (e.g. reasoning_content)"),
    "unknown_role": ("B format", "role is not system / user / assistant / tool"),
    "content_not_string": ("B format", "content is not a plain string (list / null / object)"),
    "args_not_json_string": ("B format", "tool_calls[].function.arguments is not a JSON string"),
    "args_invalid_json": ("B format", "arguments string is not valid JSON"),
    "tool_without_id": ("B format", "tool reply has no tool_call_id"),
    "tool_without_call": ("B format", "tool reply with no structured tool call before it"),
    "tool_def_bad_shape": ("B format", "a tools[] entry is not {type: function, function: {name, parameters}}"),
    "system_not_first": ("B format", "system message somewhere after the first message (flag only)"),
    # C. empty answers
    "empty_assistant": ("C empty", "assistant message with no text and no tool call"),
    "answer_empty_after_think": ("C empty", "assistant turn (no tool call) that is only <think>...</think>"),
    "final_answer_empty": ("C empty", "the LAST assistant turn shows the user nothing"),
    "think_not_closed": ("C empty", "assistant turn opens <think> but never closes it -- reasoning was cut off, no answer "
                         "(4,596 rows in posttraining open-r1 codeforces, 2026-09-24)"),
    # D. content quality
    "think_is_problem_creation": ("D quality", "<think> is about CREATING a problem while the user asked to solve one"),
    "special_token_leak": ("D quality", "chat-template special tokens written inside content (<|im_start|>, <|endoftext|>, ...)"),
    "harness_error_text": ("D quality", "assistant reply is agent-harness error/fallback text, not a real answer"),
    "degenerate_answer": ("D quality", "final answer is just 'null' / 'None' / 'NaN' AND the prompt never offered that as an option"),
    "output_limit_loop": ("D quality", "harness 'exceeded the maximum output length' error repeated 3+ times"),
    # E. info only -- reported, never a reason to drop a row
    "null_in_args": ("E info", "a tool argument value is null (real data in JSON-string arguments)"),
    "answer_in_prompt": ("E info", "user message is 'Question: ... Answer: <text>' -- a real answer leak in "
                         "posttraining opencoder_SFT (43,713 rows dropped there, checked 343/343), but in the other "
                         "datasets it is mostly legit tasks (judge a candidate answer, translate a Q&A text, "
                         "'Answer: Let's think step by step') -- so INFO only, never a reason to drop"),
    "selftalk_outside_think": ("E info", "final visible answer (outside <think>) opens like raw R1 self-talk ('I need to "
                               "solve...', 'Okay, so I...', 'Hmm') -- reasoning not inside <think>. INFO only: many hits "
                               "are normal conversational replies ('Okay, here is...', 'Let me explain...')"),
}
# every check except these is a RULE: a row failing it is dropped by sft_clean.py
INFO_CHECKS = {"null_in_args", "answer_in_prompt", "selftalk_outside_think"}

OPENAI_KEYS = {
    "system": {"role", "content", "name"},
    "user": {"role", "content", "name"},
    "assistant": {"role", "content", "name", "tool_calls"},
    "tool": {"role", "content", "tool_call_id"},
}
CREATION = re.compile(
    r"^<think>\s*The user wants (me to (create|generate|write|design)|an? (new|original|completely original)"
    r"|a completely original)[^.]{0,200}?\b(problem|question)", re.I)
USER_ASKS_CREATE = re.compile(
    r"\b(create|generate|write|design|compose|make up|come up with)\b.{0,80}\b(problem|question|exercise|puzzle)",
    re.I | re.S)
SPECIAL_TOKENS = re.compile(
    r"<\|(im_start|im_end|im_system|im_middle|im_user|im_assistant|endoftext|eot_id|begin_of_text|end_of_text"
    r"|start_header_id|end_header_id|eom_id)\|>")  # + Kimi tokens (found in Toucan, 2026-09-24)
HARNESS_TEXTS = ("Technical difficulties. Please continue with the task.",)
OUTPUT_LIMIT = "exceeded the maximum output length"
# answer leaked into the prompt: exactly one "Question:/Problem:" line and ONE
# "Answer:" line after it with real text (few-shot prompts have several, or
# end on an empty "Answer:")
Q_LINE = re.compile(r"(?m)^[ \t]*(Question|Problem)[ \t]*[:：]")
A_LINE = re.compile(r"(?m)^[ \t]*Answer[ \t]*[:：]")


def answer_in_prompt(u):
    qs, ans = Q_LINE.findall(u), list(A_LINE.finditer(u))
    return (len(qs) == 1 and len(ans) == 1 and Q_LINE.search(u).start() < ans[0].start()
            and len(u[ans[0].end():].strip()) >= 20)


SELF_TALK_OPEN = re.compile(
    r"^(I need to (solve|figure|determine|find)|I have (this|a) (problem|question)|Okay, so I|Okay, let me|Okay, I need|Hmm|"
    r"So I've got this|Alright, so I|Let me think)")


def after_think(c):
    return c.split("</think>")[-1] if "</think>" in c else c


def check_row(row):
    bad = set()
    msgs = row.get("messages")
    if not msgs:
        return {"empty_messages"}
    if set(row) - {"messages", "tools"}:
        bad.add("non_openai_top_key")
    for t in row.get("tools") or []:
        fn = t.get("function") if isinstance(t, dict) else None
        if not (isinstance(t, dict) and t.get("type") == "function" and isinstance(fn, dict)
                and isinstance(fn.get("name"), str) and isinstance(fn.get("parameters"), dict)):
            bad.add("tool_def_bad_shape")
            break

    roles = [m.get("role") for m in msgs]
    # A. core rules
    if roles[-1] != "assistant":
        bad.add("does_not_end_with_assistant")
    if roles[0] == "system" and len(roles) > 1 and roles[1] == "assistant":
        bad.add("system_then_assistant")
    if "system" in roles[1:]:
        bad.add("system_not_first")

    last_user = ""
    pending = 0
    limit_hits = 0
    for m in msgs:
        r = m.get("role")
        c = m.get("content")
        if r not in OPENAI_KEYS:
            bad.add("unknown_role")
        elif set(m) - OPENAI_KEYS[r]:
            bad.add("non_openai_msg_key")
        if not isinstance(c, str):
            bad.add("content_not_string")
            c = ""
        if SPECIAL_TOKENS.search(c):
            bad.add("special_token_leak")
        if OUTPUT_LIMIT in c:
            limit_hits += 1
        if r == "user":
            last_user = c
            pending = 0
            if answer_in_prompt(c):
                bad.add("answer_in_prompt")
        elif r == "assistant":
            tcs = m.get("tool_calls") or []
            if "<think>" in c and "</think>" not in c:
                bad.add("think_not_closed")
            if not tcs and not c.strip():
                bad.add("empty_assistant")
            elif not tcs and "</think>" in c and not after_think(c).strip():
                bad.add("answer_empty_after_think")
            if CREATION.search(c[:600]) and "</think>" in c and not USER_ASKS_CREATE.search(last_user[:800]):
                bad.add("think_is_problem_creation")
            if c.strip() in HARNESS_TEXTS:
                bad.add("harness_error_text")
            for tc in tcs:
                a = (tc.get("function") or {}).get("arguments")
                if not isinstance(a, str):
                    bad.add("args_not_json_string")
                    continue
                try:
                    a = json.loads(a)
                except Exception:
                    bad.add("args_invalid_json")
                    continue
                if isinstance(a, dict) and any(v is None for v in a.values()):
                    bad.add("null_in_args")
            pending = len(tcs)
        elif r == "tool":
            if pending <= 0:
                bad.add("tool_without_call")
            pending -= 1
            if not m.get("tool_call_id"):
                bad.add("tool_without_id")
        else:
            pending = 0
    if limit_hits >= 3:
        bad.add("output_limit_loop")

    last = msgs[-1]
    if last.get("role") == "assistant" and not last.get("tool_calls"):
        final = after_think(last.get("content") if isinstance(last.get("content"), str) else "").strip()
        if not final:
            bad.add("final_answer_empty")
        elif SELF_TALK_OPEN.match(final):
            bad.add("selftalk_outside_think")
        elif final in ("null", "None", "NaN"):
            # a real label when the prompt itself offers it ("answer Yes, No or None" -- xP3, 641K rows)
            asked = " ".join(m.get("content") or "" for m in msgs if m.get("role") in ("system", "user"))
            if final.lower() not in asked.lower():
                bad.add("degenerate_answer")
    return bad


def audit_file(args):
    ds, root, rel = args
    c = Counter()
    first = {}
    rows = 0
    p = os.path.join(root, rel)
    with (gzip.open(p, "rt") if p.endswith(".gz") else open(p)) as f:
        for ln, line in enumerate(f):
            if not line.strip():
                continue
            rows += 1
            try:
                found = check_row(json.loads(line))
            except Exception:
                found = {"bad_json"}
            for b in found:
                c[b] += 1
                first.setdefault(b, ln)
    return ds, rel, rows, dict(c), first


def main():
    names = sys.argv[1:] or list(DATASETS)
    jobs = []
    for ds in names:
        root = DATASETS[ds]
        for r, dirs, fs in os.walk(root):
            dirs[:] = [d for d in dirs if not d.startswith("_")]  # _meta, _rejects, _backup...
            jobs += [(ds, root, os.path.relpath(os.path.join(r, f), root)) for f in fs
                     if f.endswith((".jsonl", ".jsonl.gz"))]
    t0 = time.time()
    per_ds = defaultdict(lambda: {"files": 0, "rows": 0, "checks": Counter(), "per_file": {}})
    # multiprocessing.Pool, not ProcessPoolExecutor: submitting ~30K futures at once deadlocked on Python 3.10 (2026-09-25)
    with Pool(int(os.environ.get("WORKERS", 64))) as pool:
        for i, res in enumerate(pool.imap_unordered(audit_file, jobs), 1):
            ds, rel, rows, c, first = res
            d = per_ds[ds]
            d["files"] += 1
            d["rows"] += rows
            d["checks"].update(c)
            if c:
                d["per_file"][rel] = {"rows": rows, "checks": c, "first_line": first}
            if i % 200 == 0 or i == len(jobs):
                print(f"[{i}/{len(jobs)}] files audited", flush=True)

    os.makedirs(REPORTS, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    rep = {"when": stamp, "minutes": round((time.time() - t0) / 60, 1), "checks": CHECKS,
           "datasets": {k: {**v, "checks": dict(v["checks"])} for k, v in per_ds.items()}}
    json.dump(rep, open(f"{REPORTS}/audit_{stamp}.json", "w"), indent=1)

    L = [f"# SFT audit {stamp}", "", "| check | group | " + " | ".join(names) + " |",
         "|---|---|" + "---:|" * len(names)]
    for name, (grp, _) in CHECKS.items():
        L.append(f"| `{name}` | {grp} | " + " | ".join(f"{per_ds[d]['checks'].get(name, 0):,}" for d in names) + " |")
    L.append("| **rows** | | " + " | ".join(f"**{per_ds[d]['rows']:,}**" for d in names) + " |")
    L.append("| files | | " + " | ".join(f"{per_ds[d]['files']:,}" for d in names) + " |")
    L += ["", "## What each check means", ""] + [f"- `{k}` ({g}): {m}" for k, (g, m) in CHECKS.items()]
    md = "\n".join(L) + "\n"
    open(f"{REPORTS}/audit_{stamp}.md", "w").write(md)
    open(f"{REPORTS}/latest.md", "w").write(md)
    print(md)


if __name__ == "__main__":
    main()
