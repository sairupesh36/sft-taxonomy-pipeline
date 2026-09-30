"""
Read-only data-quality scan of sft_43_language_wise. Samples rows at random
byte offsets from several shards per language folder (files are multi-GB, so
no full pass), and counts candidate defects so we know which filters matter.
"""
import os
import re
import sys
import json
import random
import collections

ROOT = "/projects/data/datasets/code_data/sai_rupesh/taxonomy/sft_43_language_wise"
ROWS_PER_SHARD = 1500
random.seed(7)

THINK_OPEN = re.compile(r"<think>", re.I)
THINK_CLOSE = re.compile(r"</think>", re.I)
ANSWER_TAG = re.compile(r"</?answer>", re.I)
TOOLCALL_TEXT = re.compile(r"<tool_call>|<\|tool_call|\"tool_calls\"|<function_call>|\bfunction_call\b", re.I)
REPL_CHAR = "�"


def sample_lines(path, n):
    size = os.path.getsize(path)
    out = []
    with open(path, "rb") as f:
        for _ in range(n // 50 + 1):
            f.seek(random.randint(0, max(0, size - 1)))
            f.readline()
            for _ in range(50):
                line = f.readline()
                if not line:
                    break
                out.append(line)
    return out[:n]


def repetition_ratio(text):
    words = text.split()
    if len(words) < 20:
        return 0.0
    grams = [" ".join(words[i:i + 3]) for i in range(len(words) - 2)]
    return 1.0 - len(set(grams)) / len(grams)


def max_char_run(text):
    best = run = 1
    for a, b in zip(text, text[1:]):
        run = run + 1 if a == b else 1
        best = max(best, run)
    return best


def analyze(row, stats):
    msgs = row.get("messages")
    stats["rows"] += 1
    for k in row.keys():
        stats["top_key:" + k] += 1
    if not isinstance(msgs, list) or not msgs:
        stats["no_messages"] += 1
        return
    roles = [m.get("role") for m in msgs if isinstance(m, dict)]
    for m in msgs:
        for k in m.keys():
            stats["msg_key:" + k] += 1
    for r in set(roles):
        stats["has_role:" + str(r)] += 1
    if "user" not in roles:
        stats["no_user_turn"] += 1
    if "assistant" not in roles:
        stats["no_assistant_turn"] += 1
    if roles and roles[0] not in ("system", "user"):
        stats["starts_with_" + str(roles[0])] += 1
    if roles and roles[-1] != "assistant":
        stats["last_turn_not_assistant"] += 1
    for a, b in zip(roles, roles[1:]):
        if a == b and a in ("user", "assistant"):
            stats["consecutive_same_role"] += 1
            break

    contents = [(m.get("role"), m.get("content")) for m in msgs]
    if any(c is None or (isinstance(c, str) and not c.strip()) for r, c in contents if r in ("user", "assistant")):
        stats["empty_or_null_content"] += 1
    if any(isinstance(c, str) and c.strip() in ("None", "null", "[]") for r, c in contents if r == "assistant"):
        stats["assistant_content_None"] += 1
    # tool call lost: assistant "None"/empty directly before a tool message
    for i in range(len(msgs) - 1):
        if msgs[i].get("role") == "assistant" and msgs[i + 1].get("role") == "tool":
            c = msgs[i].get("content")
            if c is None or (isinstance(c, str) and c.strip() in ("", "None", "null")):
                if "tool_calls" not in msgs[i]:
                    stats["tool_call_dropped_before_tool_msg"] += 1
                    break
    if "tool" in roles:
        stats["has_tool_msg"] += 1
        if not any("tool_calls" in m for m in msgs):
            stats["tool_msg_but_no_tool_calls_field"] += 1
    if any("tool_calls" in m for m in msgs):
        stats["has_tool_calls_field"] += 1

    text_all = " ".join(c for r, c in contents if isinstance(c, str))
    asst = " ".join(c for r, c in contents if r == "assistant" and isinstance(c, str))
    user = " ".join(c for r, c in contents if r == "user" and isinstance(c, str))

    n_open, n_close = len(THINK_OPEN.findall(asst)), len(THINK_CLOSE.findall(asst))
    if n_open or n_close:
        stats["think_tags_in_assistant"] += 1
        if n_open != n_close:
            stats["think_unbalanced"] += 1
        if n_open > 1:
            stats["think_multiple_or_nested"] += 1
        if re.search(r"<think>\s*</think>", asst, re.I):
            stats["think_empty_block"] += 1
    if THINK_OPEN.search(user) or THINK_CLOSE.search(user):
        stats["think_tags_in_user"] += 1
    if ANSWER_TAG.search(asst):
        stats["answer_tags_in_assistant"] += 1
    if TOOLCALL_TEXT.search(text_all):
        stats["toolcall_syntax_in_text"] += 1

    if REPL_CHAR in text_all:
        stats["replacement_char"] += 1
    if re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", text_all):
        stats["control_chars"] += 1
    if asst:
        if len(asst.strip()) < 3:
            stats["assistant_tiny_lt3"] += 1
        if repetition_ratio(asst) > 0.5:
            stats["assistant_repetitive_gt50pct"] += 1
        if max_char_run(asst) >= 30:
            stats["assistant_char_run_ge30"] += 1
        if asst.endswith("...[truncated]") or asst.rstrip().endswith("[truncated]"):
            stats["marker_truncated"] += 1
        alnum = sum(ch.isalnum() for ch in asst)
        if len(asst) > 40 and alnum / len(asst) < 0.3:
            stats["assistant_low_alnum_lt30pct"] += 1
        if len(asst) > 30000:
            stats["assistant_gt30k_chars"] += 1
    if user:
        if repetition_ratio(user) > 0.5:
            stats["user_repetitive_gt50pct"] += 1
        if max_char_run(user) >= 30:
            stats["user_char_run_ge30"] += 1
        if len(user.strip()) < 3:
            stats["user_tiny_lt3"] += 1
    if user and asst and user.strip() == asst.strip():
        stats["user_equals_assistant"] += 1


def scan_folder(lang, shards_to_use=6):
    d = os.path.join(ROOT, lang)
    files = sorted(f for f in os.listdir(d) if f.endswith(".jsonl"))
    random.shuffle(files)
    stats = collections.Counter()
    user_prompts = collections.Counter()
    for fn in files[:shards_to_use]:
        for raw in sample_lines(os.path.join(d, fn), ROWS_PER_SHARD):
            try:
                row = json.loads(raw)
            except Exception:
                stats["bad_json"] += 1
                continue
            analyze(row, stats)
            try:
                u = next(m["content"] for m in row["messages"] if m["role"] == "user")
                user_prompts[u[:200]] += 1
            except Exception:
                pass
    dup_rows = sum(c for c in user_prompts.values() if c > 1)
    stats["sample_rows_sharing_identical_user_prompt"] = dup_rows
    return stats


if __name__ == "__main__":
    langs = sys.argv[1:] or ["en"]
    for lang in langs:
        s = scan_folder(lang)
        n = s["rows"] or 1
        print(f"\n===== {lang}: {s['rows']} sampled rows =====")
        for k, v in sorted(s.items(), key=lambda kv: kv[0]):
            if k == "rows":
                continue
            print(f"  {k:48s} {v:7d}  {100 * v / n:6.2f}%")
