"""
Verify the new to_full_text() against the OLD (buggy) one on real raw
messages, before touching map_batch2.py for real. Checks the exact two
problems documented in CLAUDE.md's "Known issue" section:
  1. docs where NEITHER [USER] nor [ASSISTANT] survives truncation
  2. docs where the assistant's answer gets cut off mid-sentence
"""
import json
import glob

MAX_CHARS = 3500


def to_full_text_old(messages, max_chars=MAX_CHARS):
    parts, total = [], 0
    for m in messages:
        seg = f"[{m['role'].upper()}]: {m['content']}"
        if total + len(seg) > max_chars:
            parts.append(seg[:max_chars - total] + " ...[truncated]")
            break
        parts.append(seg)
        total += len(seg)
    return "\n".join(parts)


def to_full_text_new(messages, max_chars=MAX_CHARS):
    system_parts = [f"[{m['role'].upper()}]: {m['content']}" for m in messages if m["role"] == "system"]
    other_parts = [f"[{m['role'].upper()}]: {m['content']}" for m in messages if m["role"] != "system"]

    other_text = "\n".join(other_parts)
    system_text = "\n".join(system_parts)

    if not other_text:
        if len(system_text) > max_chars:
            return system_text[:max_chars] + " ...[truncated]"
        return system_text

    if len(other_text) > max_chars:
        return other_text[:max_chars] + " ...[truncated]"

    remaining = max_chars - len(other_text) - 1
    if system_text and remaining > 0:
        if len(system_text) > remaining:
            system_text = system_text[:remaining] + " ...[truncated]"
        return system_text + "\n" + other_text
    return other_text


def has_real_content(text):
    return "[USER]:" in text or "[ASSISTANT]:" in text


def assistant_cut_off(messages, rendered_text):
    assistant_msgs = [m for m in messages if m["role"] == "assistant"]
    if not assistant_msgs:
        return False
    last_full = assistant_msgs[-1]["content"]
    # crude but matches CLAUDE.md's own check: does the full real answer appear intact?
    return last_full not in rendered_text


def main():
    files = sorted(glob.glob("sft_output_sample_batch*.jsonl"))
    files = [f for f in files if "_mapped" not in f and "_subset" not in f]
    print("checking files:", files)

    n_total = 0
    old_no_content = new_no_content = 0
    old_assistant_cut = new_assistant_cut = 0
    old_system_fully_dropped_but_had_room = 0

    for fn in files:
        with open(fn) as f:
            for line in f:
                d = json.loads(line)
                messages = d["messages"]
                n_total += 1

                old_text = to_full_text_old(messages)
                new_text = to_full_text_new(messages)

                if not has_real_content(old_text):
                    old_no_content += 1
                if not has_real_content(new_text):
                    new_no_content += 1

                if assistant_cut_off(messages, old_text):
                    old_assistant_cut += 1
                if assistant_cut_off(messages, new_text):
                    new_assistant_cut += 1

    print(f"\ntotal docs checked: {n_total}")
    print(f"OLD: no [USER]/[ASSISTANT] visible at all: {old_no_content} ({100*old_no_content/n_total:.2f}%)")
    print(f"NEW: no [USER]/[ASSISTANT] visible at all: {new_no_content} ({100*new_no_content/n_total:.2f}%)")
    print(f"OLD: assistant answer cut off: {old_assistant_cut} ({100*old_assistant_cut/n_total:.2f}%)")
    print(f"NEW: assistant answer cut off: {new_assistant_cut} ({100*new_assistant_cut/n_total:.2f}%)")


if __name__ == "__main__":
    main()
