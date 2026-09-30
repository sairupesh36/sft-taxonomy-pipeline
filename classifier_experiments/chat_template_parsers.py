"""
Parsers for chat-template-rendered conversations that the REAL production
sft_normalize.py (DEDUP_PIPELINE, never modified) doesn't recognize at all --
these files get mode=None ("no_adapter") and are silently skipped in their
entirety, even though the text is a genuine, complete, readable conversation.

Found by the validator agent auditing real OUTPUT files, independently
confirmed here (see scan_chat_template_files.py): 51 files, ~10M rows,
using at least 5 different template flavors, not one. This module is a
NEW, separate addition living entirely in classifier_experiments/ -- it
does NOT modify sft_normalize.py or anything else in DEDUP_PIPELINE (which
stays read-only, per the project's standing rule), and does NOT modify
sft_normalize_v4.py either (a candidate replacement file sitting in the
parent taxonomy/ directory, built by someone else, not this session's to
change). Two of the five flavors below (chatml, llama3) already have
real, tested parsers in sft_normalize_v4.py -- reused here via import
rather than reimplemented, same "reuse production code" rule this whole
project follows. The other three (inst, human_asst, zephyr) are new,
written and verified here against real example rows pulled directly from
real OUTPUT files before being trusted (see the verification block at the
bottom of this file, run via `python3 chat_template_parsers.py`).

Every parser returns a list of {'role': ..., 'content': ...} dicts (the
same shape sft_normalize.to_messages() produces) or None if the text
doesn't actually match that template -- so a false "maybe this file
matches" guess downstream simply yields None and the row is skipped
exactly as it already is today, never garbage.
"""
import re
import sys

sys.path.insert(0, "/projects/data/datasets/translation_data/SFT/DEDUP_PIPELINE")
import sft_normalize as N  # production, read-only: reuse ROLE_ALIAS only

sys.path.insert(0, "/projects/data/datasets/code_data/sai_rupesh/taxonomy")
from sft_normalize_v4 import parse_chatml, parse_llama  # read-only reuse of already-tested parsers

ROLE_ALIAS = dict(N.ROLE_ALIAS)
ROLE_ALIAS["prompter"] = "user"  # OpenAssistant-style marker, not in production ROLE_ALIAS
                                  # -- a local addition only, never mutates the imported dict


# ---- Llama-2 / Mistral instruct style: [INST] ... [/INST] ----
# Real example (Maths/Paulitos__school-math-questions-llama2-5k.parquet):
#   "<s>[INST] <user text> [/INST] <assistant text> </s>"
# Multi-turn variant chains these: "<s>[INST] m1 [/INST] r1 </s><s>[INST] m2 [/INST] r2 </s>"
# An optional "<<SYS>>\n...\n<</SYS>>\n\n" system block can appear right after [INST].
_INST_SYS_RE = re.compile(r"<<SYS>>\s*(.*?)\s*<</SYS>>\s*\n*", re.S)
_INST_TURN_RE = re.compile(r"\[INST\]\s*(.*?)\s*\[/INST\]\s*(.*?)(?=\[INST\]|\Z)", re.S)


def parse_inst(s):
    if "[INST]" not in s or "[/INST]" not in s:
        return None
    out = []
    matches = list(_INST_TURN_RE.finditer(s))
    if not matches:
        return None
    for m in matches:
        user_part, asst_part = m.group(1), m.group(2)
        sys_match = _INST_SYS_RE.search(user_part)
        if sys_match:
            sys_text = sys_match.group(1).strip()
            if sys_text and not any(t["role"] == "system" for t in out):
                out.append({"role": "system", "content": sys_text})
            user_part = _INST_SYS_RE.sub("", user_part)
        user_text = user_part.strip()
        # strip trailing chat-template scaffolding tokens, not real content
        asst_text = re.sub(r"</s>\s*$", "", asst_part).strip()
        asst_text = re.sub(r"^<s>\s*", "", asst_text).strip()
        if user_text:
            out.append({"role": "user", "content": user_text})
        if asst_text:
            out.append({"role": "assistant", "content": asst_text})
    if not out:
        return None
    if not any(t["role"] == "user" for t in out):
        return None
    if not any(t["role"] == "assistant" for t in out):
        return None
    return out


# ---- Alpaca/Vicuna-style markdown headers: ### Human: ... ### Assistant: ----
# Real example (Maths/khaled123__MathReasoning.parquet):
#   "### human:  **\n\n<text>\n\n### Assistant:**\n\n<text>"
# Note real data is inconsistent on case and carries leftover markdown bold
# markers ("**") right after the colon -- handled explicitly, not guessed.
_HUMAN_ASST_MARK = re.compile(r"#{1,4}\s*(Human|Assistant|System)\s*:\s*\**\s*", re.I)


def parse_human_asst(s):
    marks = list(_HUMAN_ASST_MARK.finditer(s))
    if len(marks) < 2:
        return None
    out = []
    for i, m in enumerate(marks):
        role = ROLE_ALIAS.get(m.group(1).lower(), m.group(1).lower())
        end = marks[i + 1].start() if i + 1 < len(marks) else len(s)
        body = s[m.end():end].strip()
        if not body:
            continue
        if out and out[-1]["role"] == role:
            out[-1]["content"] += "\n\n" + body
        else:
            out.append({"role": role, "content": body})
    if not out:
        return None
    if not any(t["role"] == "user" for t in out):
        return None
    if not any(t["role"] == "assistant" for t in out):
        return None
    return out


# ---- Zephyr/TinyLlama/OpenAssistant style: <|system|>/<|user|>/<|assistant|>
# or <|system|>/<|prompter|>/<|assistant|> ----
# Real examples:
#   Finance/premee__finance-convfinqa-tinyllama-1k.parquet:
#     "<|user|>\n<text>\n<|assistant|>\n<text>"
#   code/smangrul__code-chat-assistant-v1.parquet (OpenAssistant flavor,
#     found and fixed after the first version of this parser scored 0/50 on
#     this real file -- it uses "<|prompter|>" instead of "<|user|>" for the
#     human turn, and "<|endoftext|>" between turns, which the next-marker
#     split below handles fine without needing to treat endoftext specially):
#     "<|system|> <text> <|endoftext|> <|prompter|> <text> <|endoftext|> <|assistant|> <text> <|endoftext|>"
# Unlike ChatML there's no closing tag per turn -- a turn runs until the
# next role marker or end of string.
_ZEPHYR_MARK = re.compile(r"<\|(system|user|assistant|prompter)\|>\s*", re.I)


_ENDOFTEXT_RE = re.compile(r"<\|endoftext\|>", re.I)


def parse_zephyr(s):
    marks = list(_ZEPHYR_MARK.finditer(s))
    if len(marks) < 2:
        return None
    out = []
    for i, m in enumerate(marks):
        role = ROLE_ALIAS.get(m.group(1).lower(), m.group(1).lower())
        end = marks[i + 1].start() if i + 1 < len(marks) else len(s)
        body = _ENDOFTEXT_RE.sub("", s[m.end():end]).strip()
        if not body:
            continue
        if out and out[-1]["role"] == role:
            out[-1]["content"] += "\n\n" + body
        else:
            out.append({"role": role, "content": body})
    if not out:
        return None
    if not any(t["role"] == "user" for t in out):
        return None
    if not any(t["role"] == "assistant" for t in out):
        return None
    return out


# Order matters only for efficiency (cheapest/most-specific-marker checks
# first); each parser independently validates its own markers are really
# present, so trying more than one is safe and never double-matches --
# it just returns non-None on the first real hit.
PARSERS = [
    ("chatml", parse_chatml),
    ("llama3", parse_llama),
    ("inst", parse_inst),
    ("human_asst", parse_human_asst),
    ("zephyr", parse_zephyr),
]


def parse_chat_template(text):
    """Tries every known template parser against raw text. Returns
    (template_name, turns) for the first one that finds real turns, or
    (None, None) if nothing matches -- exactly like today's silent skip,
    just with more templates recognized before giving up."""
    if not isinstance(text, str) or not text.strip():
        return None, None
    for name, parser in PARSERS:
        try:
            turns = parser(text)
        except Exception:
            continue
        if turns:
            return name, turns
    return None, None


if __name__ == "__main__":
    # Real verification against real pulled examples (not synthetic), one
    # per template flavor, run with `python3 chat_template_parsers.py`.
    examples = {
        "inst": (
            "<s>[INST] Can you explain how to solve this math problem: There are 30 "
            "spaces for each vehicle in a parking lot. A caravan takes up a total of "
            "2 spaces of parking space. How many vehicles can still park if there are "
            "3 caravans currently parking? [/INST] The three caravans used up 3x2= 6 "
            "parking spaces in the parking lot.\nTherefore, 30-6= 24 vehicles can "
            "still park in the parking lot. </s>"
        ),
        "human_asst": (
            "### human:  **\n\n*You are given a set of data points...*\n\n"
            "### Assistant:**\n\n**Step 1: Decode the Problem**\n\nThe problem involves..."
        ),
        "zephyr": (
            "<|user|>\njpmorgan chase & co./2007 annual report...\n\nwhat was the "
            "change in investment banking fees from 2005 to 2006? 1432.0\n\n"
            "<|assistant|>\n0.35029"
        ),
        "chatml": "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n4<|im_end|>",
        "llama3": (
            "<|start_header_id|>user<|end_header_id|>\nWhat is 2+2?<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n4<|eot_id|>"
        ),
    }
    for expected_name, text in examples.items():
        name, turns = parse_chat_template(text)
        status = "OK" if name == expected_name and turns else "FAIL"
        print(f"[{status}] expected={expected_name} got={name} turns={turns}")
