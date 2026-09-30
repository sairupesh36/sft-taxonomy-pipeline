#!/usr/bin/env python3
"""
sft_clean_filter.py -- ONE script that turns the language-split SFT jsonl files into clean,
strict OpenAI chat-format SFT data.

Output row:  {"messages": [ {"role": "system|user|assistant|tool", "content": str,
                              "tool_calls": [...]  (assistant only, optional),
                              "tool_call_id": str  (tool only, optional)} ... ],
              "tools": [...]  (optional, only when a tool schema could be parsed)}

Every input row is either
  * REPAIRED and kept   (safe, mechanical fixes -- counted under repairs), or
  * REJECTED with a reason (counted, and a few examples saved in the report), or
  * kept unchanged.
Nothing is silently altered beyond the repairs listed below. Input files are read-only.

Rejections (reason keys)                          Repairs (repair keys)
  parse_error / no_messages / bad_role              repr_parts_joined       (content stored as "[{'type':'text',...}]")
  empty_content / bad_content_type                  repr_conversation_unwrapped (assistant turn = dump of whole chat)
  multimodal_image_part / nested_conversation_in_prompt          duplicate_think_tag_collapsed  (<think><think> / </think></think>)
  think_empty  (nothing between the tags)           answer_tag_unwrapped    (<answer>..</answer>)
  think_junk   (dots / punctuation only)            hermes_tool_calls_converted (<tool_call> text -> tool_calls field)
  think_unbalanced / think_no_answer                think_stripped          (only with --think strip)
                                                    nested_think_repaired   (<think><think>A</think> ans</think> ans)
                                                    empty_think_stripped    (only with --empty-think strip)
  tool_result_without_call / tool_mismatch
  tool_call_unbalanced / tool_call_unparseable / tool_response_missing
  tool_call_dropped  (function prompt, assistant says it calls a function, call missing)
  seq_invalid  (roles out of order / last turn not a plain assistant answer)
  control_or_replacement_char / single_char_run / dot_run / repetition_loop
  placeholder_answer / answer_equals_question / qa_no_word_overlap (strict lexical mismatch)
  prompt_missing_payload (instruction points at "the following text:" but the text is absent)
  too_long / exact_duplicate / prompt_cap

Usage
  python3 sft_clean_filter.py --selftest
  python3 sft_clean_filter.py --in DIR_OR_FILE [--in ...] --sample 200          # dry run: stats + examples, writes nothing
  python3 sft_clean_filter.py --in sft_43_language_wise/hi --out CLEAN_DIR      # real run, mirrors folder layout
Options: --think keep|strip  --empty-think drop|strip  --max-chars N  --max-per-prompt K
         --lexical-mismatch (off by default)  --code-tool NAME  --workers N  --save-rejects
"""
import argparse, ast, collections, glob, hashlib, json, os, random, re, sys, time
from multiprocessing import Pool

ROLES = ("system", "user", "assistant", "tool")
KEEP_MSG_KEYS = ("role", "content", "tool_calls", "tool_call_id", "name")

THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
THINK_BLOCK = re.compile(r"<think>(.*?)</think>", re.S | re.I)
ANSWER_TAG = re.compile(r"</?answer>", re.I)
HERMES_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)
HERMES_RESP = re.compile(r"<tool_response>\s*(.*?)\s*</tool_response>", re.S)
TOOLS_BLOCK = re.compile(r"<tools>\s*(.*?)\s*</tools>", re.S)
FUNC_PROMPT = re.compile(r"<functions>|<tools>|function[- ]calling|function signatures|access to the following (?:apis|functions|tools)", re.I)
DROPPED_MENTION = re.compile(r"\bI (?:need|will|should|have|am going|'ll) (?:to )?call (?:the )?[`'\"]?\w+|\bBy calling (?:this|the)\b|\bcall the [`'\"]?[\w\.]+[`'\"]? function\b", re.I)
CODE_FENCE = re.compile(r"```.*?```", re.S)
CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f�]")
LETTER_RUN = re.compile(r"([^\W\d_])\1{24,}")
DOT_RUN = re.compile(r"(?:\.\s){30,}|\.{300,}|\u2026{100,}")
MISSING_PAYLOAD = re.compile(r"(?:following|below|given)\s+(?:text|passage|snippet|excerpt|article|paragraph|document|context|abstract|sentence|story|extract)s?(?:\s+snippet)?\s*:?\s*$", re.I)
REPR_START = re.compile(r"^\s*\[\s*\{\s*['\"]")
STOP = frozenset("the a an of to in and or is are was were be for on with as by at from that this it its what how why when which who do does did can could would should you we they he she my your our their not yes than then so if but about into over under between during versus".split())
ALIAS = {"human": "user", "gpt": "assistant", "bot": "assistant", "model": "assistant", "function": "tool", "observation": "tool"}


class Reject(Exception):
    def __init__(self, reason):
        self.reason = reason


# ----------------------------------------------------------------------------- content repairs
def _parse_repr(s):
    s = s.strip()
    if len(s) < 8 or len(s) > 600000 or s[-1] != "]" or not REPR_START.match(s):
        return None
    for fn in (ast.literal_eval, json.loads):
        try:
            v = fn(s)
        except Exception:
            continue
        if isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
            return v
    return None


def _parts_to_text(parts):
    out = []
    for p in parts:
        t = str(p.get("type", "text")).lower()
        if "image" in t or "audio" in t or "video" in t:
            raise Reject("multimodal_image_part")
        txt = p.get("text", p.get("content"))
        if isinstance(txt, str) and txt.strip():
            out.append(txt)
    return "\n".join(out)


def normalize_content(c, rep):
    """content -> str (or None, or ('__NESTED__', list) for a whole-conversation dump)."""
    if c is None:
        return None
    if isinstance(c, list):
        if not all(isinstance(x, dict) for x in c):
            raise Reject("bad_content_type")
        rep["repr_parts_joined"] += 1
        return _parts_to_text(c)
    if not isinstance(c, str):
        raise Reject("bad_content_type")
    v = _parse_repr(c) if c.lstrip()[:1] == "[" else None
    if v is not None:
        if all(("role" in x and ("content" in x or "value" in x)) for x in v):
            return ("__NESTED__", v)
        if all(("type" in x or "text" in x) for x in v):
            rep["repr_parts_joined"] += 1
            return _parts_to_text(v)
    return c


def clean_role(r):
    r = str(r).strip().lower()
    return ALIAS.get(r, r)


def normalize_messages(msgs, rep):
    out = []
    for m in msgs:
        if not isinstance(m, dict):
            raise Reject("bad_content_type")
        role = clean_role(m.get("role"))
        if role not in ROLES:
            raise Reject("bad_role")
        c = normalize_content(m.get("content"), rep)
        nm = {"role": role, "content": c}
        for k in ("tool_calls", "tool_call_id", "name"):
            if m.get(k):
                nm[k] = m[k]
        out.append(nm)
    # an assistant turn that is a dump of the whole conversation -> unwrap it
    if out and isinstance(out[-1]["content"], tuple) and out[-1]["role"] == "assistant":
        nested = out[-1]["content"][1]
        lead = [x for x in out[:-1] if x["role"] == "system"]
        rebuilt = []
        for x in nested:
            r = clean_role(x.get("role"))
            cc = x.get("content", x.get("value"))
            if r not in ROLES or not isinstance(cc, str):
                raise Reject("repr_unparseable")
            rebuilt.append({"role": r, "content": cc})
        rep["repr_conversation_unwrapped"] += 1
        return lead + rebuilt
    if any(isinstance(x["content"], tuple) for x in out):
        raise Reject("nested_conversation_in_prompt")
    return out


# ----------------------------------------------------------------------------- think handling
def fix_think(text, role, cfg, rep):
    """returns cleaned text; raises Reject on unusable think structure."""
    low = text.lower()
    if "<think" not in low and "</think" not in low:
        return text
    if role != "assistant":
        return text                                  # e.g. a system prompt that describes the <think> format
    m2 = re.match(r"^\s*<think>\s*<think>(.*?)</think>(.*?)</think>(.*)$", text, re.S | re.I)
    if m2 and m2.group(3).strip() and m2.group(2).strip() == m2.group(3).strip():      # <think><think>A</think> ans</think> ans
        text = "<think>" + m2.group(1) + "</think>" + m2.group(3)
        rep["nested_think_repaired"] += 1
    if re.search(r"(?:<think>\s*){2,}|(?:</think>\s*){2,}", text, re.I):
        text = re.sub(r"(<think>)(?:\s*<think>)+", r"\1", text, flags=re.I)
        text = re.sub(r"(</think>)(?:\s*</think>)+", r"\1", text, flags=re.I)
        rep["duplicate_think_tag_collapsed"] += 1
    if text.lower().count(THINK_OPEN) != text.lower().count(THINK_CLOSE):
        raise Reject("think_unbalanced")
    for m in THINK_BLOCK.finditer(text):
        body = m.group(1)
        if not body.strip() or body.strip().lower() in ("none", "null", "n/a", "nan"):
            if cfg["empty_think"] == "drop":
                raise Reject("think_empty")
        elif sum(ch.isalnum() for ch in body) < 5:
            raise Reject("think_junk")
    if cfg["empty_think"] == "strip":
        before = len(THINK_BLOCK.findall(text))
        text = re.sub(r"<think>\s*(?:none|null|n/a|nan)?\s*</think>", "", text, flags=re.I).strip()
        if len(THINK_BLOCK.findall(text)) != before:
            rep["empty_think_stripped"] += 1
    if cfg["think"] == "strip":
        text = THINK_BLOCK.sub("", text).strip()
        rep["think_stripped"] += 1
    return text


# ----------------------------------------------------------------------------- tools
def parse_call_block(b, cfg):
    """one <tool_call> block -> (name, arguments) . JSON first, then tolerant fallbacks, then raw code."""
    t = re.sub(r"^(?:\\n|\s)+|(?:\\n|\s)+$", "", b)          # stray literal "\n" wrappers
    for fn in (lambda x: json.loads(x, strict=False), ast.literal_eval):
        try:
            j = fn(t)
        except Exception:
            continue
        if isinstance(j, dict) and isinstance(j.get("name"), str):
            return j["name"], j.get("arguments", j.get("parameters", {}))
        break
    if cfg.get("code_tool") and t and t[0] not in "{[":              # code-interpreter style: the block IS the code
        return cfg["code_tool"], {"code": t}
    raise Reject("tool_call_unparseable")


def convert_tools(msgs, rep, cfg):
    out, call_n, i = [], 0, 0
    has_func_prompt = any(m["role"] in ("system", "user") and isinstance(m["content"], str) and FUNC_PROMPT.search(m["content"][:3000]) for m in msgs[:2])
    while i < len(msgs):
        m = msgs[i]
        if m["role"] == "tool":
            raise Reject("tool_result_without_call")   # tool messages are consumed with their assistant call below
        if m["role"] != "assistant" or not isinstance(m["content"], str):
            out.append(m); i += 1; continue
        if m["content"].count("<tool_call>") != m["content"].count("</tool_call>"):
            raise Reject("tool_call_unbalanced")          # template artifact such as a bare "<tool_call>assistant"
        blocks = HERMES_CALL.findall(m["content"]) if "<tool_call>" in m["content"] else []
        if "tool_calls" in m and not blocks:      # already structured: pass through with its tool messages
            out.append(m); i += 1
            while i < len(msgs) and msgs[i]["role"] == "tool":
                out.append(msgs[i]); i += 1
            continue
        if not blocks:
            if has_func_prompt and DROPPED_MENTION.search(m["content"]):
                raise Reject("tool_call_dropped")
            out.append(m); i += 1; continue
        calls, ids = [], []
        for b in blocks:
            name, args = parse_call_block(b, cfg)
            if not isinstance(b, str) or not b.strip().startswith(("{", "\\n{")):
                rep["code_tool_call_converted"] += 1
            call_n += 1; cid = f"call_{call_n}"; ids.append(cid)
            calls.append({"id": cid, "type": "function", "function": {"name": name, "arguments": args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)}})
        text = HERMES_CALL.sub("", m["content"]).strip()
        rep["hermes_tool_calls_converted"] += 1
        out.append({"role": "assistant", "content": text, "tool_calls": calls})
        i += 1
        resp = []
        while i < len(msgs) and msgs[i]["role"] == "tool" and isinstance(msgs[i]["content"], str):
            found = HERMES_RESP.findall(msgs[i]["content"])
            resp += found if found else [msgs[i]["content"].strip()]
            i += 1
        if not resp and i >= len(msgs):
            continue                                 # trailing tool call: valid final assistant turn
        if len(resp) != len(ids):
            raise Reject("tool_response_missing" if not resp else "tool_mismatch")
        for cid, r in zip(ids, resp):
            out.append({"role": "tool", "tool_call_id": cid, "content": r})
    return out


def extract_tools(msgs):
    """tool schemas from a <tools>...</tools> block in the system prompt (skips the instruction sentence
    that merely mentions '<tools> </tools> XML tags')."""
    for m in msgs:
        if m["role"] == "system" and isinstance(m["content"], str):
            for b in TOOLS_BLOCK.finditer(m["content"]):
                txt = b.group(1).strip()
                if not txt:
                    continue
                try:
                    v = json.loads(txt)
                    v = v if isinstance(v, list) else [v]
                except Exception:
                    try:
                        v = [json.loads(l) for l in txt.splitlines() if l.strip()]
                    except Exception:
                        continue
                if v and all(isinstance(x, dict) for x in v):
                    return v
    return None


# ----------------------------------------------------------------------------- structure + text checks
def check_sequence(msgs):
    roles = [m["role"] for m in msgs]
    i = 1 if roles and roles[0] == "system" else 0
    if i >= len(roles) or roles[i] != "user":
        raise Reject("seq_invalid")
    expect, pending = "user", 0
    for m in msgs[i:]:
        r = m["role"]
        if r == "user" and expect == "user":
            expect = "assistant"
        elif r == "assistant" and expect == "assistant":
            pending = len(m.get("tool_calls") or [])
            expect = "tool" if pending else "user"
        elif r == "tool" and expect == "tool":
            pending -= 1
            if pending == 0:
                expect = "assistant"
        else:
            raise Reject("seq_invalid")
    last = msgs[-1]
    if last["role"] != "assistant":
        raise Reject("seq_invalid")
    if not last.get("tool_calls") and not (last["content"] or "").strip():
        raise Reject("empty_content")


def cwords(t):
    return {w for w in re.findall(r"[^\W\d_]{4,}", t.lower()) if w not in STOP}


def check_text(msgs, cfg):
    for m in msgs:
        c = m["content"] or ""
        if not c.strip() and not m.get("tool_calls"):
            raise Reject("empty_content")
        if CTRL.search(c):
            raise Reject("control_or_replacement_char")
        if m["role"] == "assistant":
            plain = CODE_FENCE.sub(" ", c)
            if LETTER_RUN.search(plain):
                raise Reject("single_char_run")
            if DOT_RUN.search(plain):
                raise Reject("dot_run")
            w = plain.split()
            if len(w) >= 150:
                g = collections.Counter(" ".join(w[k:k + 10]) for k in range(0, len(w) - 9))
                gram, cnt = g.most_common(1)[0]
                if cnt >= 12 and sum(1 for x in gram.split() if re.search(r"[^\W\d_]", x)) >= 5:
                    raise Reject("repetition_loop")
            elif len(plain) >= 400 and plain.count(" ") < len(plain) / 15:      # scripts written without spaces (CJK, Thai, ...)
                g = collections.Counter(plain[k:k + 30] for k in range(0, len(plain) - 29, 3))
                win, cnt = g.most_common(1)[0]
                if cnt >= 8 and sum(ch.isalpha() for ch in win) >= 20:
                    raise Reject("repetition_loop")
    asst = [m["content"] for m in msgs if m["role"] == "assistant" and m["content"]]
    user = [m["content"] for m in msgs if m["role"] == "user"]
    if asst and asst[-1].strip() in ("None", "null", "N/A", "nan", "[]", "{}", "NULL"):
        raise Reject("placeholder_answer")
    if user and asst and user[0].strip() == asst[-1].strip():
        raise Reject("answer_equals_question")
    total = sum(len(m["content"] or "") for m in msgs)
    if cfg["max_chars"] and total > cfg["max_chars"]:
        raise Reject("too_long")
    if len(user) == 1 and len(user[0].strip()) < 160 and MISSING_PAYLOAD.search(user[0].strip()):
        raise Reject("prompt_missing_payload")          # "Write an educational piece related to the following text snippet:" + nothing
    if cfg["lexical"] and len(user) == 1 and len(asst) == 1 and not any(m["role"] == "tool" for m in msgs):
        u, a = user[0], THINK_BLOCK.sub(" ", asst[0])
        if "```" not in a and len(u) >= 60 and len(a) >= 150:
            if sum(ch.isalpha() for ch in u) / len(u) > 0.8 and sum(ch.isalpha() for ch in a) / len(a) > 0.8:
                wu, wa = cwords(u), cwords(a)
                if len(wu) >= 6 and len(wa) >= 20 and not (wu & wa):
                    raise Reject("qa_no_word_overlap")


# ----------------------------------------------------------------------------- one row
def process_row(line, cfg, state):
    """returns (row_dict_or_None, reason_or_None, repairs Counter)"""
    rep = collections.Counter()
    raw = line if isinstance(line, bytes) else line.encode("utf-8", "replace")
    key = int.from_bytes(hashlib.blake2b(raw.strip(), digest_size=8).digest(), "little")
    if key in state["seen"]:
        return None, "exact_duplicate", rep
    try:
        d = json.loads(line)
    except Exception:
        return None, "parse_error", rep
    try:
        msgs = d.get("messages") if isinstance(d, dict) else None
        if not isinstance(msgs, list) or not msgs:
            raise Reject("no_messages")
        msgs = normalize_messages(msgs, rep)
        n0 = len(msgs)
        msgs = [m for m in msgs if not (m["role"] == "system" and not (m["content"] or "").strip())]
        if len(msgs) != n0:
            rep["empty_system_dropped"] += 1
        for m in msgs:                                  # think / answer-tag repair per message
            if isinstance(m["content"], str):
                m["content"] = fix_think(m["content"], m["role"], cfg, rep)
                if m["role"] == "assistant" and ANSWER_TAG.search(m["content"]):
                    m["content"] = ANSWER_TAG.sub("", m["content"]).strip(); rep["answer_tag_unwrapped"] += 1
        tools = extract_tools(msgs) or (d.get("tools") if isinstance(d.get("tools"), list) and d.get("tools") else None)
        msgs = convert_tools(msgs, rep, cfg)
        for m in msgs:
            if m["role"] == "assistant" and m.get("content") is None:
                m["content"] = ""
            if m["role"] == "assistant" and not m.get("tool_calls") and THINK_BLOCK.search(m["content"] or "") and not THINK_BLOCK.sub("", m["content"]).strip():
                raise Reject("think_no_answer")
        check_sequence(msgs)
        check_text(msgs, cfg)
        if cfg["max_per_prompt"]:
            first = next((m["content"] for m in msgs if m["role"] == "user"), "")
            pk = hashlib.blake2b(re.sub(r"\s+", " ", first.lower()).encode(), digest_size=8).digest()
            state["prompts"][pk] += 1
            if state["prompts"][pk] > cfg["max_per_prompt"]:
                raise Reject("prompt_cap")
        state["seen"].add(key)
    except Reject as r:
        return None, r.reason, rep
    except Exception:
        return None, "internal_error", rep
    row = {"messages": [{k: m[k] for k in KEEP_MSG_KEYS if k in m} for m in msgs]}
    if tools:
        row["tools"] = tools
    return row, None, rep


# ----------------------------------------------------------------------------- file / driver
def new_state():
    return {"seen": set(), "prompts": collections.Counter()}


def out_name(rel, idx, nchunks):
    return rel if nchunks == 1 else rel[:-6] + f"__p{idx:04d}.jsonl"        # rel always ends with ".jsonl"


def run_chunk(args):
    """one byte-range of one input file -> one output file. Lines are owned by the chunk in which they START."""
    path, root, out_dir, cfg, start, end, idx, nchunks = args
    rel = os.path.relpath(path, root) if os.path.isdir(root) else os.path.basename(path)
    name = out_name(rel, idx, nchunks)
    st = collections.Counter(); reasons = collections.Counter(); repairs = collections.Counter(); ex = collections.defaultdict(list)
    state = new_state(); t0 = time.time(); sample = cfg["sample"]; fout = frej = None; tmp = outp = meta = None
    if out_dir and not sample:
        meta = os.path.join(out_dir, "_meta", name + ".json")
        if os.path.exists(meta):                                    # resume: this chunk is already finished
            d = json.load(open(meta)); return name, d["stats"], d["reasons"], d["repairs"], {}, True
        outp = os.path.join(out_dir, name); os.makedirs(os.path.dirname(outp), exist_ok=True); tmp = outp + ".tmp"
        fout = open(tmp, "w", encoding="utf-8", buffering=1 << 22)
        if cfg["save_rejects"]:
            rp = os.path.join(out_dir, "_rejected", name); os.makedirs(os.path.dirname(rp), exist_ok=True); frej = open(rp, "w", encoding="utf-8")

    def lines():
        if sample:
            rnd = random.Random(hash(rel) % 100003); size = os.path.getsize(path); used = set()
            with open(path, "rb") as f:
                for _ in range(sample * 4):
                    if len(used) >= sample:
                        break
                    f.seek(rnd.randint(0, max(0, size - 1))); f.readline(); off = f.tell(); l = f.readline()
                    if l and off not in used:
                        used.add(off); yield l
        else:
            with open(path, "rb", buffering=1 << 22) as f:
                if start > 0:
                    f.seek(start - 1); f.readline()                   # finish the line that belongs to the previous chunk
                while f.tell() < end:
                    l = f.readline()
                    if not l:
                        break
                    yield l

    for l in lines():
        st["rows_in"] += 1
        row, reason, rep = process_row(l, cfg, state)
        repairs.update(dict.fromkeys(rep, 1))
        if row is not None:
            st["rows_kept"] += 1
            if fout:
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            if rep and len(ex["repaired"]) < 2:
                ex["repaired"].append({"before": l.decode("utf-8", "replace")[:700], "after": json.dumps(row, ensure_ascii=False)[:700]})
        else:
            reasons[reason] += 1
            if len(ex[reason]) < 3:
                ex[reason].append(l.decode("utf-8", "replace")[:600])
            if frej:
                frej.write(json.dumps({"reason": reason, "row": l.decode("utf-8", "replace").strip()}, ensure_ascii=False) + "\n")
    st["seconds"] = round(time.time() - t0, 1)
    if fout:
        fout.close(); os.replace(tmp, outp)
        os.makedirs(os.path.dirname(meta), exist_ok=True)
        json.dump({"stats": dict(st), "reasons": dict(reasons), "repairs": dict(repairs)}, open(meta + ".tmp", "w")); os.replace(meta + ".tmp", meta)
    if frej:
        frej.close()
    return name, dict(st), dict(reasons), dict(repairs), dict(ex), False


def gather(paths):
    files = []
    for p in paths:
        if os.path.isdir(p):
            files += sorted(glob.glob(os.path.join(p, "**", "*.jsonl"), recursive=True))
        else:
            files.append(p)
    return files


def print_report(tot, reasons, repairs):
    n = max(1, tot["rows_in"])
    print(f"\nrows in {tot['rows_in']:,}  kept {tot['rows_kept']:,} ({100*tot['rows_kept']/n:.2f}%)")
    print("REJECTED by reason:")
    for k, v in reasons.most_common():
        print(f"  {k:32s}{v:>14,}  {100*v/n:6.2f}%")
    print("REPAIRED (rows kept, fixed):")
    for k, v in repairs.most_common():
        print(f"  {k:32s}{v:>14,}  {100*v/n:6.2f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", action="append", default=[])
    ap.add_argument("--out"); ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--think", choices=["keep", "strip"], default="keep")
    ap.add_argument("--empty-think", dest="empty_think", choices=["drop", "strip"], default="drop")
    ap.add_argument("--max-chars", type=int, default=200000)
    ap.add_argument("--max-per-prompt", type=int, default=0)
    ap.add_argument("--lexical-mismatch", action="store_true", help="OFF by default: rejects single-turn rows whose answer shares no content word with the question. Unsafe for translation tasks and for scripts written without spaces.")
    ap.add_argument("--code-tool", dest="code_tool", default="python", help="function name used when a <tool_call> block is raw code; '' = reject such rows")
    ap.add_argument("--chunk-mb", type=int, default=256, help="big input files are split into byte ranges of this size so many workers share them")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--save-rejects", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    cfg = dict(think=a.think, empty_think=a.empty_think, max_chars=a.max_chars, max_per_prompt=a.max_per_prompt,
               lexical=a.lexical_mismatch, sample=a.sample, save_rejects=a.save_rejects, code_tool=a.code_tool)
    files = gather(a.inp)
    if not files:
        sys.exit("no input files")
    root = a.inp[0] if len(a.inp) == 1 else os.path.commonpath([os.path.dirname(f) for f in files])
    if not a.sample and not a.out:
        sys.exit("--out is required unless --sample is used")
    cb = a.chunk_mb * (1 << 20); tasks = []
    for f in files:
        size = os.path.getsize(f); n = 1 if a.sample else max(1, -(-size // cb))
        tasks += [(f, root, a.out, cfg, i * cb, (i + 1) * cb if i < n - 1 else 1 << 62, i, n) for i in range(n)]
    print(f"{len(files)} input files -> {len(tasks)} chunks, {min(a.workers, len(tasks))} workers", flush=True)
    tot = collections.Counter(); reasons = collections.Counter(); repairs = collections.Counter(); ex = collections.defaultdict(list)
    t0 = time.time(); done = skipped = 0
    with Pool(min(a.workers, len(tasks))) as pool:
        for name, st, rs, rp, e, resumed in pool.imap_unordered(run_chunk, tasks, chunksize=1):
            done += 1; skipped += resumed
            if not resumed:
                tot.update({k: v for k, v in st.items() if k != "seconds"}); reasons.update(rs); repairs.update(rp)
            for k, v in e.items():
                ex[k] += v[: max(0, 3 - len(ex[k]))]
            if done % 25 == 0 or done == len(tasks):
                el = time.time() - t0
                print(f"[{done}/{len(tasks)} chunks, {skipped} resumed] {time.strftime('%H:%M:%S')}  rows_in {tot['rows_in']:,}  kept {tot['rows_kept']:,}  elapsed {el/60:.1f} min", flush=True)
    if a.out and not a.sample:                                       # totals over EVERY finished chunk, including ones from earlier runs
        tot = collections.Counter(); reasons = collections.Counter(); repairs = collections.Counter()
        for mp in glob.glob(os.path.join(a.out, "_meta", "**", "*.json"), recursive=True):
            d = json.load(open(mp)); tot.update({k: v for k, v in d["stats"].items() if k != "seconds"}); reasons.update(d["reasons"]); repairs.update(d["repairs"])
    print_report(tot, reasons, repairs)
    rep_dir = a.out or "."
    os.makedirs(rep_dir, exist_ok=True)
    rep_path = os.path.join(rep_dir, "_sample_report.json" if a.sample else "_filter_report.json")
    json.dump({"totals": dict(tot), "rejected": dict(reasons), "repaired": dict(repairs), "examples": dict(ex), "settings": {k: v for k, v in cfg.items()}}, open(rep_path, "w"), indent=1, ensure_ascii=False)
    print("report ->", rep_path)


# ----------------------------------------------------------------------------- self test
def selftest():
    base = dict(think="keep", empty_think="drop", max_chars=200000, max_per_prompt=0, lexical=True, code_tool="python")

    def run(msgs, **kw):
        return process_row(json.dumps({"messages": msgs}), dict(base, **kw), new_state())
    U = lambda t: {"role": "user", "content": t}
    A = lambda t: {"role": "assistant", "content": t}
    S = lambda t: {"role": "system", "content": t}
    ok = lambda r: r[0] is not None
    why = lambda r: r[1]
    good = [U("What is the capital of France and why is it famous?"), A("The capital of France is Paris, famous for the Eiffel Tower, the Louvre and its cafe culture.")]
    r = run(good); assert ok(r) and r[0]["messages"] == good, "valid row must pass unchanged"
    dots = "<think> " + ". " * 300 + "</think> \n\n <answer>Equity options expire on the third Friday.</answer>"
    assert why(run([U("How do small cap stocks perform vs. large cap stocks during bear trends?"), A(dots)])) == "think_junk", "user's example must be rejected"
    assert why(run([U("hello there my friend how are you"), A("<think>\n</think>\nHello! I am fine, thank you for asking.")])) == "think_empty"
    r = run([U("hello there my friend how are you"), A("<think>\n</think>\nHello! I am fine, thank you for asking.")], empty_think="strip")
    assert ok(r) and r[0]["messages"][1]["content"] == "Hello! I am fine, thank you for asking."
    r = run([U("Solve 2+2 please and explain."), A("<think><think>Add two and two.</think></think>\nThe answer is 4 because 2 plus 2 equals 4.")])
    assert ok(r) and r[0]["messages"][1]["content"].count("<think>") == 1
    r = run([U("Solve 2+2 please and explain."), A("<think>Add two and two together.</think>\n<answer>The answer is 4.</answer>")])
    assert ok(r) and "<answer>" not in r[0]["messages"][1]["content"]
    r = run([U("Which language is this sentence?"), A("<think><think> It is Arabic script. </think> ar.</think> ar.")]); assert ok(r) and r[0]["messages"][1]["content"] == "<think> It is Arabic script. </think> ar."
    assert why(run([U("Solve 2+2 please and explain."), A("<think>Add two and two together but never finish")])) == "think_unbalanced"
    assert why(run([U("Solve 2+2 please and explain."), A("<think>Add two and two together.</think>")])) == "think_no_answer"
    assert why(run([U("hi there you"), A("None")])) == "placeholder_answer"
    assert why(run([U("hi there you"), A("梦" * 300)])) == "single_char_run"
    assert why(run([U("hi there you"), A("ok " + ". " * 60 + "done")])) == "dot_run"
    assert why(run([U("hi there you"), A("bad � char here")])) == "control_or_replacement_char"
    assert why(run([U("hi there you"), A("fine answer here"), U("another question")])) == "seq_invalid"
    assert why(run([U("hi there you"), U("second user"), A("answer text")])) == "seq_invalid"
    assert why(run([U("hi"), A("")])) == "empty_content"
    assert ok(run([U("add these numbers please"), A("The sum is " + "+ 7 + 0 + 5 + 8 + 8 + 4 + 1 " * 30 + "= 300 in total for the list.")])), "arithmetic expansion is not a loop"
    assert ok(run([U("show me the contents"), A("Table of Contents\n1. Get to Know a Spider" + "." * 60 + "1\n2. Can Spiders Hurt Me?" + "." * 60 + "9")])), "TOC dot leaders are legitimate"
    assert why(run([U("hello there my friend how are you"), A("<think>None</think>\nHello! I am fine, thank you for asking.")])) == "think_empty"
    r = run([{"role": "system", "content": "[{'type': 'text', 'content': ''}]"}, U("hello there my friend"), A("Hello! I am fine, thank you.")]); assert ok(r) and r[0]["messages"][0]["role"] == "user"
    assert why(run([U("say it"), A("This is false, so we need to increase n and check again. " * 60)])) == "repetition_loop"
    assert why(run([U("Write an educational piece related to the following text snippet:"), A("Let us begin by exploring the concept of angular momentum in physics.")])) == "prompt_missing_payload"
    assert ok(run([U("Write an educational piece related to the following text snippet:\n\nAngular momentum is conserved."), A("Let us begin by exploring the concept of angular momentum in physics.")]))
    r = run([U("[{'type': 'text', 'content': 'What is two plus two in maths?'}]"), A("[{'type': 'text', 'content': 'It is four.'}]")])
    assert ok(r) and r[0]["messages"][0]["content"] == "What is two plus two in maths?"
    assert why(run([U("[{'text': 'click here', 'type': 'text'}, {'text': None, 'type': 'image'}]"), A("clicked the button")])) == "multimodal_image_part"
    nested = "[{'content': 'Namaste, tell me about Tunisia please', 'role': 'user'}, {'content': 'Tunisia is a country in North Africa.', 'role': 'assistant'}]"
    r = run([U("Namaste, tell me about Tunisia please"), A(nested)])
    assert ok(r) and len(r[0]["messages"]) == 2 and r[0]["messages"][1]["content"] == "Tunisia is a country in North Africa."
    sysm = S("You are a function calling AI model. <tools>[{\"name\": \"get_weather\", \"parameters\": {}}]</tools>")
    call = A("<think>Need the weather.</think>\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>")
    resp = {"role": "tool", "content": "<tool_response>\n{\"temp\": 20}\n</tool_response>"}
    r = run([sysm, U("What is the weather in Paris today?"), call, resp, A("It is 20 degrees in Paris.")])
    assert ok(r), why(r)
    m = r[0]["messages"]
    assert m[2]["tool_calls"][0]["function"]["name"] == "get_weather" and m[3]["tool_call_id"] == m[2]["tool_calls"][0]["id"] and m[3]["content"] == '{"temp": 20}' and r[0]["tools"][0]["name"] == "get_weather"
    assert why(run([U("What is the weather in Paris today?"), A("I checked."), resp, A("It is 20 degrees.")])) in ("tool_result_without_call", "seq_invalid")
    assert why(run([S("You are a function calling AI model with <functions></functions> XML tags."), U("What is the weather in Paris today?"), A("To get the weather I need to call the 'get_weather' function. By calling this function I will know.")])) == "tool_call_dropped"
    assert why(run([U("How do small cap stocks behave when markets fall hard for many months?"), A("Equity options actually expire on the Saturday after the third Friday of every month, according to exchange rules published online by exchanges. Their published documentation explains settlement procedures clearly.")])) == "qa_no_word_overlap"
    codecall = A("<tool_call>\nx = 2 + 2\nprint(x)\n</tool_call>")
    r = run([U("Compute two plus two with code please."), codecall, {"role": "tool", "content": "<tool_response>\n4\n</tool_response>"}, A("The answer is 4.")])
    assert ok(r), why(r)
    assert r[0]["messages"][1]["tool_calls"][0]["function"]["name"] == "python" and json.loads(r[0]["messages"][1]["tool_calls"][0]["function"]["arguments"])["code"].startswith("x = 2")
    assert why(run([U("Compute two plus two with code please."), codecall, {"role": "tool", "content": "4"}, A("It is 4.")], code_tool="")) == "tool_call_unparseable"
    r = run([U("Compute two plus two with code please."), A("Let me call it.\n<tool_call>\n{\"name\": \"calc\", \"arguments\": {\"q\": \"2+2\"}}\n</tool_call>")]); assert ok(r) and r[0]["messages"][-1]["tool_calls"]
    sys2 = S("You may call functions inside <tools> </tools> XML tags.\n<tools>\n[{\"type\": \"function\", \"function\": {\"name\": \"get_weather\"}}]\n</tools>")
    r = run([sys2, U("What is the weather in Paris today?"), call, resp, A("It is 20 degrees.")]); assert ok(r) and r[0]["tools"][0]["function"]["name"] == "get_weather"
    assert why(run([U("Compute two plus two with code please."), A("Sure.\n<tool_call>assistant\nThe answer is four and here is why it works well.")])) == "tool_call_unbalanced"
    st = new_state()
    assert process_row(json.dumps({"messages": good}), dict(base), st)[0] and process_row(json.dumps({"messages": good}), dict(base), st)[1] == "exact_duplicate"
    c2 = dict(base, max_per_prompt=1); st = new_state()
    g2 = [good[0], A("A different but fine answer about Paris, the capital of France, a city known for art.")]
    assert process_row(json.dumps({"messages": good}), c2, st)[0] and process_row(json.dumps({"messages": g2}), c2, st)[1] == "prompt_cap"
    print("SELFTEST OK")


if __name__ == "__main__":
    main()
