import os, re, sys, json, random, collections
sys.path.insert(0, "lc_work")
import scan3
random.seed(11)
HERMES_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
OTHER_CALL = re.compile(r"<function=|<\|python_tag\|>|\"function_call\"|\"tool_calls\"|Action Input:|<invoke |\bfunction_call\b", re.I)
DROPPED_MENTION = re.compile(r"\bI (?:need|will|should|have|am going|'ll) (?:to )?call (?:the )?[`'\"]?\w+|\bBy calling (?:this|the)\b|\bcall the [`'\"]?[\w\.]+[`'\"]? function\b", re.I)
REPR_CONTENT = re.compile(r"^\s*\[\s*\{\s*['\"](?:text|type|content|role)['\"]\s*:")
def cls(r):
    m = r.get("messages") or []; out = set()
    keys = set(r) - {"messages"}
    for k in keys: out.add("extra_top_key:" + k)
    for i, x in enumerate(m):
        c = x.get("content") if isinstance(x, dict) else None
        if not isinstance(c, str): continue
        if REPR_CONTENT.match(c): out.add("content_is_python_repr_list")
        if x.get("role") == "assistant":
            has_call = bool(HERMES_CALL.search(c)) or bool(OTHER_CALL.search(c)) or "tool_calls" in x
            if HERMES_CALL.search(c): out.add("tool:hermes_call_in_text")
            if "tool_calls" in x: out.add("tool:structured_tool_calls_field")
            if DROPPED_MENTION.search(c) and not has_call: out.add("tool:assistant_mentions_calling_but_no_call")
        if x.get("role") == "tool":
            out.add("tool:tool_message_present")
            prev = m[i - 1] if i else {}
            pc = prev.get("content", "") if isinstance(prev, dict) else ""
            if not (HERMES_CALL.search(pc) or OTHER_CALL.search(pc) or (isinstance(prev, dict) and "tool_calls" in prev)) or prev.get("role") != "assistant":
                out.add("tool:tool_result_WITHOUT_preceding_call")
            if "tool_call_id" not in x: out.add("tool:tool_msg_has_no_tool_call_id")
            if isinstance(prev, dict) and prev.get("role") == "assistant" and pc.strip() in ("None", "null", ""): out.add("tool:call_turn_content_None_or_empty")
    return out
if __name__ == "__main__":
    for lang, per in (("en", 100), ("hi", 60), ("zh", 60), ("ar", 60), ("es", 60), ("vi", 60), ("kn", 60)):
        rows = scan3.sample(lang, per); c = collections.Counter(); ex = {}
        for fn, r in rows:
            for k in cls(r):
                c[k] += 1; ex.setdefault(k, r)
        n = len(rows); print(f"\n##### {lang}: {n} rows")
        for k, v in sorted(c.items(), key=lambda kv: -kv[1]): print(f"  {k:52s}{v:5d} {100*v/n:5.2f}%")
        json.dump(ex, open(f"lc_work/scan4_examples_{lang}.json", "w"))
