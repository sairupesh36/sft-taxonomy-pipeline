import sys, json, re, collections
def check(row):
    v = []
    if set(row) - {"messages", "tools"}: v.append("extra_top_keys")
    m = row.get("messages")
    if not isinstance(m, list) or not m: return ["no_messages"]
    ids_open = {}; last_asst = None
    roles = [x.get("role") for x in m]
    if any(r not in ("system", "user", "assistant", "tool") for r in roles): v.append("bad_role")
    body = [r for r in roles if r != "system"]
    if roles.count("system") > 1 or ("system" in roles[1:]): v.append("system_not_first")
    if not body or body[0] != "user": v.append("first_not_user")
    if roles[-1] != "assistant": v.append("last_not_assistant")
    for i, x in enumerate(m):
        c = x.get("content"); r = x.get("role")
        if set(x) - {"role", "content", "tool_calls", "tool_call_id", "name"}: v.append("extra_msg_keys")
        if not isinstance(c, str): v.append("content_not_str"); continue
        if r != "assistant" and not c.strip(): v.append("empty_content")
        if r == "assistant" and not c.strip() and not x.get("tool_calls"): v.append("empty_assistant")
        if "�" in c: v.append("replacement_char")
        if r == "assistant":
            if c.lower().count("<think>") != c.lower().count("</think>"): v.append("think_unbalanced")
            if re.search(r"<think>\s*</think>", c, re.I): v.append("empty_think")
            if "<tool_call>" in c: v.append("hermes_text_left")
            for tc in x.get("tool_calls") or []:
                f = tc.get("function", {})
                if not (isinstance(tc.get("id"), str) and tc.get("type") == "function" and isinstance(f.get("name"), str) and isinstance(f.get("arguments"), str)): v.append("bad_tool_call"); continue
                try: json.loads(f["arguments"])
                except Exception: v.append("arguments_not_json")
                ids_open[tc["id"]] = True
        if r == "tool":
            if x.get("tool_call_id") not in ids_open: v.append("tool_without_matching_call")
            else: ids_open.pop(x["tool_call_id"])
        if i > 0 and r in ("user", "assistant") and m[i-1].get("role") == r: v.append("consecutive_same_role")
    if any(r == "tool" for r in roles) and roles[-1] == "assistant" and m[-1].get("tool_calls") is None and ids_open: v.append("unanswered_call_mid_conversation")
    return v
if __name__ == "__main__":
    n = 0; viol = collections.Counter(); kinds = collections.Counter()
    for line in open(sys.argv[1], encoding="utf-8"):
        row = json.loads(line); n += 1; res = check(row); viol.update(set(res))
        kinds["has_tool_calls"] += any(x.get("tool_calls") for x in row["messages"]); kinds["has_tools_field"] += "tools" in row; kinds["has_system"] += row["messages"][0]["role"] == "system"; kinds["multi_turn"] += sum(x["role"] == "user" for x in row["messages"]) > 1
    print(f"validated {n} rows; violations: {dict(viol) if viol else 'NONE'}; composition: {dict(kinds)}")
