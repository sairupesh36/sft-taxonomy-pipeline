"""Detector scan: uniform random rows (1 row per seek), many rules, examples per rule."""
import os, re, sys, json, random, collections
ROOT = "/projects/data/datasets/code_data/sai_rupesh/taxonomy/sft_43_language_wise"
random.seed(int(sys.argv[2]) if len(sys.argv) > 2 else 3)
THINK = re.compile(r"<think>(.*?)</think>", re.S | re.I)
CODE = re.compile(r"```.*?```", re.S)
FUNC = re.compile(r"<functions>|function[- ]calling|function signatures|access to the following (?:apis|functions|tools)|<tools>", re.I)
STOP = set("the a an of to in and or is are was were be for on with as by at from that this it its what how why when which who do does did can could would should i you we they he she my your our their not no yes than then so if but about into over under between during vs versus".split())
def words(t): return {w for w in re.findall(r"[a-z0-9]{3,}", t.lower()) if w not in STOP}
def sample(lang, per_file):
    d = f"{ROOT}/{lang}"; rows = []
    for fn in sorted(os.listdir(d)):
        p = f"{d}/{fn}"; size = os.path.getsize(p)
        with open(p, "rb") as f:
            for _ in range(per_file):
                f.seek(random.randint(0, max(0, size - 1))); f.readline(); l = f.readline()
                if l:
                    try: rows.append((fn, json.loads(l)))
                    except Exception: pass
    return rows
def check(r):
    hits = []; m = r.get("messages")
    if not isinstance(m, list) or not m: return ["fmt_no_messages"]
    if set(r) != {"messages"}: hits.append("fmt_extra_top_keys")
    roles = []
    for x in m:
        if not isinstance(x, dict) or not isinstance(x.get("content"), str) or not x["content"].strip(): hits.append("fmt_empty_or_nonstring_content"); break
    for x in m:
        if isinstance(x, dict): roles.append(x.get("role"))
    if any(x not in ("system", "user", "assistant", "tool") for x in roles): hits.append("fmt_bad_role")
    if "system" in roles[1:]: hits.append("fmt_system_not_first")
    body = [x for x in roles if x != "system"]
    if body and body[0] != "user": hits.append("fmt_first_not_user")
    if roles and roles[-1] != "assistant": hits.append("fmt_last_not_assistant")
    if any(a == b and a in ("user", "assistant") for a, b in zip(roles, roles[1:])): hits.append("fmt_consecutive_same_role")
    if "tool" in roles: hits.append("fmt_tool_role_without_tool_calls_fields")
    txt = lambda role: "\n".join(x["content"] for x in m if isinstance(x, dict) and x.get("role") == role and isinstance(x.get("content"), str))
    u, a, s = txt("user"), txt("assistant"), txt("system")
    tb = THINK.findall(a); no, nc = a.lower().count("<think>"), a.lower().count("</think>")
    if no != nc: hits.append("think_unbalanced")
    if no > 1 or nc > 1: hits.append("think_nested_or_multiple")
    for t in tb:
        alnum = sum(c.isalnum() for c in t)
        if alnum < 15 or (len(t) > 60 and alnum / len(t) < 0.25): hits.append("think_junk_placeholder"); break
    if no and not THINK.sub("", a).strip(): hits.append("think_without_final_answer")
    if re.search(r"</?think>", u, re.I): hits.append("think_tag_in_user")
    if re.search(r"</?answer>", a, re.I): hits.append("answer_tag_wrapper")
    ans = CODE.sub(" ", THINK.sub(" ", a))
    if re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f�]", u + a): hits.append("text_control_or_replacement_char")
    if re.search(r"(?:[.\-_=~*#]\s?){40,}", ans): hits.append("text_punct_run_ge40")
    w = ans.split()
    if len(w) > 80:
        g = collections.Counter(" ".join(w[i:i + 10]) for i in range(len(w) - 9))
        if g.most_common(1)[0][1] >= 5: hits.append("text_repetition_loop")
    if ans.strip() and len(ans) > 30 and sum(c.isalnum() for c in ans) / len(ans) < 0.3: hits.append("text_low_alnum_answer")
    if a.strip() in ("None", "null", "N/A", "[]", "{}", "nan"): hits.append("text_placeholder_answer")
    if u.strip() and u.strip() == a.strip(): hits.append("text_answer_equals_question")
    if (FUNC.search(s) or FUNC.search(u[:800])) and not any(isinstance(x, dict) and (x.get("role") == "tool" or "tool_calls" in x) for x in m):
        hits.append("tool_prompt_but_no_tool_call")
    fu, fa = words(u), words(CODE.sub(" ", THINK.sub(" ", a)))
    if len(fu) >= 8 and len(fa) >= 25 and not (fu & fa): hits.append("qa_zero_word_overlap")
    return hits
if __name__ == "__main__":
    lang = sys.argv[1]; per = int(sys.argv[3]) if len(sys.argv) > 3 else 80
    rows = sample(lang, per); c = collections.Counter(); ex = collections.defaultdict(list); anyhit = 0
    for fn, r in rows:
        h = check(r); anyhit += bool(h)
        for k in set(h):
            c[k] += 1
            if len(ex[k]) < 2: ex[k].append(r)
    n = len(rows); print(f"##### {lang}: {n} rows, rows with >=1 hit: {100*anyhit/n:.1f}%")
    for k, v in sorted(c.items(), key=lambda kv: -kv[1]): print(f"  {k:44s}{v:6d}  {100*v/n:5.2f}%")
    json.dump({k: v for k, v in ex.items()}, open(f"lc_work/scan3_examples_{lang}.json", "w"))
