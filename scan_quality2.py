import os, re, sys, json, random, collections
sys.path.insert(0, '/projects/data/datasets/code_data/sai_rupesh/taxonomy/pylibs')
sys.path.insert(0, '.')
import fasttext
import scan_data_quality as S
random.seed(21)
lid = fasttext.load_model('lang_id_models/lid.176.bin')

THINK_BLOCK = re.compile(r"<think>.*?</think>", re.S | re.I)
CODE_BLOCK = re.compile(r"```.*?```", re.S)
FUNC_SYS = re.compile(r"<functions>|function[- ]calling|function signatures|you (?:have|are given) access to (?:the following )?(?:functions|tools|api)|<tools>|available tools", re.I)

def get(r, role):
    return " ".join(m['content'] for m in r['messages'] if m.get('role') == role and isinstance(m.get('content'), str))

def clean_for_lid(t):
    t = THINK_BLOCK.sub(" ", t); t = CODE_BLOCK.sub(" ", t)
    t = re.sub(r"\$[^$]*\$|\\\(.*?\\\)|\\\[.*?\\\]", " ", t, flags=re.S)
    return re.sub(r"\s+", " ", t).strip()

def lang_conf(t):
    t = t[:1000]
    if len(t) < 60: return None, None
    l, p = lid.predict(t, k=1)
    return l[0].replace('__label__', ''), float(p[0])

def loop_score(t):
    w = t.split()
    if len(w) < 60: return 0
    grams = collections.Counter(" ".join(w[i:i+8]) for i in range(0, len(w)-7))
    return grams.most_common(1)[0][1]

def run(lang, shards=6, per=1000):
    d = os.path.join(S.ROOT, lang)
    files = sorted(f for f in os.listdir(d) if f.endswith('.jsonl')); random.shuffle(files)
    c = collections.Counter(); ex = collections.defaultdict(list)
    for fn in files[:shards]:
        for raw in S.sample_lines(os.path.join(d, fn), per):
            try: r = json.loads(raw)
            except Exception: c['bad_json'] += 1; continue
            c['rows'] += 1
            sysm, u, a = get(r, 'system'), get(r, 'user'), get(r, 'assistant')
            if FUNC_SYS.search(sysm) or FUNC_SYS.search(u[:600]):
                c['functioncalling_prompt'] += 1
                if not any(m.get('role') == 'tool' or 'tool_calls' in m for m in r['messages']):
                    c['functioncalling_prompt_NO_tool_call_or_result'] += 1
                    if len(ex['fc']) < 2: ex['fc'].append((sysm[:150], a[:150]))
            if '<think>' in a:
                c['think'] += 1
                after = THINK_BLOCK.sub('', a).strip()
                if len(after) < 5:
                    c['think_but_no_final_answer'] += 1
                    if len(ex['think_noans']) < 2: ex['think_noans'].append(a[-200:])
                if a.count('<think>') > 1: c['think_nested_or_multi'] += 1
                if not a.lstrip().startswith('<think>'): c['think_not_at_start'] += 1
                tk = re.search(r"<think>(.*?)</think>", a, re.S)
                if tk and len(tk.group(1)) > 8 * max(1, len(after)) and len(tk.group(1)) > 3000: c['think_gt8x_answer_and_gt3k'] += 1
            ua, aa = clean_for_lid(u), clean_for_lid(a)
            lu, pu = lang_conf(ua); la, pa = lang_conf(aa)
            if la and lu and la != lu and pa > 0.6 and pu > 0.6:
                c['user_vs_assistant_language_mismatch'] += 1
                if len(ex['mismatch']) < 3: ex['mismatch'].append((lu, la, u[:80], a[:80]))
            if la and pa < 0.25 and len(aa) > 200:
                c['assistant_long_text_lid_conf_lt25'] += 1
                if len(ex['lowconf']) < 3: ex['lowconf'].append(aa[:160])
            if loop_score(a) >= 6:
                c['degenerate_loop_8gram_ge6'] += 1
                if len(ex['loop']) < 2: ex['loop'].append(a[-200:])
            if a.strip() and not re.search(r"[A-Za-z0-9À-￿]", a):
                c['assistant_no_alnum_at_all'] += 1
                if len(ex['noalnum']) < 3: ex['noalnum'].append(a[:60])
            if re.search(r"([A-Za-z0-9À-￿])\1{24,}", a):
                c['assistant_alnum_char_run_ge25'] += 1
                if len(ex['run']) < 2: ex['run'].append(re.search(r".{30}([A-Za-z0-9À-￿])\1{24,}.{10}", a, re.S).group(0) if re.search(r".{30}([A-Za-z0-9À-￿])\1{24,}.{10}", a, re.S) else '')
            if len(a) > 32000: c['assistant_gt32k_chars'] += 1
            if a.rstrip().endswith('[truncated]'): c['marker_truncated'] += 1
    return c, ex

if __name__ == '__main__':
    for lang in sys.argv[1:]:
        c, ex = run(lang)
        n = c['rows'] or 1
        print(f"\n===== {lang}: {c['rows']} rows =====")
        for k, v in sorted(c.items()):
            if k != 'rows': print(f"  {k:48s}{v:6d} {100*v/n:6.2f}%")
        for k, v in ex.items():
            print(f"  [ex:{k}]")
            for e in v: print("     ", e if isinstance(e, str) else e)
