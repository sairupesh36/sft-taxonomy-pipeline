"""Post-pass: make tool_calls[].function.arguments a JSON STRING (the OpenAI
spec) everywhere in a folder of SFT JSONL.

Standing user rule (2026-09-23): stored SFT data is plain OpenAI format;
anything a specific chat template needs (GLM-5.2 wants arguments as a dict
and calls .items() on them) is done later by a separate adapter script,
never in the stored data. v1 and hf-agentic-data had stored dicts for the
GLM template; this reverses that. Lossless: json.loads gives the dict back.

    python3 to_openai_args.py FOLDER [FOLDER...]

Recursive (skips "_"-prefixed dirs), streaming, rewrites only files that
change, atomic per file (tmp + os.replace).
"""

import json
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed


def fix_file(path):
    c = Counter()
    tmp = path + ".argtmp"
    with open(path) as fi, open(tmp, "w") as fo:
        for line in fi:
            if '"tool_calls"' not in line:
                fo.write(line)
                continue
            d = json.loads(line)
            changed = False
            for m in d["messages"]:
                for tc in m.get("tool_calls") or []:
                    fn = tc.get("function") or {}
                    a = fn.get("arguments")
                    if not isinstance(a, str):
                        fn["arguments"] = json.dumps(a if a is not None else {}, ensure_ascii=False)
                        c["calls_converted"] += 1
                        changed = True
            if changed:
                c["rows_changed"] += 1
                fo.write(json.dumps(d, ensure_ascii=False) + "\n")
            else:
                fo.write(line)
    if c["rows_changed"]:
        os.replace(tmp, path)
    else:
        os.remove(tmp)
    return path, dict(c)


def main():
    files = []
    for top in sys.argv[1:]:
        for root, dirs, fs in os.walk(top):
            dirs[:] = [d for d in dirs if not d.startswith("_")]
            files += [os.path.join(root, f) for f in fs if f.endswith(".jsonl")]
    tot = Counter()
    with ProcessPoolExecutor(max_workers=int(os.environ.get("WORKERS", 48))) as ex:
        futs = [ex.submit(fix_file, f) for f in sorted(files)]
        for i, fu in enumerate(as_completed(futs), 1):
            p, c = fu.result()
            tot.update(c)
            if c:
                print(f"[{i}/{len(files)}] {p} {c}", flush=True)
    print("TOTAL", dict(tot), flush=True)


if __name__ == "__main__":
    main()
