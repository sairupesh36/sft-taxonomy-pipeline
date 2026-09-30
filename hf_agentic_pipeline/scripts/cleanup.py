"""Stage 4 -- cleanup, in place, streaming (never holds a file in memory).

Same rules as v1's trim_trailing_non_assistant.py:
  * cut everything after the last assistant message (real trajectories
    logged mid-tool-call / mid-user-turn);
  * drop a row that has no assistant message at all.
Plus:
  * remove empty assistant messages (no content, no tool_calls, no
    reasoning) -- nothing to learn from, and they read as a turn where the
    model said nothing;
  * drop rows where the system prompt is followed directly by an assistant
    message. Standing user rule for ALL SFT data (2026-09-23): every row
    must end on assistant, and no assistant may come right after the system
    prompt -- always applied, not optional.
Only rewrites a file if at least one row changed.
"""

import json
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(__file__))
from common import OUT_DIR, REPORT_DIR  # noqa: E402

DROP_SYS_ASST = True  # standing user rule, always on


def clean(row):
    msgs = [m for m in row["messages"]
            if not (m["role"] == "assistant" and not m.get("tool_calls")
                    and not (m.get("content") or "").strip() and not m.get("reasoning_content"))]
    removed_empty = len(msgs) != len(row["messages"])
    last = max((i for i, m in enumerate(msgs) if m["role"] == "assistant"), default=-1)
    if last < 0:
        return None, "dropped_no_assistant"
    if DROP_SYS_ASST and len(msgs) > 1 and msgs[0]["role"] == "system" and msgs[1]["role"] == "assistant":
        return None, "dropped_system_then_assistant"
    trimmed = last < len(msgs) - 1
    msgs = msgs[: last + 1]
    if not trimmed and not removed_empty:
        return row, "unchanged"
    row = dict(row)
    row["messages"] = msgs
    return row, "trimmed" if trimmed else "removed_empty_assistant"


def process(fname):
    path = os.path.join(OUT_DIR, fname)
    tmp = path + ".cleantmp"
    c = Counter()
    with open(path) as fi, open(tmp, "w") as fo:
        for line in fi:
            row, why = clean(json.loads(line))
            c[why] += 1
            if row is not None:
                fo.write(line if why == "unchanged" else json.dumps(row, ensure_ascii=False) + "\n")
    changed = sum(v for k, v in c.items() if k != "unchanged")
    kept = sum(v for k, v in c.items() if not k.startswith("dropped"))
    if not changed:
        os.remove(tmp)
    elif kept == 0:
        os.remove(tmp)
        os.remove(path)
    else:
        os.replace(tmp, path)
    return fname, dict(c)


def main():
    files = sorted(f for f in os.listdir(OUT_DIR) if f.endswith(".jsonl"))
    total = Counter()
    per_file = {}
    with ProcessPoolExecutor(max_workers=32) as ex:
        for i, fu in enumerate(as_completed([ex.submit(process, f) for f in files]), 1):
            name, c = fu.result()
            total.update(c)
            if set(c) != {"unchanged"}:
                per_file[name] = c
            print(f"[{i}/{len(files)}] {name} {c}", flush=True)
    json.dump({"drop_system_then_assistant": DROP_SYS_ASST, "totals": dict(total), "per_file": per_file},
              open(f"{REPORT_DIR}/stage4_cleanup_report.json", "w"), indent=1)
    print("\nTOTAL", dict(total))


if __name__ == "__main__":
    main()
