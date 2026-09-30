"""Run the hf-agentic-data pipeline stages in order (same shape as v1's).

    python3 run_pipeline.py --list
    python3 run_pipeline.py --run all
    python3 run_pipeline.py --run 3          # one stage
    python3 run_pipeline.py --run 1 NAME...  # stage 1 for some datasets only

Stages:
  1  convert   convert.py    parquet -> {"messages", "tools"} JSONL
               (the v1 Stage-2 "known bug" fixes are built into the
               converter here, so there is no separate fix stage)
  3  audit     audit.py      structural audit, read-only
  4  cleanup   cleanup.py    end on assistant; no assistant right after
                             system; drop empty assistant turns
  (dedup.py exists but is NOT run -- user decision 2026-09-23: no dedup)
  5  validate  validate.py   render through the real GLM-5.2 template with
                             tools, and check nothing was silently lost
  6  report    report.py     plain-English summary + STATS
Stage 3 is re-run after 4 so the final audit describes the final data.
"""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
STAGES = [("1", "convert.py"), ("3", "audit.py"), ("4", "cleanup.py"),
          ("3", "audit.py"), ("5", "validate.py"), ("6", "report.py")]


def run(script, extra=()):
    log = os.path.join(HERE, "..", "logs", script.replace(".py", ".log"))
    print(f"==> {script}  (log: {log})", flush=True)
    with open(log, "w") as f:
        r = subprocess.run([sys.executable, os.path.join(HERE, script), *extra], stdout=f, stderr=subprocess.STDOUT)
    if r.returncode:
        sys.exit(f"{script} failed, see {log}")


def main():
    a = sys.argv[1:]
    if not a or a[0] == "--list":
        print(__doc__)
        return
    if a[0] != "--run" or len(a) < 2:
        sys.exit(__doc__)
    which, extra = a[1], a[2:]
    if which == "all":
        for _, s in STAGES:
            run(s)
    else:
        scripts = [s for n, s in STAGES if n == which]
        if not scripts:
            sys.exit(f"unknown stage {which}")
        run(scripts[0], extra)


if __name__ == "__main__":
    main()
