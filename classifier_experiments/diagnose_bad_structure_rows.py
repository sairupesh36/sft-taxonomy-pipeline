"""
READ-ONLY diagnostic: measures how much real OUTPUT data has no recognizable
single/multi-turn conversation structure, and how much of that is currently
being silently skipped by sample_from_sft_output.py's own error handling
(never touches/writes anything under OUTPUT).

Answers the user's question with real numbers instead of guessing: of the
rows in OUTPUT, how many (a) come from a file whose column schema doesn't
match ANY known conversation shape at all (whole file skipped, mode=None),
and (b) come from a recognized-schema file but still fail row-by-row
conversion (to_messages/canonicalize_turns/build throws or returns an
error)? Reused directly: N.plan/to_messages (sft_normalize.py) and
S.canonicalize_turns/build (sft_schema.py) -- the exact same real production
functions sample_from_sft_output.py already uses, never reimplemented.
"""
import os
import sys
import glob

import pyarrow.parquet as pq

sys.path.insert(0, "/projects/data/datasets/translation_data/SFT/DEDUP_PIPELINE")
import sft_schema as S
import sft_normalize as N

OUTPUT_DIR = "/projects/data/datasets/translation_data/SFT/OUTPUT"
FILES_PER_DOMAIN = 5     # bounded sample, read-only
ROWS_PER_FILE = 500      # one batch worth, matches production BATCH_SIZE


def scan_domain(domain_name):
    files = sorted(glob.glob(os.path.join(OUTPUT_DIR, domain_name, "*.parquet")))[:FILES_PER_DOMAIN]
    stats = dict(files_scanned=0, files_mode_none=0, rows_scanned=0,
                 rows_to_messages_fail=0, rows_canon_build_fail=0, rows_ok=0)

    for f in files:
        try:
            pf = pq.ParquetFile(f)
            cols = list(pf.schema_arrow.names)
        except Exception:
            continue
        stats["files_scanned"] += 1

        p = N.plan(cols)
        if p["mode"] is None:
            stats["files_mode_none"] += 1
            pf.close()
            continue

        try:
            for batch in pf.iter_batches(batch_size=ROWS_PER_FILE):
                rows = batch.to_pylist()
                for i, r in enumerate(rows):
                    stats["rows_scanned"] += 1
                    try:
                        raw = N.to_messages(r, p)
                    except Exception:
                        stats["rows_to_messages_fail"] += 1
                        continue
                    if not raw:
                        stats["rows_to_messages_fail"] += 1
                        continue
                    try:
                        msgs = S.canonicalize_turns(raw)
                        rec, err = S.build(msgs, file_name=os.path.relpath(f, OUTPUT_DIR),
                                            row_idx=i, source=os.path.basename(f)[:-8],
                                            domain=domain_name, adapter=p["mode"])
                    except Exception:
                        stats["rows_canon_build_fail"] += 1
                        continue
                    if err:
                        stats["rows_canon_build_fail"] += 1
                        continue
                    stats["rows_ok"] += 1
                break  # only first batch per file, bounded sample
        except Exception:
            pass
        finally:
            pf.close()

    return stats


def main():
    domains = sorted(os.listdir(OUTPUT_DIR))
    domains = [d for d in domains if os.path.isdir(os.path.join(OUTPUT_DIR, d))]
    print(f"scanning {len(domains)} domains, up to {FILES_PER_DOMAIN} files/domain, {ROWS_PER_FILE} rows/file (bounded sample)\n")

    totals = dict(files_scanned=0, files_mode_none=0, rows_scanned=0,
                  rows_to_messages_fail=0, rows_canon_build_fail=0, rows_ok=0)

    per_domain_bad_rate = []

    for d in domains:
        s = scan_domain(d)
        for k in totals:
            totals[k] += s[k]
        if s["rows_scanned"] > 0:
            bad = s["rows_to_messages_fail"] + s["rows_canon_build_fail"]
            rate = bad / s["rows_scanned"]
        else:
            rate = 1.0 if s["files_scanned"] > 0 else 0.0
        per_domain_bad_rate.append((d, s, rate))
        print(f"{d:30s} files={s['files_scanned']:3d} mode_none={s['files_mode_none']:3d}  "
              f"rows={s['rows_scanned']:5d} bad_rows={s['rows_to_messages_fail']+s['rows_canon_build_fail']:5d} "
              f"ok={s['rows_ok']:5d}")

    print("\n=== TOTALS (bounded sample across all domains) ===")
    print(f"files scanned: {totals['files_scanned']}")
    print(f"files with NO recognizable schema at all (mode=None, entire file skipped): {totals['files_mode_none']} ({100*totals['files_mode_none']/max(1,totals['files_scanned']):.1f}%)")
    print(f"rows scanned (from recognized-schema files only): {totals['rows_scanned']}")
    if totals["rows_scanned"]:
        print(f"  rows failing to_messages (malformed/missing role etc.): {totals['rows_to_messages_fail']} ({100*totals['rows_to_messages_fail']/totals['rows_scanned']:.1f}%)")
        print(f"  rows failing canonicalize_turns/build: {totals['rows_canon_build_fail']} ({100*totals['rows_canon_build_fail']/totals['rows_scanned']:.1f}%)")
        print(f"  rows successfully usable: {totals['rows_ok']} ({100*totals['rows_ok']/totals['rows_scanned']:.1f}%)")

    print("\n=== worst domains by bad-row rate ===")
    per_domain_bad_rate.sort(key=lambda x: -x[2])
    for d, s, rate in per_domain_bad_rate[:8]:
        print(f"  {d:30s} bad_rate={rate*100:.1f}%  (files_mode_none={s['files_mode_none']}/{s['files_scanned']})")


if __name__ == "__main__":
    main()
