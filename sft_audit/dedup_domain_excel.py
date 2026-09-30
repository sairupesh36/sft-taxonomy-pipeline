"""Before vs after DEDUP, rows per domain (user ask 2026-09-25). Rows only.

before = our 10 final datasets (pre-dedup), same domain rules as the rows-only workbook:
         non-hf datasets per file (reports/sft_domain_rows_quick_latest.xlsx, 'Files' sheet re-mapped with the
         current domain_rules), hf_domain per ROW (reports/hf_domain_before_rowdomain_per_file.jsonl).
after  = the fuzzy-dedup output split into domain folders (sft_final_deduped_domain_wise/_manifest.jsonl).
Dedup chain (Himanshu): our data copied 2026-09-24 23:41 -> exact dedup -> fuzzy (MinHash) dedup; per-dataset
numbers of both steps come from their report.json files.

    python3 dedup_domain_excel.py  -> sft_final_deduped_domain_wise/_dedup_domain_stats.xlsx (+ reports/ copy)
"""
import json, os, shutil, sys, time
from collections import Counter, defaultdict
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import domain_rules as DR                     # noqa: E402
import build_dedup_domain_folders as Bd       # noqa: E402
from sft_audit import DATASETS                # noqa: E402

DD = "/projects/data/datasets/code_data/Himanshu_sharma/Sft"
QUICK = os.path.join(HERE, "reports", "sft_domain_rows_quick_latest.xlsx")
HF_BEFORE = os.path.join(HERE, "reports", "hf_domain_before_rowdomain_per_file.jsonl")
ORDER = ["Math", "NLP Tasks (QA / Summarization / Classification)", "General Chat & Instructions",
         "Code & Software Engineering", "Reasoning & Logic", "Science & STEM", "Educational / Textbook", "Translation",
         Bd.UNTRACED, "Medical & Health", "Finance & Business", "Law & Policy", "Agentic & Tool Use",
         "Translated Code & Math (Indic)", "Data Analysis & Visualization", "Cybersecurity", "Safety & Alignment",
         "Other / Mixed"]
REV = {v: k for k, v in Bd.DS.items()}         # our dataset name -> dedup folder name


def main():
    # ---------------- before (pre-dedup)
    q = pd.read_excel(QUICK, sheet_name="Files (all final jsonl)")
    q = q[q.dataset != Bd.HF].copy()
    q["rel"] = [os.path.relpath(p, r) for p, r in zip(q["final jsonl (full path)"], q["dataset_path"])]
    q["domain"] = [DR.domain_of(DR.source_of(d, r)) for d, r in zip(q.dataset, q.rel)]
    before = [(d, dom, int(n), False) for d, dom, n in zip(q.dataset, q.domain, q.rows)]
    hf_files = 0
    for l in open(HF_BEFORE):
        x = json.loads(l); hf_files += 1
        unc = x["file"].startswith("uncertain/")
        for dom, n in x["domains"].items():
            before.append((Bd.HF, Bd.UNTRACED if dom == "Untraced" else dom, n, unc))
    B = pd.DataFrame(before, columns=["dataset", "domain", "rows", "not_given_to_dedup"])

    # ---------------- after (fuzzy output, per domain folder)
    man = [json.loads(l) for l in open(os.path.join(Bd.OUT, "_manifest.jsonl"))]
    after, files = [], []
    for m in man:
        for dom, n in m["rows"].items():
            after.append((m["dataset"], dom, n))
            files.append({"domain": dom, "domain folder": os.path.join(Bd.OUT, Bd.dirname(dom)), "dataset": m["dataset"],
                          "source": m["source"], "rows": n, "file (full path)": m["out"][dom],
                          "deduped input file (full path)": m["src_path"]})
    A = pd.DataFrame(after, columns=["dataset", "domain", "rows"])
    F = pd.DataFrame(files)

    # ---------------- domain summary
    b = B.groupby("domain")["rows"].sum(); bg = B[~B.not_given_to_dedup].groupby("domain")["rows"].sum()
    a = A.groupby("domain")["rows"].sum()
    doms = [d for d in ORDER if d in set(b.index) | set(a.index)] + sorted((set(b.index) | set(a.index)) - set(ORDER))
    rows = []
    for d in doms:
        rb, rg, ra = int(b.get(d, 0)), int(bg.get(d, 0)), int(a.get(d, 0))
        rows.append({"domain": d, "folder (full path)": os.path.join(Bd.OUT, Bd.dirname(d)),
                     "rows_before_dedup": rb, "rows_before_M": round(rb / 1e6, 2),
                     "not_given_to_dedup (hf uncertain/)": rb - rg,
                     "rows_after_dedup": ra, "rows_after_M": round(ra / 1e6, 2),
                     "rows_removed_by_dedup": rg - ra, "pct_removed (of rows given to dedup)": round(100 * (rg - ra) / rg, 2) if rg else None,
                     "pct_of_all_rows_after": None,
                     "files_after": int((F.domain == d).sum()) if len(F) else 0})
    S = pd.DataFrame(rows)
    ta = S.rows_after_dedup.sum()
    S["pct_of_all_rows_after"] = (100 * S.rows_after_dedup / ta).round(2)
    tot = {"domain": "TOTAL", "rows_before_dedup": S.rows_before_dedup.sum(), "rows_before_M": round(S.rows_before_dedup.sum() / 1e6, 2),
           "not_given_to_dedup (hf uncertain/)": S["not_given_to_dedup (hf uncertain/)"].sum(),
           "rows_after_dedup": ta, "rows_after_M": round(ta / 1e6, 2), "rows_removed_by_dedup": S.rows_removed_by_dedup.sum(),
           "pct_of_all_rows_after": 100.0, "files_after": S.files_after.sum()}
    g = tot["rows_before_dedup"] - tot["not_given_to_dedup (hf uncertain/)"]
    tot["pct_removed (of rows given to dedup)"] = round(100 * tot["rows_removed_by_dedup"] / g, 2)
    S = pd.concat([S, pd.DataFrame([tot])], ignore_index=True)

    # ---------------- dataset summary (with exact + fuzzy steps from their reports)
    ex = json.load(open(f"{DD}/exact_dedup_data/_dedup_report/report.json"))["per_source"]
    fz = json.load(open(f"{DD}/fuzzy_dedup_data/_dedup_report/report.json"))["per_source"]
    dsr = []
    for ds in DATASETS:
        k = REV.get(ds, ds)
        ours = int(B[B.dataset == ds].rows.sum()); given = ex[k]["rows_in"]; aft = int(A[A.dataset == ds].rows.sum())
        dsr.append({"dataset": ds, "our final path (before)": DATASETS[ds], "rows_ours_before": ours,
                    "rows_given_to_dedup (their copy 24-Sep 23:41)": given, "difference ours - given": ours - given,
                    "after_exact_dedup": ex[k]["kept"], "exact_removed": ex[k]["dup_removed"],
                    "after_fuzzy_dedup (report)": fz[k]["kept"], "fuzzy_removed": fz[k]["dup_removed"],
                    "fuzzy: dup within same dataset": fz[k].get("dup_within_source", 0),
                    "fuzzy: dup of another dataset": fz[k].get("dup_of_other_source", 0) or 0,
                    "rows_after_in_domain_folders (counted)": aft, "check: counted == report": aft == fz[k]["kept"],
                    "total_removed_by_dedup": given - aft, "pct_removed": round(100 * (given - aft) / given, 2)})
    D = pd.DataFrame(dsr)
    tot = {c: D[c].sum() for c in D.columns if c not in ("dataset", "our final path (before)", "check: counted == report", "pct_removed")}
    tot.update({"dataset": "TOTAL", "check: counted == report": bool(D["check: counted == report"].all()),
                "pct_removed": round(100 * tot["total_removed_by_dedup"] / tot["rows_given_to_dedup (their copy 24-Sep 23:41)"], 2)})
    D = pd.concat([D, pd.DataFrame([tot])], ignore_index=True)

    # ---------------- domain x dataset
    bb = B.groupby(["domain", "dataset"])["rows"].sum().rename("rows_before")
    ag = B[~B.not_given_to_dedup].groupby(["domain", "dataset"])["rows"].sum().rename("rows_given")
    aa = A.groupby(["domain", "dataset"])["rows"].sum().rename("rows_after")
    X = pd.concat([bb, ag, aa], axis=1).fillna(0).astype("int64").reset_index()
    X["rows_removed_by_dedup"] = X.rows_given - X.rows_after
    X["pct_removed"] = (100 * X.rows_removed_by_dedup / X.rows_given.clip(lower=1)).round(2)
    X["domain_rank"] = X.domain.map({d: i for i, d in enumerate(doms)})
    X = X.sort_values(["domain_rank", "rows_before"], ascending=[True, False]).drop(columns="domain_rank")
    X.insert(1, "folder (full path)", X.domain.map(lambda d: os.path.join(Bd.OUT, Bd.dirname(d), "")) + X.dataset)

    notes = pd.DataFrame({"note": [
        f"Output: {Bd.OUT}/<domain>/<dataset>/<same path as in the dedup output>. Rows copied byte for byte from {DD}/fuzzy_dedup_data.",
        "Folder names = the domain names; '/' cannot be in a folder name, so 'NLP Tasks (QA / Summarization / Classification)' -> "
        "'NLP Tasks (QA - Summarization - Classification)' and 'Other / Mixed' -> 'Other - Mixed'.",
        "Domain = rough rule from the source dataset / parquet NAME (sft_audit/domain_rules.py). hf_domain: per ROW, the prompt's "
        "source folder (Maths, Biology_and_Genomics, ...); rows whose prompt is in no source parquet = 'Untraced'. "
        "Before and after use exactly the same rules, so they compare 1:1.",
        "Dedup chain: our 10 final datasets were copied on 2026-09-24 23:41 -> exact dedup -> fuzzy MinHash dedup "
        "(Jaccard >= 0.8, 5-word shingles, global across all 10, source priority = posttraining A first ... datasets_sft last).",
        "hf_domain: the 187 files in uncertain/ (52,158,161 rows) were NOT in the copy given to dedup, so they are not in the "
        "output; shown as 'not_given_to_dedup' and not counted as removed.",
        "hf_traces / hf_agentic / code_math_text / B1 copies were taken before our 2026-09-25 cut-off-<think> drop, so the dedup "
        "input had those rows (315,243 + 51 + 1 + 5) -- 'difference ours - given' is negative for them.",
        "rows_removed_by_dedup = rows given to dedup - rows in the output (exact + fuzzy together).",
    ]})
    out = os.path.join(Bd.OUT, "_dedup_domain_stats.xlsx")
    with pd.ExcelWriter(out, engine="openpyxl") as xw:
        for name, df in [("Domain before vs after", S), ("Dataset before vs after", D), ("Domain x dataset", X),
                         ("Files (after, all)", F), ("Notes", notes)]:
            df.to_excel(xw, sheet_name=name[:31], index=False)
        for ws in xw.book.worksheets:
            for col in ws.columns:
                w = max((len(str(c.value)) for c in col[:300] if c.value is not None), default=8)
                ws.column_dimensions[col[0].column_letter].width = min(max(10, w + 2), 70)
            ws.freeze_panes = "A2"
    stamp = time.strftime("%Y%m%d_%H%M%S")
    shutil.copy2(out, os.path.join(HERE, "reports", f"sft_dedup_domain_stats_{stamp}.xlsx"))
    pd.set_option("display.width", 250)
    print(S[["domain", "rows_before_M", "not_given_to_dedup (hf uncertain/)", "rows_after_M", "rows_removed_by_dedup",
             "pct_removed (of rows given to dedup)"]].to_string(index=False))
    print(D[["dataset", "rows_ours_before", "rows_given_to_dedup (their copy 24-Sep 23:41)", "after_exact_dedup",
             "rows_after_in_domain_folders (counted)", "check: counted == report", "pct_removed"]].to_string(index=False))
    print("written", out)


if __name__ == "__main__":
    main()
