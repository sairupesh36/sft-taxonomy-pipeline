"""URGENT rows-only domain workbook (user ask 2026-09-25, while the o200k stats run was still going).

No tokenizing: every final file's rows are counted as lines (wc -l / pigz -dc | wc -l). Domain comes from the
same name rules as sft_domain_stats.py (domain_rules.py). sft_hf_domain_data final files are mixed shards, so its
rows per domain come from the first cleaning-loss report (rows_after per source parquet, mapped with
domain_rules.hf_source_domain); final rows not covered by that report are shown as 'Untraced'.

    python3 quick_domain_rows.py
Output: reports/sft_domain_rows_quick_<ts>.xlsx (+ reports/sft_domain_rows_quick_latest.xlsx)
"""
import json, os, shutil, subprocess, sys, time
from multiprocessing import Pool
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sft_audit import DATASETS          # noqa: E402
import domain_rules as DR               # noqa: E402

T = "/projects/data/datasets/code_data/sai_rupesh/taxonomy"
HF = "sft_hf_domain_data_openai_ready"
HF_SRC = "/projects/data/datasets/translation_data/SFT/OUTPUT"
OLD_REPORT = f"{T}/sft_loss_report/sft_cleaning_loss_report.xlsx"


def count(p):
    cmd = f"pigz -dc '{p}' | wc -l" if p.endswith(".gz") else f"wc -l < '{p}'"
    return p, int(subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip() or 0)


def main():
    files = []
    for ds, root in DATASETS.items():
        for r, dirs, fs in os.walk(root):
            dirs[:] = [x for x in dirs if not x.startswith("_")]
            files += [(ds, root, os.path.join(r, f)) for f in fs if f.endswith((".jsonl", ".jsonl.gz"))]
    files.sort(key=lambda x: -os.path.getsize(x[2]))
    print(len(files), "files", flush=True)
    rows = {}
    reuse = os.environ.get("REUSE_COUNTS")       # a previous quick workbook: take its per-file rows instead of recounting
    if reuse:
        prev = pd.read_excel(reuse, sheet_name="Files (all final jsonl)")
        rows = dict(zip(prev["final jsonl (full path)"], prev["rows"]))
        missing = [f[2] for f in files if f[2] not in rows]
        if missing:
            sys.exit(f"{len(missing)} files not in {reuse}, e.g. {missing[0]}")
        files_to_count = []
    else:
        files_to_count = [f[2] for f in files]
    with Pool(int(os.environ.get("WORKERS", 32))) as pool:
        for i, (p, n) in enumerate(pool.imap_unordered(count, files_to_count), 1):
            rows[p] = n
            if i % 2000 == 0:
                print(f"[{i}/{len(files)}]", flush=True)

    recs = []
    for ds, root, p in files:
        rel = os.path.relpath(p, root)
        src = DR.source_of(ds, rel)
        recs.append({"dataset": ds, "dataset_path": root, "final jsonl (full path)": p, "source": src,
                     "domain": DR.domain_of(src) if ds != HF else "(mixed shard)", "rows": rows[p],
                     "GB_on_disk": round(os.path.getsize(p) / 1e9, 3)})
    pf = pd.DataFrame(recs)

    # hf_domain rows by domain, from its per-source after-counts
    hf = pd.read_excel(OLD_REPORT, sheet_name="Domain (sft_hf_domain)")
    hf = hf[hf["rows_after"].fillna(0) > 0]
    hf_src = pd.DataFrame({"dataset": HF, "dataset_path": DATASETS[HF], "source": hf["source (dataset / parquet)"],
                           "raw parquet (full path)": HF_SRC + "/" + hf["source (dataset / parquet)"].astype(str),
                           "domain": hf["source (dataset / parquet)"].astype(str).map(DR.hf_source_domain),
                           "rows": hf["rows_after"].astype("int64")})
    hf_total = int(pf.loc[pf.dataset == HF, "rows"].sum())
    untraced = hf_total - int(hf_src["rows"].sum())
    hf_dom = hf_src.groupby("domain")["rows"].sum()
    if untraced > 0:
        hf_dom.loc["Untraced (hf_domain rows not matched to a source)"] = untraced

    base = pd.concat([pf[pf.dataset != HF][["dataset", "domain", "rows"]],
                      pd.DataFrame({"dataset": HF, "domain": hf_dom.index, "rows": hf_dom.values})], ignore_index=True)
    tot = base["rows"].sum()
    by_dom = base.groupby("domain")["rows"].sum().sort_values(ascending=False).reset_index()
    by_dom["pct_rows"] = (100 * by_dom["rows"] / tot).round(2)
    by_dom["rows_million"] = (by_dom["rows"] / 1e6).round(2)
    by_dom["datasets"] = by_dom["domain"].map(base[base.rows > 0].groupby("domain")["dataset"].apply(lambda s: ", ".join(sorted(set(s)))))
    by_dom = pd.concat([by_dom, pd.DataFrame([{"domain": "TOTAL", "rows": tot, "pct_rows": 100.0, "rows_million": round(tot / 1e6, 2)}])])

    piv = base.pivot_table(index="domain", columns="dataset", values="rows", aggfunc="sum", fill_value=0)
    piv["TOTAL"] = piv.sum(axis=1)
    piv = piv.sort_values("TOTAL", ascending=False)
    piv.loc["TOTAL"] = piv.sum()
    piv = piv.reset_index()

    ds_dom = base.groupby(["dataset", "domain"])["rows"].sum().reset_index()
    ds_dom["pct_of_dataset"] = (100 * ds_dom["rows"] / ds_dom.groupby("dataset")["rows"].transform("sum")).round(2)
    ds_dom.insert(1, "dataset_path", ds_dom["dataset"].map(DATASETS))
    ds_dom = ds_dom.sort_values(["dataset", "rows"], ascending=[True, False])

    src = pf[pf.dataset != HF].groupby(["dataset", "dataset_path", "source", "domain"]).agg(
        files=("rows", "size"), rows=("rows", "sum"),
        final_jsonl=("final jsonl (full path)", lambda s: " | ".join(sorted(s)))).reset_index()
    src = src.rename(columns={"final_jsonl": "final jsonl files (full path)"}).sort_values(["dataset", "rows"], ascending=[True, False])
    ds_tot = pf.groupby(["dataset", "dataset_path"]).agg(files=("rows", "size"), rows=("rows", "sum"),
                                                         GB_on_disk=("GB_on_disk", "sum")).reset_index()
    ds_tot["GB_on_disk"] = ds_tot["GB_on_disk"].round(1)

    notes = pd.DataFrame({"note": [
        "QUICK rows-only version, made while the full o200k token stats were still running. Rows = lines in each final file.",
        "Domain = rough guess from the source dataset / parquet NAME (sft_audit/domain_rules.py), not a per-row classifier.",
        f"sft_hf_domain_data: final files are mixed shards, so its domains come from rows_after per source parquet in "
        f"{OLD_REPORT}; {untraced:,} final rows are not covered there and are shown as 'Untraced'.",
        "Datasets 9/10 = the traces_team folders (final data, used as-is).",
        "Token counts and context length come in the full workbook reports/sft_domain_stats_latest.xlsx when the stats run ends.",
    ]})
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = os.path.join(HERE, "reports", f"sft_domain_rows_quick_{stamp}.xlsx")
    with pd.ExcelWriter(out, engine="openpyxl") as xw:
        for name, df in [("Domain summary (rows)", by_dom), ("Domain x dataset (rows)", piv), ("Dataset -> domain", ds_dom),
                         ("Datasets (rows, paths)", ds_tot), ("Source -> domain", src),
                         ("hf_domain sources", hf_src.sort_values(["domain", "rows"], ascending=[True, False])),
                         ("Files (all final jsonl)", pf), ("Notes", notes)]:
            df.to_excel(xw, sheet_name=name[:31], index=False)
        for ws in xw.book.worksheets:
            for row in ws.iter_rows():
                for c in row:
                    if isinstance(c.value, str) and len(c.value) > 32000:
                        c.value = c.value[:31900].rsplit(" | ", 1)[0] + f" | ... ({c.value.count(' | ') + 1:,} in total, see 'Files' sheet)"
            for col in ws.columns:
                w = max((len(str(c.value)) for c in col[:300] if c.value is not None), default=8)
                ws.column_dimensions[col[0].column_letter].width = min(max(10, w + 2), 70)
            ws.freeze_panes = "A2"
    shutil.copy2(out, os.path.join(HERE, "reports", "sft_domain_rows_quick_latest.xlsx"))
    print(by_dom.to_string(index=False))
    print(ds_tot.to_string(index=False))
    print("written", out)


if __name__ == "__main__":
    main()
