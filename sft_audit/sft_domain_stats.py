"""Domain-wise stats (rows, o200k tokens, context length) for every dataset in sft_audit.DATASETS.
Separate workbook, user ask 2026-09-25. Read-only.

    python3 sft_domain_stats.py
Output: reports/sft_domain_stats_<timestamp>.xlsx (+ reports/sft_domain_stats_latest.xlsx)

Domain = ROUGH guess from the source dataset / parquet NAME (domain_rules.py), except
sft_hf_domain_data, whose final files are mixed shards: there every ROW gets the domain of the source
parquet folder it came from (sft_loss_report/v2/hf_domain_domainwise.py). Inputs:
  reports/sft_stats_latest_per_file.csv            (all datasets, per final file)
  ../sft_loss_report/v2/hf_domain_domainwise_per_file.jsonl  (hf_domain, per file per domain)
"""
import json, os, re, sys, time, shutil
from collections import Counter, defaultdict
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sft_audit import DATASETS                       # noqa: E402
import domain_rules as DR                             # noqa: E402
from sft_stats import LEN_LABELS                      # noqa: E402

HF = "sft_hf_domain_data_openai_ready"
HF_ROWS = os.path.join(HERE, "..", "sft_loss_report", "v2", "hf_domain_domainwise_per_file.jsonl")
LEN_COLS = [f"rows_len_{l}" for l in LEN_LABELS]

LR = os.path.join(HERE, "..", "sft_loss_report")
RAW_ROOTS = {  # 'before' inventories: dataset key -> (final dataset, raw root)
    "sft_hf_domain_data": (HF, "/projects/data/datasets/translation_data/SFT/OUTPUT"),
    "sft_hf_agentic_data": ("sft_hf_agentic_data_openai_ready", "/projects/data/datasets/translation_data/Agentic_Ai/OUTPUT"),
    "sft_hf_traces_data": ("sft_hf_traces_data_openai_ready", "/projects/data/datasets/translation_data/Traces_datasets/OUTPUT"),
    "sft_rl_reference_data": ("sft_rl_reference_data_openai_ready", "/projects/data/datasets/code_data/codeLLM_data/RL_reference_data_from_SFT"),
    "sft_code_math_text": ("sft_code_math_text_openai_ready", "/projects/data/datasets/code_data/codeLLM_data/code_math_text_translations"),
}
PT_SRC = "/projects/data/datasets/code_data/codeLLM_data/posttraining_datasets"
DATA_EXT = (".parquet", ".arrow", ".jsonl", ".json")


def _csv(name):
    p = os.path.join(LR, "v2", "csv_all10", name + ".csv")
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()


def raw_paths():
    """every RAW source file of all 10 datasets with its (rough) domain, rows before and rows in the final data."""
    rows = []
    old = {"sft_hf_domain_data_openai_ready": _csv("Domain_sft_hf_domain"), "sft_hf_traces_data_openai_ready": _csv("Traces_sft_hf_traces"),
           "sft_hf_agentic_data_openai_ready": _csv("Agentic_sft_hf_agentic"), "sft_code_math_text_openai_ready": _csv("Code_math_text"),
           "sft_rl_reference_data_openai_ready": _csv("RL_reference")}
    src_after = {ds: dict(zip(df["source (dataset / parquet)"].astype(str), df["rows_after"])) for ds, df in old.items() if len(df)}
    for l in open(os.path.join(LR, "inventory_before.jsonl")):
        d = json.loads(l)
        if d["dataset"] not in RAW_ROOTS or not d["file"].endswith(DATA_EXT) or d.get("rows") is None:
            continue
        ds, root = RAW_ROOTS[d["dataset"]]
        rel = d["file"]
        if ds == HF:
            source, dom = rel, DR.hf_source_domain(rel)
        else:
            source = rel.split("/")[0] if "/" in rel else re.sub(r"(_\d+)?\.(parquet|arrow|jsonl|json)$", "", rel)
            dom = DR.domain_of(source)
        a = src_after.get(ds, {}).get(source)
        rows.append({"domain": dom, "final_dataset": ds, "source": source, "raw file (full path)": f"{root}/{rel}",
                     "rows_before": d["rows"], "rows_after (this source, all its files)": a})
    pt = _csv("Posttraining_files")
    for r in pt.itertuples(index=False):
        rel = os.path.relpath(r[1], PT_SRC)
        dom = DR.domain_of(r[0])
        outs = [x for x in (r[4], r[6], r[8]) if isinstance(x, str) and x]
        fin = "sft_posttraining_* (" + ", ".join(b for b, x in zip(("A", "B1", "B2"), (r[4], r[6], r[8])) if isinstance(x, str) and x) + ")" if outs else "(not in final)"
        rows.append({"domain": dom, "final_dataset": fin, "source": r[0], "raw file (full path)": r[1],
                     "rows_before": r[2], "rows_after (this file)": r.rows_after_total if hasattr(r, "rows_after_total") else None})
    ds_ = _csv("Datasets_sft")
    for r in ds_.to_dict("records"):
        rows.append({"domain": r["domain (from name)"], "final_dataset": "sft_datasets_sft_openai_ready", "source": r["source (raw parquet)"],
                     "raw file (full path)": r["raw file (full path)"], "rows_before": r["rows_before"], "rows_after (this file)": r["rows_after"]})
    for r in _csv("Synthetic").to_dict("records"):
        rows.append({"domain": "Data Analysis & Visualization", "final_dataset": "sft_synthetic_data_openai_ready",
                     "source": r["source (raw parquet)"], "raw file (full path)": r["raw file (full path)"],
                     "rows_before": r["rows_before"], "rows_after (this file)": r["rows_after"]})
    df = pd.DataFrame(rows)
    # posttraining rows_after per raw file
    if len(pt):
        m = dict(zip(pt["raw file (full path)"], pt["rows_after_total"]))
        sel = df["final_dataset"].str.startswith("sft_posttraining") | (df["final_dataset"] == "(not in final)")
        df.loc[sel, "rows_after (this file)"] = df.loc[sel, "raw file (full path)"].map(m)
    # hf_domain per parquet: rows_after from the Domain sheet (kept in final)
    hfa = src_after.get(HF, {})
    sel = df["final_dataset"] == HF
    df.loc[sel, "rows_after (this file)"] = df.loc[sel, "source"].map(hfa)
    df.loc[sel, "rows_after (this source, all its files)"] = None
    cols = ["domain", "final_dataset", "source", "raw file (full path)", "rows_before", "rows_after (this file)",
            "rows_after (this source, all its files)"]
    return df[[c for c in cols if c in df.columns]].sort_values(["domain", "final_dataset", "source", "raw file (full path)"])


def agg(df, keys):
    g = df.groupby(keys, dropna=False)
    out = g[["files", "rows", "tokens_o200k"] + LEN_COLS].sum()
    out["max_tokens_o200k"] = g["max_tokens_o200k"].max()
    out = out.reset_index()
    out["tokens_o200k_billion"] = (out["tokens_o200k"] / 1e9).round(3)
    out["avg_tokens_per_row"] = (out["tokens_o200k"] / out["rows"].clip(lower=1)).round(0)
    tr, tt = out["rows"].sum(), out["tokens_o200k"].sum()
    out["pct_rows"] = (100 * out["rows"] / max(tr, 1)).round(2)
    out["pct_tokens"] = (100 * out["tokens_o200k"] / max(tt, 1)).round(2)
    cols = keys + ["files", "rows", "pct_rows", "tokens_o200k", "tokens_o200k_billion", "pct_tokens",
                   "avg_tokens_per_row", "max_tokens_o200k"] + LEN_COLS
    return out[cols]


def main():
    pf = pd.read_csv(os.path.join(HERE, "reports", "sft_stats_latest_per_file.csv"))
    missing = set(DATASETS) - set(pf["dataset"])
    if missing:
        sys.exit(f"stats per-file table lacks {missing} -- run sft_stats.py first")
    pf["files"] = 1
    pf["source"] = [DR.source_of(d, f) for d, f in zip(pf["dataset"], pf["file"])]
    pf["domain"] = pf["source"].map(DR.domain_of)

    # hf_domain: row-level domain from the source-folder mapping
    hf_rows = []
    for l in open(HF_ROWS):
        d = json.loads(l)
        for dom, c in d["domains"].items():
            r = {"dataset": HF, "full_path": d["file"], "file": os.path.relpath(d["file"], DATASETS[HF]),
                 "source": "(rows traced to source folders)", "domain": dom, "files": 0,
                 "rows": c.get("rows", 0), "tokens_o200k": c.get("tokens", 0), "max_tokens_o200k": c.get("max_tokens", 0)}
            for lab, col in zip(LEN_LABELS, LEN_COLS):
                r[col] = c.get(f"len::{lab}", 0)
            hf_rows.append(r)
    hf = pd.DataFrame(hf_rows)
    hf_files = hf.groupby("full_path").ngroups
    base = pd.concat([pf[pf.dataset != HF], hf], ignore_index=True)
    n_hf_stats = int(pf.loc[pf.dataset == HF, "rows"].sum())
    n_hf_dom = int(hf["rows"].sum())

    by_dom = agg(base, ["domain"]).sort_values("rows", ascending=False)
    by_ds_dom = agg(base, ["dataset", "domain"]).sort_values(["dataset", "rows"], ascending=[True, False])
    by_ds_dom.insert(1, "dataset_path", by_ds_dom["dataset"].map(DATASETS))
    # files column for hf is 0 in 'base' rows; give it the real file count per domain
    hf_dom_files = hf[hf.rows > 0].groupby("domain")["full_path"].nunique()
    m = by_ds_dom.dataset == HF
    by_ds_dom.loc[m, "files"] = by_ds_dom.loc[m, "domain"].map(hf_dom_files).fillna(0).astype(int)
    by_dom["files"] = by_dom["domain"].map(base[base.dataset != HF].groupby("domain")["files"].sum()).fillna(0).astype(int) \
        + by_dom["domain"].map(hf_dom_files).fillna(0).astype(int)

    src = agg(base[base.dataset != HF], ["dataset", "source", "domain"]).sort_values(["dataset", "rows"], ascending=[True, False])
    paths = base[base.dataset != HF].groupby(["dataset", "source"])["full_path"].apply(lambda s: " | ".join(sorted(s)))
    src["final jsonl files (full path)"] = [paths.get((d, s), "") for d, s in zip(src["dataset"], src["source"])]
    src.insert(1, "dataset_path", src["dataset"].map(DATASETS))

    files = pf[["dataset", "full_path", "source", "domain", "rows", "tokens_o200k", "avg_tokens_o200k", "max_tokens_o200k"] + LEN_COLS].copy()
    files.loc[files.dataset == HF, ["source", "domain"]] = ["(mixed shard)", "row-level, see 'hf_domain rows by domain'"]
    hf_file_dom = agg(hf, ["full_path", "domain"]).sort_values(["full_path", "rows"], ascending=[True, False])
    folders = pd.DataFrame([{"hf_domain source folder": k, "domain": v} for k, v in sorted(DR.HF_FOLDERS.items())] +
                           [{"hf_domain source folder": "final / NL / data / generated_data / Generated_Dataset",
                             "domain": "generic folders: domain from each parquet NAME (domain_rules.RULES)"}])
    other = src[src.domain == "Other / Mixed"][["dataset", "source", "rows", "tokens_o200k", "final jsonl files (full path)"]]
    rules = pd.DataFrame([{"order": i + 1, "name contains (regex)": p, "domain": d} for i, (p, d) in enumerate(DR.RULES)])
    notes = pd.DataFrame({"note": [
        "ROUGH domain: decided from the source dataset / parquet NAME (see sheet 'Name rules'), not by reading each row -- "
        "except sft_hf_domain_data, where every row gets the domain of the source folder its prompt came from.",
        f"sft_hf_domain_data: {n_hf_dom:,} rows counted row by row (stats total {n_hf_stats:,}); rows whose prompt is in no "
        f"source parquet are 'Untraced'; {hf_files:,} files.",
        "tokens_o200k = real GPT o200k_base token count (tiktoken) of all message text + tool-call names/arguments + tools JSON "
        "(chat-template special tokens not included).",
        "Context length = o200k tokens of one row (one conversation). rows_len_<a>-<b> = rows whose length is in that range; "
        "max_tokens_o200k = the longest row.",
        "xP3 (in sft_datasets_sft) is split by its task name: translation tasks (flores, tatoeba, ...) -> Translation, code "
        "tasks -> Code, the rest -> NLP Tasks.",
        "Sheet 'Domain -> raw parquet paths': every RAW source file (parquet / arrow / jsonl) of all 10 datasets with its domain, "
        "rows before and rows that reached the final data (per file where known; for traces/agentic/code_math/rl per whole source).",
        "sft_hf_domain_data: a prompt found in source files of different domains (41.5M unique prompts) gets the domain of the "
        "first source file (by file id).",
        "Built by sft_audit/sft_domain_stats.py from reports/sft_stats_latest_per_file.csv.",
    ]})
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = os.path.join(HERE, "reports", f"sft_domain_stats_{stamp}.xlsx")
    rp = raw_paths()
    # raw parquets per domain, shown right in the summary (user ask 2026-09-25: "in the domain excel I need parquets also")
    rp_in = rp[rp["rows_after (this file)"].fillna(rp.get("rows_after (this source, all its files)", 0)).fillna(0) > 0]
    by_dom.insert(1, "raw_parquet_files", by_dom["domain"].map(rp.groupby("domain").size()).fillna(0).astype(int))
    by_dom.insert(2, "raw_parquet_files_in_final", by_dom["domain"].map(rp_in.groupby("domain").size()).fillna(0).astype(int))
    by_dom.insert(3, "raw_rows_before", by_dom["domain"].map(rp.groupby("domain")["rows_before"].sum()).fillna(0).astype("int64"))
    by_dom["raw parquet paths (full list in sheet 'Domain -> raw parquet paths')"] = by_dom["domain"].map(
        rp.groupby("domain")["raw file (full path)"].apply(lambda x: " | ".join(sorted(x)))).fillna("")
    rsum = rp.groupby(["domain", "final_dataset"]).agg(raw_parquet_files=("raw file (full path)", "count"),
                                                      rows_before=("rows_before", "sum")).reset_index()
    with pd.ExcelWriter(out, engine="openpyxl") as xw:
        by_dom.to_excel(xw, sheet_name="Domain summary (all 10)", index=False)
        rp.to_excel(xw, sheet_name="Domain -> raw parquet paths", index=False)
        rsum.to_excel(xw, sheet_name="Domain raw parquets (count)", index=False)
        by_ds_dom.to_excel(xw, sheet_name="Domain x dataset", index=False)
        src.to_excel(xw, sheet_name="Source -> domain", index=False)
        hf_file_dom.to_excel(xw, sheet_name="hf_domain rows by domain", index=False)
        files.to_excel(xw, sheet_name="Files (all final jsonl)", index=False)
        other.to_excel(xw, sheet_name="Other-Mixed (check)", index=False)
        folders.to_excel(xw, sheet_name="hf_domain folders", index=False)
        rules.to_excel(xw, sheet_name="Name rules", index=False)
        notes.to_excel(xw, sheet_name="Notes", index=False)
        for ws in xw.book.worksheets:
            for row in ws.iter_rows():                 # Excel cells hold at most 32,767 characters
                for c in row:
                    if isinstance(c.value, str) and len(c.value) > 32000:
                        c.value = c.value[:31900].rsplit(" | ", 1)[0] + f" | ... ({c.value.count(' | ') + 1:,} paths in total, full list in its own sheet)"
            for col in ws.columns:
                w = max((len(str(c.value)) for c in col[:300] if c.value is not None), default=8)
                ws.column_dimensions[col[0].column_letter].width = min(max(10, w + 2), 70)
            ws.freeze_panes = "A2"
    shutil.copy2(out, os.path.join(HERE, "reports", "sft_domain_stats_latest.xlsx"))
    print(by_dom[["domain", "files", "rows", "pct_rows", "tokens_o200k_billion", "pct_tokens", "avg_tokens_per_row",
                  "max_tokens_o200k"]].to_string(index=False))
    print("written", out)


if __name__ == "__main__":
    main()
