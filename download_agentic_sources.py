import os
import sys
import json
import time
import urllib.request
from pathlib import Path

try:
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    print("[Downloader] Installing required pyarrow and pandas...")
    os.system("pip install --quiet pyarrow pandas")
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

OUT_DIR = Path("/projects/data/datasets/code_data/sai_rupesh/taxonomy/agentic_data")
OUT_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR = OUT_DIR / ".tmp_download"
TMP_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "*/*",
}

SOURCES = [
    {"id": 1, "mixture": "internlm/Agent-FLAN", "name": "agent_flan_agent_instruct_react", "url": "https://huggingface.co/datasets/internlm/Agent-FLAN/resolve/main/data/agent_instruct_react.jsonl", "type": "jsonl_stream", "cap": 5000, "desc": "Multi-step ReAct action trajectories on AgentInstruct seeds"},
    {"id": 2, "mixture": "internlm/Agent-FLAN", "name": "agent_flan_agent_instruct_tflan", "url": "https://huggingface.co/datasets/internlm/Agent-FLAN/resolve/main/data/agent_instruct_tflan.jsonl", "type": "jsonl_stream", "cap": 5000, "desc": "Conversational tool execution flows"},
    {"id": 3, "mixture": "internlm/Agent-FLAN", "name": "agent_flan_toolbench_react_10p", "url": "https://huggingface.co/datasets/internlm/Agent-FLAN/resolve/main/data/toolbench_react_10p.jsonl", "type": "jsonl_stream", "cap": 5000, "desc": "ReAct tool execution over real-world RapidAPI REST tools"},
    {"id": 4, "mixture": "internlm/Agent-FLAN", "name": "agent_flan_toolbench_tflan_60p", "url": "https://huggingface.co/datasets/internlm/Agent-FLAN/resolve/main/data/toolbench_tflan_60p_r10r5u7.jsonl", "type": "jsonl_stream", "cap": 5000, "desc": "Multi-turn API execution dialogues with arguments"},
    {"id": 5, "mixture": "internlm/Agent-FLAN", "name": "agent_flan_toolbench_tflan_cot", "url": "https://huggingface.co/datasets/internlm/Agent-FLAN/resolve/main/data/toolbench_tflan_cot_30p.jsonl", "type": "jsonl_stream", "cap": 5000, "desc": "Chain-of-thought tool argument reasoning and selection"},
    {"id": 6, "mixture": "internlm/Agent-FLAN", "name": "agent_flan_toolbench_instruct_3k", "url": "https://huggingface.co/datasets/internlm/Agent-FLAN/resolve/main/data/toolbench_instruct_j1s1_3k.jsonl", "type": "jsonl_stream", "cap": 3000, "desc": "Precise instruction-to-API mapping (full 3k rows)"},
    {"id": 7, "mixture": "internlm/Agent-FLAN", "name": "agent_flan_toolbench_negative", "url": "https://huggingface.co/datasets/internlm/Agent-FLAN/resolve/main/data/toolbench_negative.jsonl", "type": "jsonl_stream", "cap": 5000, "desc": "Negative agent samples (learning when NOT to call tools)"},
    {"id": 8, "mixture": "Team-ACE/ToolACE", "name": "toolace_complex_tools", "url": "https://huggingface.co/datasets/Team-ACE/ToolACE/resolve/main/data.json", "type": "json_array_stream", "cap": 5000, "desc": "BFCL #1 rank nested tool calling & error recovery"},
    {"id": 9, "mixture": "glaiveai/glaive-function-calling-v2", "name": "glaive_function_calling", "url": "https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2/resolve/main/glaive-function-calling-v2.json", "type": "jsonl_stream", "cap": 5000, "desc": "Parallel and sequential function calling dialogue turns"},
    {"id": 10, "mixture": "nvidia/Nemotron-SFT-Agentic-v2", "name": "nemotron_interactive_agent", "url": "https://huggingface.co/datasets/nvidia/Nemotron-SFT-Agentic-v2/resolve/main/data/interactive_agent.jsonl", "type": "jsonl_stream", "cap": 5000, "desc": "Interactive multi-turn goal decomposition agent traces"},
    {"id": 11, "mixture": "nvidia/Nemotron-SFT-Agentic-v2", "name": "nemotron_search_agent", "url": "https://huggingface.co/datasets/nvidia/Nemotron-SFT-Agentic-v2/resolve/main/data/search.jsonl", "type": "jsonl_stream", "cap": 5000, "desc": "Multi-hop search & information retrieval agent traces"},
    {"id": 12, "mixture": "nvidia/Nemotron-SFT-Agentic-v2", "name": "nemotron_tool_calling", "url": "https://huggingface.co/datasets/nvidia/Nemotron-SFT-Agentic-v2/resolve/main/data/tool_calling.jsonl", "type": "jsonl_stream", "cap": 5000, "desc": "Single & multi-turn tool calling with observation reasoning"},
    {"id": 13, "mixture": "microsoft/orca-agentinstruct-1M-v1", "name": "agentinstruct_tool_use", "url": "https://huggingface.co/datasets/microsoft/orca-agentinstruct-1M-v1/resolve/main/data/tool_use-00000-of-00001.parquet", "type": "parquet_file", "cap": 5000, "desc": "AgentInstruct synthetic API tool usage flows"},
    {"id": 14, "mixture": "microsoft/orca-agentinstruct-1M-v1", "name": "agentinstruct_webagent_flow", "url": "https://huggingface.co/datasets/microsoft/orca-agentinstruct-1M-v1/resolve/main/data/webagent_flow-00000-of-00001.parquet", "type": "parquet_file", "cap": 5000, "desc": "Autonomous web browsing and DOM element action flows"},
    {"id": 15, "mixture": "microsoft/orca-agentinstruct-1M-v1", "name": "agentinstruct_code_agent", "url": "https://huggingface.co/datasets/microsoft/orca-agentinstruct-1M-v1/resolve/main/data/code_-00000-of-00002.parquet", "type": "parquet_file", "cap": 5000, "desc": "Code generation, debugging, refactoring, and test writing"},
    {"id": 16, "mixture": "microsoft/orca-agentinstruct-1M-v1", "name": "agentinstruct_rag_agent", "url": "https://huggingface.co/datasets/microsoft/orca-agentinstruct-1M-v1/resolve/main/data/rag-00000-of-00001.parquet", "type": "parquet_file", "cap": 5000, "desc": "Multi-hop retrieval-augmented generation and verification"},
    {"id": 17, "mixture": "microsoft/orca-agentinstruct-1M-v1", "name": "agentinstruct_analytical_reasoning", "url": "https://huggingface.co/datasets/microsoft/orca-agentinstruct-1M-v1/resolve/main/data/analytical_reasoning-00000-of-00001.parquet", "type": "parquet_file", "cap": 5000, "desc": "Goal decomposition, planning, and multi-step deduction"},
    {"id": 18, "mixture": "princeton-nlp/SWE-bench", "name": "swe_bench_software_engineering_agent", "url": "https://huggingface.co/datasets/princeton-nlp/SWE-bench/resolve/main/data/train-00000-of-00001.parquet", "type": "parquet_file", "cap": 5000, "desc": "Real GitHub issue resolution via terminal, git diff, & tests"},
]

def stream_jsonl(url, cap):
    req = urllib.request.Request(url, headers=HEADERS)
    rows = []
    with urllib.request.urlopen(req, timeout=120) as resp:
        for line in resp:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line.decode("utf-8", errors="ignore"))
                rows.append(item)
                if len(rows) >= cap:
                    break
            except Exception:
                continue
    return rows

def download_json_array(url, cap, tmp_path):
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=180) as resp, open(tmp_path, "wb") as f_out:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            f_out.write(chunk)
    with open(tmp_path, "r", encoding="utf-8", errors="ignore") as f:
        data = json.load(f)
    tmp_path.unlink(missing_ok=True)
    return data[:cap]

def download_parquet_file(url, cap, tmp_path):
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=180) as resp, open(tmp_path, "wb") as f_out:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            f_out.write(chunk)
    table = pq.read_table(tmp_path)
    if table.num_rows > cap:
        table = table.slice(0, cap)
    tmp_path.unlink(missing_ok=True)
    return table

def save_to_parquet(data, out_path):
    if isinstance(data, pa.Table):
        pq.write_table(data, out_path, compression="zstd")
        return data.num_rows
    df = pd.DataFrame(data)
    for col in df.columns:
        if df[col].apply(lambda x: isinstance(x, (dict, list))).any():
            df[col] = df[col].apply(lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, (dict, list)) else ("" if pd.isna(x) else str(x)))
    table = pa.Table.from_pandas(df)
    pq.write_table(table, out_path, compression="zstd")
    return len(df)

def main():
    print("=" * 80)
    print(" 🚀 STARTING MULTI-SOURCE AGENTIC SFT DATASET DOWNLOAD (5,000 CAP)")
    print(f" Target Directory: {OUT_DIR}")
    print("=" * 80)

    # Mirror script to /projects/data/.../taxonomy/
    tax_script = Path("/projects/data/datasets/code_data/sai_rupesh/taxonomy/download_agentic_sources.py")
    try:
        with open(__file__, "r") as f_in, open(tax_script, "w") as f_out:
            f_out.write(f_in.read())
        tax_script.chmod(0o755)
    except Exception:
        pass

    manifest = []
    total_downloaded = 0
    t_start = time.time()

    for item in SOURCES:
        src_id = item["id"]
        name = item["name"]
        cap = item["cap"]
        url = item["url"]
        stype = item["type"]
        desc = item["desc"]

        out_file = OUT_DIR / f"{name}_5k.parquet"
        print(f"\n[{src_id}/18] Processing: {name}")
        print(f"     Source: {item['mixture']} | Cap Target: {cap:,}")

        if out_file.exists() and out_file.stat().st_size > 0:
            existing_rows = pq.read_table(out_file).num_rows
            print(f"     ✓ Already exists with {existing_rows:,} rows. Skipping.")
            manifest.append({
                "id": src_id,
                "name": name,
                "mixture": item["mixture"],
                "file": out_file.name,
                "rows": existing_rows,
                "status": "COMPLETED",
                "desc": desc
            })
            total_downloaded += existing_rows
            continue

        tmp_file = TMP_DIR / f"{name}.tmp"
        t0 = time.time()

        try:
            if stype == "jsonl_stream":
                data = stream_jsonl(url, cap)
            elif stype == "json_array_stream":
                data = download_json_array(url, cap, tmp_file)
            elif stype == "parquet_file":
                data = download_parquet_file(url, cap, tmp_file)
            else:
                raise ValueError(f"Unknown type: {stype}")

            saved_rows = save_to_parquet(data, out_file)
            elapsed = time.time() - t0
            file_mb = out_file.stat().st_size / (1024 * 1024)
            print(f"     ✅ Downloaded {saved_rows:,} rows ({file_mb:.2f} MB) in {elapsed:.1f}s")

            manifest.append({
                "id": src_id,
                "name": name,
                "mixture": item["mixture"],
                "file": out_file.name,
                "rows": saved_rows,
                "size_mb": round(file_mb, 2),
                "status": "COMPLETED",
                "desc": desc
            })
            total_downloaded += saved_rows

        except Exception as e:
            print(f"     ❌ Error downloading {name}: {e}")
            manifest.append({
                "id": src_id,
                "name": name,
                "mixture": item["mixture"],
                "file": out_file.name,
                "rows": 0,
                "status": f"FAILED: {e}",
                "desc": desc
            })

    try:
        TMP_DIR.rmdir()
    except Exception:
        pass

    manifest_path = OUT_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    stats_md_path = Path("/projects/data/datasets/code_data/sai_rupesh/taxonomy/agentic_sources_stats.md")
    total_time = time.time() - t_start

    md_lines = [
        "# Agentic SFT Datasets: Multi-Source 5k Capped Manifest",
        f"**Completed At**: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
        f"**Storage Directory**: `{OUT_DIR}`",
        f"**Total Samples Downloaded**: **{total_downloaded:,} rows**",
        f"**Total Execution Time**: **{total_time / 60:.1f} minutes**",
        "",
        "| # | Sub-Source Name | Original Mixture | Rows | File Size (MB) | Trace Type / Description |",
        "|---|---|---|:---:|:---:|---|",
    ]

    for m in manifest:
        sz = m.get("size_mb", "-")
        md_lines.append(f"| {m['id']} | `{m['name']}` | `{m['mixture']}` | {m['rows']:,} | {sz} | {m['desc']} |")

    with open(stats_md_path, "w") as f:
        f.write("\n".join(md_lines) + "\n")

    print("\n" + "=" * 80)
    print(f" 🎉 AGENTIC DOWNLOAD COMPLETED! Total Rows: {total_downloaded:,}")
    print(f" Manifest saved: {manifest_path}")
    print(f" Stats report:   {stats_md_path}")
    print("=" * 80)

if __name__ == "__main__":
    main()
