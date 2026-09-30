#!/usr/bin/env python3
"""
Parallel Sharded Fine-Grained 3-Level Domain Hierarchy Extraction for 17 Agentic Sources
Supports: Kubernetes Indexed Jobs (JOB_COMPLETION_INDEX 0..5)
Extracts: domain, subdomain, sub_subdomain (Fine-Grained Domain Ontology, No Vague Buckets)
Model: GPT-OSS-120B (18 Cluster Replicas via SGLang Router)
Output: CSV per shard (agentic_domains_shard_X.csv)
"""

import asyncio
import aiohttp
import csv
import json
import os
import re
import sys
import time
import argparse
import glob
from pathlib import Path
from collections import Counter
import pyarrow.parquet as pq

# ── Cluster SGLang Router Configuration (18 Replicas of GPT-OSS-120B) ────────
API_URL     = "http://sglang-router.sglang.svc.cluster.local:30000/v1/chat/completions"
API_KEY = os.environ.get("SGLANG_API_KEY", "")  # set this env var before running -- never hardcode the real key
MODEL_NAME  = "openai/gpt-oss-120b"
CONCURRENCY = 12          # 12 per pod x 6 pods = 72 domain slots (144 total with tasks, leaving 128 slots for team)
MAX_TOKENS  = 550         # Nested JSON requires up to 400-500 tokens

BASE_DIR    = Path("/projects/data/datasets/code_data/sai_rupesh/taxonomy")
DATA_DIR    = BASE_DIR / "agentic_data"
OUT_DIR     = BASE_DIR / "agentic_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Exact same prompt as Tulu-3 Domain Extraction
SYSTEM_PROMPT = """You are an expert academic, scientific, and industry taxonomist.
Classify the user instruction into a highly specific, fine-grained 3-level domain hierarchy:
1. domain: The precise academic, scientific, or professional discipline (e.g. Distributed Systems & Cloud Architecture, Organic Chemistry, Microeconomics & Consumer Theory, Intellectual Property Law, Pediatric Cardiology, Renaissance Art History). Never use vague umbrella buckets like "Everyday Life", "General Knowledge", "Miscellaneous", "Science", "Other", or "UNKNOWN".
2. subdomain: The specific sub-discipline, branch, or field.
3. sub_subdomain: The exact granular subject topic or research area.

RULES:
- Be highly specific, descriptive, and fine-grained at all 3 levels.
- Never output "UNKNOWN", "OTHER", "N/A", "NONE", or vague generic words.
- Output strictly valid JSON with these 3 keys:
{"domain": "...", "subdomain": "...", "sub_subdomain": "..."}"""

BANNED_VALUES = {
    "UNKNOWN", "OTHER", "N/A", "NONE", "", "NULL", "UNDEFINED",
    "GENERAL", "MISCELLANEOUS", "EVERYDAY LIFE", "DAILY LIFE", "GENERAL KNOWLEDGE"
}

def sanitize_val(val, fallback):
    if not val or not isinstance(val, str):
        return fallback
    clean = val.strip()
    if clean.upper() in BANNED_VALUES:
        return fallback
    return clean

def extract_text_from_row(row, max_chars=3500):
    for col in ["messages", "conversations", "trajectory", "turns", "dialogue"]:
        if col in row and row[col]:
            val = row[col]
            if isinstance(val, str):
                try:
                    val = json.loads(val)
                except Exception:
                    return val[:max_chars]
            if isinstance(val, list):
                parts = []
                total = 0
                for m in val:
                    if isinstance(m, dict):
                        role = m.get("role") or m.get("from") or "UNKNOWN"
                        content = m.get("content") or m.get("value") or m.get("text") or ""
                        if isinstance(content, list):
                            content = " ".join(c.get("text", "") if isinstance(c, dict) else str(c) for c in content)
                        seg = f"[{str(role).upper()}]: {content}"
                    else:
                        seg = str(m)
                    if total + len(seg) > max_chars:
                        parts.append(seg[:max_chars - total] + " ...[truncated]")
                        break
                    parts.append(seg)
                    total += len(seg)
                return "\n".join(parts)

    combined = []
    for col in ["problem_statement", "instruction", "input", "prompt", "question", "task", "code"]:
        if col in row and row[col] and str(row[col]).strip():
            combined.append(f"[{col.upper()}]: {row[col]}")
    for col in ["hints_text", "patch", "output", "response", "solution", "answer"]:
        if col in row and row[col] and str(row[col]).strip():
            combined.append(f"[{col.upper()}]: {row[col]}")

    if combined:
        return "\n\n".join(combined)[:max_chars]

    return " | ".join(f"{k}: {v}" for k, v in row.items() if v)[:max_chars]

def fallback_from_text(text):
    low = text.lower()
    if any(k in low for k in ("def ", "import ", "python", "javascript", "code", "function", "class ", "git", "repo")):
        return ("Computer Science & Software Engineering", "Applied Systems Programming", "Software Repository Maintenance")
    if any(k in low for k in ("calculate", "solve", "equation", "derivative", "integral", "matrix", "math")):
        return ("Mathematical & Quantitative Sciences", "Applied Mathematics", "Computational Problem Solving")
    if any(k in low for k in ("sql", "database", "table", "schema", "relational")):
        return ("Data Architecture & Information Systems", "Database Management Systems", "Relational Query Processing")
    if any(k in low for k in ("tool", "api", "function_call", "json", "endpoint")):
        return ("Systems Architecture & Middleware", "API Integration & Web Services", "Automated Service Orchestration")
    if any(k in low for k in ("search", "web", "browser", "shopping", "ecommerce")):
        return ("Information Retrieval & Web Systems", "Electronic Commerce & Search", "Autonomous Web Navigation")
    return ("Artificial Intelligence & Intelligent Systems", "Autonomous Agent Architectures", "Multi-Agent System Planning")

class ShardMonitor:
    def __init__(self, shard_id, total):
        self.shard_id = shard_id
        self.total = total
        self.done = 0
        self.errors = 0
        self.start_time = time.time()

    def update(self, success=True):
        self.done += 1
        if not success:
            self.errors += 1
        elapsed = time.time() - self.start_time
        rate = self.done / elapsed if elapsed > 0 else 0
        eta_min = ((self.total - self.done) / rate) / 60 if rate > 0 else 0
        pct = (self.done / self.total * 100) if self.total else 0
        print(f"\r[Agentic Domains Shard {self.shard_id}: {self.done:,}/{self.total:,} ({pct:.1f}%)] "
              f"Rate: {rate:4.1f} req/s | Errors: {self.errors} | ETA: {eta_min:4.1f}m", end="", flush=True)

async def worker(queue, session, sem, csv_writer, csv_lock, out_f, monitor):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            break

        row_id, source, text = item
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Source: {source}\nTrajectory / Interaction:\n{text}"}
            ],
            "temperature": 0.1,
            "max_tokens": MAX_TOKENS
        }

        async with sem:
            success = False
            domain, subdomain, sub_subdomain = None, None, None
            for attempt in range(2):
                try:
                    async with session.post(API_URL, json=payload, headers=headers, timeout=40) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            msg = data["choices"][0]["message"]
                            raw = msg.get("content", "") or msg.get("reasoning_content", "") or ""
                            
                            # 1. JSON parsing
                            s_idx = raw.find('{')
                            e_idx = raw.rfind('}')
                            if s_idx != -1 and e_idx != -1 and e_idx > s_idx:
                                try:
                                    parsed = json.loads(raw[s_idx:e_idx+1])
                                    domain = sanitize_val(parsed.get("domain"), None)
                                    subdomain = sanitize_val(parsed.get("subdomain"), None)
                                    sub_subdomain = sanitize_val(parsed.get("sub_subdomain"), None)
                                except Exception:
                                    pass

                            # 2. Regex fallback
                            if not domain:
                                m = re.search(r'"domain"\s*:\s*"([^"]+)"', raw)
                                if m: domain = sanitize_val(m.group(1), None)
                            if not subdomain:
                                m = re.search(r'"subdomain"\s*:\s*"([^"]+)"', raw)
                                if m: subdomain = sanitize_val(m.group(1), None)
                            if not sub_subdomain:
                                m = re.search(r'"sub_subdomain"\s*:\s*"([^"]+)"', raw)
                                if m: sub_subdomain = sanitize_val(m.group(1), None)

                            if domain and subdomain:
                                success = True
                                break
                except Exception:
                    await asyncio.sleep(0.3 * (attempt + 1))

            if not success or not domain:
                domain, subdomain, sub_subdomain = fallback_from_text(text)

            if not sub_subdomain:
                sub_subdomain = f"{subdomain} Applications"

            async with csv_lock:
                csv_writer.writerow([row_id, source, domain, subdomain, sub_subdomain])
                out_f.flush()

            monitor.update(success=success)
            queue.task_done()

async def main(shard_idx):
    all_files = sorted([f for f in glob.glob(str(DATA_DIR / "*.parquet")) if os.path.getsize(f) > 5000])
    num_shards = 6
    
    assigned_files = [f for i, f in enumerate(all_files) if i % num_shards == shard_idx]
    
    output_csv = OUT_DIR / f"agentic_domains_shard_{shard_idx}.csv"
    print(f"\n[Worker Shard {shard_idx}] Assigned {len(assigned_files)} files:")
    for f in assigned_files:
        print(f"   • {Path(f).name}")
    print(f"[Worker Shard {shard_idx}] Output file: {output_csv.name}")

    completed_ids = set()
    write_header = not output_csv.exists() or output_csv.stat().st_size == 0
    if output_csv.exists() and output_csv.stat().st_size > 0:
        with open(output_csv, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)
            for r in reader:
                if r: completed_ids.add(r[0])
        print(f"[Worker Shard {shard_idx}] Resuming: {len(completed_ids):,} rows already in shard output")

    total_shard_rows = sum(pq.ParquetFile(f).metadata.num_rows for f in assigned_files)
    remaining_rows = total_shard_rows - len(completed_ids)
    print(f"[Worker Shard {shard_idx}] Total rows in shard: {total_shard_rows:,} | Remaining to do: {remaining_rows:,}")

    if remaining_rows <= 0:
        print(f"[Worker Shard {shard_idx}] All {total_shard_rows:,} rows already completed!")
        return

    monitor = ShardMonitor(shard_idx, total_shard_rows)
    monitor.done = len(completed_ids)

    queue = asyncio.Queue(maxsize=CONCURRENCY * 3)
    sem = asyncio.Semaphore(CONCURRENCY)
    csv_lock = asyncio.Lock()

    out_mode = "a" if not write_header else "w"
    with open(output_csv, out_mode, newline="", encoding="utf-8") as out_f:
        csv_writer = csv.writer(out_f)
        if write_header:
            csv_writer.writerow(["id", "source", "domain", "subdomain", "sub_subdomain"])
            out_f.flush()

        connector = aiohttp.TCPConnector(limit=CONCURRENCY * 2, ttl_dns_cache=300)
        async with aiohttp.ClientSession(connector=connector) as session:
            workers = [
                asyncio.create_task(worker(queue, session, sem, csv_writer, csv_lock, out_f, monitor))
                for _ in range(CONCURRENCY)
            ]

            for pf_path in assigned_files:
                src_name = Path(pf_path).stem.replace("_5k", "")
                pfile = pq.ParquetFile(pf_path)
                for batch in pfile.iter_batches(batch_size=256):
                    df = batch.to_pandas()
                    for idx, row in df.iterrows():
                        rid = str(row.get("id", row.get("instance_id", f"{src_name}_{idx}")))
                        if rid in completed_ids:
                            continue
                        text = extract_text_from_row(row)
                        await queue.put((rid, src_name, text))

            for _ in range(CONCURRENCY):
                await queue.put(None)

            await queue.join()
            for w in workers:
                w.cancel()
            out_f.flush()

    print(f"\n[Worker Shard {shard_idx}] Shard extraction completed! Saved to {output_csv}")

if __name__ == "__main__":
    k8s_index = os.environ.get("JOB_COMPLETION_INDEX", "0")
    shard_id = int(k8s_index)
    asyncio.run(main(shard_id))
