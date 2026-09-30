#!/usr/bin/env python3
"""
⚡ Unified 36-Source Open-Ended Domain Taxonomy Discovery Pipeline
Covers 19 Tulu-3 SFT sub-datasets + 17 Agentic SFT sources (capped at 5,000 samples per source).
Model: openai/gpt-oss-120b (SGLang Router)
Extracts: Multi-label weighted nested JSON hierarchy:
  domains -> subdomains -> sub_subdomains
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
import pandas as pd
import pyarrow.parquet as pq

API_URL     = "http://sglang-router.sglang.svc.cluster.local:30000/v1/chat/completions"
API_KEY = os.environ.get("SGLANG_API_KEY", "")  # set this env var before running -- never hardcode the real key
MODEL_NAME  = "openai/gpt-oss-120b"
CONCURRENCY = 24
MAX_TOKENS  = 550
MAX_PER_SOURCE = 5000

BASE_DIR        = Path("/projects/data/datasets/code_data/sai_rupesh/taxonomy")
TULU_DATA_DIR   = BASE_DIR / "data"
AGENT_DATA_DIR  = BASE_DIR / "agentic_data"
OUTPUT_DIR      = BASE_DIR / "discovery_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DOMAIN_SYSTEM_PROMPT = """You are an expert academic, scientific, and industry taxonomist doing OPEN-ENDED DISCOVERY. You are helping build a taxonomy from scratch, so propose specific, granular categories freely — do not limit yourself to any existing list.

Analyze the input and identify EVERY distinct domain it touches, not just the primary one. Most inputs will have 1 domain; some will genuinely span 2-3. Do not force a single domain onto multi-domain content, and do not inflate a single-domain input into multiple domains just to fill the list.

For each domain identified, go one level deeper into subdomain, and one level deeper into sub_subdomain, following the same principle (usually 1, sometimes 2, rarely more).

RULES:
- Be highly specific and fine-grained (e.g. "Distributed Systems & Consensus Algorithms", not "Computer Science" or "Technology").
- Never use vague umbrella buckets: "General", "Miscellaneous", "Other", "Everyday Life", "UNKNOWN".
- Max 3 domains, max 2 subdomains per domain, max 2 sub_subdomains per subdomain. Only include a domain/subdomain/sub_subdomain if it has genuine, non-trivial presence in the content (weight >= 0.15) — do not list something the input just barely touches in passing.
- weight is relative to its parent (subdomain weights sum to ~1 within their domain).

Output strictly valid JSON, no text outside it:
{
  "domains": [
    {
      "domain": "...",
      "weight": 0.0-1.0,
      "subdomains": [
        {
          "subdomain": "...",
          "weight": 0.0-1.0,
          "sub_subdomains": [
            {"name": "...", "weight": 0.0-1.0}
          ]
        }
      ]
    }
  ]
}"""

def format_text(messages_or_text, max_chars=4000):
    if isinstance(messages_or_text, str):
        return messages_or_text[:max_chars]
    parts = []
    total = 0
    if isinstance(messages_or_text, list):
        for msg in messages_or_text:
            if isinstance(msg, dict):
                role = msg.get("role", "?").upper()
                content = msg.get("content", "")
                if isinstance(content, list):
                    content = " ".join(c.get("text", "") if isinstance(c, dict) else str(c) for c in content)
                seg = f"[{role}]: {content}"
            else:
                seg = str(msg)
            if total + len(seg) > max_chars:
                parts.append(seg[:max_chars - total] + " ...[truncated]")
                break
            parts.append(seg)
            total += len(seg)
    return "\n".join(parts)

def parse_domain_json(raw):
    s_idx = raw.find('{')
    e_idx = raw.rfind('}')
    if s_idx != -1 and e_idx != -1 and e_idx > s_idx:
        try:
            parsed = json.loads(raw[s_idx:e_idx+1])
            if "domains" in parsed and isinstance(parsed["domains"], list) and len(parsed["domains"]) > 0:
                return parsed
        except Exception:
            pass
    return None

def fallback_domain(text):
    low = text.lower()
    if any(k in low for k in ("def ", "function", "class ", "algorithm", "quicksort", "dijkstra")):
        d, sd, ssd = "Computer Science", "Algorithms & Data Structures", "Procedural Programming"
    elif any(k in low for k in ("bug", "error", "traceback", "exception", "fix", "debug")):
        d, sd, ssd = "Software Engineering", "Application Debugging", "Error Diagnostic Systems"
    elif any(k in low for k in ("calculate", "solve", "equation", "derivative", "matrix")):
        d, sd, ssd = "Mathematics", "Mathematical Analysis", "Algebraic Derivation"
    elif any(k in low for k in ("biology", "medical", "gene", "protein", "dna", "clinical")):
        d, sd, ssd = "Biomedical Sciences", "Molecular Biology", "Clinical Genetics"
    elif any(k in low for k in ("physics", "quantum", "gravity", "force", "thermodynamics")):
        d, sd, ssd = "Physical Sciences", "Theoretical Physics", "Thermodynamics"
    elif any(k in low for k in ("history", "war", "century", "empire", "revolution")):
        d, sd, ssd = "Humanities", "World History", "Historical Analysis"
    elif any(k in low for k in ("business", "stock", "finance", "market", "revenue", "investment")):
        d, sd, ssd = "Economics & Finance", "Financial Markets", "Corporate Valuation"
    else:
        d, sd, ssd = "Applied Science & Technology", "Information Systems", "Technical Literature"
    
    return {
        "domains": [
            {
                "domain": d,
                "weight": 1.0,
                "subdomains": [
                    {
                        "subdomain": sd,
                        "weight": 1.0,
                        "sub_subdomains": [{"name": ssd, "weight": 1.0}]
                    }
                ]
            }
        ]
    }

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
                {"role": "system", "content": DOMAIN_SYSTEM_PROMPT},
                {"role": "user", "content": f"Source: {source}\nText:\n{text}"}
            ],
            "temperature": 0.1,
            "max_tokens": MAX_TOKENS
        }

        async with sem:
            success = False
            domain_obj = None
            for attempt in range(2):
                try:
                    async with session.post(API_URL, json=payload, headers=headers, timeout=60) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            msg = data["choices"][0]["message"]
                            raw = msg.get("content", "") or msg.get("reasoning_content", "") or ""
                            domain_obj = parse_domain_json(raw)
                            if domain_obj:
                                success = True
                                break
                except Exception:
                    await asyncio.sleep(0.3 * (attempt + 1))

            if not success or not domain_obj:
                domain_obj = fallback_domain(text)

            # Extract primary values for fast sorting
            try:
                p_dom = domain_obj["domains"][0]["domain"]
                p_sub = domain_obj["domains"][0]["subdomains"][0]["subdomain"]
                p_ssd = domain_obj["domains"][0]["subdomains"][0]["sub_subdomains"][0]["name"]
                p_wt = domain_obj["domains"][0].get("weight", 1.0)
            except Exception:
                p_dom, p_sub, p_ssd, p_wt = "General Domain", "Field", "Topic", 1.0

            tax_json_str = json.dumps(domain_obj, ensure_ascii=False)

            async with csv_lock:
                csv_writer.writerow([row_id, source, p_dom, p_sub, p_ssd, p_wt, tax_json_str])
                out_f.flush()

            monitor.update(success=success)
            queue.task_done()

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
        pct = (self.done / self.total * 100) if self.total else 0
        print(f"\r[Discovery Domain Shard {self.shard_id}: {self.done:,}/{self.total:,} ({pct:.1f}%)] "
              f"Rate: {rate:4.1f} req/s | Errors: {self.errors}", end="", flush=True)

async def main(shard_idx):
    output_csv = OUTPUT_DIR / f"discovery_domains_shard_{shard_idx}.csv"
    print(f"[Discovery Domains Shard {shard_idx}] Output file: {output_csv.name}")

    completed_ids = set()
    write_header = not output_csv.exists() or output_csv.stat().st_size == 0
    if not write_header:
        with open(output_csv, "r", encoding="utf-8", errors="ignore") as f:
            r = csv.reader(f)
            next(r, None)
            for row in r:
                if row:
                    completed_ids.add(row[0])

    # Build work items for this shard
    work_items = []
    
    # 1. Tulu-3 Shard File
    tulu_files = sorted(glob.glob(str(TULU_DATA_DIR / "*.parquet")))
    if shard_idx < len(tulu_files):
        tulu_file = tulu_files[shard_idx]
        print(f"[Shard {shard_idx}] Loading Tulu-3 shard: {Path(tulu_file).name}")
        pfile = pq.ParquetFile(tulu_file)
        src_counts = Counter()
        for batch in pfile.iter_batches(batch_size=512, columns=["id", "messages", "source"]):
            df_batch = batch.to_pandas()
            for _, row in df_batch.iterrows():
                row_id = str(row["id"])
                if row_id in completed_ids:
                    continue
                source = str(row.get("source", "unknown")).strip()
                if src_counts[source] >= MAX_PER_SOURCE:
                    continue
                src_counts[source] += 1
                text = format_text(row.get("messages", []))
                work_items.append((row_id, source, text))

    # 2. Agentic Sources (partitioned across the 6 shards)
    agent_files = sorted(glob.glob(str(AGENT_DATA_DIR / "*.parquet")))
    assigned_agents = [f for i, f in enumerate(agent_files) if i % 6 == shard_idx]
    for af in assigned_agents:
        print(f"[Shard {shard_idx}] Loading Agentic source: {Path(af).name}")
        table = pq.read_table(af)
        df_agent = table.to_pandas()
        src_name = Path(af).stem.replace("_5k", "")
        for idx, row in df_agent.iterrows():
            row_id = f"{src_name}_{idx}"
            if row_id in completed_ids:
                continue
            if "messages" in row and row["messages"]:
                raw_msg = row["messages"]
                if isinstance(raw_msg, str):
                    try: raw_msg = json.loads(raw_msg)
                    except Exception: pass
                text = format_text(raw_msg)
            elif "conversation" in row and row["conversation"]:
                text = format_text(row["conversation"])
            else:
                text = str(row.to_dict())[:4000]
            work_items.append((row_id, src_name, text))

    total_rows = len(work_items)
    print(f"[Shard {shard_idx}] Total items to extract: {total_rows:,} (already done: {len(completed_ids):,})")
    if total_rows == 0:
        print(f"[Shard {shard_idx}] All items completed!")
        return

    monitor = ShardMonitor(shard_idx, total_rows)
    queue = asyncio.Queue(maxsize=CONCURRENCY * 3)
    sem = asyncio.Semaphore(CONCURRENCY)
    csv_lock = asyncio.Lock()

    out_mode = "a" if not write_header else "w"
    with open(output_csv, out_mode, newline="", encoding="utf-8") as out_f:
        csv_writer = csv.writer(out_f)
        if write_header:
            csv_writer.writerow(["id", "source", "primary_domain", "primary_subdomain", "primary_sub_subdomain", "primary_weight", "taxonomy_json"])
            out_f.flush()

        connector = aiohttp.TCPConnector(limit=CONCURRENCY * 2, ttl_dns_cache=300)
        async with aiohttp.ClientSession(connector=connector) as session:
            workers = [
                asyncio.create_task(worker(queue, session, sem, csv_writer, csv_lock, out_f, monitor))
                for _ in range(CONCURRENCY)
            ]

            for item in work_items:
                await queue.put(item)

            for _ in range(CONCURRENCY):
                await queue.put(None)

            await queue.join()
            for w in workers:
                w.cancel()
            out_f.flush()

    print(f"\n[Shard {shard_idx}] Domain discovery completed! Output saved to {output_csv}")

if __name__ == "__main__":
    k8s_index = os.environ.get("JOB_COMPLETION_INDEX", "0")
    shard_id = int(k8s_index)
    asyncio.run(main(shard_id))
