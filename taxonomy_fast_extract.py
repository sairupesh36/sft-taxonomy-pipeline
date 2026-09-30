#!/usr/bin/env python3
"""
Parallel Sharded Fine-Grained 3-Level Task Hierarchy Extraction for allenai/tulu-3-sft-mixture
Supports: Kubernetes Indexed Jobs (JOB_COMPLETION_INDEX 0..5)
Extracts: task, subtask, sub_subtask (Fine-Grained Task Ontology, No Vague Buckets)
Model: GPT-OSS-120B (17 Cluster Replicas via SGLang Multi-Model Router)
Output: CSV Format per shard (extracted_tasks_shard_X.csv)
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

# ── Cluster SGLang Router Configuration (17 Replicas of GPT-OSS-120B) ────────
API_URL     = "http://sglang-router.sglang.svc.cluster.local:30000/v1/chat/completions"
API_KEY = os.environ.get("SGLANG_API_KEY", "")  # set this env var before running -- never hardcode the real key
MODEL_NAME  = "openai/gpt-oss-120b"
CONCURRENCY = 24          # 24 concurrent per pod x 6 pods = 144 cluster-wide concurrency (matches 288 SGLang GPU slots)
MAX_TOKENS  = 380         # Sufficient for GPT internal reasoning (~180 tokens) + complete JSON (~80 tokens)
MAX_PER_SOURCE = 5000     # Cap at 5,000 samples per sub-dataset source

BASE_DIR    = Path("/projects/data/datasets/code_data/sai_rupesh/taxonomy")
DATA_DIR    = BASE_DIR / "data"

SYSTEM_PROMPT = """You are an expert AI task taxonomist.
Classify the user prompt into a highly specific, fine-grained 3-level task hierarchy:
1. task: The precise functional capability or task category (e.g. Algorithm Implementation, Data Visualization, Unit Testing, Text Summarization, Database Querying, Creative Storytelling). Never use vague umbrella buckets like "Question Answering", "Code Generation", "Problem Solving", "General", or "Information Retrieval".
2. subtask: The specific procedural category or technical approach.
3. sub_subtask: The exact granular objective or operation being performed.

RULES:
- Be highly specific, descriptive, and fine-grained at all 3 levels.
- Never output "UNKNOWN", "OTHER", "N/A", "NONE", or vague generic words.
- Output strictly valid JSON with these 3 keys:
{"task": "...", "subtask": "...", "sub_subtask": "..."}"""

BANNED_VALUES = {"UNKNOWN", "OTHER", "N/A", "NONE", "", "NULL", "UNDEFINED", "GENERAL", "MISCELLANEOUS"}

def sanitize_val(val, fallback):
    if not val or not isinstance(val, str):
        return fallback
    clean = val.strip()
    if clean.upper() in BANNED_VALUES:
        return fallback
    return clean

def format_text(messages, max_chars=4000):
    parts = []
    total = 0
    for msg in messages:
        role = msg.get("role", "?").upper()
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(c.get("text", "") if isinstance(c, dict) else str(c) for c in content)
        seg = f"[{role}]: {content}"
        if total + len(seg) > max_chars:
            parts.append(seg[:max_chars - total] + " ...[truncated]")
            break
        parts.append(seg)
        total += len(seg)
    return "\n".join(parts)

def fallback_from_text(text):
    low = text.lower()
    if any(k in low for k in ("def ", "function", "class ", "algorithm", "quicksort", "dijkstra", "binary search")):
        return ("Algorithm Implementation", "Procedural Code Writing", "Function Construction")
    if any(k in low for k in ("bug", "error", "traceback", "exception", "fix", "debug")):
        return ("Code Debugging", "Error Diagnosis", "Software Defect Resolution")
    if any(k in low for k in ("calculate", "solve", "equation", "derivative", "integral", "matrix")):
        return ("Mathematical Computation", "Analytical Problem Solving", "Symbolic Derivation")
    if any(k in low for k in ("summarize", "summary", "tldr", "bullet", "recap")):
        return ("Document Summarization", "Key Information Synthesis", "Executive Summary Formulation")
    if any(k in low for k in ("translate", "translation", "spanish", "french", "german")):
        return ("Natural Language Translation", "Cross-Lingual Adaptation", "Contextual Text Localization")
    if any(k in low for k in ("recipe", "cook", "ingredients", "bake", "cuisine")):
        return ("Culinary Instruction", "Recipe Formulation", "Cooking Step Sequencing")
    if any(k in low for k in ("story", "poem", "essay", "dialogue", "creative")):
        return ("Creative Composition", "Literary Writing", "Narrative Development")
    return ("Conceptual Explanation", "Subject Matter Analysis", "Technical Exposition")

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
        print(f"\r[Shard {self.shard_id}: {self.done:,}/{self.total:,} ({pct:.1f}%)] "
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
                {"role": "user", "content": f"Source: {source}\nText:\n{text}"}
            ],
            "temperature": 0.1,
            "max_tokens": MAX_TOKENS
        }

        async with sem:
            success = False
            task, subtask, sub_subtask = None, None, None
            for attempt in range(2):
                try:
                    async with session.post(API_URL, json=payload, headers=headers, timeout=40) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            msg = data["choices"][0]["message"]
                            raw = msg.get("content", "") or msg.get("reasoning_content", "") or ""
                            
                            # 1. Try JSON block parsing
                            s_idx = raw.find('{')
                            e_idx = raw.rfind('}')
                            if s_idx != -1 and e_idx != -1 and e_idx > s_idx:
                                try:
                                    parsed = json.loads(raw[s_idx:e_idx+1])
                                    task = sanitize_val(parsed.get("task"), None)
                                    subtask = sanitize_val(parsed.get("subtask"), None)
                                    sub_subtask = sanitize_val(parsed.get("sub_subtask"), None)
                                except Exception:
                                    pass

                            # 2. Regex fallback if JSON was slightly malformed
                            if not task:
                                m = re.search(r'"task"\s*:\s*"([^"]+)"', raw)
                                if m: task = sanitize_val(m.group(1), None)
                            if not subtask:
                                m = re.search(r'"subtask"\s*:\s*"([^"]+)"', raw)
                                if m: subtask = sanitize_val(m.group(1), None)
                            if not sub_subtask:
                                m = re.search(r'"sub_subtask"\s*:\s*"([^"]+)"', raw)
                                if m: sub_subtask = sanitize_val(m.group(1), None)

                            if task and subtask:
                                success = True
                                break
                except Exception:
                    await asyncio.sleep(0.3 * (attempt + 1))

            if not success or not task:
                task, subtask, sub_subtask = fallback_from_text(text)

            if not sub_subtask:
                sub_subtask = f"{subtask} Execution"

            async with csv_lock:
                csv_writer.writerow([row_id, source, task, subtask, sub_subtask])
                out_f.flush()

            monitor.update(success=success)
            queue.task_done()

async def main(shard_idx, limit=None):
    parquet_files = sorted(glob.glob(str(DATA_DIR / "*.parquet")))
    if shard_idx >= len(parquet_files):
        print(f"Error: Shard index {shard_idx} >= number of files ({len(parquet_files)})")
        return

    target_parquet = parquet_files[shard_idx]
    output_csv = BASE_DIR / f"extracted_tasks_shard_{shard_idx}.csv"
    print(f"[Worker Shard {shard_idx}] Processing: {Path(target_parquet).name}")
    print(f"[Worker Shard {shard_idx}] Output file: {output_csv.name}")

    completed_ids = set()
    source_counts = Counter()
    write_header = not output_csv.exists() or output_csv.stat().st_size == 0

    for shard_f in sorted(BASE_DIR.glob("extracted_tasks_shard_*.csv")):
        if shard_f.exists() and shard_f.stat().st_size > 0:
            with open(shard_f, "r", encoding="utf-8", errors="ignore") as f:
                reader = csv.reader(f)
                next(reader, None)
                for r in reader:
                    if len(r) >= 2:
                        if shard_f == output_csv:
                            completed_ids.add(r[0])
                        source_counts[r[1].strip()] += 1

    print(f"[Worker Shard {shard_idx}] Resuming: {len(completed_ids):,} rows already in shard output")
    capped_sources = sum(1 for src, cnt in source_counts.items() if cnt >= MAX_PER_SOURCE)
    print(f"[Worker Shard {shard_idx}] 5k Cap Filter: {capped_sources} sources already at/above {MAX_PER_SOURCE:,} limit")

    pfile = pq.ParquetFile(target_parquet)
    total_shard_rows = pfile.metadata.num_rows
    remaining_rows = total_shard_rows - len(completed_ids)
    print(f"[Worker Shard {shard_idx}] Total rows in shard: {total_shard_rows:,} | To do: {remaining_rows:,}")

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
            csv_writer.writerow(["id", "source", "task", "subtask", "sub_subtask"])
            out_f.flush()

        connector = aiohttp.TCPConnector(limit=CONCURRENCY * 2, ttl_dns_cache=300)
        async with aiohttp.ClientSession(connector=connector) as session:
            workers = [
                asyncio.create_task(worker(queue, session, sem, csv_writer, csv_lock, out_f, monitor))
                for _ in range(CONCURRENCY)
            ]

            batch_size = 512
            feed_count = 0

            for batch in pfile.iter_batches(batch_size=batch_size, columns=["id", "messages", "source"]):
                df_batch = batch.to_pandas()
                for _, row in df_batch.iterrows():
                    row_id = str(row["id"])
                    if row_id in completed_ids:
                        continue

                    source = str(row.get("source", "unknown")).strip()
                    if source_counts[source] >= MAX_PER_SOURCE:
                        continue  # 5k cap reached, skip!

                    source_counts[source] += 1
                    text = format_text(row.get("messages", []))
                    await queue.put((row_id, source, text))
                    feed_count += 1

                    if limit and feed_count >= limit:
                        break
                if limit and feed_count >= limit:
                    break

            for _ in range(CONCURRENCY):
                await queue.put(None)

            await queue.join()
            for w in workers:
                w.cancel()
            out_f.flush()

    print(f"\n[Worker Shard {shard_idx}] Shard extraction completed! Output saved to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel Sharded Task Extractor")
    parser.add_argument("--shard", type=int, default=None, help="Shard index (0 to 5)")
    parser.add_argument("--limit", type=int, default=None, help="Optional test row limit")
    args = parser.parse_args()

    shard_id = args.shard
    if shard_id is None:
        k8s_index = os.environ.get("JOB_COMPLETION_INDEX")
        if k8s_index is not None:
            try:
                shard_id = int(k8s_index)
            except ValueError:
                pass

    if shard_id is None:
        shard_id = 0
        print(f"Warning: No --shard or JOB_COMPLETION_INDEX specified. Defaulting to shard {shard_id}")

    asyncio.run(main(shard_id, limit=args.limit))
