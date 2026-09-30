#!/usr/bin/env python3
"""
Antigravity Summarization Benchmarking Script
Benchmarks GPT, Qwen, and Gemma on diverse conversation samples
(Diverse samples across 19 Tulu-3 sources + 17 Agentic sources)
Evaluates: Speed, Latency, Output Quality, Key Information Retention, Conciseness
"""

import os
import sys
import time
import json
import glob
import re
import asyncio
import aiohttp
from pathlib import Path
from collections import defaultdict
import pyarrow.parquet as pq

BASE_DIR = Path("/projects/data/datasets/code_data/sai_rupesh/taxonomy")
TULU_DIR = BASE_DIR / "data"
AGENTIC_DIR = BASE_DIR / "agentic_data"
RESULTS_DIR = BASE_DIR / "benchmark_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Endpoints Configuration ──────────────────────────────────────────────────
CANDIDATE_ENDPOINTS = {
    "gpt": {
        "urls": [
            "http://sglang-router.sglang.svc.cluster.local:30000/v1",
            "http://sglang-gpt-oss-120b.sglang.svc.cluster.local:30003/v1",
            "http://10.150.38.6:30000/v1"
        ],
        "default_model": "openai/gpt-oss-120b",
        "api_key": os.environ.get("SGLANG_API_KEY", "")
    },
    "qwen": {
        "urls": [
            "http://sglang-qwen3.sglang.svc.cluster.local:8000/v1",
            "http://sglang-qwen3.sglang.svc.cluster.local:30000/v1",
            "http://10.150.142.57:8000/v1",
            "http://10.150.142.57:30000/v1",
            "http://172.17.99.1:31184/v1"
        ],
        "default_model": "qwen3-8-27b",
        "api_key": "sk-litellm-master-key"
    },
    "gemma": {
        "urls": [
            "http://sglang-gemma-4-31b.sglang.svc.cluster.local:30000/v1",
            "http://10.150.159.182:30000/v1"
        ],
        "default_model": "gemma-4-31b",
        "api_key": os.environ.get("SGLANG_API_KEY", "")
    }
}

SUMMARIZATION_PROMPT = """You are an expert technical and conversational summarizer.
Summarize the following interaction (which may be a multi-turn user conversation, instruction response, or autonomous agent trajectory).

RULES:
1. Provide a clear, comprehensive, and well-structured summary.
2. Capture the core objective, key constraints, actions/tools used, code/technical decisions, and final outcomes.
3. Do NOT omit critical technical details, entity names, parameters, or resolution steps.
4. Keep the summary concise, objective, and dense with information (aim for 3-6 bullet points or 1-2 tight paragraphs)."""

def extract_text_from_row(row, max_chars=3500):
    # Try common conversation / text column names
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

    # Try single text fields (SWE-bench, code, reasoning)
    combined = []
    for col in ["problem_statement", "instruction", "input", "prompt", "question", "task", "code"]:
        if col in row and row[col] and str(row[col]).strip():
            combined.append(f"[{col.upper()}]: {row[col]}")
    for col in ["hints_text", "patch", "output", "response", "solution", "answer"]:
        if col in row and row[col] and str(row[col]).strip():
            combined.append(f"[{col.upper()}]: {row[col]}")

    if combined:
        return "\n\n".join(combined)[:max_chars]

    # Fallback: stringify all non-empty columns
    return " | ".join(f"{k}: {v}" for k, v in row.items() if v)[:max_chars]

# ── 1. Discover Active Model Endpoints ─────────────────────────────────────────
async def discover_endpoints(session):
    active_endpoints = {}
    print("=== DISCOVERING ACTIVE MODEL ENDPOINTS ===", flush=True)
    
    for model_key, cfg in CANDIDATE_ENDPOINTS.items():
        found = False
        for base_url in cfg["urls"]:
            models_url = f"{base_url}/models"
            headers = {"Authorization": f"Bearer {cfg['api_key']}"}
            try:
                async with session.get(models_url, headers=headers, timeout=5) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        model_id = cfg["default_model"]
                        if "data" in data and len(data["data"]) > 0:
                            # Prefer explicit model matching
                            ids = [m.get("id") for m in data["data"] if isinstance(m, dict)]
                            if cfg["default_model"] in ids:
                                model_id = cfg["default_model"]
                            else:
                                model_id = ids[0]
                        print(f"✅ {model_key.upper()} endpoint active: {base_url} (Model: {model_id})", flush=True)
                        active_endpoints[model_key] = {
                            "base_url": base_url,
                            "chat_url": f"{base_url}/chat/completions",
                            "model": model_id,
                            "api_key": cfg["api_key"]
                        }
                        found = True
                        break
            except Exception:
                continue
        if not found:
            print(f"⚠️ {model_key.upper()}: No /models response, using default fallback: {cfg['urls'][0]}", flush=True)
            active_endpoints[model_key] = {
                "base_url": cfg["urls"][0],
                "chat_url": f"{cfg['urls'][0]}/chat/completions",
                "model": cfg["default_model"],
                "api_key": cfg["api_key"]
            }
    return active_endpoints

# ── 2. Collect Diverse Samples across all 36 sources ─────────────────────────
def collect_benchmark_samples(samples_per_source=3):
    print(f"\n=== COLLECTING {samples_per_source} SAMPLES PER SOURCE (36 Sources) ===", flush=True)
    samples = []
    
    # 19 Tulu-3 Sources
    tulu_files = sorted(glob.glob(str(TULU_DIR / "*.parquet")))
    tulu_source_counts = defaultdict(int)
    
    for pf in tulu_files:
        pfile = pq.ParquetFile(pf)
        # Check which columns exist in parquet
        avail_cols = [c for c in ["id", "messages", "source"] if c in pfile.schema.names]
        for batch in pfile.iter_batches(batch_size=256, columns=avail_cols):
            df = batch.to_pandas()
            for _, row in df.iterrows():
                src = str(row.get("source", "unknown")).strip()
                if tulu_source_counts[src] < samples_per_source:
                    rid = str(row.get("id", f"{src}_{tulu_source_counts[src]}"))
                    text = extract_text_from_row(row)
                    if len(text.strip()) > 30:
                        samples.append({
                            "id": rid,
                            "source": src,
                            "group": "Tulu-3",
                            "text": text
                        })
                        tulu_source_counts[src] += 1
            if len(tulu_source_counts) >= 19 and all(cnt >= samples_per_source for cnt in tulu_source_counts.values()):
                break

    print(f"Collected {len(samples)} Tulu-3 samples across {len(tulu_source_counts)} sources.", flush=True)

    # 17 Agentic Sources
    agentic_files = sorted(glob.glob(str(AGENTIC_DIR / "*.parquet")))
    agentic_source_counts = defaultdict(int)

    for pf in agentic_files:
        pfile = pq.ParquetFile(pf)
        src_name = Path(pf).stem.replace("_5k", "")
        # Read available batch
        for batch in pfile.iter_batches(batch_size=50):
            df = batch.to_pandas()
            for idx, row in df.iterrows():
                if agentic_source_counts[src_name] < samples_per_source:
                    rid = str(row.get("id", row.get("instance_id", f"{src_name}_{agentic_source_counts[src_name]}")))
                    text = extract_text_from_row(row)
                    if len(text.strip()) > 30:
                        samples.append({
                            "id": rid,
                            "source": src_name,
                            "group": "Agentic",
                            "text": text
                        })
                        agentic_source_counts[src_name] += 1
                if agentic_source_counts[src_name] >= samples_per_source:
                    break
            if agentic_source_counts[src_name] >= samples_per_source:
                break

    print(f"Total benchmark samples ready: {len(samples)} across {len(tulu_source_counts) + len(agentic_source_counts)} sources.", flush=True)
    return samples

# ── 3. Benchmarking Inference Worker ──────────────────────────────────────────
async def summarize_sample(session, endpoint_info, text, source):
    payload = {
        "model": endpoint_info["model"],
        "messages": [
            {"role": "system", "content": SUMMARIZATION_PROMPT},
            {"role": "user", "content": f"Source: {source}\n\nInteraction to Summarize:\n{text}"}
        ],
        "temperature": 0.2,
        "max_tokens": 400
    }
    headers = {
        "Authorization": f"Bearer {endpoint_info['api_key']}",
        "Content-Type": "application/json"
    }
    
    t0 = time.perf_counter()
    try:
        async with session.post(endpoint_info["chat_url"], json=payload, headers=headers, timeout=45) as resp:
            elapsed = time.perf_counter() - t0
            if resp.status == 200:
                data = await resp.json()
                msg = data["choices"][0]["message"]
                content = msg.get("content", "") or msg.get("reasoning_content", "") or ""
                usage = data.get("usage", {})
                completion_tokens = usage.get("completion_tokens", len(content.split()) * 1.3)
                tok_per_sec = completion_tokens / elapsed if elapsed > 0 else 0
                return {
                    "success": True,
                    "summary": content.strip(),
                    "latency_sec": elapsed,
                    "completion_tokens": completion_tokens,
                    "tokens_per_sec": tok_per_sec,
                    "error": None
                }
            else:
                err_txt = await resp.text()
                return {"success": False, "summary": "", "latency_sec": elapsed, "completion_tokens": 0, "tokens_per_sec": 0, "error": f"HTTP {resp.status}: {err_txt[:100]}"}
    except Exception as e:
        elapsed = time.perf_counter() - t0
        return {"success": False, "summary": "", "latency_sec": elapsed, "completion_tokens": 0, "tokens_per_sec": 0, "error": str(e)}

# ── 4. Quality Evaluation Metrics ─────────────────────────────────────────────
def evaluate_quality(original_text, summary):
    if not summary:
        return {"key_info_retention": 0.0, "conciseness_score": 0.0, "word_count": 0, "density": 0.0}
    
    orig_words = set(re.findall(r'\b[a-zA-Z]{4,}\b', original_text.lower()))
    sum_words = set(re.findall(r'\b[a-zA-Z]{4,}\b', summary.lower()))
    
    overlap = len(orig_words.intersection(sum_words))
    retention = (overlap / min(len(orig_words), 50)) if orig_words else 0.0
    retention = min(1.0, retention)
    
    summary_len = len(summary.split())
    if 50 <= summary_len <= 160:
        conciseness = 1.0
    elif summary_len < 50:
        conciseness = summary_len / 50.0
    else:
        conciseness = max(0.2, 1.0 - (summary_len - 160) / 250.0)
        
    density = overlap / summary_len if summary_len > 0 else 0.0
    
    return {
        "key_info_retention": round(retention, 3),
        "conciseness_score": round(conciseness, 3),
        "word_count": summary_len,
        "density": round(density, 3)
    }

# ── 5. Main Benchmark Execution ───────────────────────────────────────────────
async def main():
    connector = aiohttp.TCPConnector(limit=60)
    async with aiohttp.ClientSession(connector=connector) as session:
        active_endpoints = await discover_endpoints(session)
        # Take 3 samples per source across all 36 sources = 108 highly diverse samples
        samples = collect_benchmark_samples(samples_per_source=3)
        
        print(f"\n=== STARTING SUMMARIZATION BENCHMARK ===", flush=True)
        print(f"Models: {list(active_endpoints.keys())}", flush=True)
        print(f"Total Trajectories: {len(samples)}\n", flush=True)
        
        results = {m: [] for m in active_endpoints.keys()}
        
        # Process in batches with progress logging
        for idx, sample in enumerate(samples, 1):
            tasks = [
                summarize_sample(session, ep, sample["text"], sample["source"])
                for m_key, ep in active_endpoints.items()
            ]
            model_keys = list(active_endpoints.keys())
            responses = await asyncio.gather(*tasks)
            
            for m_key, res in zip(model_keys, responses):
                q_eval = evaluate_quality(sample["text"], res["summary"])
                results[m_key].append({
                    "sample_id": sample["id"],
                    "source": sample["source"],
                    "group": sample["group"],
                    "latency_sec": res["latency_sec"],
                    "tokens_per_sec": res["tokens_per_sec"],
                    "completion_tokens": res["completion_tokens"],
                    "success": res["success"],
                    "error": res["error"],
                    "quality": q_eval,
                    "summary": res["summary"]
                })
            
            if idx % 5 == 0 or idx == len(samples):
                completed = [r for r in results[model_keys[0]] if r["success"]]
                avg_l = sum(r["latency_sec"] for r in completed) / len(completed) if completed else 0
                print(f"[{idx}/{len(samples)}] Evaluated sample from '{sample['source']}' | Last Latency: {avg_l:.2f}s", flush=True)
        
        # ── Print Complete Analytical Table ───────────────────────────────────
        print("\n" + "=" * 85, flush=True)
        print("                  🏆 REAL EMPIRICAL BENCHMARK RESULTS 🏆", flush=True)
        print("=" * 85, flush=True)
        print(f"{'Model':<10} | {'Latency':<9} | {'Throughput':<12} | {'Retention':<11} | {'Conciseness':<12} | {'Words':<8} | {'Score'}", flush=True)
        print("-" * 85, flush=True)
        
        summary_stats = {}
        for m_key, res_list in results.items():
            successful = [r for r in res_list if r["success"]]
            if not successful:
                print(f"{m_key.upper():<10} | FAILED ({res_list[0]['error'][:30]})", flush=True)
                continue
                
            avg_latency = sum(r["latency_sec"] for r in successful) / len(successful)
            avg_tps = sum(r["tokens_per_sec"] for r in successful) / len(successful)
            avg_retention = sum(r["quality"]["key_info_retention"] for r in successful) / len(successful)
            avg_conciseness = sum(r["quality"]["conciseness_score"] for r in successful) / len(successful)
            avg_words = sum(r["quality"]["word_count"] for r in successful) / len(successful)
            
            speed_score = min(1.0, avg_tps / 100.0)
            composite_score = (avg_retention * 0.45) + (avg_conciseness * 0.25) + (speed_score * 0.30)
            
            summary_stats[m_key] = {
                "model_name": active_endpoints[m_key]["model"],
                "avg_latency_sec": round(avg_latency, 2),
                "avg_tokens_per_sec": round(avg_tps, 1),
                "key_info_retention_pct": round(avg_retention * 100, 1),
                "conciseness_score_pct": round(avg_conciseness * 100, 1),
                "avg_word_count": round(avg_words, 1),
                "composite_score": round(composite_score * 100, 1)
            }
            
            print(f"{m_key.upper():<10} | {avg_latency:6.2f} s | {avg_tps:6.1f} tok/s | {avg_retention*100:6.1f} %   | {avg_conciseness*100:6.1f} %    | {avg_words:6.1f} | {composite_score*100:5.1f}/100", flush=True)
            
        print("=" * 85, flush=True)
        
        # Print qualitative sample comparisons
        print("\n=== SAMPLE SUMMARIZATION OUTPUT COMPARISONS ===", flush=True)
        sample_idx = min(3, len(samples) - 1)
        print(f"Source: {samples[sample_idx]['source']} ({samples[sample_idx]['group']})")
        print("-" * 70)
        for m_key in active_endpoints.keys():
            m_res = results[m_key][sample_idx]
            print(f"[{m_key.upper()} - {active_endpoints[m_key]['model']}]:")
            print(m_res["summary"][:400] + ("..." if len(m_res["summary"]) > 400 else ""))
            print()
            
        out_json = RESULTS_DIR / "summarization_benchmark_report.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump({"summary_stats": summary_stats, "detailed_results": results}, f, indent=2)
            
        print(f"Full benchmark report saved to: {out_json}", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
