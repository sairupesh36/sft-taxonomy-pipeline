#!/usr/bin/env python3
"""
Stage 1+ (LLM-driven): map-reduce clustering that turns the deduped, counted
raw value tables from taxonomy_extract_unique.py into a frozen, closed,
MECE 3-level enum tree per axis (domain: domain/subdomain/sub_subdomain,
task: task/subtask/sub_subtask) -- fully derived from the real discovery data.

Round-robins calls across the 3 live cluster models (gpt-oss-120b, qwen3-8-27b,
gemma-4-31b) all served through the single SGLang router.

Usage:
  python3 taxonomy_cluster.py --axis domain --stage l1
  python3 taxonomy_cluster.py --axis domain --stage l2
  python3 taxonomy_cluster.py --axis domain --stage l3
  python3 taxonomy_cluster.py --axis task --stage all
"""
import os
import argparse
import asyncio
import aiohttp
import json
import itertools
from collections import Counter, defaultdict
from pathlib import Path

BASE_DIR = Path("/projects/data/datasets/code_data/sai_rupesh/taxonomy")
UV_DIR = BASE_DIR / "unique_values"
OUT_DIR = BASE_DIR / "taxonomy_build"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ROUTER_URL = "http://sglang-router.sglang.svc.cluster.local:30000/v1/chat/completions"
API_KEY = os.environ.get("SGLANG_API_KEY", "")  # set this env var before running -- never hardcode the real key
MODELS = ["openai/gpt-oss-120b", "qwen3-8-27b", "gemma-4-31b"]  # round-robin across all 3

BATCH_SIZE = 120
CONCURRENCY = 24
MAX_TOKENS = 2500

BANNED = {"other", "miscellaneous", "general", "everyday life", "daily life",
          "general knowledge", "unknown", "n/a", "none", "various"}

LEVEL_LABEL = {"domain": ("domain", "subdomain", "sub_subdomain"),
               "task": ("task", "subtask", "sub_subtask")}


def build_system_prompt(axis, level, parent_context=None):
    l1n, l2n, l3n = LEVEL_LABEL[axis]
    if level == 1:
        target = "20-40"
        what = ("top-level " + l1n + "s (broad academic / professional / functional fields, "
                "NOT overly narrow sub-specialties)")
        ctx = ""
    elif level == 2:
        target = "3-8"
        what = f"{l2n}s (sub-fields/branches) under the fixed parent {l1n} '{parent_context}'"
        ctx = f"All input values were already classified under the parent {l1n} '{parent_context}'. Only decide the {l2n} split.\n"
    else:
        target = "3-10"
        what = f"{l3n}s (precise granular topics) under the fixed parent path '{parent_context}'"
        ctx = f"All input values were already classified under the fixed path '{parent_context}'. Only decide the {l3n} split.\n"

    return f"""You are an expert taxonomist building a strict MECE (mutually exclusive, collectively exhaustive) \
enum taxonomy that will be used to classify billions of SFT training samples.

{ctx}You will be given a list of raw, noisy, LLM-extracted labels (with occurrence counts) that need to be \
merged into a clean closed set of {what}.

RULES:
- Merge near-duplicates, synonyms, and narrower/broader phrasings of the SAME underlying concept into one canonical cluster \
(e.g. "Computer Science" and "Computer Science & Software Engineering" and "CS" are the SAME cluster).
- Aim for roughly {target} final clusters. Fewer, well-separated clusters are better than many overlapping ones.
- Every cluster gets a short canonical name (2-5 words, Title Case, no punctuation soup, no vague words like {sorted(BANNED)}).
- Every single input value must be assigned to EXACTLY ONE cluster (MECE) -- do not drop any, do not invent new ones.
- Do not copy counts into the output.

Output STRICT JSON only, no prose, no markdown fences:
{{"clusters": [{{"canonical": "Name", "members": ["exact input value 1", "exact input value 2"]}}, ...]}}"""


def build_user_prompt(items):
    lines = [f'- "{it["value"]}" (n={it["count"]})' for it in items]
    return "Raw values to cluster:\n" + "\n".join(lines)


async def call_llm(session, sem, model, system_prompt, user_prompt):
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system_prompt},
                     {"role": "user", "content": user_prompt}],
        "temperature": 0.0,
        "max_tokens": MAX_TOKENS,
    }
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    async with sem:
        for attempt in range(3):
            try:
                async with session.post(ROUTER_URL, json=payload, headers=headers, timeout=90) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        raw = data["choices"][0]["message"].get("content", "") or ""
                        s, e = raw.find("{"), raw.rfind("}")
                        if s != -1 and e != -1 and e > s:
                            return json.loads(raw[s:e + 1])
            except Exception:
                pass
            await asyncio.sleep(0.5 * (attempt + 1))
    return None


async def cluster_one_batch(session, sem, model_cycle, axis, level, parent_context, items):
    system_prompt = build_system_prompt(axis, level, parent_context)
    user_prompt = build_user_prompt(items)
    model = next(model_cycle)
    result = await call_llm(session, sem, model, system_prompt, user_prompt)
    clusters = result.get("clusters") if result else None
    if not clusters:
        # fallback: identity clusters (each item is its own canonical) so nothing is dropped
        return [{"canonical": it["value"][:60], "members": [it["value"]]} for it in items]
    return clusters


async def map_reduce(session, sem, model_cycle, axis, level, parent_context, items,
                      target_max=40, max_rounds=6, batch_size=BATCH_SIZE):
    """items: list[{value,count}]. Returns (final_items, chain) where chain is a list of
    per-round {round_value: canonical_value} dicts to resolve raw -> final canonical."""
    chain = []
    current = items
    for round_idx in range(max_rounds):
        batches = [current[i:i + batch_size] for i in range(0, len(current), batch_size)]
        tasks = [cluster_one_batch(session, sem, model_cycle, axis, level, parent_context, b) for b in batches]
        results = await asyncio.gather(*tasks)

        round_map = {}
        agg = Counter()
        for clusters in results:
            for c in clusters:
                canon = c.get("canonical", "").strip()
                if not canon:
                    continue
                for m in c.get("members", []):
                    round_map[m] = canon
        # anything the model silently dropped keeps its own value as canonical
        for it in current:
            if it["value"] not in round_map:
                round_map[it["value"]] = it["value"]
        for it in current:
            agg[round_map[it["value"]]] += it["count"]

        chain.append(round_map)
        current = [{"value": v, "count": c} for v, c in agg.most_common()]
        print(f"    round {round_idx + 1}: {len(batches)} batches -> {len(current)} clusters")

        if len(current) <= target_max or len(current) == sum(len(b) for b in batches):
            break

    return current, chain


def resolve_chain(original_value, chain):
    cur = original_value
    for round_map in chain:
        cur = round_map.get(cur, cur)
    return cur


def make_model_cycle():
    return itertools.cycle(MODELS)


async def run_l1(axis):
    items = json.loads((UV_DIR / f"{axis}_l1.json").read_text())
    print(f"[{axis}] L1: clustering {len(items)} raw values")
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=CONCURRENCY * 2)) as session:
        sem = asyncio.Semaphore(CONCURRENCY)
        final_items, chain = await map_reduce(session, sem, make_model_cycle(), axis, 1, None, items, target_max=35)

    raw_to_canonical = {it["value"]: resolve_chain(it["value"], chain) for it in items}
    out = {
        "canonical_l1": [it["value"] for it in final_items],
        "counts": {it["value"]: it["count"] for it in final_items},
        "raw_to_canonical": raw_to_canonical,
    }
    (OUT_DIR / f"{axis}_l1_final.json").write_text(json.dumps(out, indent=1))
    print(f"[{axis}] L1 DONE: {len(final_items)} canonical top-level values -> {OUT_DIR / f'{axis}_l1_final.json'}")


async def run_l2(axis):
    l1_final = json.loads((OUT_DIR / f"{axis}_l1_final.json").read_text())
    raw_l1_to_canon = l1_final["raw_to_canonical"]

    l2_items = json.loads((UV_DIR / f"{axis}_l2.json").read_text())
    buckets = defaultdict(Counter)
    for it in l2_items:
        canon_l1 = raw_l1_to_canon.get(it["l1"], it["l1"])
        buckets[canon_l1][it["value"]] += it["count"]

    result = {"per_l1": {}}
    raw_to_canonical = {}  # (canon_l1, raw_l2) -> canon_l2, stored as "canon_l1||raw_l2"

    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=CONCURRENCY * 2)) as session:
        sem = asyncio.Semaphore(CONCURRENCY)
        for canon_l1, counter in buckets.items():
            items = [{"value": v, "count": c} for v, c in counter.most_common()]
            print(f"[{axis}] L2 under '{canon_l1}': clustering {len(items)} raw values")
            final_items, chain = await map_reduce(session, sem, make_model_cycle(), axis, 2, canon_l1, items,
                                                    target_max=8, batch_size=100)
            result["per_l1"][canon_l1] = [it["value"] for it in final_items]
            for it in items:
                raw_to_canonical[f"{canon_l1}||{it['value']}"] = resolve_chain(it["value"], chain)

    result["raw_to_canonical"] = raw_to_canonical
    (OUT_DIR / f"{axis}_l2_final.json").write_text(json.dumps(result, indent=1))
    total_l2 = sum(len(v) for v in result["per_l1"].values())
    print(f"[{axis}] L2 DONE: {total_l2} canonical (l1,l2) pairs -> {OUT_DIR / f'{axis}_l2_final.json'}")


async def run_l3(axis):
    l1_final = json.loads((OUT_DIR / f"{axis}_l1_final.json").read_text())
    l2_final = json.loads((OUT_DIR / f"{axis}_l2_final.json").read_text())
    raw_l1_to_canon = l1_final["raw_to_canonical"]
    raw_l2_to_canon = l2_final["raw_to_canonical"]  # key "canon_l1||raw_l2"

    l3_items = json.loads((UV_DIR / f"{axis}_l3.json").read_text())
    buckets = defaultdict(Counter)  # key (canon_l1, canon_l2)
    for it in l3_items:
        canon_l1 = raw_l1_to_canon.get(it["l1"], it["l1"])
        canon_l2 = raw_l2_to_canon.get(f"{canon_l1}||{it['l2']}", it["l2"])
        buckets[(canon_l1, canon_l2)][it["value"]] += it["count"]

    result = {"per_l1_l2": {}}
    raw_to_canonical = {}

    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=CONCURRENCY * 2)) as session:
        sem = asyncio.Semaphore(CONCURRENCY)
        for (canon_l1, canon_l2), counter in buckets.items():
            items = [{"value": v, "count": c} for v, c in counter.most_common()]
            key = f"{canon_l1}||{canon_l2}"
            print(f"[{axis}] L3 under '{key}': clustering {len(items)} raw values")
            final_items, chain = await map_reduce(session, sem, make_model_cycle(), axis, 3, key, items,
                                                    target_max=10, batch_size=90)
            result["per_l1_l2"][key] = [it["value"] for it in final_items]
            for it in items:
                raw_to_canonical[f"{key}||{it['value']}"] = resolve_chain(it["value"], chain)

    result["raw_to_canonical"] = raw_to_canonical
    (OUT_DIR / f"{axis}_l3_final.json").write_text(json.dumps(result, indent=1))
    total_l3 = sum(len(v) for v in result["per_l1_l2"].values())
    print(f"[{axis}] L3 DONE: {total_l3} canonical (l1,l2,l3) triples -> {OUT_DIR / f'{axis}_l3_final.json'}")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", choices=["domain", "task"], required=True)
    ap.add_argument("--stage", choices=["l1", "l2", "l3", "all"], required=True)
    args = ap.parse_args()

    stages = ["l1", "l2", "l3"] if args.stage == "all" else [args.stage]
    for stage in stages:
        if stage == "l1":
            await run_l1(args.axis)
        elif stage == "l2":
            await run_l2(args.axis)
        elif stage == "l3":
            await run_l3(args.axis)


if __name__ == "__main__":
    asyncio.run(main())
