#!/usr/bin/env python3
"""
Classifies every real discovered raw value (domain/task axes, L1 + L2) against
the FIXED closed taxonomy_v4.json schema, using the live cluster models.
Single-pass closed-set classification (no map-reduce merging needed since the
target category list is already fixed) -> per-category volume coverage +
raw->canonical mapping (doubles as labeled training data for the classifier stage).

Usage:
  python3 taxonomy_validate_v4.py --axis domain --level 1
  python3 taxonomy_validate_v4.py --axis domain --level 2
  python3 taxonomy_validate_v4.py --axis task --level 1
  python3 taxonomy_validate_v4.py --axis task --level 2
"""
import os
import argparse
import asyncio
import aiohttp
import json
from collections import defaultdict, Counter
from pathlib import Path

BASE_DIR = Path("/projects/data/datasets/code_data/sai_rupesh/taxonomy")
UV_DIR = BASE_DIR / "unique_values"
OUT_DIR = BASE_DIR / "taxonomy_build"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ROUTER_URL = "http://sglang-router.sglang.svc.cluster.local:30000/v1/chat/completions"
API_KEY = os.environ.get("SGLANG_API_KEY", "")  # set this env var before running -- never hardcode the real key
MODELS = ["gemma-4-31b"]  # gpt-oss-120b / qwen3-8-27b timing out as of last health check; re-add once fixed

BATCH_SIZE = 40
CONCURRENCY = 24
MAX_TOKENS = 2000
TIMEOUT_S = 60
RETRIES = 2

V4 = json.loads((BASE_DIR / "taxonomy_v4.json").read_text())


def rows_to_options(rows):
    # rows[0] is header e.g. ["value","description"]; skip it
    return [{"value": r[0], "description": r[-1]} for r in rows[1:]]


AXIS_CONFIG = {
    "domain": {
        "l1_unique": "domain_l1.json",
        "l2_unique": "domain_l2.json",
        "l1_options": rows_to_options(V4["primary_domain_values"]),
        "l2_by_parent": None,  # built at runtime from domain_subdomain_values
        "l2_source_rows": V4["domain_subdomain_values"],
        "l1_label": "primary_domain",
        "l2_label": "subdomain",
    },
    "task": {
        "l1_unique": "task_l1.json",
        "l2_unique": "task_l2.json",
        "l1_options": rows_to_options(V4["task_family_values"]),
        "l2_by_parent": None,
        "l2_source_rows": V4["task_subfamily_values"],
        "l1_label": "task_family",
        "l2_label": "task_subfamily",
    },
}

for cfg in AXIS_CONFIG.values():
    by_parent = defaultdict(list)
    for row in cfg["l2_source_rows"][1:]:
        parent, value, desc = row[0], row[1], row[2]
        by_parent[parent].append({"value": value, "description": desc})
    cfg["l2_by_parent"] = dict(by_parent)


def build_system_prompt(level, options, parent_label=None, parent_value=None):
    opt_lines = "\n".join(f'- "{o["value"]}": {o["description"]}' for o in options)
    scope = f" (already known to belong under '{parent_label}' = '{parent_value}')" if parent_value else ""
    return f"""You are classifying raw, noisy, LLM-extracted labels{scope} into ONE fixed closed set of categories.

Allowed categories (pick the single best match for each input; if genuinely nothing fits, use "other"):
{opt_lines}

RULES:
- For every input value, output its EXACT category value string from the allowed list above (nothing else, no new categories).
- Do not skip any input value.
Output STRICT JSON only, no prose, no markdown fences:
{{"assignments": [{{"input": "exact input value", "category": "exact allowed category value"}}, ...]}}"""


def build_user_prompt(items):
    lines = [f'- "{it["value"]}" (n={it["count"]})' for it in items]
    return "Input values:\n" + "\n".join(lines)


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
        for attempt in range(RETRIES):
            try:
                async with session.post(ROUTER_URL, json=payload, headers=headers, timeout=TIMEOUT_S) as resp:
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


def make_model_cycle():
    import itertools
    return itertools.cycle(MODELS)


async def classify_batch(session, sem, model_cycle, items, options, parent_label=None, parent_value=None):
    system_prompt = build_system_prompt(2 if parent_value else 1, options, parent_label, parent_value)
    user_prompt = build_user_prompt(items)
    model = next(model_cycle)
    result = await call_llm(session, sem, model, system_prompt, user_prompt)
    valid_values = {o["value"] for o in options}
    out = {}
    if result and result.get("assignments"):
        for a in result["assignments"]:
            cat = a.get("category")
            if cat in valid_values:
                out[a.get("input")] = cat
    for it in items:
        if it["value"] not in out:
            out[it["value"]] = "other"  # fallback for anything the model missed/hallucinated
    return out


async def run_l1(axis):
    cfg = AXIS_CONFIG[axis]
    items = json.loads((UV_DIR / cfg["l1_unique"]).read_text())
    options = cfg["l1_options"]
    print(f"[{axis}] L1: classifying {len(items)} raw values against {len(options)} fixed {cfg['l1_label']} categories")

    mapping = {}
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=CONCURRENCY * 2)) as session:
        sem = asyncio.Semaphore(CONCURRENCY)
        model_cycle = __import__("itertools").cycle(MODELS)
        batches = [items[i:i + BATCH_SIZE] for i in range(0, len(items), BATCH_SIZE)]
        tasks = [classify_batch(session, sem, model_cycle, b, options) for b in batches]
        results = await asyncio.gather(*tasks)
        for r in results:
            mapping.update(r)

    coverage = Counter()
    for it in items:
        coverage[mapping[it["value"]]] += it["count"]

    out = {"raw_to_category": mapping, "volume_by_category": dict(coverage.most_common())}
    (OUT_DIR / f"{axis}_l1_v4mapping.json").write_text(json.dumps(out, indent=1))

    total = sum(coverage.values())
    other_pct = 100 * coverage.get("other", 0) / total if total else 0
    print(f"[{axis}] L1 DONE. total_volume={total:,}  'other' volume={coverage.get('other', 0):,} ({other_pct:.1f}%)")
    print(f"  top categories by volume: {coverage.most_common(10)}")


async def run_l2(axis):
    cfg = AXIS_CONFIG[axis]
    l1_map = json.loads((OUT_DIR / f"{axis}_l1_v4mapping.json").read_text())["raw_to_category"]
    items = json.loads((UV_DIR / cfg["l2_unique"]).read_text())

    buckets = defaultdict(list)
    for it in items:
        parent = l1_map.get(it["l1"], "other")
        buckets[parent].append({"value": it["value"], "count": it["count"]})

    ckpt_path = OUT_DIR / f"{axis}_l2_v4mapping.json"
    mapping = {}
    coverage_by_parent = {}
    done_parents = set()
    if ckpt_path.exists():
        prev = json.loads(ckpt_path.read_text())
        mapping = prev.get("raw_to_category", {})
        coverage_by_parent = prev.get("volume_by_parent_category", {})
        done_parents = set(prev.get("done_parents", []))
        print(f"[{axis}] L2 resuming: {len(done_parents)} parents already done ({sorted(done_parents)})")

    def checkpoint():
        out = {"raw_to_category": mapping, "volume_by_parent_category": coverage_by_parent,
               "done_parents": sorted(done_parents)}
        ckpt_path.write_text(json.dumps(out, indent=1))

    todo = {p: its for p, its in buckets.items() if p not in done_parents}
    print(f"[{axis}] L2: {len(todo)} parents left to classify (of {len(buckets)} total)")

    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=CONCURRENCY * 2)) as session:
        sem = asyncio.Semaphore(CONCURRENCY)
        model_cycle = make_model_cycle()

        for parent, parent_items in todo.items():
            options = cfg["l2_by_parent"].get(parent, cfg["l2_by_parent"].get("other", [{"value": "other", "description": "fallback"}]))
            batches = [parent_items[i:i + BATCH_SIZE] for i in range(0, len(parent_items), BATCH_SIZE)]
            print(f"[{axis}] L2 under '{parent}': {len(parent_items)} raw values, {len(batches)} batches", flush=True)
            tasks = [classify_batch(session, sem, model_cycle, b, options, cfg["l1_label"], parent) for b in batches]
            results = await asyncio.gather(*tasks)
            local_map = {}
            for r in results:
                local_map.update(r)
            cov = Counter()
            for it in parent_items:
                cat = local_map[it["value"]]
                mapping[f"{parent}||{it['value']}"] = cat
                cov[cat] += it["count"]
            coverage_by_parent[parent] = dict(cov.most_common())
            done_parents.add(parent)
            checkpoint()
            print(f"[{axis}] L2 under '{parent}': DONE, checkpointed ({len(done_parents)}/{len(buckets)} parents)", flush=True)

    print(f"[{axis}] L2 ALL DONE -> {ckpt_path}")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", choices=["domain", "task"], required=True)
    ap.add_argument("--level", choices=["1", "2"], required=True)
    args = ap.parse_args()
    if args.level == "1":
        await run_l1(args.axis)
    else:
        await run_l2(args.axis)


if __name__ == "__main__":
    asyncio.run(main())
