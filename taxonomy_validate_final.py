#!/usr/bin/env python3
"""
Same purpose as taxonomy_validate_v4.py, retargeted at taxonomy_tree_final.xlsx
(48 domains/270 subdomains, 62 task families/261 subfamilies) instead of the
smaller taxonomy_v4.json. Supports multi-label output (Domain/Subdomain/Task
Family/Task Subfamily are all "Multiple" cardinality per the Main sheet) --
each raw value can be assigned up to 2 categories. L2 bucketing uses the
primary (first) category to keep the job tractable; full multi-label is
preserved in the output mapping for the real per-document pipeline to use.

Usage:
  python3 taxonomy_validate_final.py --axis domain --level 1
  python3 taxonomy_validate_final.py --axis domain --level 2
  python3 taxonomy_validate_final.py --axis task --level 1
  python3 taxonomy_validate_final.py --axis task --level 2
"""
import os
import argparse
import asyncio
import aiohttp
import itertools
import json
from collections import defaultdict, Counter
from pathlib import Path
import pandas as pd

BASE_DIR = Path("/projects/data/datasets/code_data/sai_rupesh/taxonomy")
UV_DIR = BASE_DIR / "unique_values"
OUT_DIR = BASE_DIR / "taxonomy_build_final"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ROUTER_URL = "http://sglang-router.sglang.svc.cluster.local:30000/v1/chat/completions"
API_KEY = os.environ.get("SGLANG_API_KEY", "")  # set this env var before running -- never hardcode the real key
MODELS = ["gemma-4-31b", "openai/gpt-oss-120b", "qwen3-8-27b"]  # reasoning disabled via
                          # _reasoning_kwargs() below -- verified this actually fixes both models.

BATCH_SIZE = 20   # smaller batches -> lower per-call latency, more reliable under current load
CONCURRENCY = 50  # 3 healthy models sharing load (~17 concurrent each). Tested clean at 60 on a
                  # single job; the earlier crash at 90 was the 60s timeout + 2 silently-dead
                  # models, both since fixed.
MAX_TOKENS = 1400
TIMEOUT_S = 180   # measured real latency for a 35-item/49-option call was 110-120s under
                  # current load; 60s was killing valid calls before they could finish.
RETRIES = 3
MAX_LABELS = 2  # cap multi-label output per raw value

XLSX_PATH = BASE_DIR / "taxonomy_tree_final.xlsx"
_sheets = pd.read_excel(XLSX_PATH, sheet_name=None)


def rows_to_options(df, value_col, desc_col="description"):
    return [{"value": str(r[value_col]), "description": str(r[desc_col])} for _, r in df.iterrows()]


AXIS_CONFIG = {
    "domain": {
        "l1_unique": "domain_l1.json",
        "l2_unique": "domain_l2.json",
        "l1_options": rows_to_options(_sheets["domain_values"], "domain"),
        "l2_source_df": _sheets["domain_subdomain_values"],
        "l1_col": "domain", "l2_col": "domain_subdomain",
        "l1_label": "domain",
        "l2_label": "subdomain",
    },
    "task": {
        "l1_unique": "task_l1.json",
        "l2_unique": "task_l2.json",
        "l1_options": rows_to_options(_sheets["task_family_values"], "task_family"),
        "l2_source_df": _sheets["task_subfamily_values"],
        "l1_col": "task_family", "l2_col": "task_subfamily",
        "l1_label": "task_family",
        "l2_label": "task_subfamily",
    },
}

for cfg in AXIS_CONFIG.values():
    by_parent = defaultdict(list)
    for _, row in cfg["l2_source_df"].iterrows():
        by_parent[row[cfg["l1_col"]]].append({"value": row[cfg["l2_col"]], "description": row["description"]})
    cfg["l2_by_parent"] = dict(by_parent)


NO_FIT_SENTINEL = "NO_CONFIDENT_MATCH"
GAP_MARKER = "__GAP__"  # internal only -- never written to the tree, used to measure real coverage gaps


def build_system_prompt(options, parent_label=None, parent_value=None):
    opt_lines = "\n".join(f'- "{o["value"]}": {o["description"]}' for o in options)
    scope = f" (already known to belong under '{parent_label}' = '{parent_value}')" if parent_value else ""
    return f"""You are classifying raw, noisy, LLM-extracted labels{scope} into a FIXED closed set of categories.
Multiple categories may genuinely apply to a single input -- assign up to {MAX_LABELS}, ranked with the single
best-fit category FIRST. Most inputs need only 1 category; only use 2 when the input is genuinely cross-cutting.

Allowed categories (use the exact value string):
{opt_lines}

If, and only if, NONE of the categories above genuinely fits an input, output the special value
"{NO_FIT_SENTINEL}" instead of forcing the closest-but-wrong category. Do not use this as a shortcut --
only use it when the input truly does not belong anywhere in the list above.

RULES:
- For every input value, output a list of 1-{MAX_LABELS} exact category value string(s) from the allowed list
  above, OR ["{NO_FIT_SENTINEL}"] if nothing fits.
- Do not skip any input value. Do not invent new categories.
Output STRICT JSON only, no prose, no markdown fences:
{{"assignments": [{{"input": "exact input value", "categories": ["best_category", "second_category_optional"]}}, ...]}}"""


def build_user_prompt(items):
    lines = [f'- "{it["value"]}" (n={it["count"]})' for it in items]
    return "Input values:\n" + "\n".join(lines)


def _reasoning_kwargs(model):
    # qwen3 and gpt-oss are reasoning models that otherwise burn MAX_TOKENS on hidden
    # reasoning_content and never emit the real answer (finish_reason="length", content="").
    # Verified experimentally: chat_template_kwargs.enable_thinking=False and
    # reasoning_effort="low" both actually work on this SGLang deployment.
    if model == "qwen3-8-27b":
        return {"chat_template_kwargs": {"enable_thinking": False}}
    if model == "openai/gpt-oss-120b":
        return {"reasoning_effort": "low"}
    return {}


async def call_llm(session, sem, model, system_prompt, user_prompt):
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system_prompt},
                     {"role": "user", "content": user_prompt}],
        "temperature": 0.0,
        "max_tokens": MAX_TOKENS,
        **_reasoning_kwargs(model),
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
    return itertools.cycle(MODELS)


async def classify_batch(session, sem, model_cycle, items, options, parent_label=None, parent_value=None):
    system_prompt = build_system_prompt(options, parent_label, parent_value)
    user_prompt = build_user_prompt(items)
    model = next(model_cycle)
    result = await call_llm(session, sem, model, system_prompt, user_prompt)
    valid_values = {o["value"] for o in options}
    out = {}
    if result and result.get("assignments"):
        for a in result["assignments"]:
            raw_cats = a.get("categories") or []
            if NO_FIT_SENTINEL in raw_cats:
                out[a.get("input")] = [GAP_MARKER]
                continue
            cats = [c for c in raw_cats if c in valid_values][:MAX_LABELS]
            if cats:
                out[a.get("input")] = cats
    for it in items:
        if it["value"] not in out:
            # model failed to respond, or hallucinated something not in the list --
            # a real gap in our knowledge, not a reason to silently pick options[0]
            out[it["value"]] = [GAP_MARKER]
    return out


async def run_l1(axis):
    cfg = AXIS_CONFIG[axis]
    items = json.loads((UV_DIR / cfg["l1_unique"]).read_text())
    options = cfg["l1_options"]
    print(f"[{axis}] L1: classifying {len(items)} raw values against {len(options)} fixed {cfg['l1_label']} categories (multi-label)", flush=True)

    mapping = {}
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=CONCURRENCY * 2)) as session:
        sem = asyncio.Semaphore(CONCURRENCY)
        model_cycle = make_model_cycle()
        batches = [items[i:i + BATCH_SIZE] for i in range(0, len(items), BATCH_SIZE)]
        tasks = [classify_batch(session, sem, model_cycle, b, options) for b in batches]
        results = await asyncio.gather(*tasks)
        for r in results:
            mapping.update(r)

    coverage = Counter()
    for it in items:
        for cat in mapping[it["value"]]:
            coverage[cat] += it["count"]

    out = {"raw_to_categories": mapping, "volume_by_category": dict(coverage.most_common())}
    (OUT_DIR / f"{axis}_l1_finalmapping.json").write_text(json.dumps(out, indent=1))

    total = sum(it["count"] for it in items)
    gap_pct = 100 * coverage.get(GAP_MARKER, 0) / total if total else 0
    print(f"[{axis}] L1 DONE. total_rows={total:,}  gap volume={coverage.get(GAP_MARKER, 0):,} ({gap_pct:.1f}%)", flush=True)
    print(f"  top categories by volume: {coverage.most_common(12)}", flush=True)


async def run_l2(axis):
    cfg = AXIS_CONFIG[axis]
    l1_map = json.loads((OUT_DIR / f"{axis}_l1_finalmapping.json").read_text())["raw_to_categories"]
    items = json.loads((UV_DIR / cfg["l2_unique"]).read_text())

    buckets = defaultdict(list)
    for it in items:
        cats = l1_map.get(it["l1"], [GAP_MARKER])
        parent = cats[0]  # primary category only, to keep this job tractable
        buckets[parent].append({"value": it["value"], "count": it["count"]})

    ckpt_path = OUT_DIR / f"{axis}_l2_finalmapping.json"
    mapping = {}
    coverage_by_parent = {}
    done_parents = set()
    if ckpt_path.exists():
        prev = json.loads(ckpt_path.read_text())
        mapping = prev.get("raw_to_categories", {})
        coverage_by_parent = prev.get("volume_by_parent_category", {})
        done_parents = set(prev.get("done_parents", []))
        print(f"[{axis}] L2 resuming: {len(done_parents)} parents already done", flush=True)

    def checkpoint():
        out = {"raw_to_categories": mapping, "volume_by_parent_category": coverage_by_parent,
               "done_parents": sorted(done_parents)}
        ckpt_path.write_text(json.dumps(out, indent=1))

    todo = {p: its for p, its in buckets.items() if p not in done_parents}
    print(f"[{axis}] L2: {len(todo)} parents left to classify (of {len(buckets)} total)", flush=True)

    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=CONCURRENCY * 2)) as session:
        sem = asyncio.Semaphore(CONCURRENCY)
        model_cycle = make_model_cycle()

        for parent, parent_items in todo.items():
            options = cfg["l2_by_parent"].get(parent, [{"value": GAP_MARKER, "description": "no confident L1 match was found for this item"}])
            batches = [parent_items[i:i + BATCH_SIZE] for i in range(0, len(parent_items), BATCH_SIZE)]
            print(f"[{axis}] L2 under '{parent}': {len(parent_items)} raw values, {len(batches)} batches", flush=True)
            tasks = [classify_batch(session, sem, model_cycle, b, options, cfg["l1_label"], parent) for b in batches]
            results = await asyncio.gather(*tasks)
            local_map = {}
            for r in results:
                local_map.update(r)
            cov = Counter()
            for it in parent_items:
                cats = local_map[it["value"]]
                mapping[f"{parent}||{it['value']}"] = cats
                for cat in cats:
                    cov[cat] += it["count"]
            coverage_by_parent[parent] = dict(cov.most_common())
            done_parents.add(parent)
            checkpoint()
            print(f"[{axis}] L2 under '{parent}': DONE, checkpointed ({len(done_parents)}/{len(buckets)} parents)", flush=True)

    print(f"[{axis}] L2 ALL DONE -> {ckpt_path}", flush=True)


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
