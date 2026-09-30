#!/usr/bin/env python3
"""
Automated quality validation for taxonomy_map_document.py -- no manual human
labeling required. For each sampled real document:
  1. gemma-4-31b classifies it (the production mapping pipeline).
  2. qwen3-8-27b -- a genuinely different model, not grading itself -- independently
     judges each assigned field as correct/incorrect given the doc text + the
     category's own description, and proposes a fix when it disagrees.
Aggregates per-field agreement rate and lists concrete disagreements for review.
This is meant to be re-run any time the tree or the mapping prompts change,
without requiring the user to hand-label anything.
"""
import argparse
import asyncio
import aiohttp
import json
from collections import Counter, defaultdict
from pathlib import Path

import taxonomy_map_document as mapper

BASE_DIR = Path("/projects/data/datasets/code_data/sai_rupesh/taxonomy")
JUDGE_MODEL = "qwen3-8-27b"
ROUTER_URL = mapper.ROUTER_URL
API_KEY = mapper.API_KEY

JUDGE_SYSTEM = """You are an independent quality auditor for a document taxonomy classifier. You did NOT
produce the classification being checked -- your job is to catch its mistakes, not rubber-stamp it.

You will see: the document text, and the taxonomy fields another model assigned to it (with each
assigned category's own official description). For EACH field, judge whether the assignment is
correct given the document and the category's description. Be skeptical -- if a category's
description doesn't clearly match the document, mark it incorrect.

Output STRICT JSON only:
{"verdicts": [{"field": "domain", "value": "assigned_value", "correct": true, "reason": "short reason"}, ...]}
One verdict entry per assigned value (a field with 2 values gets 2 verdict entries)."""


def build_judge_prompt(text, result):
    lines = [f"DOCUMENT:\n{text[:2500]}\n\nASSIGNED TAXONOMY (verify each):"]
    field_to_sheet_lookup = {
        "domain": ("domain_values", "domain"),
        "task_family": ("task_family_values", "task_family"),
        "domain_subdomain": ("domain_subdomain_values", "domain_subdomain"),
        "task_subfamily": ("task_subfamily_values", "task_subfamily"),
    }
    for field, (sheet, col) in field_to_sheet_lookup.items():
        vals = result.get(field) or []
        df = mapper._sheets[sheet]
        for v in vals:
            row = df[df[col] == v]
            desc = row["description"].values[0] if len(row) else "(no description found)"
            lines.append(f'- {field} = "{v}": {desc}')
    for field in ["interaction_mode", "complexity", "task_composition", "tool_requirement"]:
        v = result.get(field)
        if v:
            lines.append(f'- {field} = "{v}"')
    return "\n".join(lines)


async def call_judge(session, sem, text, result):
    payload = {
        "model": JUDGE_MODEL,
        "messages": [{"role": "system", "content": JUDGE_SYSTEM},
                     {"role": "user", "content": build_judge_prompt(text, result)}],
        "temperature": 0.0, "max_tokens": 1200,
    }
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    async with sem:
        for attempt in range(4):
            try:
                async with session.post(ROUTER_URL, json=payload, headers=headers, timeout=90) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        raw = data["choices"][0]["message"].get("content", "") or ""
                        s, e = raw.find("{"), raw.rfind("}")
                        if s != -1 and e != -1:
                            return json.loads(raw[s:e + 1])
            except Exception:
                pass
            await asyncio.sleep(1.5 * (attempt + 1))
    return None


async def sample_docs(n, source):
    return mapper.sample_docs_stratified(n, source)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--source", choices=["tulu3", "agentic", "both"], default="both")
    args = ap.parse_args()

    sources = ["tulu3", "agentic"] if args.source == "both" else [args.source]
    rows = []
    for s in sources:
        rows += await sample_docs(args.n // len(sources), s)

    sem = asyncio.Semaphore(16)
    field_stats = defaultdict(lambda: [0, 0])  # field -> [correct, total]
    disagreements = []

    async def process_one(session, row):
        text = mapper.extract_text_from_row(row)
        if not text.strip():
            return None
        result = await mapper.map_document(session, sem, text)
        if not result:
            return None
        verdict = await call_judge(session, sem, text, result)
        if not verdict:
            return None
        return text, verdict

    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=40)) as session:
        outcomes = await asyncio.gather(*[process_one(session, row) for row in rows])

    for outcome in outcomes:
        if not outcome:
            continue
        text, verdict = outcome
        for v in verdict.get("verdicts", []):
            field, correct = v.get("field"), v.get("correct")
            field_stats[field][1] += 1
            if correct:
                field_stats[field][0] += 1
            else:
                disagreements.append({"text": text[:150], "field": field, "value": v.get("value"),
                                       "reason": v.get("reason")})

    print("\n" + "=" * 70)
    print(f"JUDGE REPORT -- {len(rows)} docs sampled, judged by {JUDGE_MODEL} (independent of gemma classifier)")
    print("=" * 70)
    overall_correct = sum(c for c, t in field_stats.values())
    overall_total = sum(t for c, t in field_stats.values())
    print(f"\nOVERALL accuracy: {overall_correct}/{overall_total} = {100*overall_correct/overall_total:.1f}%" if overall_total else "no verdicts")
    for field, (c, t) in sorted(field_stats.items()):
        pct = 100 * c / t if t else 0
        print(f"  {field:20s}: {c:3d}/{t:3d} = {pct:5.1f}%")

    print(f"\n{len(disagreements)} flagged disagreements:")
    for d in disagreements[:30]:
        print(f"  [{d['field']}] \"{d['value']}\" -- {d['reason']}  (doc: {d['text'][:80]}...)")

    (BASE_DIR / "taxonomy_judge_report.json").write_text(json.dumps(
        {"field_stats": {k: {"correct": v[0], "total": v[1]} for k, v in field_stats.items()},
         "disagreements": disagreements}, indent=1))
    print(f"\nFull report -> taxonomy_judge_report.json")


if __name__ == "__main__":
    asyncio.run(main())
