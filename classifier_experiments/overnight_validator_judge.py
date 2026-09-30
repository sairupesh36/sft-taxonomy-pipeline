#!/usr/bin/env python3
"""
Overnight skeptical-validator audit script.

Does NOT re-run gemma. Pulls documents that were ALREADY labeled by gemma
(the "ground truth" the FastText classifier is being trained against) from
an existing classifier_experiments/*.jsonl file, and asks two genuinely
independent models (qwen3-8-27b and openai/gpt-oss-120b -- never gemma
itself, and never gpt-oss-20b, already ruled out for quality reasons in the
parent project) to independently judge whether each assigned field is
actually correct, given the real document text and that category's own
official description from taxonomy_tree_final.xlsx.

Reuses the exact judging pattern already proven in ../taxonomy_judge.py --
this just adapts it to read pre-labeled records instead of calling
map_document live, and extends field coverage to every axis (not just
domain/task_family/subdomain/subfamily), and runs two independent judges
instead of one so a flagged disagreement can be cross-checked between them.

Usage:
  python3 overnight_validator_judge.py --n 300 --infile sft_output_sample_combined_98436.jsonl --out audit_run1.json
"""
import argparse
import asyncio
import aiohttp
import json
import random
from collections import defaultdict
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import taxonomy_map_document as mapper

BASE_DIR = Path(__file__).resolve().parent
ROUTER_URL = mapper.ROUTER_URL
API_KEY = mapper.API_KEY

JUDGE_MODELS = {
    # Both models do hidden chain-of-thought "reasoning" before writing the actual JSON answer,
    # which eats into max_tokens -- confirmed directly for qwen3-8-27b here (not just the
    # already-documented gpt-oss case): a real test call used 1,430 of a 1,500 max_tokens budget
    # purely on reasoning_content, leaving the JSON answer truncated/invalid. Budget generously
    # for both rather than assuming only gpt-oss needs the higher ceiling.
    "qwen3-8-27b": {"max_tokens": 3000, "extra": {}},
    "openai/gpt-oss-120b": {"max_tokens": 3000, "extra": {"reasoning_effort": "low"}},
}

JUDGE_SYSTEM = """You are an independent quality auditor for a document taxonomy classifier. You did NOT
produce the classification being checked -- your job is to catch its mistakes, not rubber-stamp it.

You will see: the full document text (user/assistant turns), and EVERY taxonomy field another model
assigned to it, with each assigned category's own official description where one exists. For EACH
assigned value, judge whether it is correct given the document and the category's description. Be
skeptical -- if a category's description doesn't clearly match the document, or a field looks
missing/wrong, mark it incorrect and say what it should have been instead in "reason".

Output STRICT JSON only, nothing else:
{"verdicts": [{"field": "domain", "value": "assigned_value", "correct": true, "reason": "short reason"}, ...]}
One verdict entry per assigned scalar/list-item value actually present in the record. Skip fields that
are empty lists (nothing to judge there)."""


def sheet_desc(sheet, col, val, parent_col=None, parent_vals=None):
    """Look up a category's own official description. `domain_subdomain_values` and
    `task_subfamily_values` have a SEPARATE row (with its own description) for the shared
    value "other" per parent (one "other" per domain, one per task_family) -- and possibly
    other value-name collisions across parents too. A naive `df[col]==val` lookup grabs
    whichever matching row happens to sort first (e.g. agriculture's "other"), regardless of
    the real document's actual domain/task_family -- confirmed this was happening for real
    (an automated judge run flagged "other -- description refers to agriculture" for a legal
    document and a UML/software document, both false alarms caused by this exact bug, not by
    the mapper). Scope the lookup by parent_col/parent_vals (e.g. domain) whenever given, and
    only fall back to the unscoped lookup if that yields nothing."""
    df = mapper._sheets[sheet]
    if parent_col and parent_vals:
        scoped = df[(df[col] == val) & (df[parent_col].isin(parent_vals))]
        if len(scoped):
            return scoped["description"].values[0]
    row = df[df[col] == val]
    return row["description"].values[0] if len(row) else None


def build_judge_prompt(text, result):
    # 3000 chars silently cut off real task-defining content on at least one real, longer
    # document tonight (a "financial agent" agentic task whose second, most-specific tool
    # name fell just past the old cutoff) -- causing an independent judge to flag 10
    # different fields as "wrong" on a document where Gemma's labels were actually all
    # correct, simply because the judge was never shown enough of the text to know that.
    # Raised generously; the model's own max_tokens budget is the real constraint elsewhere,
    # not this.
    lines = [f"DOCUMENT:\n{text[:6000]}\n\nASSIGNED TAXONOMY (verify each field below):"]
    list_field_sheets = {
        "domain": ("domain_values", "domain", None),
        "task_family": ("task_family_values", "task_family", None),
        "domain_subdomain": ("domain_subdomain_values", "domain_subdomain", ("domain", result.get("domain"))),
        "task_subfamily": ("task_subfamily_values", "task_subfamily", ("task_family", result.get("task_family"))),
        "tool_category": ("tool_category_values", "tool_category", None),
        "constraints": ("constraints_values", "constraint", None),
    }
    for field, (sheet, col, parent) in list_field_sheets.items():
        vals = result.get(field) or []
        parent_col, parent_vals = parent if parent else (None, None)
        for v in vals:
            desc = sheet_desc(sheet, col, v, parent_col, parent_vals)
            if desc:
                lines.append(f'- {field} = "{v}": {desc}')
            else:
                lines.append(f'- {field} = "{v}": (no description found -- flag if this looks like a hallucinated category)')
    scalar_field_sheets = {
        "interaction_mode": ("interaction_mode_values", "interaction_mode"),
        "tool_requirement": ("tool_requirement_values", "tool_requirement"),
        "complexity": ("complexity_values", "complexity"),
        "task_composition": ("task_composition_values", "task_composition"),
        "response_behavior": ("response_behavior_values", "response_behavior"),
    }
    for field, (sheet, col) in scalar_field_sheets.items():
        v = result.get(field)
        if v:
            desc = sheet_desc(sheet, col, v)
            lines.append(f'- {field} = "{v}"' + (f": {desc}" if desc else ""))
    for io_field in ("input", "output"):
        io = result.get(io_field) or {}
        if io:
            lines.append(f'- {io_field}.type = "{io.get("type")}", {io_field}.format = "{io.get("format")}", '
                          f'{io_field}.language = "{io.get("language")}"')
    lines.append("\nAlso separately note (as a verdict with field=\"interaction_mode_turn_count\") whether the "
                  "literal number of [USER]: turns in the document text is consistent with the assigned "
                  "interaction_mode value.")
    return "\n".join(lines)


async def call_judge(session, sem, model, text, result):
    cfg = JUDGE_MODELS[model]
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": JUDGE_SYSTEM},
                     {"role": "user", "content": build_judge_prompt(text, result)}],
        "temperature": 0.0, "max_tokens": cfg["max_tokens"],
        **cfg["extra"],
    }
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    async with sem:
        for attempt in range(4):
            try:
                async with session.post(ROUTER_URL, json=payload, headers=headers, timeout=120) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        raw = data["choices"][0]["message"].get("content", "") or ""
                        s, e = raw.find("{"), raw.rfind("}")
                        if s != -1 and e != -1:
                            try:
                                return json.loads(raw[s:e + 1])
                            except json.JSONDecodeError:
                                pass
            except Exception:
                pass
            await asyncio.sleep(1.5 * (attempt + 1))
    return None


def stratified_sample(infile, n, seed=42):
    """Reservoir-style per-source sample so the audit isn't dominated by whichever
    source happens to sort first in the file (same concern the parent project's
    sample_docs_stratified() was built to address)."""
    by_source = defaultdict(list)
    with open(infile) as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except Exception:
                continue
            if not row.get("ok") or not row.get("result"):
                continue
            src = row.get("orig_source") or row.get("src_file") or "unknown"
            by_source[src].append(row)
    rng = random.Random(seed)
    sources = list(by_source.keys())
    per_source = max(1, n // max(1, len(sources)))
    picked = []
    for src in sources:
        pool = by_source[src]
        rng.shuffle(pool)
        picked.extend(pool[:per_source])
    rng.shuffle(picked)
    return picked[:n]


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--infile", default=str(BASE_DIR / "sft_output_sample_combined_98436.jsonl"))
    ap.add_argument("--out", default=str(BASE_DIR / "overnight_validator_audit.json"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--concurrency", type=int, default=24)
    args = ap.parse_args()

    rows = stratified_sample(args.infile, args.n, args.seed)
    print(f"Sampled {len(rows)} already-labeled docs from {args.infile} across "
          f"{len(set(r.get('orig_source') or r.get('src_file') for r in rows))} sources", flush=True)

    sem = asyncio.Semaphore(args.concurrency)
    field_stats = {m: defaultdict(lambda: [0, 0]) for m in JUDGE_MODELS}
    disagreements = {m: [] for m in JUDGE_MODELS}

    async def process_one(session, model, row):
        text = row.get("full_text") or ""
        result = row.get("result") or {}
        if not text.strip():
            return None
        verdict = await call_judge(session, sem, model, text, result)
        if not verdict:
            return None
        return row.get("uid"), row.get("orig_source"), text, verdict

    connector = aiohttp.TCPConnector(limit=args.concurrency * 2)
    async with aiohttp.ClientSession(connector=connector) as session:
        for model in JUDGE_MODELS:
            print(f"\n--- Running judge model: {model} ---", flush=True)
            outcomes = await asyncio.gather(*[process_one(session, model, row) for row in rows])
            n_ok = sum(1 for o in outcomes if o)
            print(f"{model}: {n_ok}/{len(rows)} judged successfully", flush=True)
            for outcome in outcomes:
                if not outcome:
                    continue
                uid, src, text, verdict = outcome
                for v in verdict.get("verdicts", []):
                    field, correct = v.get("field"), v.get("correct")
                    if field is None:
                        continue
                    field_stats[model][field][1] += 1
                    if correct:
                        field_stats[model][field][0] += 1
                    else:
                        disagreements[model].append({
                            "uid": uid, "orig_source": src, "field": field,
                            "value": v.get("value"), "reason": v.get("reason"),
                            "text_preview": text[:200],
                        })

    report = {"n_sampled": len(rows), "field_stats": {}, "disagreements": {}}
    for model in JUDGE_MODELS:
        print(f"\n{'='*70}\nJUDGE REPORT -- model {model}\n{'='*70}")
        fs = field_stats[model]
        overall_c = sum(c for c, t in fs.values())
        overall_t = sum(t for c, t in fs.values())
        if overall_t:
            print(f"OVERALL: {overall_c}/{overall_t} = {100*overall_c/overall_t:.1f}%")
        for field, (c, t) in sorted(fs.items()):
            pct = 100 * c / t if t else 0
            print(f"  {field:28s}: {c:4d}/{t:4d} = {pct:5.1f}%")
        print(f"{len(disagreements[model])} flagged disagreements (showing up to 40):")
        for d in disagreements[model][:40]:
            print(f"  [{d['field']}] \"{d['value']}\" -- {d['reason']}  (uid={d['uid']}, {d['text_preview'][:70]}...)")
        report["field_stats"][model] = {k: {"correct": v[0], "total": v[1]} for k, v in fs.items()}
        report["disagreements"][model] = disagreements[model]

    Path(args.out).write_text(json.dumps(report, indent=1))
    print(f"\nFull report -> {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
