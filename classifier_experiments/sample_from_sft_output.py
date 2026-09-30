"""
Samples NEW, deduped rows from /projects/data/datasets/translation_data/SFT/OUTPUT
(the same source the original 15,722-doc training set came from) to expand the
classifier's training pool. READ-ONLY against OUTPUT -- never writes, moves, or
deletes anything there; all output goes into this project's own
classifier_experiments/ folder.

Reuses the real, production column-detection and row-canonicalization code from
DEDUP_PIPELINE (sft_normalize.plan/to_messages, sft_schema.canonicalize_turns/
validate/build) exactly as normalize_job.py (the real Stage-1 job) uses it --
not a reimplementation. Also respects DEDUP_PIPELINE's own benchmarks_exclude.txt
so eval/benchmark files never enter training data, same as the real pipeline.

Sampling strategy is round-robin at TWO levels, not one:
  1. Across all 22 domains, one row at a time.
  2. WITHIN a domain, across its real distinct datasets ("families") --
     e.g. songlab__gnomad_169.parquet, _114.parquet, _242.parquet... are all
     shards of the SAME real dataset, collapsed to one family by stripping
     the trailing shard number, exactly the way the original 15,722-doc
     sampling run's own coverage report tracked "families_hit" as its real
     diversity metric. A first version of this script only round-robinned
     across raw files with a per-file cap -- that let a domain fill its
     entire quota from just ~7 distinct real datasets (confirmed on a test
     run: Maths touched only 6 families here vs. 336 in the original run),
     because a family split into many shards looked like many different
     "files" to sample from even though it's really one dataset. Grouping
     by family first, with a low per-family cap, fixes that: a domain now
     has to spread across dozens of real distinct datasets to fill its
     quota, not just a handful.
Both levels are self-balancing: once a family (or a whole domain) runs out
of real usable content, it drops out and the remaining budget naturally
flows to whatever still has real content, without a separate backfill pass.

Deduplication: content_hash (the same hash sft_schema.py itself uses) is
checked against every row already in the existing 15,722-doc sft_output_sample/
corpus, AND against every row picked so far this run, so no duplicate or
near-duplicate document is added twice.

Per-file/family sampling is randomized (files shuffled, rows shuffled within
each batch) so this isn't "the first N rows of whichever file sorts first" --
the same bug this project's own taxonomy_map_document.py had to fix once
already (see _interleave_trim / reservoir sampling there).
"""
import os
import re
import sys
import ast
import json
import glob
import random
import collections

sys.path.insert(0, "/projects/data/datasets/translation_data/SFT/DEDUP_PIPELINE")
import sft_schema as S
import sft_normalize as N
import pyarrow.parquet as pq

OUTPUT_DIR = "/projects/data/datasets/translation_data/SFT/OUTPUT"
EXCLUDE_FILE = "/projects/data/datasets/translation_data/SFT/DEDUP_PIPELINE/benchmarks_exclude.txt"
EXISTING_HASHES_FILE = "/projects/data/datasets/code_data/sai_rupesh/taxonomy/classifier_experiments/existing_content_hashes.json"
OUT_FILE = "/projects/data/datasets/code_data/sai_rupesh/taxonomy/classifier_experiments/sft_output_sample_batch4.jsonl"

# 2026-09-20 fix: some raw source files have a SEPARATE grounding/context
# column (e.g. Finance's virattt/financial-qa-10K has 'question','context',
# 'answer') that sft_normalize.plan()/to_messages() (real production code,
# not touched here) has no slot for -- it's not the exact alpaca
# 'instruction'+'input' pair, and it's not in any USER/SYSTEM column-name
# list either, so it's silently dropped from `messages` entirely. In the
# real DEDUP_PIPELINE it survives into an `extra` JSON blob, but this
# sampler calls S.build() without passing extra=..., so even that fallback
# is empty here. Net effect: for these rows, the question references a
# passage that never reaches the model, sometimes making the question
# nonsensical on its own (confirmed on real data: found and verified by the
# parallel validator agent -- e.g. "Could you please provide the case number
# for L vs Kandla..." with the actual case-record passage silently dropped).
# CONTEXT_DROP_AFFECTED_FILES.json lists the exact 57 known-affected source
# files and which column(s) got dropped for each (derived from a real scan
# of output_column_inventory.json, not guessed). This map is intentionally
# narrow/known-file-based rather than a generic "any column named *context*"
# heuristic, so it only ever changes behavior for files already confirmed to
# have this problem -- it cannot silently start altering unrelated files.
CONTEXT_DROP_AFFECTED_FILES_JSON = "/projects/data/datasets/code_data/sai_rupesh/taxonomy/classifier_experiments/context_drop_affected_files.json"


def load_context_drop_map():
    """rel path ('Finance/virattt__financial-qa-10K.parquet') -> list of
    dropped column names to splice back in, if the file exists."""
    if not os.path.exists(CONTEXT_DROP_AFFECTED_FILES_JSON):
        return {}
    with open(CONTEXT_DROP_AFFECTED_FILES_JSON) as f:
        entries = json.load(f)
    return {e["path"]: e["dropped_cols"] for e in entries}


def splice_dropped_context(raw, row, dropped_cols):
    """Prepends any real-text dropped column(s) onto the first user turn's
    content, so the question is coherent again instead of silently missing
    the passage it depends on. Skips columns whose value isn't real text
    (e.g. miriad's 'passage_position' is just a short numeric index, not
    the actual passage -- checked on real rows before deciding this filter;
    only 'passage_text' there is the real dropped grounding content)."""
    ctx_parts = []
    for col in dropped_cols:
        val = row.get(col)
        if isinstance(val, str) and len(val.strip()) > 20 and not val.strip().isdigit():
            ctx_parts.append(val.strip())
    if not ctx_parts:
        return raw
    context_block = "\n\n".join(ctx_parts)
    for m in raw:
        if m["role"] == "user":
            m["content"] = f"Context: {context_block}\n\nQuestion: {m['content']}"
            break
    return raw


# 2026-09-20, second real bug found the same night (validator agent, again
# verified by me before acting): some Tool_Use chat-format datasets store an
# assistant's tool invocation with content=None (OpenAI-style -- the actual
# call lives in a separate 'tool_calls'/'function_calls' field, not in
# 'content'). sft_normalize.to_messages()'s generic chat-turn loop does
# `cont = t.get('content') or t.get('value') or t.get('text') or
# t.get('message') or ''` then `if not str(cont).strip(): continue` --
# confirmed directly against the real production function (not guessed):
# when content is real None and no other key matches, cont='' and the WHOLE
# TURN IS SILENTLY DROPPED. The tool's result turn survives (it has real
# text), so what's left looks like a tool result appearing out of nowhere
# with no explanation of what was called. Found on 6 real Tool_Use files
# (824,555 raw rows; 642 rows already sampled into the 98,436-doc training
# file). Same fix shape as the context-drop bug: pre-process the RAW chat
# list (using sft_normalize.coerce_chat(), read-only reuse of production
# code) BEFORE calling to_messages(), synthesizing real text for any
# null-content turn that has real tool-call data sitting in a different
# field, then handing the patched list back so to_messages() runs normally
# and no longer has anything to silently drop. Verified against real rows
# from all 6 files (not just one) before trusting this -- see
# overnight_builder1_log.md for the by-file check.
TOOL_CALL_DROP_AFFECTED_FILES_JSON = "/projects/data/datasets/code_data/sai_rupesh/taxonomy/classifier_experiments/tool_call_dropped_affected_sources.json"


def load_tool_call_drop_files():
    """Set of relative OUTPUT paths known to have this problem. Deliberately
    a small, named set (same safety reasoning as CONTEXT_DROP_MAP) -- checked
    on real rows from each of the 6 files before being trusted, not a
    generic 'any row with a tool_calls field' rule that could misfire on
    files that already render tool calls as visible text (13 of the 19
    Tool_Use files checked do this fine already, e.g. inline
    <tool_call>{...}</tool_call>, and must NOT be touched)."""
    if not os.path.exists(TOOL_CALL_DROP_AFFECTED_FILES_JSON):
        return set()
    with open(TOOL_CALL_DROP_AFFECTED_FILES_JSON) as f:
        sources = json.load(f)
    # sources.json lists dataset ids (e.g. "stindardlogic/tool-calling-english-100k");
    # convert to the same relpath shape used elsewhere ("Tool_Use/stindardlogic__tool-calling-english-100k.parquet").
    paths = set()
    for src in sources:
        fname = src.replace("/", "__") + ".parquet"
        # search once for the actual domain folder rather than hardcoding "Tool_Use"
        for hit in glob.glob(os.path.join(OUTPUT_DIR, "*", fname)):
            paths.add(os.path.relpath(hit, OUTPUT_DIR))
    return paths


def synthesize_tool_call_content(turn):
    """Given a chat turn dict with no real 'content', returns a synthetic
    text description of its tool call if one is findable in a known field,
    else None. Handles the 2 real shapes found across the 6 affected files:
    a plain-text 'function_calls' field that's already human-readable
    (NewEden/jacobmorrison/allenai's Dolci family), and an OpenAI-style
    'tool_calls' field stored as a Python-repr string of
    [{'function': {'name':.., 'arguments':..}}] (stindardlogic, zake7749,
    nvidia/Nemotron)."""
    fc = turn.get("function_calls")
    if fc and str(fc).strip() and str(fc).strip().lower() != "none":
        return f"[Tool call(s)]\n{str(fc).strip()}"
    tc = turn.get("tool_calls")
    if tc and str(tc).strip() and str(tc).strip().lower() not in ("none", "[]"):
        parsed = None
        if isinstance(tc, str):
            try:
                parsed = ast.literal_eval(tc)
            except Exception:
                parsed = None
        elif isinstance(tc, (list, tuple)):
            parsed = tc
        if isinstance(parsed, list) and parsed:
            parts = []
            for call in parsed:
                if not isinstance(call, dict):
                    continue
                fn = call.get("function") if isinstance(call.get("function"), dict) else call
                name = fn.get("name", "unknown_function")
                args = fn.get("arguments", "")
                parts.append(f"{name}({args})")
            if parts:
                return "[calls " + "; ".join(parts) + "]"
        # parsing failed but there's clearly real (non-empty, non-"None") data -- use it raw rather than lose it
        return f"[Tool call(s)]\n{str(tc).strip()}"
    return None


def patch_null_content_tool_turns(raw_list):
    """Mutates raw_list in place: any turn with no real content gets a
    synthesized one if a tool-call field explains what really happened;
    turns that already have real content, or have no findable tool-call
    data either, are left untouched."""
    for t in raw_list:
        if not isinstance(t, dict):
            continue
        content = t.get("content")
        if content is not None and str(content).strip():
            continue
        synth = synthesize_tool_call_content(t)
        if synth:
            t["content"] = synth
    return raw_list


TARGET_TOTAL = 400_000  # bumped cap accordingly -- batch3 already achieved full coverage at
                         # 40/family, so this run doubles MAX_ROWS_PER_FAMILY to pull genuinely
                         # new rows per family (dedup against existing_content_hashes.json
                         # still skips anything already used) rather than re-walking the same
                         # first-40-per-family ground
MAX_ROWS_PER_FAMILY = 80      # low cap for real diversity across many distinct datasets per
                               # domain. Previously looked "too slow" at this value, but that
                               # slowness was actually the round-robin-across-families design
                               # keeping hundreds of files open at once (fixed below by draining
                               # one family at a time) -- not the cap itself. Confirmed on a real
                               # 3,000-row test: 40 here now runs at ~190 rows/sec.
                               # real distinct datasets instead of filling up on a few
MAX_BATCHES_SCANNED_PER_FILE = 20  # bound worst-case work on a huge low-yield shard
BATCH_SIZE = 500

_SHARD_SUFFIX = re.compile(r"_\d+$")
CONTEXT_DROP_MAP = {}  # populated in main(); module-level default so family_generator() always has something to .get() against
TOOL_CALL_DROP_FILES = set()  # populated in main()


def load_excludes():
    if not os.path.exists(EXCLUDE_FILE):
        return set()
    with open(EXCLUDE_FILE) as fh:
        return {l.strip() for l in fh if l.strip() and not l.startswith("#")}


def family_generator(domain_name, shard_files, rng):
    """Yields valid canonical rows from across ALL shards of ONE real dataset,
    up to MAX_ROWS_PER_FAMILY total -- so a 169-shard family and a 1-file
    family are capped the same way, by real dataset identity, not file count.

    BUG FIXED HERE: an earlier version never explicitly closed the ParquetFile
    handles below, including on the frequent early `break` (hitting
    MAX_ROWS_PER_FAMILY or MAX_BATCHES_SCANNED_PER_FILE mid-file) -- pyarrow
    doesn't release the underlying file handle just because the Python
    reference is dropped, so every file this generator ever touched stayed
    open for the rest of the run. Confirmed directly: a real run had 1,000+
    simultaneously open parquet files after 45 minutes, and got progressively
    slower the entire time -- exactly the shape of a file-handle leak, not
    "this one file is slow." Every ParquetFile is now closed in a finally
    block, whether the file finishes naturally or is bailed out of early."""
    shard_files = list(shard_files)
    rng.shuffle(shard_files)
    taken = 0
    for f in shard_files:
        if taken >= MAX_ROWS_PER_FAMILY:
            return

        # Cheap schema-only check (reads file metadata, not row data) -- skips
        # non-text files (e.g. raw molecular property tables) instantly,
        # without paying for a full read of a potentially 1M-row shard.
        schema_pf = None
        try:
            schema_pf = pq.ParquetFile(f)
            cols = list(schema_pf.schema_arrow.names)
        except Exception:
            continue
        finally:
            if schema_pf is not None:
                schema_pf.close()

        p = N.plan(cols)
        if p["mode"] is None:
            continue

        rel = os.path.relpath(f, OUTPUT_DIR)
        source = os.path.basename(f)[:-8].replace("__", "/")
        dropped_cols = CONTEXT_DROP_MAP.get(rel)
        is_tool_call_drop_file = p["mode"] == "chat" and rel in TOOL_CALL_DROP_FILES

        pf = None
        try:
            pf = pq.ParquetFile(f)
            idx = 0
            for batch_num, batch in enumerate(pf.iter_batches(batch_size=BATCH_SIZE)):
                if batch_num >= MAX_BATCHES_SCANNED_PER_FILE or taken >= MAX_ROWS_PER_FAMILY:
                    break
                rows = batch.to_pylist()
                rng.shuffle(rows)
                for r in rows:
                    i = idx
                    idx += 1
                    if taken >= MAX_ROWS_PER_FAMILY:
                        break
                    if is_tool_call_drop_file:
                        # Pre-patch the RAW chat list (before to_messages()
                        # ever sees it) so a null-content tool-call turn has
                        # real text by the time the generic per-turn loop
                        # checks it -- see patch_null_content_tool_turns()
                        # above for why this must happen before, not after.
                        try:
                            coerced = N.coerce_chat(r.get(p["col"]))
                        except Exception:
                            coerced = None
                        if coerced:
                            patch_null_content_tool_turns(coerced)
                            r = dict(r)
                            r[p["col"]] = coerced
                    # DEDUP_PIPELINE's to_messages() isn't defensive against every
                    # malformed row shape (e.g. a chat-format row missing 'role' on
                    # a non-first turn raises KeyError) -- that's their code, not
                    # ours to touch, so skip the bad row here rather than let one
                    # oddly-shaped row crash an hours-long run.
                    try:
                        raw = N.to_messages(r, p)
                    except Exception:
                        continue
                    if not raw:
                        continue
                    if dropped_cols:
                        raw = splice_dropped_context(raw, r, dropped_cols)
                    try:
                        msgs = S.canonicalize_turns(raw)
                        rec, err = S.build(msgs, file_name=rel, row_idx=i, source=source,
                                            domain=domain_name, adapter=p["mode"])
                    except Exception:
                        continue
                    if err:
                        continue
                    taken += 1
                    yield rec
        except Exception:
            continue
        finally:
            if pf is not None:
                pf.close()


def domain_generator(domain_name, excludes, seen_hashes, rng):
    """Drains one real dataset ("family") at a time -- shuffled order, so which
    ones come first is random, but only ONE family's file is ever open at once
    per domain. An earlier version round-robinned across ALL of a domain's
    families simultaneously (one row per family per round), which meant every
    single family got its first file opened almost immediately and held open
    for up to MAX_ROWS_PER_FAMILY rounds -- with domains that have hundreds of
    real families (Maths alone has 482), that's not a handful of open files,
    it's genuinely hundreds-to-thousands at once, confirmed directly on a real
    run (1,000+ open files within 22 minutes, still climbing). That was never
    really a "leak" (every file WAS eventually closed) -- it was the algorithm
    legitimately keeping that many files simultaneously in progress. Draining
    families one at a time instead bounds this to about 1 open file per
    domain, ~22 total across the whole run, with no loss of real diversity --
    the SET of families touched to fill a domain's share is identical either
    way, only the ORDER they're drained in changes."""
    files = sorted(glob.glob(os.path.join(OUTPUT_DIR, domain_name, "*.parquet")))
    files = [f for f in files if os.path.relpath(f, OUTPUT_DIR) not in excludes]

    families = collections.defaultdict(list)
    for f in files:
        fam = _SHARD_SUFFIX.sub("", os.path.basename(f)[:-len(".parquet")])
        families[fam].append(f)

    family_names = list(families.keys())
    rng.shuffle(family_names)

    for fam in family_names:
        for rec in family_generator(domain_name, families[fam], rng):
            if rec["content_hash"] in seen_hashes:
                continue
            seen_hashes.add(rec["content_hash"])
            yield rec


def main():
    rng = random.Random(42)
    excludes = load_excludes()
    print(f"loaded {len(excludes)} benchmark-exclude entries", flush=True)
    global CONTEXT_DROP_MAP, TOOL_CALL_DROP_FILES
    CONTEXT_DROP_MAP = load_context_drop_map()
    print(f"loaded {len(CONTEXT_DROP_MAP)} known context-drop-affected files to splice-fix", flush=True)
    TOOL_CALL_DROP_FILES = load_tool_call_drop_files()
    print(f"loaded {len(TOOL_CALL_DROP_FILES)} known tool-call-drop-affected files to patch", flush=True)

    with open(EXISTING_HASHES_FILE) as f:
        seen_hashes = set(json.load(f))
    print(f"loaded {len(seen_hashes)} existing content hashes to dedup against", flush=True)

    domains = sorted(os.listdir(OUTPUT_DIR))
    domains = [d for d in domains if os.path.isdir(os.path.join(OUTPUT_DIR, d))]
    print(f"domains: {len(domains)}", flush=True)

    gens = {d: domain_generator(d, excludes, seen_hashes, rng) for d in domains}
    active = list(gens.keys())
    per_domain_count = {d: 0 for d in domains}
    per_domain_families = {d: set() for d in domains}

    kept = 0
    with open(OUT_FILE, "w", encoding="utf-8") as out_f:
        round_num = 0
        while kept < TARGET_TOTAL and active:
            round_num += 1
            for d in list(active):
                if kept >= TARGET_TOTAL:
                    break
                try:
                    rec = next(gens[d])
                except StopIteration:
                    active.remove(d)
                    continue
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out_f.flush()
                kept += 1
                per_domain_count[d] += 1
                per_domain_families[d].add(_SHARD_SUFFIX.sub("", rec["source"].replace("/", "__")))
                if kept % 500 == 0:
                    print(f"  [{kept}] rows so far, {len(active)} domains still active, round {round_num}", flush=True)

    print(f"\nDONE: {kept} new rows -> {OUT_FILE}", flush=True)
    print("per-domain counts (rows / distinct real datasets touched):", flush=True)
    for d, c in sorted(per_domain_count.items(), key=lambda x: -x[1]):
        print(f"  {d:30s} {c:5d} rows  /  {len(per_domain_families[d]):4d} families", flush=True)


if __name__ == "__main__":
    main()
