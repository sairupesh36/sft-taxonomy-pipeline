#!/usr/bin/env python3
"""
The real mapping pipeline: takes a raw SFT/agentic document (actual text, not
a pre-extracted label) and classifies it against the full frozen taxonomy_tree_final.xlsx
schema in 2 LLM calls:
  Call A (coarse): domain(s), task_family(ies), interaction_mode (includes agentic_loop),
                    tool_requirement, complexity, task_composition
  Call B (fine, scoped to A's domain/task_family picks): subdomain(s), task_subfamily(ies),
                    tool_category, constraints, input/output type+format+language

This is the per-document classifier -- distinct from taxonomy_validate_final.py, which
only canonicalizes the short pre-extracted discovery LABELS (cheap, used for tree
gap-finding). This script reads real document text, for real per-doc taxonomy tagging.
"""
import os
import argparse
import asyncio
import aiohttp
import json
import random
import re
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq

BASE_DIR = Path("/projects/data/datasets/code_data/sai_rupesh/taxonomy")
XLSX_PATH = BASE_DIR / "taxonomy_tree_final.xlsx"

ROUTER_URL = "http://sglang-router.sglang.svc.cluster.local:30000/v1/chat/completions"
API_KEY = os.environ.get("SGLANG_API_KEY", "")  # set this env var before running -- never hardcode the real key
MODEL = "gemma-4-31b"
TIMEOUT_S = 90
RETRIES = 4
MAX_TOKENS_A = 400
MAX_TOKENS_B = 500

_sheets = pd.read_excel(XLSX_PATH, sheet_name=None)


def opt_lines(df, val_col, desc_col="description"):
    return "\n".join(f'- "{r[val_col]}": {r[desc_col]}' for _, r in df.iterrows())


DOMAIN_OPTS = opt_lines(_sheets["domain_values"], "domain")
TASK_FAMILY_OPTS = opt_lines(_sheets["task_family_values"], "task_family")
INTERACTION_OPTS = opt_lines(_sheets["interaction_mode_values"], "interaction_mode")
TOOL_REQ_OPTS = opt_lines(_sheets["tool_requirement_values"], "tool_requirement")
COMPLEXITY_OPTS = opt_lines(_sheets["complexity_values"], "complexity")
COMPOSITION_OPTS = opt_lines(_sheets["task_composition_values"], "task_composition")
TOOL_CAT_OPTS = opt_lines(_sheets["tool_category_values"], "tool_category")
CONSTRAINTS_OPTS = opt_lines(_sheets["constraints_values"], "constraint")

_dsub = _sheets["domain_subdomain_values"]
_tsub = _sheets["task_subfamily_values"]
SUBDOMAIN_BY_DOMAIN = {d: g[["domain_subdomain", "description"]].values.tolist() for d, g in _dsub.groupby("domain")}
SUBFAMILY_BY_FAMILY = {f: g[["task_subfamily", "description"]].values.tolist() for f, g in _tsub.groupby("task_family")}


def _parse_inputoutput_sheet():
    """The 'inputoutput' sheet packs 3 separate tables (LANGUAGE VALUES, TYPE -> VALID
    FORMAT MAPPING, FORMAT VALUES) into one sheet with section-header rows and blank
    separators -- pandas can't read it as one dataframe, so parse it by hand."""
    # keep_default_na=False: pandas otherwise silently reads the literal string "NA"
    # (Nauru's real ISO 639-1 code) as a missing value and drops the row entirely.
    raw = pd.read_excel(XLSX_PATH, sheet_name="inputoutput", header=None, keep_default_na=False, na_values=[])
    languages, types, formats = {}, {}, {}
    section = None
    for _, row in raw.iterrows():
        cells = [c for c in row.tolist()]
        first = str(cells[0]) if cells[0] is not None else ""
        if first == "LANGUAGE VALUES":
            section = "lang"; continue
        if first == "TYPE → VALID FORMAT MAPPING":
            section = "type"; continue
        if first == "FORMAT VALUES":
            section = "format"; continue
        if first in ("language_code", "type", "format", "", "nan") or cells[0] is None:
            continue
        if section == "lang":
            languages[cells[0]] = {"name": cells[1], "description": cells[3]}
        elif section == "type":
            valid_formats = [f.strip() for f in str(cells[2]).split(",")]
            types[cells[0]] = {"used_in": cells[1], "valid_formats": valid_formats, "description": cells[3]}
        elif section == "format":
            formats[cells[0]] = {"used_in": cells[1], "description": cells[2]}
    return languages, types, formats


LANGUAGES, IO_TYPES, IO_FORMATS = _parse_inputoutput_sheet()
LANGUAGE_OPTS = "\n".join(f'- "{code}": {v["description"]}' for code, v in LANGUAGES.items())
TYPE_OPTS = "\n".join(f'- "{t}": {v["description"]} Valid formats: {", ".join(v["valid_formats"])}.'
                      for t, v in IO_TYPES.items())

# Common markdown code-fence language tags -> this project's closed CODE format codes, for the
# deterministic output-format override in deterministic_fixes(). Every value here must be a real
# member of IO_TYPES["CODE"]["valid_formats"] -- built from aliases onto that live set, not hardcoded
# separately, so it can't drift out of sync with the tree.
_CODE_FORMATS = set(IO_TYPES["CODE"]["valid_formats"])
_FENCE_ALIASES = {
    "python": "PY", "py": "PY", "python3": "PY", "javascript": "JS", "js": "JS", "node": "JS",
    "typescript": "TS", "ts": "TS", "tsx": "TSX", "jsx": "JSX", "bash": "SH", "sh": "SH",
    "shell": "SH", "zsh": "SH", "shellscript": "SH", "sql": "SQL", "mysql": "SQL", "postgresql": "SQL",
    "csharp": "CS", "cs": "CS", "c#": "CS", "cpp": "CPP", "c++": "CPP", "golang": "GO", "go": "GO",
    "rust": "RS", "rs": "RS", "ruby": "RB", "rb": "RB", "php": "PHP", "kotlin": "KT", "kt": "KT",
    "swift": "SWIFT", "java": "JAVA", "html": "HTML", "css": "CSS",
    "dockerfile": "DOCKERFILE", "powershell": "POWERSHELL", "ps1": "POWERSHELL", "matlab": "MATLAB",
    "perl": "PERL", "scala": "SCALA", "lua": "LUA", "r": "R", "dart": "DART", "elixir": "ELIXIR",
    "haskell": "HS", "hs": "HS", "objectivec": "OBJC", "objc": "OBJC", "graphql": "GRAPHQL",
    "makefile": "MAKEFILE", "ini": "INI", "toml": "TOML", "latex": "LATEX", "tex": "LATEX",
}

# Phrases that open a genuine decline/refusal, checked against the first 300 chars of the assistant's
# reply for the deterministic compliance-constraint fix in deterministic_fixes(). Kept deliberately narrow
# (clear refusal openers only) to avoid false-positiving on responses that merely discuss inability in
# passing (e.g. "I cannot verify this without more context" mid-explanation).
REFUSAL_MARKERS = [
    "I'm sorry, but I cannot", "I'm sorry, but I can't", "I am sorry, but I cannot",
    "I cannot assist", "I can't assist", "I cannot help", "I can't help",
    "I'm unable to assist", "I am unable to assist", "I'm unable to help", "I am unable to help",
    "I cannot provide", "I can't provide", "I'm not able to provide", "I am not able to provide",
    "I cannot fulfill", "I can't fulfill", "I cannot comply", "I can't comply",
    "I cannot generate", "I can't generate", "I'm not able to assist", "I am not able to assist",
    "I must decline", "I have to decline", "I won't be able to help", "I will not be able to help",
]
FENCE_LANG_TO_CODE_FORMAT = {tag: fmt for tag, fmt in _FENCE_ALIASES.items() if fmt in _CODE_FORMATS}

# response_behavior sub-classification, checked in this order against the final assistant reply whenever
# REFUSAL_MARKERS already matched (see deterministic_fixes()). More specific patterns are checked first;
# whatever's left inside the REFUSAL_MARKERS match that isn't clarification or capability is a policy_refusal.
# Built and verified against 68 real declined_request documents from this project's own validation batches.
CLARIFICATION_MARKERS = [
    "is not clear", "isn't clear", "request is unclear", "your request is unclear",
    "need more context", "more context to provide", "is underspecified", "please clarify",
    "please provide more context", "could you clarify", "could you please clarify",
]
CAPABILITY_MARKERS = [
    "don't have real-time", "do not have real-time", "don't have access to real-time",
    "do not have access to real-time", "cannot browse the internet", "can't browse the internet",
    "unable to browse the internet", "not capable of experiencing", "don't have personal experiences",
    "do not have personal experiences", "as an ai language model, i don't have",
    "as an ai language model, i do not have", "text-based ai", "i'm a text-based",
    "pretraining cutoff", "training data cutoff", "knowledge cutoff", "don't have the capability to",
    "do not have the capability to", "cannot fetch", "can't fetch", "cannot download", "can't download",
    "don't have access to current", "do not have access to current", "unable to provide real-time",
    "no real-time access", "i don't have direct access", "i do not have direct access",
]

# Phrases indicating a tool/API call actually failed, checked (lowercased) against the assistant's reply
# for the deterministic tool_failure_handling constraint. Only checked when tool_category is already
# non-empty (a tool was genuinely in play), so generic words like "error" in an unrelated context can't
# false-positive on documents with no tool involved at all.
TOOL_FAILURE_MARKERS = [
    "error occurred", "failed to", "timeout", "timed out", "rate limit", "unable to retrieve",
    "request failed", "an error", "try again", "no response from", "connection error",
    "call failed", "api error", "encountered an error", "something went wrong",
]

# Explicit RAG-style "answer strictly from this retrieved passage" template markers, checked against the
# input (pre-[ASSISTANT]:) portion of the text for the deterministic REFERENCE_DATA input-type override.
# Found via a real 2000-doc audit: documents matching one of these exact templates got input.type=
# "REFERENCE_DATA" only 23/65 times (35%) when left purely to the LLM -- the other 42 fell through to
# generic TEXT/FREE_TEXT despite being structurally identical to the 23 correct ones (verified by reading
# both sets in full). These are precise, multi-word phrases from real recurring templates, not generic
# single words, to avoid false-positiving on unrelated content.
REFERENCE_DATA_MARKERS = [
    "document id: 1", "document id:1",
    "i have been asked the following",
    "the following query requires an answer that is derived from the context given",
    "it is imperative that the answer is not based on external knowledge",
    "you will be shown biomedical passages",
    "excerpts from biomedical research papers",
    "a list of paragraphs is provided as context to help you answer the question",
]

VALID = {
    "domain": set(_sheets["domain_values"]["domain"]),
    "task_family": set(_sheets["task_family_values"]["task_family"]),
    "domain_subdomain": set(_dsub["domain_subdomain"]),
    "task_subfamily": set(_tsub["task_subfamily"]),
    "interaction_mode": set(_sheets["interaction_mode_values"]["interaction_mode"]),
    "tool_requirement": set(_sheets["tool_requirement_values"]["tool_requirement"]),
    "tool_category": set(_sheets["tool_category_values"]["tool_category"]),
    "constraints": set(_sheets["constraints_values"]["constraint"]),
    "complexity": set(_sheets["complexity_values"]["complexity"]),
    "task_composition": set(_sheets["task_composition_values"]["task_composition"]),
}


MAX_LABELS = 2  # domain/task_family/domain_subdomain/task_subfamily are prompted as "1-2 values";
                # tool_category/constraints are prompted as "0+ values" and stay uncapped.
CAPPED_FIELDS = {"domain", "task_family", "domain_subdomain", "task_subfamily"}


def validate_result(result):
    """Drop any value the model returned that isn't actually in the allowed sheets --
    closed-enum guarantee must hold even if the model hallucinates. Also enforces the
    1-2 label cap on domain/task_family/domain_subdomain/task_subfamily: the prompt asks
    for at most 2, but nothing previously stopped the model from returning more."""
    if not result:
        return result
    flagged = []
    for field in ["domain", "task_family", "domain_subdomain", "task_subfamily", "tool_category", "constraints"]:
        if field in result and isinstance(result[field], list):
            kept = [v for v in result[field] if v in VALID[field]]
            dropped = [v for v in result[field] if v not in VALID[field]]
            if dropped:
                flagged.append((field, dropped))
            if field in CAPPED_FIELDS and len(kept) > MAX_LABELS:
                flagged.append((f"{field}_over_cap", kept[MAX_LABELS:]))
                kept = kept[:MAX_LABELS]
            result[field] = kept
    for field in ["interaction_mode", "tool_requirement", "complexity", "task_composition"]:
        if field in result and result[field] not in VALID[field]:
            flagged.append((field, result[field]))
            result[field] = None
    for io_field in ["input", "output"]:
        io = result.get(io_field)
        if isinstance(io, dict):
            t = io.get("type")
            if t not in IO_TYPES:
                flagged.append((f"{io_field}.type", t))
                io["type"] = None
            else:
                valid_formats = set(IO_TYPES[t]["valid_formats"])
                if io.get("format") not in valid_formats:
                    flagged.append((f"{io_field}.format", io.get("format")))
                    io["format"] = None
            if io.get("language") not in LANGUAGES:
                flagged.append((f"{io_field}.language", io.get("language")))
                io["language"] = None
    if flagged:
        result["_hallucinated_dropped"] = flagged
    return result


def deterministic_fixes(result, text):
    """Cross-field/objective corrections that must not be left to LLM judgment -- the model can produce
    a self-contradictory or objectively-countable-wrong record even when every individual field value is
    independently valid. Runs after validate_result() so it only ever tightens an already-schema-valid
    result; never introduces a value that isn't in the closed sets."""
    if not result:
        return result
    # 1. tool_category non-empty is direct evidence a tool was actually named/used in the document -- it
    #    cannot coexist with tool_requirement="none" (that specific self-contradiction was found recurring
    #    in external audits of this pipeline's output). Bump to the more defensible non-"none" value rather
    #    than guessing "required" outright -- "required" only when the interaction is an actual tool loop.
    if result.get("tool_category") and result.get("tool_requirement") == "none":
        result["tool_requirement"] = "required" if result.get("interaction_mode") == "agentic_loop" else "optional"
    # 2. Turn count is an objective, countable property of the text, not a judgment call -- don't leave
    #    it purely to LLM estimation. If the text plainly shows 2+ user turns but the model said
    #    single_turn, that's objectively wrong regardless of how it read the content otherwise. Only
    #    correct FROM single_turn (never override agentic_loop -- a real tool loop also has multiple
    #    [USER]/[ASSISTANT] markers for its steps, and that's a different, deliberate axis value).
    if result.get("interaction_mode") == "single_turn":
        user_turns = text.count("[USER]:")
        if user_turns >= 2:
            result["interaction_mode"] = "multi_turn"
    # 2b. The reverse direction of the check above: interaction_mode="multi_turn" with only 0-1 real
    #     "[USER]:" turns is objectively wrong MOST of the time (real, measured: 1,377 of 1,445 real cases in
    #     the 98,436-doc batch) -- the model pattern-matched a word like "conversation" in an instruction/
    #     system-style opener ("Engage in a conversation to...", "A conversation between User and
    #     Assistant...") rather than the text actually being multi-turn. But a blind reverse rule
    #     ("[USER]: count <=1 -> single_turn") would also wrongly downgrade a real, distinct case: documents
    #     whose ENTIRE prior conversation history is embedded as plain text inside one single "[USER]:" block
    #     (input.type=CONVERSATION_HISTORY-shaped), which really is multi-turn even though it never repeats
    #     the literal "[USER]:" marker. Checked directly: input.type=="CONVERSATION_HISTORY" alone is NOT a
    #     reliable signal to gate this on -- the same root-cause pattern-matching bug also over-assigns
    #     CONVERSATION_HISTORY itself to single-turn documents that merely mention "conversation" (e.g. a
    #     templated "A conversation between User and Assistant..." tool-use preamble), so gating on that field
    #     alone would still leave most of the 1,377 real errors uncorrected. The actual reliable signal is
    #     objective and text-internal, same spirit as the [USER]: count itself: does the text contain a real
    #     embedded exchange -- at least 2 "User:"/"Human:" (or "<|user|>"/"<|human|>") markers AND at least 1
    #     "Assistant:"/"AI:" (or "<|assistant|>"/"<|ai|>") marker, case-insensitively? A document merely
    #     describing "a conversation between User and Assistant" in prose has zero literal "User:"/"Human:"
    #     marker occurrences (no colon follows the word), so it doesn't false-trigger this. Verified on real
    #     data: this signal correctly keeps 721/1,445 as multi_turn (spot-checked broadly at random, all
    #     genuine multi-exchange dialogues, including a non-English example) and downgrades the other 724 to
    #     single_turn (spot-checked broadly at random, all genuinely single question/answer exchanges or
    #     template preambles, zero false downgrades found).
    elif result.get("interaction_mode") == "multi_turn" and text.count("[USER]:") <= 1:
        user_marks = len(re.findall(r"\b(?:User|Human)\s*:", text, re.I))
        asst_marks = len(re.findall(r"\b(?:Assistant|AI)\s*:", text, re.I))
        user_pipe_marks = len(re.findall(r"<\|\s*(?:user|human)\s*\|>", text, re.I))
        asst_pipe_marks = len(re.findall(r"<\|\s*(?:assistant|ai)\s*\|>", text, re.I))
        has_genuine_embedded_history = (
            (user_marks >= 2 and asst_marks >= 1)
            or (user_pipe_marks >= 2 and asst_pipe_marks >= 1)
        )
        if not has_genuine_embedded_history:
            result["interaction_mode"] = "single_turn"
    # 3. Output format is objectively detectable from the response's literal prefix in the clear-cut case
    #    (the whole response IS a code block or a JSON blob, not prose that merely mentions one) -- don't
    #    spend LLM judgment on what's a deterministic string check. Only overrides the generic default
    #    (TEXT/FREE_TEXT); never overrides a more specific choice the model already made (e.g. MARKDOWN for
    #    prose-with-an-embedded-snippet, which is often the semantically correct call and shouldn't be
    #    clobbered by this check).
    out = result.get("output")
    if isinstance(out, dict) and out.get("type") == "TEXT" and out.get("format") == "FREE_TEXT":
        last = text.rsplit("[ASSISTANT]:", 1)
        if len(last) == 2:
            reply = last[1].strip()
            fence = re.match(r"```(\w+)\s*\n", reply)
            if fence:
                lang = fence.group(1).lower()
                if lang == "json":
                    out["type"], out["format"] = "STRUCTURED_DATA", "JSON"
                else:
                    fmt = FENCE_LANG_TO_CODE_FORMAT.get(lang)
                    if fmt:
                        out["type"], out["format"] = "CODE", fmt
            elif reply[:1] in "{[":
                try:
                    json.loads(reply)
                    out["type"], out["format"] = "STRUCTURED_DATA", "JSON"
                except (ValueError, json.JSONDecodeError):
                    pass
    # 4. Any input/output type where the model couldn't name a format that actually validated against that
    #    type's real closed list got its format stripped to None by validate_result() -- that combination
    #    (a specific type with no valid format) usually means the model over-called a specific type (CODE,
    #    REFERENCE_DATA, etc.) for what's really generic prose/text. Recover to the generic text default on
    #    BOTH input and output (checked separately) rather than leaving a permanently null format. Originally
    #    only checked input.type=="CODE"; generalized after finding real cases of output.type=="CODE" and
    #    input.type=="REFERENCE_DATA" hitting the exact same dead end.
    for io_field in ("input", "output"):
        io = result.get(io_field)
        if isinstance(io, dict) and io.get("format") is None:
            io["type"], io["format"] = "TEXT", "FREE_TEXT"
    # 5. A refusal/decline for safety, legal, medical, or policy reasons must carry the "compliance"
    #    constraint regardless of what domain/task the underlying request maps to (rule 11 already keeps
    #    the domain/task on-topic through a refusal -- this ensures the refusal itself doesn't go
    #    unmarked). Detected deterministically from the response's own opening, not left to the model to
    #    remember on top of everything else it's already tracking. Also tags "declined_request" -- a
    #    narrower, distinct signal from "compliance": compliance can independently apply to fully-answered
    #    content bound by a policy/regulation (confirmed in real data: most compliance-tagged docs are NOT
    #    refusals), while declined_request specifically flags that the ask itself went unfulfilled --
    #    needed as its own signal so a refusal can be found/filtered without being conflated with every
    #    other compliance-bound-but-answered response.
    def _ensure_constraint(value):
        constraints = result.get("constraints")
        if not isinstance(constraints, list):
            constraints = []
            result["constraints"] = constraints
        if value not in constraints:
            constraints.append(value)

    last = text.rsplit("[ASSISTANT]:", 1)
    reply = last[1] if len(last) == 2 else ""
    if any(p in reply[:300] for p in REFUSAL_MARKERS):
        _ensure_constraint("compliance")
        _ensure_constraint("declined_request")
    # 6. tool_failure_handling: real, measured signal (a meaningful share of tool-using documents in this
    #    project's own validation batch show visible tool-error language) that was previously invisible in
    #    the schema -- a response that had to report/work around/recover from an errored or timed-out tool
    #    call is doing real, distinct work beyond a normal successful tool call, and downstream ablation
    #    needs to be able to find these. Unlike the refusal check above, this looks at the WHOLE visible
    #    text, not just the final assistant turn -- a tool failure is a property of the trajectory (it can
    #    happen mid-conversation and get recovered from before the final answer), not just the last
    #    utterance. Only applies when a tool was actually in play (tool_category non-empty) AND the text
    #    shows real failure language -- don't guess this from topic alone.
    if result.get("tool_category") and any(p in text.lower() for p in TOOL_FAILURE_MARKERS):
        _ensure_constraint("tool_failure_handling")
    # 7. response_behavior: a separate, standalone signal for "what actually happened when the model
    #    responded" -- deliberately NOT folded into "constraints" (constraints are things the USER's request
    #    required of the response; this describes an outcome, a different kind of fact entirely -- conflating
    #    the two would make "constraints" unusable for its original purpose). Trigger condition is
    #    "declined_request" already being present in constraints at this point -- NOT a fresh independent
    #    REFUSAL_MARKERS check -- because declined_request can get set two ways: my deterministic check above,
    #    OR the model choosing it itself in Call B (it's a real closed-set constraint option now, visible in
    #    its own prompt, and the model does sometimes correctly self-select it on phrasings REFUSAL_MARKERS
    #    doesn't cover, e.g. "I am not capable of experiencing sensory input" has no "but I cannot"/"I can't").
    #    Re-deriving from REFUSAL_MARKERS alone silently missed those model-caught cases in testing (38/68
    #    real documents) -- keying off the combined signal instead fixed that. Sub-classified in order from
    #    most to least specific: a genuinely unclear/ambiguous request (clarification_required) and an
    #    inherent AI capability gap (capability_limitation) are checked first since they're narrower and more
    #    identifiable; anything else defaults to policy_refusal. Verified against 68 real documents.
    reply_lower = reply.lower()
    if "declined_request" in (result.get("constraints") or []):
        if any(p in reply_lower[:400] for p in CLARIFICATION_MARKERS):
            result["response_behavior"] = "clarification_required"
        elif any(p in reply_lower[:400] for p in CAPABILITY_MARKERS):
            result["response_behavior"] = "capability_limitation"
        else:
            result["response_behavior"] = "policy_refusal"
    else:
        result["response_behavior"] = "normal"
    # 8. REFERENCE_DATA input override: an explicit "answer strictly from this retrieved passage" RAG-style
    #    template (see REFERENCE_DATA_MARKERS) is an objective, literal textual fact about the input, not a
    #    judgment call -- yet left to the LLM alone it only got this right 35% of the time (23/65 real
    #    documents) despite being the exact same template every time. Checked against the input portion only
    #    (before the first "[ASSISTANT]:"), so an assistant reply that happens to mention these phrases can't
    #    false-positive this. Only overrides when the model called it generic "TEXT" (FREE_TEXT or
    #    MARKDOWN -- these templates often have "### Task:"/"### Context:" markdown-style headers, which
    #    can make the model pick MARKDOWN even though the real answer is still REFERENCE_DATA; checked and
    #    broadened after finding real cases stuck on this exact gap) -- never clobbers a more specific type
    #    the model already correctly chose (e.g. TABULAR_DATA for a table-shaped input).
    input_text = text.split("[ASSISTANT]:", 1)[0].lower()
    inp = result.get("input")
    if (isinstance(inp, dict) and inp.get("type") == "TEXT"
            and any(p in input_text for p in REFERENCE_DATA_MARKERS)):
        inp["type"], inp["format"] = "REFERENCE_DATA", "TXT"
    return result


SYSTEM_A_RULES = """RULES (apply these before assigning any category -- violating one of these is the most
common way this task gets a category wrong, so check each one deliberately, don't just pattern-match):

1. "general" is a narrow fallback, not a default. Use it ONLY for content with no specialized domain at all
   (small talk, trivia, capability/refusal statements). If a more specific domain plausibly fits, use that
   instead, even if the fit isn't perfect.
2. Creative writing, fiction, storytelling, and poetry belong under "humanities" (literature). Reviews or
   discussion of film/TV/music/games belong under "media_and_entertainment" or "gaming" -- NOT
   "arts_and_culture", unless the content is specifically about visual/performing arts or crafts themselves.
3. Classify a writing task by WHAT KIND OF TASK it is, not by what topic it happens to mention. Composing an
   essay, story, poem, script, or reflective/personal narrative is a literature/composition task under
   "humanities" even if its subject matter references another domain (e.g. a reflective essay about working
   in healthcare is still a literature task, not a healthcare one).
4. "agentic_loop" (an interaction_mode value) covers autonomous multi-step loops against tools or an
   environment. Use it instead of "single_turn"/"multi_turn" whenever the model is looping (plan, act,
   observe, repeat) rather than producing one direct response.
5. Before assigning "other" at any level, actively check every sibling option in the given list and rule
   each one out -- "other" is a genuine last resort, not a shortcut for an imperfect-but-real fit.
6. Only assign a second value (domain, task_family, or any field prompted as "1-2 values") when the content
   is genuinely about both -- not as a hedge when you're unsure which single value is right. Pick the best
   one confidently instead. Concrete test before adding a second domain/task_family: "would this document
   still deserve this second label if the assistant had solved/answered it a completely different way?" If
   yes, the second label is genuinely independent and belongs. If no -- the label only applies because of
   the particular METHOD the assistant happened to use to solve an otherwise single-domain task -- leave it
   off. The clearest recurring case: a pure math/logic/reasoning problem where the assistant happens to
   write and run a Python snippet to compute the answer is NOT also a "software_engineering"/
   "code_generation" task -- the user never asked for code, and the same question answered by hand would be
   the identical task. Tag the actual tool usage on tool_category ("code_execution") and tool_requirement
   ("optional"/"required" as appropriate) instead -- those axes exist precisely to capture HOW something was
   solved without contaminating WHAT was asked into a second domain/task_family.
7. For input/output type, format, and language: choose only from the exact closed lists given -- never
   invent a value, abbreviation, or variant spelling not literally present in the list.
8. Tracing, running-through, or explaining what a piece of code does step by step (e.g. "trace this loop
   and explain the output") is an "explanation_and_tutoring" task, NOT "analysis" -- "analysis" is for
   examining data/trends/sentiment/patterns, not walking through code execution.
9. Natural language inference / entailment tasks (given a premise, judge whether a hypothesis is true,
   false, or unknown/uncertain) are a "classification" task (pick the class: entailment/contradiction/
   neutral, or however many options are given), NOT "question_answering" -- the model isn't retrieving or
   answering a question, it's assigning one of a fixed set of judgment labels.
10. Two specific domain-routing calls that are easy to get wrong:
   a. Extracting entities/relations from, or answering questions grounded in, biomedical/life-sciences
      RESEARCH LITERATURE (journal abstracts, paper excerpts) is "science_and_research", NOT
      "pharmaceuticals_and_life_sciences" -- the latter is for drug discovery/trials/regulatory/biotech/
      genomics R&D work itself, not for processing text about research.
   b. A task whose content IS the mechanics of calling/integrating/orchestrating an API or software tool,
      with no specific enterprise system, network, cloud platform, helpdesk ticket, or asset being managed,
      is "software_engineering", NOT "information_technology" -- information_technology's subdomains are
      about operating enterprise IT (support, networks, cloud infra, enterprise systems, asset management),
      not about generic API/tool plumbing as a topic in itself.
11. Classify what the USER asked for, not what the assistant happened to produce. Two ways this goes wrong:
   a. If the assistant declines/refuses a request, the domain and task_family still reflect the substance of
      what was ASKED, not "general"/"generation" as a side effect of the refusal. A declined question about
      synthesizing a chemical is still science_and_research/pharmaceuticals content, not general_assistance;
      a declined request for someone's private phone number is still whatever task type was asked for
      (e.g. search_and_retrieval), not a reason to blank out the domain. Only fall back toward "general" if
      the REQUEST ITSELF (not the refusal) genuinely has no specialized subject matter.
   b. If the assistant answers in an unexpected style the user didn't ask for (e.g. responds in rhyme/verse
      to a plain factual question), task_family reflects what was ASKED (e.g. question_answering), not the
      incidental style of the response (creative_generation) -- don't let the assistant's stylistic choice
      override the task type that was actually requested.
12. A coding or math task should only get a second domain when that domain is genuinely what the work is
   FOR (e.g. building a financial risk model, analyzing real sports statistics data) -- not when a
   domain-flavored variable name, scenario, or word-problem dressing is merely incidental context for what
   is otherwise a generic programming/arithmetic exercise. "Write a function to compute a batting average
   from a list of at-bats" or "calculate year-over-year revenue growth given these numbers" are generic
   programming/calculation tasks (software_engineering / mathematics only) -- the sports or finance flavor
   is illustrative dressing, not the actual subject of the work, so don't add sports_and_recreation or
   finance as a second domain just because the word problem mentions one.
13. "cross_domain" is for a task that genuinely spans multiple real subject domains at once -- it is NOT
   for a multi-hop reasoning chain that stays within one topic (or no specialized topic at all) just because
   answering it requires looking up several connected facts. A multi-hop trivia/geography question ("which
   country is the district that borders the district containing city X twinned with Y") is still a single
   general-knowledge or topical lookup task, not cross_domain, merely because the reasoning traverses
   several entities -- multi-hop-ness belongs on task_subfamily (multi_hop_qa), not on the domain axis.
14. Operating/navigating a webpage (choosing which link/button/element to click next, e.g. from a parsed
   accessibility tree or DOM snapshot) is a completely different task from building or modifying that
   webpage's code. Don't default this to "frontend_development"/"software_engineering" just because HTML,
   DOM, or accessibility-tree syntax appears in the text -- that's the OBSERVATION FORMAT the navigation
   task is presented in, not evidence the task is about writing frontend code. For this kind of task:
   a. task_family is "tool_use_and_function_calling" (choosing the correct next browser action) -- do NOT
      also add "extraction" just because the response identifies which numbered/labeled element to click;
      identifying the right element is how the navigation action gets chosen, not a separate extraction
      task in its own right.
   b. domain/subdomain should reflect what the page's actual CONTENT is about, the same way browsing an
      e-commerce site gets "ecommerce_and_retail" -- not "software_engineering" by default. If the page's
      subject matter is genuinely generic/unclear from what's shown, "general" is the right fallback, not
      frontend_development.
15. When domain (or task_family) has two values, list the PRIMARY one first: the domain of the user's
   substantive objective -- what they're actually trying to accomplish. List any SECONDARY domain(s) after
   it -- the contextual/application setting the objective happens to sit in (e.g. which industry a database
   query is about, which platform an API belongs to). This ordering is meaningful, not arbitrary -- always
   put the substantive-objective domain first.
16. LSAT-style analytical-reasoning puzzles: a set of items that must be sequenced, assigned, or selected
   subject to a list of logical rules/clues (e.g. "X must happen before Y", "A cannot be paired with B"),
   where solving it requires ONLY the stated clues and pure deduction -- no real outside knowledge of
   whatever the cover-story topic is. These almost always come wrapped in an arbitrary, swappable cover
   story (a physician conference, a physics lab, a speech-therapy classroom, three colored boxes -- the
   puzzle is mechanically identical no matter which). Don't let that cover story drive domain or
   task_family/subfamily -- tag these consistently as domain="mathematics", task_family=
   "proof_and_formal_reasoning", task_subfamily="logical_deduction", regardless of what the story is about.
   This is different from a genuine domain question that merely requires deduction (e.g. "given these facts
   and this law, is the defendant liable" IS really a legal question, since the legal content is
   substantively necessary to answer it, not decorative) -- those keep their real domain. The test: could
   you swap the cover story for an unrelated one (assign colors to boxes, letters to seats) without changing
   the actual puzzle at all? If yes, it's a pure logic puzzle -- tag it as one, not by its costume.
17. Column-header / table-schema identification tasks (e.g. "given this table data and a list of candidate
   headers, identify the correct header for each column", matching or mapping columns between two tables or
   schemas) belong to domain="data_and_information_management" (subdomain "schema_and_column_structure_
   matching") even when the table's own row/cell CONTENT is about an unrelated topic (sports, movies, people,
   etc.) -- the task is about the table's structure, not its subject matter, so don't default this to
   "general" just because the content looks like ordinary/miscellaneous data.
18. Formally-specified programming/algorithmic problems: text that gives an explicit Input/Output
   specification (e.g. "Input: N (1 <= N <= 1000), then N lines of...", "Output: two integers...") together
   with one or more worked input/output examples -- even when wrapped in an unrelated cover story (an
   airport's runway scheduling, a greenhouse's harvest scheduling, a hospital's patient/MRI scheduling, a
   warehouse robot's routing, a barcode-matching system) -- is domain="software_engineering",
   task_family="code_generation", NOT "mathematics"/"proof_and_formal_reasoning" AND NOT "mathematics"/
   "calculation" either -- this applies no matter which mathematics task_family the content might otherwise
   suggest, regardless of whether the assistant's answer is literal code or reasoning toward a solution, and
   regardless of whether the underlying technique is a discrete-math/graph/DP/counting/combinatorics
   algorithm. A numeric-sounding ask ("compute nCr mod 1e9+7", "find the maximum number of X that fit under a
   weight limit") does NOT make it "calculation" once a formal Input/Output spec plus worked examples are
   present -- that combination always signals a general procedure a program must implement for arbitrary
   future inputs, not a one-off arithmetic/combinatorics computation. This is different from rule 16's
   LSAT-style puzzles: those ask for ONE answer deduced from stated clues about a fixed, specific scenario,
   with no I/O specification and no notion of arbitrary future inputs. These instead define a general
   PROCEDURE that must work for any input of the stated shape -- a size bound like "1 <= N <= 1000" describes
   the range of possible INPUTS a program must handle, not a single puzzle instance -- and that procedural
   framing is what makes the true deliverable an algorithm/program, so it belongs under code_generation even
   when no line explicitly says "write code" or "implement a function"."""

SYSTEM_A = f"""You are an expert taxonomist tagging a real SFT/agentic training document against a fixed schema.

{SYSTEM_A_RULES}

Assign:
1. domain: 1-2 values from this list (most tasks need only 1; use 2 only if genuinely cross-domain):
{DOMAIN_OPTS}
2. task_family: 1-2 values from this list (same rule):
{TASK_FAMILY_OPTS}
3. interaction_mode: exactly 1 value (note "agentic_loop" covers autonomous multi-step tool/environment loops --
   use it instead of "single_turn"/"multi_turn" whenever the model is looping against tools or an environment):
{INTERACTION_OPTS}
4. tool_requirement: exactly 1 value:
{TOOL_REQ_OPTS}
5. complexity: exactly 1 value:
{COMPLEXITY_OPTS}
6. task_composition: exactly 1 value:
{COMPOSITION_OPTS}

Output STRICT JSON only, exact value strings from the lists above:
{{"domain": ["..."], "task_family": ["..."], "interaction_mode": "...",
  "tool_requirement": "...", "complexity": "...", "task_composition": "..."}}"""


def build_system_b(domains, task_families):
    sub_opts = []
    for d in domains:
        for val, desc in SUBDOMAIN_BY_DOMAIN.get(d, []):
            sub_opts.append(f'- "{val}" (under {d}): {desc}')
    subfam_opts = []
    for f in task_families:
        for val, desc in SUBFAMILY_BY_FAMILY.get(f, []):
            subfam_opts.append(f'- "{val}" (under {f}): {desc}')

    return f"""You are tagging the same document at a finer grain, already known to be domain={domains}, task_family={task_families}.

RULES (same discipline as the coarse pass -- check each deliberately):
1. Before assigning "other" at either the subdomain or subfamily level, actively check every sibling option
   in the given list and rule each one out. "other" is a genuine last resort, not a shortcut for an
   imperfect-but-real fit. Concrete case this is frequently missed on: when domain includes
   "artificial_intelligence_and_machine_learning" because the task involves building, training, fine-tuning,
   or evaluating an ML model (e.g. "build an ensemble model", "train a classifier"), the subdomain for that
   AI/ML aspect is "model_training_and_finetuning" ("Training and adapting AI and machine learning models")
   -- not "other" -- even when the model's subject matter is about an unrelated industry (finance, healthcare,
   etc., which gets its own separate domain/subdomain). Another frequently-missed case: a GitHub-style issue
   or project ticket (often starting "[PROBLEM_STATEMENT]:", or containing template headers like "What is
   the expected enhancement?", "Current behavior", "Add option ... to") that requests a new feature, an
   enhancement, or a doc improvement for an open-source project -- when domain is "software_engineering",
   the subdomain for this is "software_project_management" ("Managing a software project's feature requests,
   enhancement proposals, bug reports, and issue-tracker process"), not "other". The same GitHub-issue-style
   content also has a specific task_subfamily home under "requirements_analysis": "enhancement_request" ("A
   specific, already-scoped request to add, change, or improve one capability of an existing system") -- do
   NOT default this to "functional_requirements_gathering", which is for discovering/documenting a system's
   needed functions from scratch, not responding to an already-formed feature/enhancement ask. A third
   frequently-missed case (see coarse-pass rule 18): a formally-specified programming/algorithmic problem
   (explicit Input/Output spec plus worked examples, e.g. a scheduling or routing problem wrapped in some
   cover story) that coarse-pass rule 18 correctly routed to domain="software_engineering"/task_family=
   "code_generation" -- the subdomain for this is "general_purpose_programming" ("standalone utility
   functions, data structures, algorithms, or example classes not tied to a specific application layer"),
   and the task_subfamily is "function_generation" -- not "other" for either, even though the problem isn't
   framed as building an application feature.
2. Only assign a second subdomain/subfamily value when the content is genuinely about both, not as a hedge.
   Same test as the coarse pass: would this still deserve the second label if solved a different way? If
   the second label only applies because of the tool/method the assistant happened to use (e.g. a math
   problem solved via a Python snippet doesn't also need "general_purpose_programming"/"function_generation"
   -- that's what tool_category="code_execution" is for), leave it off. Specific recurring case: under the
   "generation" task_family, "text_generation" is the generic member of that family -- "creative_generation",
   "structured_content_generation", and "template_based_generation" are all just more specific kinds of
   generating text. Don't pair "text_generation" with one of its own more-specific siblings for a single
   integrated writing request (e.g. "write two funny product descriptions" is creative_generation ALONE, not
   creative_generation + text_generation) -- that's tagging the same single action as both the general case
   and the specific case, not two genuinely separate asks. Only keep both if the document has a real, second,
   independently-meaningful generation task alongside the specific one.
3. For input/output type, format, and language: choose only from the exact closed lists given below -- never
   invent a value, abbreviation, or variant spelling not literally present in the list.
4. tool_category and constraints are open-ended (0 or more) -- only include a value if it's actually true of
   this specific document, don't pad the list speculatively.
5. Pick task_subfamily by what the task fundamentally asks for, not by how the output happens to be serialized.
   E.g. a request to write a SQL query is "function_generation" under code_generation whether the answer is
   returned as raw SQL text or wrapped inside a JSON object (e.g. {{"sql": "..."}}) -- don't fall back to
   "other" just because the output format isn't the raw code type.
6. "reasoning_scratchpad" specifically (this rule affects ONLY this one value -- keep applying every other
   tool_category value, e.g. code_execution/database/api, strictly by its own normal criteria no matter what
   this rule says): tag "reasoning_scratchpad" when the document uses an explicit LABELED reasoning step as
   part of its format (a line literally starting "Think:", "Thought:", "cot:", "COT:", or equivalent) before
   an action -- one visible instance of that labeled format is enough, it doesn't need to repeat multiple
   times in the excerpt you're given. Do NOT tag "reasoning_scratchpad" when the response only explains its
   steps in ordinary prose with no such label (e.g. "Let's break this down: 1. ... 2. ... 3. ..." or "Step 1:
   ... Step 2: ..." followed by a code snippet or answer) -- ordinary numbered/step prose is not the
   "reasoning_scratchpad" tool, only an explicitly labeled Think/Thought/cot-style line is.

Assign:
1. domain_subdomain: 1-2 values from this list (scoped to the given domain(s)):
{chr(10).join(sub_opts) if sub_opts else '- "other": no subdomain list available'}
2. task_subfamily: 1-2 values from this list (scoped to the given task_family(ies)):
{chr(10).join(subfam_opts) if subfam_opts else '- "other": no subfamily list available'}
3. tool_category: 0+ values (empty list if no tools involved):
{TOOL_CAT_OPTS}
4. constraints: 0+ values (empty list if none apply):
{CONSTRAINTS_OPTS}
5. input: {{"type": "...", "format": "...", "language": "..."}} -- type must be one of:
{TYPE_OPTS}
   format must be one of that type's valid formats listed above. language must be an exact code from:
{LANGUAGE_OPTS}
6. output: same structure as input, describing what the model produces.

Output STRICT JSON only:
{{"domain_subdomain": ["..."], "task_subfamily": ["..."], "tool_category": [], "constraints": [],
  "input": {{"type": "TEXT", "format": "FREE_TEXT", "language": "EN"}},
  "output": {{"type": "TEXT", "format": "FREE_TEXT", "language": "EN"}}}}"""


def extract_text_from_row(row, max_chars=3500):
    for col in ["messages", "conversations", "conversation", "trajectory", "turns", "dialogue"]:
        if col in row and row[col] is not None:
            val = row[col]
            if isinstance(val, str):
                try:
                    val = json.loads(val)
                except Exception:
                    return val[:max_chars]
            if hasattr(val, "tolist"):
                val = val.tolist()
            if isinstance(val, list):
                parts, total = [], 0
                for m in val:
                    if isinstance(m, dict):
                        role = m.get("role") or m.get("from") or "UNKNOWN"
                        content = m.get("content") or m.get("value") or m.get("text") or ""
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
    for col in ["problem_statement", "instruction", "input", "prompt", "question"]:
        if col in row and row[col] is not None and str(row[col]).strip():
            combined.append(f"[{col.upper()}]: {row[col]}")
    return "\n\n".join(combined)[:max_chars] if combined else ""


TULU3_SOURCES = [
    "ai2-adapt-dev/coconot_converted", "ai2-adapt-dev/evol_codealpaca_heval_decontaminated",
    "ai2-adapt-dev/flan_v2_converted", "ai2-adapt-dev/no_robots_converted",
    "ai2-adapt-dev/numinamath_tir_math_decontaminated", "ai2-adapt-dev/oasst1_converted",
    "ai2-adapt-dev/personahub_code_v2_34999", "ai2-adapt-dev/personahub_ifdata_manual_seed_v3_29980",
    "ai2-adapt-dev/personahub_math_v5_regen_149960", "ai2-adapt-dev/tulu_hard_coded_repeated_10",
    "ai2-adapt-dev/tulu_v3.9_aya_100k", "ai2-adapt-dev/tulu_v3.9_open_math_2_gsm8k_50k",
    "ai2-adapt-dev/tulu_v3.9_personahub_math_interm_algebra_20k", "ai2-adapt-dev/tulu_v3.9_sciriff_10k",
    "ai2-adapt-dev/tulu_v3.9_synthetic_finalresp_wildguardmixtrain_decontaminated_50k",
    "ai2-adapt-dev/tulu_v3.9_table_gpt_5k", "ai2-adapt-dev/tulu_v3.9_wildchat_100k",
    "ai2-adapt-dev/tulu_v3.9_wildjailbreak_decontaminated_50k",
    "allenai/tulu-3-sft-personas-math-grade",
]


def _interleave_trim(buckets_in_order, n_total):
    """Round-robin across buckets (one from each in turn) instead of slicing
    the flattened list positionally -- a positional rows[:n_total] slice
    systematically drops every source that sorts late in bucket order
    whenever total collected > n_total. Returns at most n_total rows, with
    every non-empty bucket represented as evenly as possible."""
    result = []
    idx = 0
    while len(result) < n_total and any(idx < len(b) for b in buckets_in_order):
        for b in buckets_in_order:
            if idx < len(b):
                result.append(b[idx])
                if len(result) >= n_total:
                    break
        idx += 1
    return result


def _reservoir_add(reservoir, seen_count, item, quota, rng):
    """Standard reservoir sampling: after seeing `seen_count` items (1-indexed for this
    one), keep a uniform random `quota`-size sample of everything seen so far in one pass."""
    if len(reservoir) < quota:
        reservoir.append(item)
    else:
        j = rng.randint(0, seen_count - 1)
        if j < quota:
            reservoir[j] = item


def sample_docs_stratified(n_total, source_type, seed=None):
    """Draws roughly n_total//num_sources rows PER SOURCE via reservoir sampling, so
    repeated calls return DIFFERENT random documents instead of always the same first-N
    rows encountered (that was the original bug -- fixed here). Pass an explicit `seed`
    for reproducibility; omit it (default) to get a fresh random sample each call."""
    rng = random.Random(seed)
    if source_type == "tulu3":
        data_dir = BASE_DIR / "data"
        files = sorted(data_dir.glob("*.parquet"))
        quota = max(2, n_total // len(TULU3_SOURCES) + 1)
        buckets = {s: [] for s in TULU3_SOURCES}
        seen = {s: 0 for s in TULU3_SOURCES}

        for f in files:
            pf = pq.ParquetFile(f)
            for batch in pf.iter_batches(batch_size=500):
                df = batch.to_pandas()
                for _, row in df.iterrows():
                    src = row.get("source")
                    if src in buckets:
                        seen[src] += 1
                        _reservoir_add(buckets[src], seen[src], row, quota, rng)
        return _interleave_trim(list(buckets.values()), n_total)
    else:
        data_dir = BASE_DIR / "agentic_data"
        files = sorted([f for f in data_dir.glob("*.parquet") if f.stat().st_size > 5000])
        quota = max(1, n_total // len(files) + 1)
        buckets = []
        for f in files:
            pf = pq.ParquetFile(f)
            bucket, seen_count = [], 0
            for batch in pf.iter_batches(batch_size=500):
                df = batch.to_pandas()
                for _, row in df.iterrows():
                    seen_count += 1
                    _reservoir_add(bucket, seen_count, row, quota, rng)
            buckets.append(bucket)
        return _interleave_trim(buckets, n_total)


async def call_llm(session, sem, system_prompt, user_prompt, max_tokens):
    payload = {"model": MODEL, "messages": [{"role": "system", "content": system_prompt},
                                             {"role": "user", "content": user_prompt}],
               "temperature": 0.0, "max_tokens": max_tokens}
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


def _valid_labels(values, field):
    return [v for v in (values or []) if v in VALID[field]][:MAX_LABELS]


async def map_document(session, sem, text):
    a = await call_llm(session, sem, SYSTEM_A, text, MAX_TOKENS_A)
    if not a:
        return None
    # Validate domain/task_family against the real closed sets BEFORE using them to scope Call B --
    # using the raw unvalidated values here let a hallucinated domain (e.g. the model returning a
    # task_family value like "database_operations" in the domain field) silently force build_system_b()
    # into an empty subdomain list (forcing "other"), and only get caught by the final validate_result()
    # call, by which point domain ends up permanently empty with no recovery. Retry Call A once if either
    # comes back empty after filtering; only fall back to defaults as a last resort.
    domains = _valid_labels(a.get("domain"), "domain")
    task_families = _valid_labels(a.get("task_family"), "task_family")
    if not domains or not task_families:
        a_retry = await call_llm(session, sem, SYSTEM_A, text, MAX_TOKENS_A)
        if a_retry:
            a = a_retry
            domains = _valid_labels(a.get("domain"), "domain") or domains
            task_families = _valid_labels(a.get("task_family"), "task_family") or task_families
    domains = domains or ["general"]
    task_families = task_families or ["question_answering"]
    a["domain"] = domains
    a["task_family"] = task_families
    system_b = build_system_b(domains, task_families)
    b = await call_llm(session, sem, system_b, text, MAX_TOKENS_B)
    if not b:
        b = {}
    return deterministic_fixes(validate_result({**a, **b}), text)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10, help="number of sample docs to test")
    ap.add_argument("--source", choices=["tulu3", "agentic"], default="tulu3")
    args = ap.parse_args()

    rows = sample_docs_stratified(args.n, args.source)

    # Bumped from 8 -> 32: benchmarked against the real router (see taxonomy_map_document_fast.py's
    # bench_concurrency.py experiment), throughput keeps improving up to ~32-64 concurrent requests
    # with diminishing returns past that point. Pure client-side concurrency change -- doesn't touch
    # any prompt content, so it carries zero quality risk.
    sem = asyncio.Semaphore(32)
    async with aiohttp.ClientSession() as session:
        for row in rows:
            text = extract_text_from_row(row)
            if not text.strip():
                continue
            result = await map_document(session, sem, text)
            print("=" * 80)
            print(text[:200].replace("\n", " "), "...")
            print(json.dumps(result, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
