# hf_agentic_pipeline -- project notes

## What this is

The pipeline that turns `/projects/data/datasets/translation_data/Agentic_Ai/OUTPUT`
(110 parquet files, 78 Hugging Face datasets, 206GB, 38,979,199 rows; read-only)
into OpenAI-format SFT JSONL at
`/projects/data/datasets/code_data/sai_rupesh/taxonomy/hf-agentic-data/`.
Built 2026-09-23, modelled on the v1 traces pipeline (`../sft_traces_v1/`) but
not a copy of it -- see "Differences from v1".

Row format: `{"messages": [...], "tools": [...]}` -- `tools` only when the
source carries tool definitions, OpenAI style
(`{"type": "function", "function": {"name", "description", "parameters"}}`).
Messages keep v1 conventions: `tool_calls[].function.arguments` is a DICT
(the GLM-5.2 template calls `.items()` on it), `reasoning_content` as its
own field, `tool_call_id` / `name` on tool replies.

Final result: 69 files, **36,327,623 rows**, 239GB (du) / 256GB (decimal).
31,064,744 of those are cosmopedia (single-turn synthetic textbooks,
user-approved); ~5.26M are everything else, ~1.7M of it real agent
trajectories. 844,975 rows carry a `tools` list; 2,406,444 tool calls.
Full numbers: `STATS.md`.

## Layout

```
hf_agentic_pipeline/
  CLAUDE.md, STATS.md
  scripts/
    run_pipeline.py   orchestrator (--list / --run all / --run N)
    common.py         shared message/tool normalisation (all the fixes)
    convert.py        Stage 1: per-dataset converters + SKIP list w/ reasons
    audit.py          Stage 3: read-only structural audit (also takes a
                      folder arg, used to audit traces_v1_final)
    cleanup.py        Stage 4: the standing role rules (below)
    validate.py       Stage 5: GLM-5.2 render + "nothing silently lost"
    report.py         Stage 6: STATS.md
    dedup.py          written but NOT run -- user said no dedup (2026-09-23)
    scan_v1_null_args_and_empty_assistant.py  one-off read-only v1 scan
  reports/  stage JSON reports;  logs/  run logs
```

Run: `cd scripts && python3 run_pipeline.py --run all`. For a subset,
`python3 convert.py NAME...` merges into the existing stage-1 report.
Launch long runs with `nohup` -- a session restart killed an un-nohup'd
chain once (cleanup had finished; audit had to be re-run).

## Standing user rules (apply to every SFT dataset)

1. Every row ends on an assistant message (trim after the last assistant;
   drop rows with none).
2. No assistant message directly after the system prompt -- drop the row.
   This removed all 1,200 `fuvty__tau-bench-synthetic` rows (their
   "Hi! How can I help you today?" opener).
Final audit: 0 violations of either.

## Decisions (user)

- Benchmarks/eval sets are excluded (MATH-500, MMLU, mt_bench, MBPP,
  AgentIF, AgentWorldBench, MCP-Atlas, CRMArena, ...). MATH-500 problems
  are also filtered OUT of hendrycks_math (500 rows), which contains them.
- Cosmopedia is included (single-turn is fine).
- WildChat: no toxicity filter needed -- both WildChat files already have
  0 `toxic=True` rows and 0 OpenAI-moderation flags in the first 40K rows.
- Tool definitions are kept (`tools` key).
- No dedup.

## Per-dataset findings (each converter written after reading real rows)

- **Parquet struct-union nulls** (ToolMind, SWE-Gym, SWE-QA-Pro tools):
  a STRUCT column materialises every key seen anywhere in the column as
  null in every row (a 3-arg call arrives with 20 args, 17 null). Stripped
  only for struct-sourced values; nulls parsed from JSON strings are real
  data and kept (the 8,695 `null_in_args` the final audit reports).
- **Legacy `function_call` / role `function`** and **role `tool_call`**
  (content = `{"name","arguments"}`) are converted to `tool_calls`.
  Unconverted, the GLM template has no branch for them and drops the call
  silently -- SWE-QA-Pro lost 22,963 calls this way until the first audit
  caught it. Role `developer` -> `system`.
- **Toucan-1.5M**: kept `overall_score >= 4.0` (750,223 of 1,646,546;
  278K rows have no score and are dropped). Its system prompt is Qwen's
  tool template -- removed, tools carried in `tools` instead (else tools
  render twice in two call syntaxes). Some arguments contain Python `...`;
  `loads_loose` rejects anything not JSON-serialisable.
- **tau-bench-synthetic**: 4,270 rows are per-round prefixes; kept only
  `reward == 1` (1,200) -- all then dropped by rule 2.
- **SWE-Gym OpenHands**: kept `resolved == True` only (491 of 6,055).
- **claude-code-traces**: per-API-request logs = growing prefixes of one
  session; longest row per session kept (901 of 32,133). Anthropic
  `tool_use`/`tool_result` blocks converted. `assistant_response` column
  unused (text only, its tool call is lost).
- **Exgentic**: OpenTelemetry spans; the span with the longest input
  history + its output = the full trajectory. MCP `[{"type":"text"}]`
  result wrappers unwrapped.
- **glaive**: tools parsed from the system text; `<functioncall>` with
  single-quoted JSON arguments parsed.
- **Excluded on quality**: `artillerywu__DeepResearch-9K` (every tool
  observation replaced by a copy of the question, 12,596/12,974 rows);
  `Srijan-Chakraborty__GLM-5.2-Agent-Distilled` (tool results with no
  preceding tool call, 315/318); HALO, gpt-5.5-agent, DEBATE (too small /
  not user-assistant). Duplicated sources: H4 and argilla UltraFeedback
  (allenai cleaned version kept).
- Full skip list with reasons: `SKIP` in `convert.py` (40 datasets).

## Differences from v1

No separate fix stage (fixes live in the converter); audit checks more
(unknown roles, empty assistant, orphan tool replies, non-dict args);
validate renders WITH `tools` and checks every message's content, every
tool call and every tool name actually appears in the rendered text --
v1's "renders without an exception" check passed on silently-dropped data.
`tool_without_call` (3,972 rows: SWE-Next, glaive, Jarrodbarnes) is left
as-is: those datasets write the call inside assistant TEXT, and like v1
we don't parse text-embedded calls.

## Upstream gap

The source download failed for 117 datasets (see the source folder's
`pipeline.log` / `skipped_datasets.txt`), e.g. tau2-bench-trajectories,
AgentInstruct, CC-Bench. They are simply absent here.
