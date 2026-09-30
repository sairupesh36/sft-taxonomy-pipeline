# SFT Taxonomy Project

## Purpose
Build a MECE, finegrained, closed-enum taxonomy to classify SFT + agentic
training data (eventually ~1.2B documents) for downstream ablation studies.
The tree covers domain/subdomain, task_family/task_subfamily, plus 6 smaller
mechanism axes (interaction_mode, tool_requirement, tool_category,
constraints, complexity, task_composition) and input/output (type, format,
language). It must be derived from the actual discovered data, not
hand-authored from general knowledge, and every addition must be MECE-checked
against siblings -- and across axes -- with zero redundancy before being
added. `agentic_behaviour` was a 7th axis that existed early on; it was
deliberately removed and folded into `interaction_mode`'s `agentic_loop`
value -- do not re-add it, that was a settled decision, not an oversight.

## Source of truth
`taxonomy_tree_final.xlsx` -- counts drift as gaps get fixed, so check the
live file rather than trust a number here; as of the last major pass: 49
domains / 328 domain_subdomain rows, 63 task_families / 330 task_subfamily
rows, plus the 6 mechanism-axis sheets and the `inputoutput` sheet (183 ISO
639-1 language codes including "NA" for Nauru, ~100+ format values). Edited
in place directly (via pandas/openpyxl) as gaps are found and fixed --
**every `pd.read_excel` call touching this file must pass
`keep_default_na=False, na_values=[]`**, or the literal string "NA" (Nauru's
real language code) gets silently read as a missing value and the row is
dropped on any round-trip write. This has bitten this project multiple times;
treat it as non-negotiable, not a case-by-case judgment call.

## Pipeline files
- `taxonomy_extract_unique.py` -- dedupes all raw discovery-output CSVs into
  unique value+count tables per axis/level (`unique_values/*.json`). No LLM.
- `taxonomy_validate_final.py` -- classifies those unique raw strings against
  the fixed tree categories (multi-label, checkpointed per parent so a
  kill/restart loses nothing). This is the gap-finding tool: run it, look at
  the "other" volume per category, sample the raw strings landing in "other"
  to decide if the tree is missing something real. Round-robins across
  `gemma-4-31b`, `openai/gpt-oss-120b`, `qwen3-8-27b` -- this script only,
  not the per-document mapper below.
- `taxonomy_map_document.py` -- the real per-document classifier, **gemma-4-31b
  only** (`MODEL = "gemma-4-31b"`, temperature 0.0). Reads actual document
  text (not pre-extracted labels) and tags it against the full tree in 2 LLM
  calls (Call A coarse: domain/task_family/interaction_mode/tool_requirement/
  complexity/task_composition; Call B fine: subdomain/subfamily/tool_category/
  constraints/input/output, scoped to Call A's picks). `validate_result()`
  strips anything not in the real closed sets (no hallucinated categories
  survive) and enforces the 1-2 label cap; `deterministic_fixes()` runs after
  it and corrects two objectively-checkable things no LLM judgment call
  should be trusted with: a `tool_category` present while
  `tool_requirement="none"` (self-contradictory), and `interaction_mode=
  "single_turn"` when the text plainly contains 2+ `[USER]:` turns.
  `sample_docs_stratified()` draws a proper reservoir-sampled, per-source
  sample across all 19 tulu3 + 17 agentic sources -- always use it, never
  take the first N rows of the first file(s) (that draws 100% from one
  source, and even a positional slice of a per-source draw systematically
  drops whichever sources sort last -- see `_interleave_trim()`).
- `taxonomy_judge.py` -- automated quality check, no manual labeling needed.
  gemma maps a document, then a genuinely different model (qwen3-8-27b)
  independently audits each assigned field against the category's own
  description and flags disagreements.

## External audit reports -- how to read them
Two external model audits of this pipeline exist in this directory
(`taxonomy_mapping_evaluation_report.md` from Gemini, `gpt-report.txt` from
GPT), both run against `1000_samples_FULL_conversations.txt`, a **frozen
snapshot** generated partway through this project. Both reports flag the
`domain: []` empty-domain pipeline bug (Call B scoped from Call A's raw
unvalidated domain) as their #1 critical finding -- **that exact bug was
already found and fixed independently, earlier than either report's
snapshot** (see `deterministic_fixes`/the validate-before-scope-Call-B logic
above); re-verified stable (3/3) against the report's own cited MySQL-agent
example on the date this note was written. Lesson: an external audit run
against a saved output file is auditing a point-in-time snapshot, not the
live script -- before acting on any finding from one, re-run the actual
current `taxonomy_map_document.py` against the report's own cited example
text and see whether the fault still reproduces. Don't assume a report's
severity stats (error rates, "N samples affected") still describe the
current pipeline; they describe the snapshot it read.

Real, still-live findings from those two reports that led to fixes (see
`taxonomy_map_document.py` git history / `deterministic_fixes()` /
`SYSTEM_A_RULES` items 11-13): tool_requirement/tool_category
self-contradiction (fixed deterministically, not just via prompting -- an
LLM call can't be trusted to keep two of its own fields consistent), the
turn-count-vs-interaction_mode mismatch (fixed deterministically, since turn
count is objectively countable from the text, not a judgment call), refusals
and stylistic quirks erasing the request's real topic (fixed via a targeted
SYSTEM_A_RULES addition: classify what the user asked, not what the
assistant happened to produce), incidental topic-flavored word problems in
code/math tasks spuriously adding a second domain (fixed via a targeted
rule), and multi-hop reasoning chains being mistagged `cross_domain` when
they're really a multi-hop query within one topic (fixed via a targeted
rule; multi-hop-ness belongs on `task_subfamily`, not the domain axis).

Recommendations from those reports that were deliberately evaluated and
**declined**, so a future session doesn't need to re-litigate them from
scratch: splitting `task_subfamily` into multiple orthogonal facet axes
(QA-grounding/reasoning-depth/answer-type etc.), adding a `response_behavior`
axis (refusal/partial_refusal/tool_failure/...), splitting domain into
explicit primary/secondary fields, adding a `tool_usage` (attempted/
successful/failed) axis separate from `tool_requirement`, and adding
per-field confidence scores. These are legitimate architectural ideas, not
wrong ones, but they're schema redesigns the user hasn't asked for, they'd
invalidate a large amount of already-hand-verified mapping behavior, and the
core complaints behind most of them (refusals erasing topic, style
overriding task type) were addressed more surgically via targeted prompt
rules instead of a new axis. Also declined: pruning the ~23 task families
that had 0 samples in the 1000-sample snapshot -- that snapshot draws from a
specific, narrow slice of sources, and 1000 samples isn't remotely enough
evidence that a category is truly dead across the eventual 1.2B-document,
much-more-diverse corpus; revisit only with real volume behind the question.
Also treat the reports' "wildly non-deterministic across repeated identical
inputs" claim (e.g. the same question mapped 20+ different ways) with real
skepticism before acting on it: `MODEL` calls run at `temperature=0.0`, and
this project's own extensive repeat-run verification (dozens of 3-5x
reproducibility checks across many rounds) never found that kind of
instability except on genuinely ambiguous single cases -- the reports' "same
example" cases likely conflate documents that share a common template
opening (e.g. the same ToolBench-style tool-list preamble) with documents
that are actually byte-identical but have different real underlying tasks
further down in the text, a distinction this project had to learn the hard
way earlier (see `taxonomy_map_document.py`'s idx7-vs-idx9 template
investigation in session history).

### Round 2 (post-fix re-audit, `taxonomy_v2_evaluation_report.md` / `gpt-report-v2.txt`)
A second pair of external audits, run against a genuinely fresh, unseeded
1000-sample batch (`1000_samples_v2_FULL_conversations.txt`) generated after
the round-1 fixes above, confirmed all of round 1's deterministic fixes at
0 occurrences across the full batch (empty domain, tool contradiction,
turn-count mismatch, browser/frontend_development mistagging). Measured
directly against the batch (not just the reports' claims) and fixed:
- **Task-vs-execution-method conflation**: a math/logic problem the assistant
  happened to solve by writing a Python snippet was getting a spurious second
  domain/task_family (`software_engineering`/`code_generation`) and matching
  subdomain/subfamily, even though the user never asked for code and the same
  question solved by hand would be identical. Fixed via a sharpened,
  falsifiable test added to both `SYSTEM_A_RULES` rule 6 and
  `build_system_b`'s rule 2: "would this document still deserve this second
  label if solved a completely different way?" -- if the label only applies
  because of the tool/method used, it doesn't belong on domain/task_family/
  subdomain/subfamily; `tool_category`/`tool_requirement` already exist to
  capture HOW something was solved without contaminating WHAT was asked.
  Verified: the exact combinatorics example both reports flagged now stays
  pure `mathematics`/`calculation` with `tool_category=code_execution`
  correctly preserved, 4/4 stable; a genuine dual-domain case (building an
  actual ML model) was checked for regression and still correctly gets both
  domains, since "build a model" really is an AI/ML task regardless of how
  it's solved.
- **Refusals missing the `compliance` constraint**: real and measured --
  only 13% (5/39) of refusal-like responses in the v2 batch carried
  `compliance`, down from the broad, consistent application found earlier in
  the project (before the refusal-topic-preservation rule existed). Fixed
  deterministically in `deterministic_fixes()`: a refusal opener (see
  `REFUSAL_MARKERS`) in the response's first 300 chars adds `compliance` to
  `constraints` if not already present, independent of whatever domain/task
  the request maps to. Verified 4/4 stable on a reconstructed version of the
  reports' own "dangerous activity" refusal example.
- **`input.type="CODE"` with format stripped to `None`**: small-scale (1
  case in the v2 batch, not the ~15 either report estimated) but a real,
  cheap, precise fix -- `deterministic_fixes()` now recovers this specific
  combination back to `TEXT`/`FREE_TEXT` rather than leaving a permanently
  null format.
- **Primary-domain-first ordering convention**: added as `SYSTEM_A_RULES`
  rule 15 -- when domain/task_family has two values, the substantive
  objective goes first, the contextual/application setting second. A
  lightweight way to address both reports' "primary vs secondary domain"
  recommendation using the list order that already exists, instead of adding
  new schema fields.

Declined again in this round, same reasoning as round 1 plus: a `risk_type`
field alongside `response_behavior` (round 2 GPT report) -- still a new-axis
ask, and the concrete problem behind it (refusals losing their compliance
signal) got fixed via the existing `constraints` axis instead of a new one.
Also declined: further pruning/consolidating unused task families (round 2
Gemini report repeats this at "25 dead of 63") -- same standing reasoning,
insufficient evidence from a 1000-sample slice of a much larger eventual
corpus.

### Round 3 -- careful, additive-only pass over declined ideas
User explicitly asked to reconsider declined suggestions specifically for
anything that could be added WITHOUT touching or risking the already-tested
domain/task_family/subdomain/subfamily logic. Re-measured against real batch
data (not just re-reading the reports) rather than guessing, per this
project's standing data-derived-not-hand-authored rule:
- Confirmed `compliance` and "the response was a refusal" are genuinely
  different things in this project's real data (10/15 `compliance`-tagged
  docs in the v2 batch were NOT detected refusals -- `compliance` also fires
  on fully-answered content bound by a policy/regulation). So a `refusal`
  signal is not redundant with `compliance` after all.
- Confirmed 8% of tool-using documents in the v2 batch show real tool-error/
  timeout language nowhere captured in the schema.
Added two new CLOSED-SET VALUES to the existing `constraints_values` sheet
(not a new axis, not a schema change, nothing else touched):
`declined_request` (the response declines/refuses some or all of the ask --
narrower than `compliance`, which can apply without any refusal) and
`tool_failure_handling` (the response had to report/work around an errored
or timed-out tool call). Both wired into `deterministic_fixes()` using the
same reliable pattern as the round-1/2 fixes: `declined_request` piggybacks
on the existing `REFUSAL_MARKERS` check (final assistant turn only, same as
`compliance`); `tool_failure_handling` uses its own `TOOL_FAILURE_MARKERS`
list, checked against the WHOLE visible text (not just the final turn) and
gated on `tool_category` already being non-empty -- unlike a refusal, a tool
failure is a property of the trajectory (can happen mid-conversation and get
recovered from before the final answer), so scoping it to only the last
turn undercaught real cases 6/15 on first pass; widening to the whole text
fixed it to 17/17 on the same real documents, with 0/15 false positives on a
separate sample of documents with no tool failure. This is the template for
any future "is this suggestion safe to add" question: check whether it's
genuinely additive (new value on an existing axis) vs. structural (new axis,
new field, changed decision logic for something already verified), require
real batch-data evidence before adding anything, and verify precision
(catch rate AND false-positive rate) against real triggering documents, not
just a hand-picked example, before considering it done.

### Round 4 (`taxonomy_v3_evaluation_report.md` / `gpt-report-v3.txt`, audits of
`1000_samples_v3_FULL_conversations.txt`) -- the reports themselves had a real
error rate this round, more than prior rounds. Every specific claim was
checked against the real saved batch (`/tmp/batch_1000_results_v3.json`)
before acting, which is what caught this:
- **Gemini's "8 missed refusal flags" claim was wrong**: checked idx 264 and
  494 (its own cited examples) directly against the batch -- both already
  correctly carried `declined_request` + `compliance`. The two "CAPTCHA
  refusal" examples (idx 545, 970) also checked out as NOT refusals at all --
  they're browser agents narrating a webpage obstacle (a CAPTCHA/security
  challenge blocking navigation), not the assistant declining to help the
  user. Took no action on this claim.
- **2 of Gemini's 3 "domain hallucination" examples were the report's own
  analysis error, not mapper bugs**: idx 506 (cited as "crypto exchange
  wrongly mapped to hospitality_and_food_services") is actually a food-tour/
  restaurant-recommendation question in Uruguay -- `hospitality_and_food_
  services` is exactly correct; the "changenow_crypto_exchange" API the
  report fixated on is just one of many unused tools listed in a ToolBench
  system prompt, never actually invoked. idx 591 (cited as "football
  highlight API wrongly mapped to business_and_management") is actually a
  question about logo-design tools for a new business -- `business_and_
  management` is correct; "football_highlight" is again an irrelevant unused
  tool name. Lesson reinforced: when a ToolBench-style prompt lists many
  tools, judge domain from the actual `[USER]` question and which tool was
  actually invoked, never from the first/most attention-grabbing tool name in
  the system prompt -- this is exactly the same class of error this project
  had to learn from its own mistakes earlier (see the idx7-vs-idx9 note
  above), now recurring in an *auditor's* analysis instead of the mapper's.
  Given 2/3 of the report's cited examples were false positives, did NOT
  implement its suggested keyword-based domain override (crypto->finance,
  bahn->transportation, football->sports) -- that would have broken these two
  correctly-classified real documents to fix a problem that mostly doesn't
  exist.
- **Real, verified, fixed**: `deterministic_fixes()`'s None-format recovery
  (originally only checked `input.type=="CODE"`) was generalized to check
  both `input` and `output`, for any type, whenever `format` ended up `None`
  -- found real cases of `output.type=="CODE"` and `input.type==
  "REFERENCE_DATA"` hitting the identical dead end (8 real documents,
  verified against the exact idx list the report cited, which this time was
  accurate). Verified 8/8 fixed, 4/4 regression-checked on valid CODE/PY and
  REFERENCE_DATA/JSON combinations (untouched).
- **Real, verified, fixed**: a genuine MECE gap -- GitHub-style project
  issues/feature-requests/enhancement-requests (recognizable by a
  `[PROBLEM_STATEMENT]:` prefix or template headers like "What is the
  expected enhancement?") were landing in `software_engineering` /
  `domain_subdomain=other` because none of the 14 existing subdomains
  (backend/frontend/mobile/devops/etc.) cover the process of managing a
  project's issue tracker, as opposed to a specific technical layer of the
  code itself. Confirmed real and repeating (5/10 of all software_engineering
  `other` docs matched this exact pattern, all genuine enhancement/doc
  requests, no actual bugs among them). Added `software_project_management`
  as a new domain_subdomain (MECE-checked against all 14 siblings). Merely
  adding the tree value wasn't enough -- the model didn't spontaneously pick
  a new 16th option in a long list; needed a concrete worked-example rule
  (same pattern as the earlier `model_training_and_finetuning` fix) before it
  fired reliably. Verified 5/5 on the real triggering documents, no
  regression on a normal generic-programming control case.
- **Declined**: GPT's `reasoning_scratchpad`-removal request directly
  contradicts an earlier, deliberately evidenced decision in this project
  (added to `tool_category` after finding it was needed -- see the
  reasoning_scratchpad fix earlier in this file/session history); this
  report re-raises the same objection without new evidence, so the prior
  decision stands. Declined again, same reasoning as rounds 1-3: `response_
  behavior` axis, `primary_domain`/`secondary_domains` as separate fields,
  a `tools_used` axis, a `mapping_status` (taxonomy_gap/ambiguous) axis,
  modality-aware generation subtypes, empty-input handling as its own field.
  Checked two single-instance claims for repeat-pattern evidence before
  deciding: the "logical/constraint reasoning wrongly mapped to mathematics"
  claim (physician-conference LSAT-style logic puzzle, idx 864) is a
  genuinely debatable single instance (constraint-satisfaction puzzles are
  arguably discrete-math-adjacent, not obviously wrong) -- not acted on
  without a second confirmed instance. The "Armstrong number label inflation"
  example (idx 888) turned out to be a document with a completely EMPTY
  `[ASSISTANT]` response -- not a labeling-discipline bug, just malformed
  input data the model had to guess at; not representative of a real pattern.

### Round 5 -- exhaustive verification pass + the first genuine new axis added
User asked for an actual exhaustive read-through (not a sample-and-extrapolate
check) of 100 real documents, plus a systematic redundancy sweep across the
whole tree. Both done as real, separate pieces of work rather than assumed:
- **100-document exhaustive review** (fork, list-positions 0-99 of the v3
  batch, genuinely balanced 52 tulu3 / 48 agentic thanks to
  `asyncio.as_completed()` not preserving submission order -- worth knowing if
  auditing "the first N" of a batch file again, since it is NOT "the first N
  by source order"): 93/100 correct or defensible, 0/100 confirmed flat-out
  wrong. The only real finding was **consistency, not correctness** -- the
  same underlying task type (AI self-identity questions; LSAT-style
  constraint-logic puzzles) doesn't reliably get the same domain/task
  treatment across near-identical instances. Not acted on (each individual
  instance was independently defensible, and the sample size per pattern was
  small) but worth knowing this class of gap exists.
- **Systematic redundancy sweep**: compared every value's description against
  every other value's description (within-axis and cross-axis) using text
  similarity as a first-pass filter, then manually reviewed everything it
  flagged. Result: no new confirmed duplicates. The vast majority of ~400
  flagged candidate pairs were false positives from shared template wording
  between legitimately distinct siblings (e.g. "backend_development" vs
  "mobile_development" score high only because both are worded "Development
  of X"). One genuinely close pair was found and deliberately NOT touched:
  `transformation/schema_mapping` ("map fields between schemas" -- an execute
  action) vs `matching_and_resolution/schema_matching` ("determine
  correspondences" -- an identify action) mirror the real do-vs-identify
  split each of their own sibling families already uses; close enough to
  watch, not confirmed redundant, and neither has any real usage yet.
- **Real fix, found via user-directed investigation, not a report**: the
  `generation` task_family's `text_generation` subfamily is its own generic
  member -- `creative_generation`, `structured_content_generation`, and
  `template_based_generation` are all just more specific kinds of the same
  thing. Confirmed real and repeating (3 of 4 non-broken real examples showed
  `text_generation` stacked with a more-specific sibling for what was clearly
  ONE integrated writing request, not two). Fixed via a targeted addition to
  `build_system_b`'s rule 2. Verified 3/3 stable on the real triggering
  documents, including confirming the rule correctly does NOT strip the
  second label on a genuine composed case (one document asking for both a
  structured deliverable AND a separate prose paragraph -- two real asks, not
  one relabeled twice).
- **The first genuine new axis added this project: `response_behavior`**
  (sheet: `response_behavior_values`; values: `normal`, `policy_refusal`,
  `capability_limitation`, `clarification_required`). This was the most
  repeated architectural request across all four external audit rounds, and
  this time it was built -- but only after two real course-corrections driven
  directly by the user's own questions, not by re-reading the reports harder:
  1. The user asked "constraints are USER-defined constraints, right? how do
     these fit there?" about `declined_request`/`tool_failure_handling`
     already living in `constraints_values`. This was a genuinely correct
     catch: `constraints` are things the user's request required of the
     response (every original entry reads "the response must..."); a refusal
     or a tool failure is not something the user asked for, it's a fact about
     what happened. Confirmed this matters beyond naming purity: cramming
     both concepts into one field means "give me every document with a
     user-required constraint" can no longer be answered cleanly without
     separately filtering out the outcome-descriptor values. `response_
     behavior` was deliberately NOT added as more `constraints` values for
     this reason -- it's its own small standalone field instead.
  2. Before building it, manually read and classified all 68 real
     `declined_request` documents in the v3 batch by hand (not guessed):
     ~51 were genuine policy/safety refusals, ~12 were AI capability limits
     ("I can't browse the internet", "I have a training cutoff"), and ~5 were
     the request itself being unclear/ambiguous -- a real, distinct category
     the original design (copied from the external reports) hadn't
     separated out. Dropped a 5th proposed category, "unavailable_
     information", after the real data showed it wasn't cleanly separable
     from capability_limitation in practice (nearly every real example of
     "missing one specific fact" was actually "no real-time/external data
     access" -- the same root cause) -- don't invent a category the data
     doesn't actually support distinguishing.
  3. First implementation bug, caught by testing against all 68 real
     documents before declaring it done (not just a few hand-picked ones):
     classifying `response_behavior` from a fresh, independent `REFUSAL_
     MARKERS` check missed 38/68 real cases. Root cause: `declined_request`
     is now a real closed-set value the MODEL can see in its own prompt and
     sometimes correctly self-selects even on phrasings `REFUSAL_MARKERS`
     doesn't cover (e.g. "I am not capable of experiencing sensory input" has
     no "but I cannot"/"I can't"). Fixed by keying `response_behavior`'s
     trigger off `"declined_request" in constraints` (the combined signal,
     already covering both the deterministic check and the model's own
     picks) instead of re-deriving independently. After the fix: 68/68 real
     documents correctly classified into one of the three non-normal
     categories (0 fell through to a wrong "normal" default), 0/20 false
     positives on a separate sample of genuinely normal documents, and a
     live end-to-end run on fresh data confirmed it works in the real
     pipeline, not just in isolated function tests.

## Infra
All LLM calls go through the SGLang router at
`http://sglang-router.sglang.svc.cluster.local:30000/v1/chat/completions`
(API key hardcoded in each script). No local GPU access in this environment --
everything must go through this API. Models available: `gemma-4-31b`,
`openai/gpt-oss-120b`, `qwen3-8-27b`, and (added later, see "Classifier
distillation" section) `openai/gpt-oss-20b`. Check health before assuming a
model is up (`GET /v1/models` lists what's currently registered, but a model
can be listed there and still return `no_available_workers` if its pods
haven't finished starting) -- scripts should round-robin across whichever are
actually responding. **gpt-oss models do hidden chain-of-thought reasoning
before writing their answer; gemma does not** (confirmed directly: gemma's
`reasoning_tokens` is always 0, gpt-oss's is not). This matters because
`reasoning_tokens` count against the same `max_tokens` budget as the visible
answer -- a `max_tokens` value tuned for gemma (this project's
`MAX_TOKENS_A`/`MAX_TOKENS_B` = 400/500) can let a gpt-oss model burn the
whole budget thinking and return empty `content`. When calling a gpt-oss
model: raise `max_tokens` well above what gemma needs, and pass
`"reasoning_effort": "low"` in the request body (confirmed via direct API
test that SGLang honors this for gpt-oss and it roughly halves
`reasoning_tokens` while still answering correctly) -- this is also a real,
free throughput lever for gpt-oss specifically, not just a correctness fix.

## Data sources
19 tulu3 sub-datasets + 17 active agentic sub-datasets (1 of 18,
`glaive_function_calling`, downloaded 0 rows and is correctly excluded)
already discovery-extracted into free-text domain/subdomain/sub_subdomain and
task/subtask/sub_subtask labels: `extracted_domains_shard_*.csv`,
`extracted_tasks_shard_*.csv` (tulu3) and `agentic_results/agentic_{domains,
tasks}_shard_*.csv` (agentic). 215,448 domain-labeled + 315,741 task-labeled
rows as of the last full extraction.

## Gotchas
- `pip install` needs `--break-system-packages` in this environment.
- Never pipe a long-running background job through `| tail -N`: tail buffers
  everything until the piped command exits, so you get zero progress
  visibility and a killed job looks like it produced no output at all.
  Redirect straight to a file (or let the run_in_background mechanism capture
  stdout directly) instead.
- Before relaunching a killed/restarted job, verify with `ps aux` that the old
  process is actually gone -- a failed `pkill` match can leave a duplicate
  process running concurrently against the same checkpoint file.
- Multi-hour jobs started as Claude background tasks DIE when the Claude session ends (2026-09-25: the
  o200k stats run stopped at 5,000/31,486 files). Start them with `setsid nohup ./script.sh > /dev/null 2>&1 &`
  and have the script write its own progress/done files.
- Long-running consolidation jobs (`taxonomy_validate_final.py`) checkpoint
  per parent-bucket into `taxonomy_build_final/*.json` and resume automatically
  -- safe to kill and restart (e.g. to bump concurrency) without losing progress.

## Round 6 -- working through the external reports' suggestions one by one with
the user (not all at once), checking real data before each decision
Going through GPT's v3 priority list item by item, in order, deciding each on
its own merits rather than a blanket accept/decline:
- **#1 `response_behavior`**: built (see Round 5) -- the strongest, most
  repeated suggestion across all 4 audit rounds.
- **#2 remove `reasoning_scratchpad` from `tool_category`**: declined again.
  `tool_category`'s real scope (per what's already in it, e.g.
  `code_execution` isn't strictly "external" either) is "what structured
  capability/resource did this response draw on," not literally "external
  API only" -- reasoning_scratchpad fits that broader, already-established
  reading. Different in kind from the `constraints`/`response_behavior`
  mistake (that was a genuine category error); this is a defensible broader
  interpretation of an existing field.
- **#3 primary/secondary domain as explicit separate fields**: declined.
  Already solved cheaply via the list-ordering convention (Round 4, rule 15).
  A dedicated `primary_domain` field would just always equal `domain[0]` --
  restating existing information under a new name, not adding any.
- **#4 separate `tools_used` from `tool_requirement`**: declined. Checked the
  exact gap case (tool_requirement=required + tool_category=[]，18 real
  docs) -- 10/18 are already explained by `response_behavior`
  (capability_limitation/policy_refusal). The remaining 8 are a narrower,
  different, lower-priority gap (mostly ALFWorld household-agent docs where
  none of the existing tool_category values describe "interact with a
  simulated environment") -- a possible future tool_category *value*, not
  evidence for a whole new axis.
- **#5 kill multi-label inflation**: already mostly handled by the existing
  "would this deserve the label if solved differently" rule plus the
  text_generation fix below. Checked real numbers (7.4% docs have 2
  task_families, 17.4% have 2 subfamilies) and spot-checked a sample --
  mostly genuine composed tasks. Found one more instance of the same
  generic-sibling-stacking pattern one level up (`generation` alongside
  `summarization`, task_family level not subfamily level) but only 1
  real instance so far -- watching, not fixing yet (same 3-instance bar
  as everything else).
- **Real fix found via this process, not from a report**: `text_generation`
  (the generic member of the `generation` task_family) was getting stacked
  with its own more-specific siblings (`creative_generation`,
  `structured_content_generation`) for a single integrated writing request --
  confirmed on 3 real non-broken examples, fixed via a targeted addition to
  `build_system_b` rule 2, verified 3/3 stable, and confirmed the fix
  correctly does NOT strip the second label when a document has a genuinely
  separate second deliverable (one real example asked for both a structured
  matrix AND a separate prose paragraph -- kept both, correctly).
- **#6 formalize `other` as taxonomy_gap vs ambiguous (`mapping_status`
  field)**: declined. Checked a fresh, previously-unexamined sample of
  `other`-tagged documents -- every single one this whole project, across
  every round, turned out to be either a genuine gap or (this round) a case
  that looked wrong at a glance but was defensible once the full text was
  read (a roleplay-framed jailbreak-style prompt that looked like a simple
  music question in preview). Never found a case of "other" picked despite
  an obviously correct sibling being available. The premise motivating this
  suggestion isn't supported by the actual data; also, correctly
  distinguishing "genuine gap" from "model was unsure" would require the
  model to reliably self-report its own uncertainty, the same unreliable
  self-assessment problem confidence scores have.
- **#7 real fix: "reasoning method ≠ domain," refined with a real
  distinction the general rule misses**: GPT's own framing ("a legal question
  requiring deduction remains law") is right for genuine domain questions
  where the subject matter is substantively necessary to solve it, but
  checked against 3 real LSAT-style analytical-reasoning puzzles found by the
  100-document exhaustive review (Round 5) -- a physician-conference
  committee-selection puzzle, a physics-superconductor assignment puzzle, and
  a speech-therapy teaching-sequence puzzle, all with the identical
  clue-based constraint structure -- and found the opposite pattern actually
  matters more here: these cover stories are decorative and swappable (the
  puzzle is mechanically identical no matter what the story is about), unlike
  a genuine domain question. 2/3 were already consistently tagged
  `mathematics`/`proof_and_formal_reasoning`/`logical_deduction`; the third
  incorrectly followed its cover story into
  `education_and_edtech`/`question_answering`. Added `SYSTEM_A_RULES` rule 16
  with an explicit test (could you swap the cover story for an unrelated one
  without changing the puzzle at all? if yes, it's a pure logic puzzle).
  Verified all 3 now consistently classified the same way (2/2 runs each),
  and regression-checked that a genuine domain question requiring deduction
  (a real California landlord/tenant legal-notice question) correctly still
  stays in its real domain, not pulled into mathematics.

## Round 7 -- user pushback on `requirements_analysis` "feeling generic" led to a
real cross-family redundancy fix, plus a full sweep confirming the rest is fine
User questioned whether `requirements_analysis` (a task_family) is too generic,
prompted by discussing item #8. Investigated honestly rather than just
reassuring:
- The category's generic wording IS grounded in real discovered raw labels
  (e.g. "Software Requirements Engineering", "Client Requirement Elicitation"),
  so it's not hand-invented. But some of that original grounding evidence in
  `unique_values/task_l1.json` is keyword-coincidence noise, not genuine
  concept matches -- e.g. "Hydration Requirement Calculation" and "Visa
  Requirement Information" got counted as supporting evidence purely because
  they contain the word "requirement," when they're really calculation/QA
  tasks. However, checked against real per-document usage
  (`/tmp/batch_1000_results_v3.json`): 6/6 real documents tagged
  `requirements_analysis` were genuine software requirement/feature asks --
  0 contamination in actual live classification. Conclusion: an
  evidence-quality wrinkle from how the category was originally justified on
  paper, not a live mapper bug. **Decided NOT to narrow the wording to
  software-only** -- no real evidence of the hypothetical failure mode ever
  happening, and `task_family` categories are deliberately built to work
  across domains (same as `calculation` staying domain-agnostic despite being
  mostly math in practice); narrowing on a hypothetical would risk excluding
  a real non-software case later without any real problem to justify it.
- User then asked to sweep the WHOLE tree for similar cases. Delegated to a
  fork (checked ~40 generically-named task_families against raw discovery
  evidence, plus did a full close-read of the two already-known
  correlated-but-distinct clusters flagged in the plan file:
  diagnosis/root_cause_analysis/forensics_and_investigation/incident_management,
  and comparison/matching_and_resolution/reconciliation). Results:
  - **Real, live, fixed bug**: `diagnosis`'s own subfamily
    `root_cause_diagnosis` ("Determine likely underlying causes of observed
    problems") directly contradicted `diagnosis`'s own family-level
    description ("without necessarily tracing a causal chain") and fully
    duplicated the separate `root_cause_analysis` family that already exists
    for this. Not theoretical: in the real batch, 2 of `diagnosis`'s 3 real
    documents (idx 549, 566) picked this contradictory subfamily while
    `root_cause_analysis` itself got 0 real usage -- the model was
    consistently reaching for the wrong-but-available option. Removed
    `root_cause_diagnosis` from `task_subfamily_values` (328 rows after, was
    329). Verified live on the 3 real triggering documents, 2/2 runs each:
    idx 532/549 now correctly land on `diagnosis`/`fault_diagnosis`; idx 566
    now lands on `code_debugging`/`root_cause_debugging`, an existing
    category that fits even better than `root_cause_analysis` would have.
  - **`forensics_and_investigation`/`incident_management`**: clean, no
    overlap found -- their disambiguation clauses genuinely work in practice.
  - **`comparison`/`matching_and_resolution`/`reconciliation`**: mostly
    clean; one minor near-duplicate found inside `reconciliation` itself
    (`data_reconciliation` vs `cross_source_reconciliation` read almost
    identically) but 0 real usage so far -- watching, not fixing, same
    3-instance bar as everything else in this project.
  - **7 more evidence-quality-only contamination cases** (same class as
    requirements_analysis, not live bugs): `matching_and_resolution`
    ("match" also matches sports match schedules), `validation_and_
    verification` ("verif" also matches CAPTCHA "human verification"
    browser text), `ranking` (matrix rank vs ordering items), `localization`
    (geographic/math root-localization vs language localization),
    `security_assessment` ("food security" vs actual security),
    `content_moderation` (the plain adjective "moderate" vs content
    moderation), `compliance_assessment` (generic instruction-following vs
    regulatory compliance). For the two of these with real usage so far
    (`matching_and_resolution`, `validation_and_verification`), checked
    every real document by hand -- 7/7 and 7/7 correctly classified, zero
    actual contamination in live use. The rest have no real usage yet to
    confirm either way. No action taken on any of these -- same reasoning as
    `requirements_analysis` itself: real per-document usage is what matters,
    and where we could check it, it was clean.

### Round 8 -- item #8 from GPT's v3 priority list, closed out
"Add missing software-engineering request types." Checked all 6 proposed
categories (`bug_report`, `feature_request`, `enhancement_request`,
`performance_optimization`, `refactoring`, `technical_design`) against the
tree and real data individually, rather than adding the whole list:
`bug_report` -> already covered by `code_debugging/bug_identification`;
`refactoring` -> already exists (`code_transformation/refactoring`);
`technical_design` -> already covered by `software_design/architecture_
design`; `performance_optimization` -> checked the real batch, 0 instances,
no evidence, not added. Only `feature_request`/`enhancement_request` was a
real, repeating gap: 6 real GitHub-issue-style documents (idx 583, 753, 600,
668, 515, 702) were forced into `requirements_analysis/functional_
requirements_gathering`, which is really about discovering a system's needs
from scratch, not responding to an already-scoped feature ask. Added
`enhancement_request` as a new task_subfamily under the existing
`requirements_analysis` family (329 task_subfamily rows now), MECE-checked
against its 4 siblings (functional_requirements_gathering/non_functional_
requirements_analysis/user_story_creation/requirements_prioritization) --
none overlap. Wired into `build_system_b`'s rule 1, alongside the existing
`software_project_management` worked example (same GitHub-issue trigger
pattern covers both subdomain and subfamily). Verified 6/6 stable on the
real triggering documents, and regression-checked with a constructed
genuine "gather requirements for a new system from scratch" case -- correctly
stayed `functional_requirements_gathering`, 3/3, confirming the new value
didn't over-fire and swallow its sibling's real territory.

**#9 (modality-aware generation -- splitting text/code_generation into
image/audio/video/structured_generation subtypes for future multimodal
data)**: explicitly left as a watch item, not decided against, not acted on.
Reason: there's currently zero evidence of image/audio/video SFT documents
anywhere in this project's real data (all 19 tulu3 + 17 agentic sources are
text/code), so there's nothing real to verify a fix against yet -- same
"don't act without real data" standard as everything else in this file.
Revisit if/when a multimodal data source is actually added to the corpus.

This closes out the full one-by-one walkthrough of GPT's v3 priority list
(items #1-#9).

### Round 9 -- same one-by-one walkthrough treatment applied to Gemini's v3
report (`taxonomy_v3_evaluation_report.md`), 5 items
1. **"8 missed refusal flags"**: already-verified-wrong claim (see the Round 4
   note above) -- reconfirmed no action needed, `response_behavior` already
   covers this need better than a raw keyword net would have.
2. **ToolBench domain-override suggestion (crypto->finance, bahn->
   transportation, football->sports)**: declined, with a 3rd confirmed
   report error added to the 2 already found in Round 4. idx 761 ("Deutsche
   Bahn train stations") was cited as a hallucination
   (hospitality_and_food_services instead of transportation), but the real
   user request is genuine European vacation road-trip planning (train
   stations AND frequent-flyer programs "for our future flights") -- a
   textbook `travel_and_tourism` case, same pattern as the idx 506/591
   errors already found. Now 3/3 of Gemini's cited "hallucination" examples
   are the report's own analysis error (judging domain from the tool name
   instead of the actual user request), not mapper bugs. Confirmed: do not
   add the keyword-override rules.
3. **"General domain dumping ground" (~25-30 of 209 `general`-tagged docs
   claimed to have a real vertical lazily ignored)**: checked the report's 4
   cited examples (already defensible, correct task-level tags) plus an
   additional random sample of 18 more `general`-domain documents never
   examined before. Found nothing egregious across all 22 -- the real
   content is AI self-identity questions, common-sense/NLI reasoning tasks,
   refused harmful requests, and genuine ToolBench multi-utility blends, all
   legitimately domain-agnostic. No action needed.
4. **Subdomain "other" dead-ends (cybersecurity/veterinary/mythology/
   professional_communication cited)**: of the 4, only `cybersecurity` had a
   real, repeating (3/3) pattern -- the other three were each a single real
   instance among several unrelated "other" docs in the same domain, below
   the project's standing 3-instance action bar. **Fixed**: added
   `offensive_security` as a new domain_subdomain under `cybersecurity`
   (330 domain_subdomain rows now) -- "Requests involving exploiting,
   attacking, or gaining unauthorized access to systems, whether framed as
   legitimate penetration testing/red-teaming or as a malicious request,
   regardless of whether the response complies or refuses." MECE-checked
   against its 7 siblings (all defensive-side: threat_intelligence/
   incident_response/vulnerability_management/network_security/
   application_security/identity_and_access_management/
   security_operations_soc) -- no overlap, offense vs. defense is a clean
   split. Unlike `software_project_management`/`enhancement_request`, this
   one fired correctly straight from the tree description alone, no
   worked-example prompt rule needed -- verified 2/2 stable on all 3 real
   triggering documents (idx 473, 52, 413). Regression-checked with a
   constructed genuine incident-response document (suspected compromised
   host investigation) -- correctly stayed `incident_response`, 3/3, not
   pulled into `offensive_security`.
5. **I/O format null-drop recommendation**: already fixed in Round 4 (the
   `deterministic_fixes()` None-format generalization to both input/output,
   any type) -- no further action.
### Round 10 -- speed optimization experiment, kept separate from the real pipeline
User asked to speed up `taxonomy_map_document.py` (a 1000-doc batch was taking
~23-26 minutes) WITHOUT touching the original or risking quality. Created
`taxonomy_map_document_fast.py` as an explicit copy for testing -- never edit
this in place expecting it to silently become the new production script; it's
a testbed, promote a change to the real file only after it's proven safe.
Benchmarked properly rather than guessing:
- A raw trivial LLM call (temp 0, 5 max_tokens) takes ~0.24s -- confirms the
  router/model itself is fast; the real per-doc cost is prompt-processing
  time from this project's necessarily-large system prompts (SYSTEM_A alone
  is ~24K chars; Call B's scoped prompt is ~21K chars), not network latency
  or client-side queueing.
- Client-side concurrency (semaphore size) has a real but diminishing-returns
  effect: measured 0.577/0.638/0.689/0.733 docs/sec at concurrency 8/16/32/64
  on the same 40 real docs. Past ~32 the router itself is clearly saturated
  (compute-bound), so raising concurrency further buys little. **Bumped the
  fast copy's default concurrency from 8 to 32** -- this is a pure
  client-side change that never touches prompt content, so it carries zero
  quality risk by construction; adopted with confidence.
- **Tested and explicitly REJECTED**: compressing `LANGUAGE_OPTS` (183
  languages, each currently a templated "Natural-language content in X."
  sentence -- 8276 chars total, ~39% of Call B's prompt, that adds zero
  disambiguating information beyond the language name itself) down to a
  compact "CODE=Name" form. This gave a real ~15% additional throughput
  gain and is 100% information-preserving on paper -- but a proper A/B test
  proved it isn't quality-neutral in practice. Methodology: ran the
  UNCHANGED original script against itself twice on the same 50 real docs
  (20 deliberately non-English-heavy + 30 random) to measure the pipeline's
  existing baseline flip-rate at temp=0.0 (LLM serving backends with dynamic
  batching aren't perfectly deterministic even at temp=0 -- this project
  already knew a small fraction of borderline docs are unstable, see Round
  5's consistency finding) -- baseline was 4/50 docs (8%) disagreeing on at
  least one field. Then ran original-vs-compressed-language-version on the
  identical 50 docs: 9/50 docs (18%) disagreed -- roughly DOUBLE the natural
  noise floor, even though every individual mismatch was an ambiguous
  toss-up case (not a flatly wrong answer) and several fields that
  disagreed had nothing to do with language at all. Conclusion: even a
  purely reformatted, informationally-identical prompt section measurably
  perturbs the model's behavior on unrelated fields beyond baseline noise --
  changing ANY part of a shared system prompt is not automatically
  quality-neutral just because the change looks lossless on paper. Reverted
  this change after presenting the tradeoff to the user; they chose safety
  over the extra ~15%. **Lesson for any future prompt-editing-for-speed
  work: always A/B against a same-script self-consistency baseline
  measured on the identical sample, not just "does it look informationally
  equivalent" -- a change can look lossless and still not be
  behavior-neutral.**
- Net result: `taxonomy_map_document_fast.py` differed from the real
  pipeline by exactly one line (semaphore 8->32, verified via `diff`). Real,
  measured, zero-risk speedup: roughly 15-20% faster at concurrency 32 vs.
  the original's default of 8 (~20 min instead of ~24-26 min for a
  1000-doc batch). **Promoted into the real `taxonomy_map_document.py`**
  after the user approved -- same one-line change (`asyncio.Semaphore(8)` ->
  `asyncio.Semaphore(32)` in `main()`), confirmed zero prompt impact.
- Pushed further to find the real ceiling: benchmarked concurrency 32/64/
  96/128 on 60 real docs -- throughput plateaus right at 32
  (0.85/0.80/0.82/0.83 docs/sec respectively, essentially flat past 32).
  Confirmed 32 is the actual sweet spot for this one process against this
  router, not an arbitrary stopping point -- going higher buys nothing.
- **On 1.2B-doc scaling** (user asked directly, this wasn't volunteered):
  did the honest math -- at ~0.7-0.85 docs/sec single-process throughput,
  1.2B documents would take on the order of decades. This single-process
  concurrency tuning was never going to be a real lever for production
  scale; the only real lever is horizontal scaling (multiple pods/workers
  each hitting the router independently, which the user raised much
  earlier in this project's history) -- an infrastructure sizing question,
  not something further script-level tuning can solve. Flagged as the next
  real conversation once/if the user wants to revisit scaling (still
  formally deprioritized per the standing "finalize quality before scaling"
  preference, but answered directly since they asked outright).
- Ran a 2-pod simulation (2 separate processes hitting the router at once,
  concurrency=32 each, same real docs) to check whether the backend has real
  spare GPU capacity or is already saturated. Combined throughput was ~0.95
  docs/sec vs. a single process's own ~0.83-0.85 docs/sec ceiling -- only
  ~11-14% better with double the pods, each process's own rate dropped by
  ~40% once sharing. Confirms the current `gemma-4-31b` backend (4 GPUs) is
  already close to saturated; more client pods alone won't give proportional
  gains without also adding more GPU replicas behind the router.
- **Diagnosed WHY each call is slow, with real router telemetry** (not
  guesswork): inspected the raw API response fields directly.
  `reasoning_content` is empty and `reasoning_tokens: 0` on every call --
  gemma-4-31b is doing zero hidden chain-of-thought here, ruling that out as
  a cause. Actual completion length is tiny (54-91 tokens) against a
  400-token cap -- nowhere near the cap, so lowering `MAX_TOKENS_A`/
  `MAX_TOKENS_B` would save nothing. The real number: `prompt_tokens` on a
  single Call A request is ~5900-5950 -- i.e. ~99% of every request's total
  tokens is the fixed category-list system prompt, ~1% is the actual
  document + answer. This is what actually explains the per-call latency,
  not reasoning or verbose output.
- **Tested and REJECTED: batching multiple documents into one LLM call** to
  amortize that fixed ~5900-token system prompt across several documents
  instead of paying it once per document. This looked like the obvious next
  lever from the diagnosis above, but the actual measurement said otherwise.
  Methodology: ran batch sizes 1/4/8/16 (Call A only, same 160 real docs,
  same concurrency=32, enough total calls at every batch size to properly
  fill that concurrency -- an earlier first pass with too few total calls at
  the largest batch size gave a misleadingly bad read for a different
  reason, caught and corrected before trusting the result). Real result:
  batch_size=1 (4.185 docs/sec) beat every batched size outright -- 4
  (3.634), 8 (3.656), 16 (2.755) -- larger batches were WORSE, monotonically.
  Root cause: token GENERATION is autoregressive (strictly sequential, one
  token at a time) and cannot be parallelized within a single call, so
  batching N docs into one call forces the model to write all N answers as
  one long serial stream (completion tokens scaled ~linearly with batch
  size: 91/322/566/1122). Meanwhile, sending N separate single-doc calls
  lets the server's continuous-batching scheduler interleave and parallelize
  all of their generation simultaneously -- confirmed this is actually
  happening effectively here. So batching traded a real prefill saving for a
  bigger loss of generation-level parallelism. **Do not re-attempt manual
  request-batching against this router/model setup without new evidence the
  server-side scheduling behavior has changed** -- the single-doc-per-call,
  high-concurrency approach already in production (concurrency=32) is the
  actually-correct shape for this backend, confirmed empirically, not just
  assumed.
- Also directly answered along the way: this API is stateless -- the full
  system prompt is resent and reprocessed on every single call, there is no
  cross-call "the model remembers the rules" mechanism via the standard chat
  completions endpoint used here. Checked whether SGLang's RadixAttention
  prefix caching was silently providing this for free: sent 5 back-to-back
  calls with the byte-identical system prompt at low concurrency and saw no
  speedup on the repeats (~4.1-4.2s each) -- caching either isn't enabled or
  isn't effective in this deployment, reinforcing that per-call prompt cost
  is real and unavoidable without a structural change.

### Round 11 -- user's own "8.1% other is too high" pushback, exhaustive 115-doc read
User rejected the v4 batch's "other" rate (8.1% domain_subdomain, 3.5%
task_subfamily) as too high and asked for an exhaustive read of every single
"other"-tagged document (not a sample) -- all 115 unique real documents
(81 domain_subdomain=other + 35 task_subfamily=other, 1 overlap). Delegated
to a fork with the standing 3-instance action bar; 75/115 were read directly
(every cluster with 3+ repeats -- the only ones that could clear the bar;
the remaining ~40 were scattered 1-2-instance singletons across ~25
different domains, structurally incapable of forming a pattern).

**5 real, evidenced (3+) gaps found and added, all MECE-checked against
siblings and verified live on their real triggering documents:**
- `customer_support` domain_subdomain -- new `policy_driven_service_agent`
  (15 real instances: an autonomous tool-calling agent executing a
  transactional request strictly per business policy -- booking,
  cancellation, billing change -- across many simulated verticals; none of
  the 6 existing subdomains describe an autonomous policy agent, closest
  was `live_chat_and_helpdesk` which is human-mediated). Verified 5/5 live.
- `search_and_retrieval` task_subfamily -- new `personal_information_lookup`
  (8 instances: requests to find someone's private address/phone/contact
  info, usually declined). Verified 5/5 live.
- `sports_and_recreation` domain_subdomain -- new `live_sports_data`
  (6 instances: live scores, in-play status, rankings, schedules -- none of
  the existing subdomains, sports_analytics/sports_history_and_trivia/
  sports_broadcasting, cover real-time data). User pushed back asking if
  this duplicates tool_category=api/tool_requirement -- verified it doesn't:
  checked the 6 real docs, 5/6 already correctly have tool_category=['api'],
  confirming the tool axis is already tracked separately; domain_subdomain
  answers a different question (WHAT the topic is) than tool_category
  (HOW it was fetched) -- same non-redundancy pattern as the tool_use vs
  tool_requirement/tool_category case checked earlier this session (idx
  952). Verified 6/6 live.
- `transformation` task_subfamily -- new `row_or_column_reordering`
  (3 instances: reorder/sort/swap rows or columns in a table, unchanged
  format/schema/values -- none of the 5 existing subfamilies, all
  conversion-oriented, cover pure reordering). Verified 3/3 live.
- `data_and_information_management` domain_subdomain -- new
  `schema_and_column_structure_matching` (3 instances: matching/mapping
  column headers or table schema structure between datasets). Verified
  2/3 live initially -- 1 real case (idx 552, "identify the correct header
  for each column from this table") fell through to domain="general"
  instead, because Call A was distracted by the table's incidental row
  content (sports/movie/people data) rather than recognizing the task is
  about the table's STRUCTURE. Fixed via new `SYSTEM_A_RULES` rule 17
  (structure vs. content-topic distinction, same "cover story vs.
  substance" family of fix as rule 16's logic puzzles). Verified 3/3 stable
  after the fix, and confirmed via a 10-doc live sanity run that it does
  NOT over-fire on ordinary tables that are genuinely about their row
  content (e.g. a sports-attendance lookup with an incidental data table
  stayed correctly in sports_analytics/question_answering, not pulled into
  this new category).

**Not acted on (below the bar, documented so it isn't re-investigated from
scratch):** `analysis` task_family had exactly 3 clean qualitative/
interpretive-analysis instances (literary, thematic, philosophical) against
5 existing subfamilies that are all quantitative -- right at the bar, flagged
as worth a second look with more volume, not confidently added yet. 2 real
GitHub bug reports had `task_family` land on `question_answering` instead of
the already-existing `code_debugging/bug_identification` -- only 2 instances,
below the action bar, noted as a watch item.

### Round 12 -- user pushed on the I/O schema, found and fixed a real,
substantial inconsistency (REFERENCE_DATA vs TEXT/FREE_TEXT)
User asked directly whether the input/output schema (type/format/language)
had room to improve. Investigated the one soft spot flagged as worth
checking: whether `REFERENCE_DATA` (explicit RAG-style "answer strictly
from this retrieved passage" content) is applied consistently against
`TEXT/FREE_TEXT`. First pass (only reading the 23 real `REFERENCE_DATA`
docs in the v3+v4 batches) looked clean/consistent -- but that was an
incomplete check: it only looked at the correctly-tagged side, not whether
the SAME template was being missed elsewhere. Rechecking by searching for
the identical template markers ("Document Id: 1", "I have been asked the
following... retrieved the following information", "the following query
requires an answer... not based on external knowledge", biomedical-passage
templates) across ALL 2000 real documents surfaced 42 MORE documents with
the exact same structural template that fell through to generic
`TEXT/FREE_TEXT` instead -- a real, substantial, ~65% miss rate (42/65),
not a fuzzy edge case. Lesson reinforced: checking only the
correctly-tagged side of a category gives a false sense of consistency --
always also check the miss side (does the same pattern exist untagged
elsewhere) before concluding something is fine.
**Fixed deterministically** (this is an objective, literal textual fact --
same class of fix as the turn-count/tool-contradiction checks, not left to
LLM judgment): added `REFERENCE_DATA_MARKERS` (8 precise, multi-word phrases
from the real recurring templates, not generic single words) and a new
`deterministic_fixes()` step 8 that overrides `input.type` to `REFERENCE_DATA`
(`format="TXT"`) when one of these markers appears in the input portion of
the text (before `[ASSISTANT]:`) and the model had only called it generic
`TEXT` (either `FREE_TEXT` or `MARKDOWN` -- broadened from `FREE_TEXT` only
after finding real cases where the template's own "### Task:"/"### Context:"
markdown-style headers made the model pick `MARKDOWN` format despite the
type still needing to be `REFERENCE_DATA`). Verified 64/65 (98.5%) of the
real matching documents now correctly get `REFERENCE_DATA` (re-running
`deterministic_fixes()` against the already-collected batch data, no new
LLM calls needed for verification) -- the 1 remaining is a single-instance,
differently-worded case below the 3-instance action bar, not chased.
Regression-checked across all 2000 real documents: exactly 41 newly flagged
(matches the 42 real misses minus the 1 excluded edge case, with zero
unexpected extra flags) -- confirms the fix is precise, not over-firing.

Also surfaced along the way, not from either report: `customer_support` has
7 real "other" documents, a few of which (order complaints, late delivery,
a dispute) look like they should plausibly fit the existing
`complaint_resolution` subdomain but didn't get it -- flagged as an
inconclusive watch item, not acted on, since several of the 7 are genuinely
odd one-offs and it isn't as clean a repeating pattern as the cybersecurity
case was.

### Round 13 -- external GPT audit found a real, systemic confusion between
formally-specified programming problems and logic puzzles
An external GPT audit flagged competitive-programming-style problems (the
kind with an explicit "Input:"/"Output:" spec and worked examples, like a
LeetCode/Codeforces problem) being misclassified as `mathematics`/
`proof_and_formal_reasoning` instead of `software_engineering`/
`code_generation`. Checked against real triggering documents before acting,
same standard as every other round in this file -- confirmed real: these
were getting caught by Round 6's rule 16 (the LSAT-style logic-puzzle rule,
"could you swap the cover story and the puzzle stays the same?"), which is
the wrong rule for them. The two look superficially similar (both are
self-contained problems with a precise structure) but are substantively
different: a logic puzzle's cover story is decorative and swappable, while a
programming problem's Input/Output spec and examples ARE the actual task --
you cannot solve it without them, and the deliverable is code, not a
one-shot answer. Fixed via a new `SYSTEM_A_RULES` rule 18 that gives Call A
an explicit test to tell the two apart (presence of a formal Input/Output
spec + worked examples -> `software_engineering`/`code_generation`, not
`mathematics`), plus a matching worked example added to `build_system_b`'s
rule 1 so the fine-grained pass correctly picks `general_purpose_
programming`/`function_generation` instead of falling through to `other`.
Verified by pattern-matching the bug's signature (regex `Input:.{0,400}?
Output:` combined with a mathematics/proof_and_formal_reasoning tag) against
the already-labeled 15,722-doc sample used for classifier training (see
"Classifier distillation" section below): found 66 real documents matching
the bug, re-ran the corrected mapper on just those 66, all 66 correctly
flipped to `software_engineering`/`code_generation`/`function_generation`.
Original file backed up before the fix
(`sft_output_sample_mapped.jsonl.backup_before_rule18_fix`) rather than
overwritten blind.

## Classifier distillation (domain / task_family fastText, for 1.2B-doc scale)
Purpose: the real per-document mapper above (`taxonomy_map_document.py`,
2 Gemma calls/doc) is far too slow to ever run on the eventual 1.2B-document
corpus (measured ~0.7-0.85 docs/sec single-process, ~decades at that rate).
This effort distills Gemma's own domain/task_family judgments into a cheap
CPU-only fastText classifier (measured ~60-77K docs/sec multi-process) that
can realistically run at that scale. Gemma's labels are the ground truth
this classifier is trained and judged against -- the goal is never
"get fastText's own opinion," it's "get fastText to mimic what Gemma would
have said." Lives mostly under `classifier_experiments/` and `lc_work/`
(the latter is this session's scratch/experiment area, not yet reorganized).

**Model lineage**: `domain_fasttext.bin`/`task_family_fasttext.bin` (first
baseline, trained on `sft_output_sample_mapped.jsonl`) ->
`_optuna_best.bin` (hyperparameter-tuned, see below, NOT used going forward)
-> `domain_fasttext_v2_AB.bin`/`task_family_fasttext_v2_AB.bin` in
`lc_work/models/` (current production model as of this pass: base 98,436-doc
clean set + 8,780-doc weak-categories set + two active-learning batches A/B,
3,000 docs each) -> `_v4.bin` (this round's retrain, adds a new 21,406-doc
batch; see below, not yet fully validated). `labeled_T.jsonl` (1,000 docs,
built by `select_and_label.py`) is the one genuinely unbiased, always-held-out
test set across every version -- never trained on.

**Alternative architectures were tried and did not beat fastText**, checked
before assuming fastText was the right call: TF-IDF+logistic-regression
(`tfidf_baseline.py`) and frozen-sentence-embeddings+logistic-regression
(`embedding_baseline.py`/`gte_head_baseline.py`, tested with
`BAAI/bge-small-en-v1.5`) both scored lower than fastText on the same
98,436-doc split (domain F1 0.871 fastText vs 0.832 TF-IDF vs 0.768
embeddings) -- and fastText is also the only one fast enough for 1.2B docs
regardless. Not revisited unless fastText's own ceiling turns out to be the
real blocker.

### The core problem this round: micro-F1 vs macro-F1
User caught that every evaluation script in this project
(`fasttext_baseline.py`, `arms.py`, `fasttext_optuna.py`) only ever computed
**micro-F1** (true/false positives pooled across every document, one F1 from
the pooled total) -- never **macro-F1** (each label's own F1 computed
separately, then averaged across labels, unweighted). This mattered for a
concrete reason: label volume here is extremely skewed (`mathematics` alone
is roughly half of all production rows), so micro-F1 is dominated by the
huge classes and can look strong while doing badly on small ones. Measured
directly: `v2_AB`'s micro-F1 looked good (domain 0.88, task_family 0.82 on
`target_test`) but real macro-F1 was far lower (domain 0.42, task_family
0.49), with 13-15 of roughly 35-50 real classes on that test scoring a flat
0.000 F1. **Standing rule now: always report macro-F1 alongside micro-F1 for
these classifiers** (also saved as a cross-session memory) -- micro-F1 alone
is not sufficient evidence a multi-label classifier is working.

Root cause, confirmed via a real per-class training-count audit across every
labeled source used so far: the weak classes are weak because they have
almost no real training examples, not because of an algorithm defect --
e.g. `task_family` values `localization`/`compilers_and_systems`/
`threat_analysis` each had exactly 1 real labeled example; domain
`mining_and_natural_resources` had 11. fastText's `loss="ova"` also has no
per-class loss weighting knob (unlike e.g. scikit-learn's
`class_weight="balanced"`), so training is naturally dominated by whichever
classes have the most rows.

**Tested and found INSUFFICIENT on its own**: duplicating existing rare-class
training rows (oversample to a floor of ~400 effective occurrences per
class, capped at 25x duplication -- `lc_work/retrain_balanced.py`). Gave a
real macro-F1 gain on the bigger, same-distribution validation sets (domain
`clean_val` 0.578->0.592, `weak_val` 0.566->0.610; task_family `clean_val`
0.471->0.532) but was flat-to-slightly-worse on the smallest, most realistic
sample (domain `target_test` 0.419->0.406). Lesson: duplicating a handful of
real examples teaches the model to repeat the same exact wording more
confidently, it does not manufacture genuine topic understanding -- more
*unique* real documents were needed, not more copies of the same ones.

### Multi-label cap bug (found and fixed live in the 502.8M-row production output)
`predict_with_fallback` (`classifier_experiments/fasttext_predict_utils.py`)
has no cap on how many labels can clear threshold, because `loss="ova"`
gives every label its own independent score (they don't have to sum to 1
like a softmax would). Checked directly against the real classified output
(`sft_43_language_wise_classified`, 502,832,310 rows): 76,529 domain rows
and 120,071 task_family rows had 3+ labels, some as high as 53 (task_family
only has 61 total values) -- almost always a genuine failure, not a
real multi-topic document (spot-checked the actual 2-label cases, which are
~1-2% of rows, and those did look like genuine dual-topic documents). Added
`predict_capped()` alongside the original function (kept the original intact
since other scripts depend on its exact behavior) -- it keeps fastText's own
top-`max_labels` (default 2) by score, since `model.predict(..., k=-1)`
already returns labels sorted by descending confidence, matching the same
1-2 label cap the real Gemma mapper's `validate_result()` already enforces.
Wired into `sft_classify_domain_task.py`'s `classify_row` for all future
runs. Fixed the ALREADY-PRODUCED 502.8M-row output surgically
(`sft_cap_labels_postpass.py`) rather than re-running the whole job -- only
re-scores the ~0.02% of rows that actually had 3+ labels, using the model's
own fresh scores to keep its real top-2, atomic per-file rewrite.

### Corpus-wide search for real examples of the weak classes
`classifier_experiments/keywords_weak_categories_v2.py` extends the existing
keyword file for the ~78 weak domain/task_family classes not yet covered,
grounded in each category's own official `taxonomy_tree_final.xlsx`
description (same standing rule as the taxonomy tree itself: evidence-based,
not hand-invented). `lc_work/weak_candidate_scan.py` searches the full clean
corpus (`sft_43_language_wise_clean`, 517M rows) for real candidate
documents. Two real bugs found and fixed while building this, both worth
remembering for any future full-corpus text scan:
- **A single combined regex (620 literal phrases joined with `|`) is
  catastrophically slow in Python** against this corpus's long lines --
  benchmarked at under 416 lines/sec, didn't finish 50,000 lines in 2
  minutes. Switching to a plain `phrase in lowered_line` loop (CPython's
  C-optimized substring search) was 6-20x faster (2,575 lines/sec on a
  typical file). Prefer plain substring checks over a large regex
  alternation for literal-phrase matching at corpus scale.
- **Reloading a dedup hash-set from disk inside the per-file worker function
  instead of once per worker** (via a `multiprocessing.Pool(initializer=...)`)
  cost 12 minutes of zero visible progress before being caught -- any
  "load once, reuse across many files" state must be built in the pool
  initializer, not inside the per-task function.

Final run: 4,928 files, ~89 minutes, found 21,407 unique real candidate
documents, hit the 350-per-label cap for 73 of 78 weak labels (5 fell
genuinely short on real supply: `content_moderation`,
`matching_and_resolution`, `knowledge_management`, `workflow_automation`,
`selection_and_filtering`). All 21,407 were then sent through the real
Gemma mapper (`map_document`) for verification, since a keyword match is a
guess, not a label -- confirmed working correctly on a real example (a
"weather balloon" mention inside an Arabic->Polish translation exercise
correctly got redirected to `humanities`/`translation`, not
`weather_and_climate_services`). Result: 21,406/21,407 succeeded
(`lc_work/labeled_weak_candidates.jsonl`). Real confirm rates varied hugely
-- some categories confirmed well (`cybersecurity` 1,023,
`explanation_and_tutoring` 1,981, `recommendation` 650) while others stayed
thin or came back completely empty even after checking their FULL real
candidate pool: task_family `content_moderation` and
`file_and_storage_operations` got **0** confirmed hits anywhere in the whole
517M-row corpus; `compilers_and_systems`/`localization` got 1 each. This is
real evidence some categories are genuinely rare or absent in this project's
current 19 tulu3 + 17 agentic source mix, not a search-methodology failure
-- the honest next step for these specific categories is an external source
(Hugging Face), not more internal searching.

Local `OUTPUT`-folder mining (`/projects/data/datasets/translation_data/SFT/
OUTPUT`, read-only, per the project's standing no-touch rule on that path)
was tried in parallel: a filename search across all 3,472 real data files
found genuine wins for a few categories (`cybersecurity`: 14 real CVE/
phishing/malware datasets; `hospitality`/`religion`/`manufacturing`: a
handful each) but ~24 of 32 weak domains had zero local matches, and two
checked candidate files turned out unusable as-is (one pre-tokenized with no
recoverable text, one image-based bearing-fault-diagnosis dataset with no
text description of the image, and this project's classifier is text-only).
Deprioritized once the corpus-internal scan alone hit the full candidate
quota for nearly all 78 weak labels -- revisit this folder only for whichever
specific categories remain short after a Hugging Face pass.

### Retrain v4 -- real result, partially validated, NOT yet promoted to production
`lc_work/retrain_v4.py` combines every real labeled source (98,436 + 8,780
weak-categories + 3,000+3,000 active-learning A/B + the new 21,406 =
~135K rows before oversampling) with the same oversampling recipe as before,
saved as `lc_work/models/{domain,task_family}_fasttext_v4.bin`. Also added a
NEW held-out eval set (`weak_candidates_val`, a 15% holdout carved from the
fresh weak-class batch itself) because `target_test` only has 1,000 docs and
near-zero real support for most weak classes -- not enough statistical power
to judge them. Real, measured result vs `v2_AB`:
- **domain**: macro-F1 improved on every eval set, including the untouched
  `target_test` (0.412->0.476) and especially the new weak-focused set
  (0.353->0.547).
- **task_family**: strong improvement on the bigger/weak-focused sets
  (`clean_val` +0.063, `weak_val` +0.059, new weak-focused set +0.174), BUT
  `target_test` itself went DOWN (0.488->0.452). **Flagged as a real,
  unresolved result -- not yet explained, not yet acted on.** Plausibly
  noise from `target_test`'s tiny per-class support (many task_family
  classes have 1-6 true occurrences in only 1,000 docs) rather than a
  genuine regression, but this has NOT been confirmed either way. Do not
  treat `v4` as validated/final until this is investigated -- e.g. by
  checking on a larger/fresher unbiased sample, or by identifying which
  specific classes drove the drop.

### Per-class thresholds and the real coverage/macro-F1 ceiling
Since `loss="ova"` gives every label its own independent score (not a shared
softmax), different labels can genuinely need different confidence cutoffs
-- checked directly on `v2_AB`: letting each domain label pick its own
best-F1 threshold (instead of one shared number) raised macro-F1 by a real
+0.033 (0.416->0.449), fixing classes like `arts_and_culture` (0.000->0.667)
that a shared threshold was suppressing. Task_family's result in the same
quick check was flat/inconclusive -- flagged as unreliable because both
tuning AND evaluating happened on the same small 1,000-doc test set (a real
overfitting risk, not yet corrected). **Standing plan, not yet done**: a
genuine 3-way split (train / threshold-tuning validation / held-out test)
before trusting a final per-class-threshold number.

Separately, measured the real coverage-vs-macro-F1 trade-off on `v2_AB`
(`lc_work/coverage_vs_macro_f1.py`): abstaining on (leaving unlabeled) the
~20-25% of documents the model is least confident about roughly doubles
macro-F1 on the documents that remain (domain 0.37->0.66 at ~78% coverage;
task_family 0.45->0.66 at ~65% coverage). Real but important caveat found in
the same check: part of that apparent gain is the hardest, rarest classes
silently dropping out of the macro-F1 average entirely once the model never
confidently predicts them anymore -- not purely individual documents getting
more correct. Any reported "macro-F1 at X% coverage" number must also state
how many classes are still actually being scored, or it overstates the real
improvement.

`fasttext_optuna.py`'s hyperparameter search also only ever optimized
micro-F1 (confirmed by reading its `objective()`/`score()` functions) --
its prior real result was a tiny +0.008/+0.009 micro-F1 gain over the plain
baseline after 5 trials per axis, consistent with optimizing a metric that
was already dominated by the easy majority classes. `domain_fasttext_
optuna_best.bin`/`task_family_fasttext_optuna_best.bin` are NOT used going
forward for this reason. **Standing plan, not yet done**: rerun Optuna with
its objective changed to macro-F1 once the `v4` data situation above is
settled, keeping its existing "tune the confidence threshold jointly with
training hyperparameters" design (that part is sound, it was just scoring
the wrong thing).

### Exhaustive per-category correctness read (459 real documents, all 49 domains + 61 task_families)
User pushed back that only scattered spot-checks had been done, not a real per-category read --
correct: prior checks this round were ~15-20 hand-picked documents chasing specific bugs, not a
systematic sweep. Pulled 5 real samples per category from the actual 517M-row classified output
(`lc_work/sample_all_categories2.py` -- first attempt was single-threaded and would have taken
hours since some categories are genuinely rare in the output; rewritten as a `Pool`-parallel scan,
same lesson as the corpus-wide keyword scan above) and delegated the actual read-and-judge pass to
a fork (mirrors this project's own Round 5 precedent for exhaustive reviews). Real result across
all 459 documents: **354 correct (77.1%), 37 questionable (8.1%), 68 wrong (14.8%)** -- domain
noticeably more reliable (82.1% correct) than task_family (71.7% correct), consistent with
everything else found this round. Critically, the errors are NOT random noise -- they cluster into
a few repeating, fixable root causes:
- **One document template caused 8+ wrong labels across 4 different task_family categories.** A
  "does this table SUPPORT or REFUTE this claim" fact-checking template is correctly tagged
  `fact_checking_and_claim_verification` in its own right, but the same template also got
  mislabeled `prediction_and_forecasting` (triggered by the literal word "predict" in its
  boilerplate), `transformation`, `ranking`, and drove **4 of 5** of `reporting_and_documentation`'s
  sampled documents wrong outright.
- **Severe keyword-collision domain misses**: `defence_and_national_security` (4/5 wrong -- real
  content was a home-security chat between roommates, triggered by "safe"/"secure"/"threats") and
  `legal` (4/5 wrong -- real content was history/musicology papers using legal-adjacent words like
  "codification"). Also `real_estate` content getting mislabeled `hospitality_and_food_services`
  (2/5 wrong in that category).
- **The already-fixed "swappable cover story" rule (Round 6 rule 16) does not transfer to
  fastText.** Gemma's prompt has this rule; fastText only ever learns "these words correlate with
  this label" from training examples, so a pure math/logic puzzle wrapped in a board-game or
  cable-TV-cost cover story still gets tagged `gaming`/`media_and_entertainment` alongside
  `mathematics`. **General lesson: a prompt-level fix to the Gemma ground-truth mapper does NOT
  automatically reach the distilled fastText classifier** -- fastText needs actual contrastive
  training examples (e.g. "math problem with a sports cover story -> `mathematics` ONLY") before it
  can learn the same distinction, since it has no access to the reasoning rule itself.
- **Several task_family categories are currently mostly-noise, not occasionally-wrong**:
  `classification` (3/5 wrong -- idiom/phrase-meaning questions mislabeled instead of
  `explanation_and_tutoring`/`question_answering`), `code_review` (1 correct + 3 questionable --
  heavily overlapping with `code_debugging` in practice), `generation` (3/5 wrong -- actually
  `question_answering`), `tool_use_and_function_calling` (4/5 wrong -- no real tool/function call
  visible in the sampled documents), `summarization` (2 wrong + 2 questionable).
- 1 domain (`mining_and_natural_resources`) and 17 task_family categories had 0 samples in the
  classified output at all -- consistent with, not new evidence beyond, the already-documented
  finding that these are genuinely rare/absent in the current source mix.
- Full per-category verdicts and every concrete wrong example are preserved in the fork's report
  (not re-copied here in full); re-run this same read against whichever model is eventually
  promoted to confirm these specific errors are actually gone, not just described.

### Not yet done (tracked here so a future session doesn't restart from scratch)
- Add targeted CONTRASTIVE training examples for the specific confusions found in the 459-doc
  correctness read above (cover-story math puzzles, keyword-collision defence/legal misses, the
  SUPPORTS/REFUTES template's cross-category leakage) -- this is now evidenced, not hypothetical.
- Investigate the task_family `target_test` regression in `v4` before
  promoting it.
- Targeted Hugging Face search, only for the categories that came back
  empty/near-empty even after the full internal corpus scan (task_family
  `content_moderation`, `file_and_storage_operations`,
  `compilers_and_systems`, `localization`, plus a handful of others under
  ~20 confirmed real examples) -- not a blanket search across all 78, most
  of those are already fine.
- Proper 3-way-split per-class threshold tuning (see above).
- Optuna re-aimed at macro-F1 (see above).
- Once all of the above is validated: re-run the FULL 517M-row
  classification job (`sft_classify_domain_task.py`) with the final model --
  the current `sft_43_language_wise_classified` output reflects the OLD
  (`v2_AB`) weak-class accuracy and should be treated as a first-draft/
  throwaway version, not something to build on, specifically for whichever
  of the ~78 weak categories a document's true label might be.

## Post-training datasets -> OpenAI SFT conversion (separate side project)
Source (read-only): `/projects/data/datasets/code_data/codeLLM_data/posttraining_datasets`
(1.8 TiB, 381.8M rows). Converted by `convert_posttraining_to_openai.py` into
`posttraining_openai_sft/` as `.jsonl.gz` (gzip because `/projects/data` is ~100% full).
Rows are plain OpenAI format (`messages` + top-level `tools`), both SFT role rules applied.
Per-file stats and drop reasons: `posttraining_openai_sft/_convert_log.jsonl`.
- `A_native_openai/` (14.72M rows, 332 GB raw, ~88 GB gz): nemotron cascade_sft stage1/2,
  science_v1, math_proofs_v1 (455,782 rows with empty `messages` dropped -- no proof),
  open-r1 codeforces-cots, cascade_RM_Training (`chosen` side only). `reasoning_content`
  merged into content as `<think>...</think>`. tool_calling: Hermes `<tools>`/`<tool_call>`/
  `<tool_response>` text converted to real `tools`/`tool_calls`/`tool` messages.
- `B1_marker_mapped/` (236.4M rows, 990 GB raw, ~260 GB gz): KodCode SFT/RL, deepmind
  code_contests, opencoder, and nemotron text blobs split only on reliable markers
  (`<extra_id_1>User/Assistant`, `input: ... output:`, leading `Question:/Problem:` +
  `Answer:/Solution:`, `prompt\n\n<think>` with a closed `</think>`, Scientific-Coding
  RESPONSE GUIDELINES).
- Deliberately excluded: `cascade_sft_swe` (byte-identical duplicate of stage2 swe_*),
  `cascade_RL_SWE` (prompt asks for SEARCH/REPLACE in `<think>/<solution>`, stored answer is a
  git diff -- format mismatch), RL prompt-only sets, Math-Textbooks/Wiki-Rewrite (documents).
- Not converted ("B2", heuristic split, ~390 GB): MCQ rows split after the last option line
  (most of Nemotron-SFT-General and all of STEM-SFT), prompt+code-fence rows, and a generic
  "Answer:/Solution: anywhere in text" rule that was found unreliable (matches inside
  reasoning or "Reference Solution (Wrong):" in prompts). RQA rows without `Answer:` are
  mostly reasoning traces cut off mid-sentence -- not usable.
- Lesson: an earlier version of the tools regex matched the literal "<tools></tools> XML tags"
  phrase in the system prompt and silently dropped every tool list; caught only by validating
  output rows. Always validate converted output, not just the per-file success count.

## Final OpenAI-format SFT datasets + the single audit/cleaner (2026-09-24)
Five final SFT datasets live in this folder (renamed by the user on
2026-09-24 -- older notes may use the old names in brackets):
`sft_hf_domain_data_openai_ready` (521.76M rows; was sft_43_language_wise_clean),
`sft_hf_traces_data_openai_ready` (7.17M; was traces_v1_final),
`sft_hf_agentic_data_openai_ready` (36.29M; was hf-agentic-data),
`sft_code_math_text_openai_ready` (1.95M; from traces_team/code_math_text_translations_openai_sft),
`sft_rl_reference_data_openai_ready` (30.65M; from traces_team/RL_reference_data_from_SFT_openai_sft).
Standing user rules: plain OpenAI chat format only (arguments = JSON string,
top-level `tools` in OpenAI shape, reasoning as `<think>...</think>` inside
content; chat-template tweaks go in a separate later adapter, never the data);
every row ends on assistant; no assistant right after system.
- **ONE audit**: `sft_audit/sft_audit.py` (all datasets registered in
  `DATASETS`; groups A rules / B format / C empty / D quality / E info).
  "The rules" = every check except `INFO_CHECKS`, not just the two role rules.
  New problem types get added here as checks; always run on ALL datasets.
  Final run 20260924_155055: every rule = 0 on all five datasets.
- **ONE cleaner**: `sft_audit/sft_clean.py SOURCE OUTPUT` (never touches the
  source) applies every fix found so far (OpenAI-format pass, tool-shape/
  Kimi/null-answer fixes, tool-id linking, think fixes, role rules) and then
  drops any row still failing an audit rule. `sft_audit/fixes/` holds the
  in-place passes used on the already-built datasets (backups kept in each
  dataset's `_backup_*` folders -- "_" dirs are skipped by audit and cleaner).
- Bugs found this round worth remembering: parquet struct-union nulls in tool
  args; legacy `function_call` / role `tool_call` silently dropped by the GLM
  template; problem-CREATION reasoning attached to solve answers (math team
  generated_* data); answers trapped inside `<think>`; countdown RL rollouts
  (only 44% correct, verified by evaluating the expression); Kimi
  `<|im_system|>tool_declare` system messages; flat tool definitions.
- Not done yet: chat-template (GLM) adapter + render check, max token length
  decision, benchmark-contamination check, PII check for v1/agentic data.
- **Update 2026-09-24 -- A, B1, B2 all cleaned + audited.** Final (clean) folders, all
  registered in `sft_audit.py` DATASETS: `sft_posttraining_a_native_openai_ready` (14,717,011
  rows), `sft_posttraining_b1_marker_mapped_openai_ready` (236,311,230),
  `sft_posttraining_b2_heuristic_mapped_openai_ready` (72,622,378). B2 was built by
  `convert_posttraining_b2.py` (MCQ split after the FIRST option block, ending at the first blank
  line since options can be out of order; code-fence split on the last code block; replies not
  ending cleanly dropped) -- checked on ~165K sampled rows before the full run. B2 reasoning is NOT
  wrapped in `<think>` (source has no reasoning/answer boundary). Audit
  `sft_audit/reports/audit_20260924_194136.md`: every A-D rule = 0 on all 8 datasets.
- **Update 2026-09-24 (later) -- think fix.** B2 re-converted (v2): raw R1 self-talk replies now
  have reasoning inside `<think>` (everything up to the LAST self-talk paragraph), clean
  explanation + `\boxed{}` stays visible -- 33.58M of B2's 72.62M rows got `<think>`. A re-cleaned
  with new audit RULE `think_not_closed` (<think> opened, never closed = cut-off reasoning, no
  answer): 4,597 open-r1 codeforces rows dropped -> 14,712,432 rows. New INFO check
  `selftalk_outside_think` (many hits are normal "Okay, here is..." replies, so info only).
  Old versions kept in `posttraining_openai_sft/_old_*`. Audit `audit_20260924_222158`: all 3
  posttraining sets pass every rule; the new rule found real cut-off reasoning in OTHER datasets
  not yet re-cleaned: hf_traces 315,243 (mostly dyve_plus_120k, DCAgent staqc terminus, limo --
  final turn truncated mid-thought), hf_agentic 51, code_math_text 1, B1 5. smolagents
  codeagent-traces has ~100 rows where <think> opens in one turn and closes in a later one.
- **Update 2026-09-25 -- 10 datasets.** Added `sft_datasets_sft_openai_ready` (748,913,055 rows,
  from `traces_team/datasets_sft_openai_sft/by_dataset`) and `sft_synthetic_data_openai_ready`
  (158,214 rows, from `traces_team/synthetic_data_openai_sft`). Total 1,670,538,731 rows, 2.74 TB on
  disk / 4.18 TB text, ~925B est. tokens (chars/4). Audit `audit_20260925_002933`: 6 datasets pass
  every rule; `think_not_closed` still open in hf_traces (315,243), datasets_sft (94), hf_agentic
  (51), B1 (5), code_math_text (1) -- not re-cleaned yet, waiting on the user. Stats:
  `sft_audit/reports/sft_stats_20260925_004709.xlsx`.
- **Deadlock fix 2026-09-25:** `sft_audit.py` / `sft_stats.py` / `sft_clean.py` used
  `ProcessPoolExecutor` + submitting every file at once; at ~31K files (after datasets_sft) the
  audit hung forever on Python 3.10 (main process stuck in pipe_write, workers idle). All three now
  use `multiprocessing.Pool.imap_unordered`. If an audit log stays empty for minutes, check worker
  CPU / open files before waiting on it.

## The 10 final OpenAI SFT datasets -- paths, rows, raw sources (as of 2026-09-25)
All under `/projects/data/datasets/code_data/sai_rupesh/taxonomy/`. Rows are after the
`think_not_closed` drop; audit `sft_audit/reports/audit_20260925_022850`: every A-D rule = 0 on
all 10. Files are `.jsonl` except #6-8 (`.jsonl.gz`).

| # | Folder | Rows | Size | Raw (unconverted) source |
|---|---|---|---|---|
| 1 | `sft_hf_domain_data_openai_ready` | 521,764,468 | 938.8 GB | `translation_data/SFT/OUTPUT` |
| 2 | `sft_hf_traces_data_openai_ready` | 6,850,008 | ~218 GB | `translation_data/Traces_datasets/OUTPUT` |
| 3 | `sft_hf_agentic_data_openai_ready` | 36,294,330 | ~253 GB | `translation_data/Agentic_Ai/OUTPUT` |
| 4 | `sft_code_math_text_openai_ready` | 1,948,295 | 15.0 GB | via `traces_team/code_math_text_translations_openai_sft` |
| 5 | `sft_rl_reference_data_openai_ready` | 30,649,026 | 64.2 GB | via `traces_team/RL_reference_data_from_SFT_openai_sft` |
| 6 | `sft_posttraining_a_native_openai_ready` | 14,712,432 | 94.3 GB gz | `codeLLM_data/posttraining_datasets` |
| 7 | `sft_posttraining_b1_marker_mapped_openai_ready` | 236,311,225 | 278.3 GB gz | `codeLLM_data/posttraining_datasets` |
| 8 | `sft_posttraining_b2_heuristic_mapped_openai_ready` | 72,622,378 | 161.9 GB gz | `codeLLM_data/posttraining_datasets` |
| 9 | **`/projects/data/datasets/traces_team/datasets_sft_openai_sft/by_dataset`** (name in DATASETS: `sft_datasets_sft_openai_ready`) | 737,962,197 (after traces_team within-file dedupe) | 662 GB | `translation_data/datasets_sft/parquet_files` |
| 10 | **`/projects/data/datasets/traces_team/synthetic_data_openai_sft`** (name in DATASETS: `sft_synthetic_data_openai_ready`) | 158,214 | 0.66 GB | combined parquet in `curated_SFT_datasets/synthetic_data` |

**#9/#10 final paths (user, 2026-09-25):** the traces_team folders ARE the final data for these two;
all audits and stats use them (`sft_audit.py` DATASETS points there). The taxonomy copies
`sft_datasets_sft_openai_ready/` (748,912,961 rows, copied 2026-09-24 21:06 BEFORE traces_team ran a
within-file dedupe at 22:55 that removed 10,959,672 dup rows) and `sft_synthetic_data_openai_ready/`
(byte-identical to traces_team) are stale and not used. We have no write access to traces_team, so
rule failures there cannot be fixed in place. Audit `sft_audit/reports/audit_20260925_121401`
(traces_team paths): synthetic passes every rule; datasets_sft (737,962,197 rows, 20,662 files) FAILS
rules -- system_then_assistant 5,868, special_token_leak 2,213, degenerate_answer 633,
think_not_closed 94, answer_empty_after_think 1, final_answer_empty 1 (our old copy had these dropped;
the traces_team dedupe did not drop them). User decision 2026-09-25: IGNORE these ~8,810 rows -- the traces_team folder is used as-is (owner says it is already cleaned; his clean = dedupe + format validation only). Do not re-raise them.

Raw inventory for #9/#10 (`sft_loss_report/v2/inventory_before_new5.jsonl`): datasets_sft raw =
20,039 parquet / 872,516,871 rows; synthetic folder = 18 parquet / 632,850 rows, but only the one
combined parquet (158,214 rows) was converted. Posttraining raw scan:
`sft_loss_report/v2/posttraining_raw_scan.jsonl`.

### Stats / before-after / domain reports (user asks 2026-09-25)
Run all together, detached, by `sft_audit/run_stats_domain_all.sh` (`setsid nohup`; writes
`run_stats_domain_all.progress` and `.done`; stats log `sft_audit/stats_run_o200k_v3.log`):
1. `sft_audit/sft_stats.py` -> `sft_audit/reports/sft_stats_latest.xlsx` (+ `_per_file.csv`):
   rows, GB, GPT `o200k_base` token counts (tiktoken `encode_ordinary_batch`), context-length
   buckets, per-file max tokens; every sheet has a `dataset_path` column and "Per file" has the
   full path of every jsonl.
2. `sft_loss_report/v2/hf_domain_domainwise.py count` -> hf_domain rows/tokens per domain
   (row-level: each final prompt key mapped to its source folder via `sft_loss_report/domain_work`
   keys, `hf_final_key_domain.npy`; 41.5M keys appear in several domains -> first source wins;
   4.09M final keys untraced).
3. `sft_loss_report/v2/build_before_after_all10.py` -> `sft_loss_report/v2/sft_before_after_all10.xlsx`
   (+ `csv_all10/`): before/after ROWS only (user said rows are enough there) for all 10.
4. `sft_audit/sft_domain_stats.py` -> SEPARATE workbook `sft_audit/reports/sft_domain_stats_latest.xlsx`:
   rows, o200k tokens, context length per domain, plus which raw parquet paths and which final
   jsonl paths belong to each domain.
Domain = rough guess from dataset/file/folder NAMES (`sft_audit/domain_rules.py`: 17 DOMAINS,
ordered regex RULES, first match wins; `HF_FOLDERS` for hf_domain folders) -- not a per-row
classifier. Unmatched names land in "Other / Mixed" and are listed in the workbook for checking.
Name-rule gotcha: match on a path RELATIVE to the source root, not the full path (the full path
contains "translation_data" and made everything "Translation").
- **Stats speed fix 2026-09-25:** `sft_stats.py` gave each FILE to one process; the biggest file
  (`sft_hf_traces_data_openai_ready/.../nvidia__Open-SWE-Traces.jsonl`, 122.9 GB) alone would have taken
  ~6.5 h at ~5 MB/s/process (o200k tokenizing is the cost) while the other cores sat idle. Plain `.jsonl`
  files > 512 MB are now split into byte ranges (a range owns the lines that START inside it; per-file
  counters merged, `tokens_max` merged as max). Verified identical to whole-file counts. `.jsonl.gz` files
  (posttraining) can't seek, so they stay whole -- they are small. Backup: `_sft_stats.py.bak_before_chunks`.
- **Domain workbook parquets (2026-09-25):** `sft_domain_stats.py` "Domain summary (all 10)" now also has
  `raw_parquet_files`, `raw_parquet_files_in_final`, `raw_rows_before` and the joined raw parquet paths per domain;
  sheet 2 = "Domain -> raw parquet paths" (one row per raw file, 69,373 files), sheet 3 = counts per domain x dataset.
  Any cell > 32,000 chars is cut with a "(N paths in total...)" note (Excel limit 32,767). "Other / Mixed" has ~42K raw
  files because hf_traces raw holds non-chat data (spec_cpu branch traces 21,123 files, FLAME-MoE 20,920) that never
  reached the final data -- expected, not a rules gap.
- **Domain rule bug fixed 2026-09-25 (aya / Updesh):** the name rule `translat|aya_|updesh` put
  `CohereForAI_aya_collection` (509.6M rows in datasets_sft) and `microsoft_Updesh_beta` (8.95M) in "Translation"
  (32.6% of all rows). Checked real rows: aya subsets are named `translated_*`/`templated_*` because the DATA was
  machine-translated into 100+ languages -- the tasks are dialogue (soda 178M), sentence rewriting (wiki_split 117M),
  continuation (xlel_wd 78M), QA (hotpotqa, nqopen, ...), summarization (cnn_dailymail), ... `source_of` now returns
  `aya::<subset>` / `updesh::<task>` and RULES map by task (like xP3). Translation fell to 25.1M (real MT tasks only).
  Lesson: a name-based domain for a huge single source must be checked on real rows before reporting.
  Quick rows-only workbook: `sft_audit/quick_domain_rows.py` -> `reports/sft_domain_rows_quick_latest.xlsx`
  (`REUSE_COUNTS=<old quick xlsx>` rebuilds without recounting).

## Deduped data, domain-wise (2026-09-25)
Himanshu deduped our 10 final datasets: copy at `/projects/data/datasets/code_data/Himanshu_sharma/Sft/raw` (taken
2026-09-24 23:41) -> `exact_dedup_data` -> `fuzzy_dedup_data` (MinHash Jaccard >= 0.8, global, source priority
posttraining A first ... datasets_sft last; reports in each `_dedup_report/report.json`). Their copy is NOT identical
to our final data: hf_domain `uncertain/` (187 files, 52,158,161 rows) was not included, and hf_traces / agentic /
code_math / B1 were copied before our cut-off-<think> drop (+315,243 / +51 / +1 / +5 rows).
- `sft_audit/build_dedup_domain_folders.py` -> `sft_final_deduped_domain_wise/<Domain>/<dataset>/<same rel path>`
  (byte copy; hf_domain split per ROW by prompt key -> source-folder domain, other datasets per file by source name;
  "/" in a domain name becomes " - " in the folder name). `_manifest.jsonl` = rows per domain per input file (resumes).
- `sft_audit/hf_before_rowdomain.py` -> row-level domain counts of OUR pre-dedup hf_domain (same rule, no tokens).
- `sft_audit/dedup_domain_excel.py` -> `sft_final_deduped_domain_wise/_dedup_domain_stats.xlsx`: rows before vs after
  dedup per domain / dataset / domain x dataset, exact + fuzzy steps, every output file path.
