# Overnight Builder 1 Log

Plain-English running log. Dated entries, newest at the bottom. This is my
own scratch record -- the human will fold anything useful into CLAUDE.md
in the morning, so I'm not editing that file myself.

## 2026-09-20, start of session

Read both CLAUDE.md files fully before starting anything, per instructions.

### First surprise: a lot of this was already done by an earlier session

Before touching anything, I looked at the folder and found files timestamped
*after* the parent CLAUDE.md's "NOT YET FIXED" note for the truncation bug
(map_batch2.py, fasttext_predict_utils.py, fix_multilabel_zero_predictions.py,
verify_truncation_fix.py, remediate_truncation_dryrun.py, all Sep 17,
after the CLAUDE.md's own Sep 17 11:39 save time). So some earlier work
happened that never made it into CLAUDE.md's written notes. I read all of it
carefully instead of assuming and redoing work. Summary of what I found
already in place:

- **Truncation bug (my Task 1): already fixed in code.** `map_batch2.py`'s
  `to_full_text()` was rewritten to keep the real conversation ([USER]/
  [ASSISTANT]/[TOOL] messages) in full first, and only truncates/drops the
  system prompt if there isn't room. This is exactly the fix direction the
  parent CLAUDE.md asked for. There's also a `verify_truncation_fix.py` that
  compares old vs new logic on real data and found the old bug affected
  1.34% of documents (1,778/132,610) with zero [USER]/[ASSISTANT] content
  surviving, and the new logic brings that down to 0/132,610. I re-ran the
  numbers myself below to double check, rather than just trusting the
  earlier session's claim.
  - `retry_failed_80962.py` does NOT have its own truncation logic (it just
    re-sends the already-saved `full_text` from a failed row), so it wasn't
    affected by this bug and didn't need a fix.

- **Multi-label under-prediction (my Task 3): already investigated and
  fixed.** `fix_multilabel_zero_predictions.py` measured the real gap
  (13.6-13.8% of validation docs got ZERO predicted labels at the default
  0.5 threshold, even though every real document has at least 1 true label)
  and tested a fix: if nothing clears the confidence threshold, fall back to
  the model's single best guess instead of returning nothing. This was
  turned into a shared helper (`fasttext_predict_utils.py`) and wired into
  both `fasttext_baseline.py` and `fasttext_optuna.py`. Measured result
  (already in the code's own comments, I re-checked it): +5.3 percentage
  points exact-match for domain, +5.7pp for task_family, "with essentially
  no change to precision-heavy F1." I still need to double-check this
  specifically on single-label-only documents (the task instructions asked
  for that split explicitly) since I didn't see that split done yet -- see
  below.

- **What was NOT done yet**: actually recovering and re-labeling the
  documents damaged by the old truncation bug (Task 2), and re-running the
  full baseline/tuning cycle with all fixes in place (Task 4). The dataset
  files (`domain_fasttext.bin`, `sft_output_sample_combined_98436.jsonl`,
  etc.) are all still timestamped Sep 16, before any of these fixes existed.
  There was also a not-yet-run `remediate_truncation_dryrun.py` (a "can we
  even recover the original text" check, no LLM calls) sitting ready to go.
  I ran it myself (see below) rather than assume it had already been run.

### A validator agent is running in parallel (as expected)

Got a message that a validator agent (`a628ca92b8dedfdcb`) is auditing this
work independently. It had already left a file in this folder,
`GEMMA_98K_TAXONOMY_MAPPING_AUDIT_REPORT.md`, dated Sep 18, with several
claims about the 98,436-doc dataset. Per this whole project's standing rule
("check real data, don't trust a report blindly"), I did NOT act on any of
its claims without checking them myself first. It also messaged me directly
partway through with 3 independently-confirmed findings. Here's where things
landed:

- **Truncation claim (1,341 blind rows in the 98,436 set)**: I verified this
  count myself directly against the file -- confirmed, exactly 1,341. This
  matches Task 2 of my job (see below).
- **Reverse multi_turn bug** (docs wrongly tagged `multi_turn` when they're
  really a single real turn): the validator found this and asked me to check
  one specific risk before anyone acts on it -- whether some real multi-turn
  conversations store their history as plain embedded text instead of
  separate "[USER]:" lines (a `input.type == "CONVERSATION_HISTORY"` case),
  which a blind "count [USER]: lines" fix would wrongly break. I checked:
  yes, real risk -- 728 of 2,333 such documents would be wrongly downgraded
  by a naive fix. I confirmed this back to the validator. **This bug lives
  in `taxonomy_map_document.py`** (the shared, main-project production
  mapper), which is explicitly outside my job tonight (I was only told to
  touch that file if the TRUNCATION bug turned out to be inside it, which it
  isn't). Flagging this for the human to decide on, not fixing it myself.
- **Missing `number_theory` subdomain** (a taxonomy-tree gap, 4 real docs
  affected) and **`analysis`/`transformation` task_subfamily "other"
  collapse** (a taxonomy-tree coverage gap, ~63%/52% of those task families
  landing in "other"): both confirmed real by the validator with exact
  counts. These are taxonomy-tree changes (editing `taxonomy_tree_final.xlsx`
  with a full MECE check), which is a bigger, separate piece of work than
  what I was asked to do tonight. Not touching the tree tonight -- flagging
  for the human.

None of these 3 flagged items block my actual 4 tasks, since `domain` and
`task_family` (the two axes currently being trained) aren't affected by any
of them -- they're about `interaction_mode`, `domain_subdomain`
(number_theory only, 4 rows), and `task_subfamily`, none of which FastText
is being trained on right now (see parent CLAUDE.md: "only two axes have
enough per-class volume to be meaningful right now").

### Task 1 -- truncation fix: confirming, not redoing

Re-ran the before/after check myself against the real batch files (batch2,
batch3, batch4 -- the actual raw sample data): confirms the old logic
lost real content on 1,778/132,610 docs (1.34%), the new logic loses 0. Code
already matches what the parent task asked for (prioritize [USER]/
[ASSISTANT]/[TOOL], truncate the system prompt first). Marking Task 1 done
(inherited from the earlier session, independently re-verified by me).

### Task 2 -- handling the already-affected documents

Checked BOTH files as instructed:
- `sft_output_sample_combined_80962.jsonl`: 1,147 severe rows (1.42%) --
  matches the parent CLAUDE.md's number exactly.
- `sft_output_sample_combined_98436.jsonl`: 1,341 severe rows (1.36%) --
  matches the validator's independently-confirmed count.
- Checked overlap: 1,133 of the 80,962 file's affected rows are the SAME
  documents (by content hash / uid) still affected in the 98,436 file. The
  other 14 aren't present in the 98,436 file at all (dropped somewhere in
  how that file was built up) -- since 98,436 is the current, actively-used
  training file (`fasttext_baseline.py` reads from it, not from 80,962), and
  those 14 docs aren't even part of it, I'm not chasing them further. The
  80,962 file itself isn't used for anything going forward once 98,436's
  fixed version replaces it, so fixing 98,436 properly covers both files in
  practice.

Ran `remediate_truncation_dryrun.py` (already-written, not-yet-run
dry-run/no-LLM-calls check): of the 1,341 severe rows in 98,436, **1,330 had
their original raw messages recoverable** from the raw batch files
(`sft_output_sample_batch2/3/4.jsonl` or the original 15,722-doc sample
folder). Only **11 could not be recovered** (their uid doesn't match
anything in the raw files still on disk).

Per the instructions ("prefer re-labeling over exclusion when raw messages
are still available"), I wrote `relabel_severe_98436.py`:
1. Finds the 1,341 severe rows.
2. Recovers original raw messages for 1,330 of them.
3. Re-renders `full_text` using the FIXED `to_full_text()` (imported
   straight from the already-fixed `map_batch2.py`, not reimplemented).
4. Re-runs the real Gemma mapper (`map_document`, imported directly from
   `taxonomy_map_document.py`) on the fixed text.
5. The 11 unrecoverable docs get marked `ok=False` (excluded from training)
   since there's no real text to re-label from and their old label was a
   blind guess.

Writes a new file, `sft_output_sample_combined_98436_fixed.jsonl` (does not
overwrite the original), plus a `relabel_severe_98436_report.json` listing
exactly which uids were relabeled vs excluded.

Launched this as a background job (`relabel_severe_98436.log`). Status/
results to follow once it completes -- 1,330 real LLM calls at concurrency
192, should be a couple minutes, not hours.

## Update -- additional task added mid-session by the coordinator

Got a message adding a 5th task: benchmark the TRAINED classifier's own
prediction speed (not Gemma's speed, already measured elsewhere) and do the
honest 1.2B-document math with it. Doing this in parallel with the rest.
See "FastText inference speed" section below.

## Task 3 -- multi-label fix: checked the earlier fix more carefully, found
a real precision cost, made it better

The earlier session's fix (always guess the single best label when nothing
clears the confidence threshold) does work -- but I split the numbers by
single-label vs multi-label documents like the instructions asked, and found
something the earlier session's own notes didn't mention: on documents that
truly have only ONE correct label, this fix measurably HURTS precision:
- domain: precision drops from 0.928 -> 0.862 on single-label docs
- task_family: precision drops from 0.868 -> 0.815 on single-label docs

Why: when the model doesn't clear the confidence bar, its "best guess" is
only right 44-47% of the time (checked directly). Always guessing turns a
guaranteed miss into a coin flip. Overall F1 still goes up a tiny bit
(+0.001 domain, +0.008 task_family) because it also fixes real 2-label docs
that were getting 0 labels, but "barely positive with a real precision cost"
isn't a clean fix.

**What I changed**: added a minimum confidence floor before the fallback
guess is allowed to fire (if the model's best guess scores below 0.10, still
return nothing rather than force a guess). Tested floor values from 0 to
0.30 on the real validation data. 0.10 is right at the best F1 for both
axes, AND cuts the single-label precision damage roughly in half (0.928 ->
0.909 domain, 0.868 -> 0.845 task_family, vs 0.862/0.815 with no floor), AND
actually makes single-label F1 better than before the fix existed at all
(0.874 -> 0.882 domain, 0.826 -> 0.835 task_family). This is a strictly
better version of the same idea, done in-place in
`fasttext_predict_utils.py` (the one shared file `fasttext_baseline.py` and
`fasttext_optuna.py` both already import from), so nothing else needed to
change. **Task 3 marked done.**

## Task 1 -- truncation fix: re-confirmed, nothing more to do

Re-ran the real before/after check against the actual raw sample files
(batch2/3/4, the real source data): old logic loses all real content on
1,778/132,610 docs (1.34%), the already-fixed `to_full_text()` in
`map_batch2.py` loses 0/132,610. Code matches exactly what was asked
(keep the real conversation first, cut the system prompt first if there's
no room). **Task 1 confirmed done** (inherited fix, independently
re-verified, no changes needed).

## Task 2 -- relabeling the truncation-damaged documents: in progress

Checked both files as asked:
- 80,962-doc file: 1,147 damaged rows (1.42%) -- matches the number already
  written in CLAUDE.md.
- 98,436-doc file (the CURRENT one actually used for training): 1,341
  damaged rows (1.36%). 1,133 of these are the exact same documents as in
  the 80,962 file; the other 14 damaged docs from the 80,962 file aren't
  even part of the 98,436 file, so they don't matter for training going
  forward.

For the 1,341 damaged rows in the 98,436 file, I could recover the
ORIGINAL raw conversation for 1,330 of them from the untouched raw sample
files still on disk. Wrote `relabel_severe_98436.py`, which:
1. finds the damaged rows,
2. recovers their original raw messages,
3. rebuilds the text using the ALREADY-FIXED `to_full_text()`,
4. sends that fixed text to the real Gemma pipeline again for a real label.

The 11 rows with no recoverable original text get marked as excluded (their
old label was a pure guess and there's nothing left to fix it with).

This is running now in the background (`relabel_severe_98436.log`) --
1,330 real calls to Gemma, started ~17:29, at roughly 1.5 docs/sec as of
this update (~900/1330 done). Will write the final counts here once it
finishes. Output goes to a NEW file
(`sft_output_sample_combined_98436_fixed.jsonl`), the original 98,436 file
is left untouched.

## A second real bug found by the validator agent, not from the audit
report file -- a dropped "context" column, separate from truncation

The validator (agent `a628ca92b8dedfdcb`) messaged me mid-session with a
different, real bug: for 56 real source files (Finance, biology/medicine,
some legal datasets), the sampler script picks ONE column as "the question"
and has nowhere to put a second column that holds background/context text
the question depends on -- so that context column gets silently thrown
away. Example real row: a question asks for a specific court case number,
the case-record text that has the answer in it never reaches Gemma. This
is a DIFFERENT bug from the truncation one (this loses content because the
column was never even collected in the first place, not because of a
character-count cutoff).

I checked this myself before acting on it (didn't just trust the report):
confirmed all 3,209 already-affected rows really are in the current
98,436-doc training file, and hand-checked 2 real examples -- both
genuinely nonsensical/underspecified without the missing context, exactly
as described.

**What I did about it:**
1. **For the 3,209 rows already sampled with this problem**: excluding them
   from training rather than trying to patch them up. Retroactively finding
   and re-attaching the exact right passage to an already-sampled row is
   easy to get subtly wrong under time pressure (matching the wrong row's
   context to the wrong question), so the safe move tonight is exclude, not
   guess-and-fix. The validator sent me the exact 3,209 IDs
   (`context_drop_affected_uids.json`) so we're excluding precisely the set
   they verified, not a re-derived list that might not match.
2. **Fixed the root cause in my own sampler script** (`sample_from_sft_
   output.py`, NOT the shared production DEDUP_PIPELINE code, which nobody
   on this project is allowed to touch) so this stops happening to NEW rows
   sampled tonight and beyond. For the 57 known-affected files (a list the
   validator derived and I independently re-checked), the sampler now
   glues the dropped context text onto the front of the question, e.g.:
   `"Context: <passage>\n\nQuestion: <original question>"`. Verified this
   directly against a real file (`virattt/financial-qa-10K`) -- before the
   fix, the question arrived alone; after, the context is right there in
   front of it. Deliberately narrow: only touches the 57 specific files
   already confirmed to have this problem, not a generic "any column with
   'context' in its name" rule that could misfire on files that are fine.

Excluding the 3,209 rows will happen as part of building the final,
cleaned-up training file (combining this with the truncation relabeling
above) -- see next update.

## Task 2 update -- relabeling finished, retrying the API leftovers

`relabel_severe_98436.py` finished. Results out of the 1,341 damaged rows:
- 1,146 got a real new label from Gemma on the fixed text -- success.
- 184 recovered their real text fine but the Gemma call itself failed
  (network/timeout, the same kind of thing `retry_failed_80962.py` saw
  before -- most of those succeeded on a second try).
- 11 had no recoverable original text at all -- excluded, nothing to fix.

Wrote `retry_relabel_severe_failed.py` to give those 184 a second try
(same pattern as the existing `retry_failed_80962.py` -- reuses the
already-recovered text, no need to re-fetch anything). Running now.

## FastText inference speed -- final, cleaner numbers

Redid the benchmark with 12x more documents per run (5,000 -> 60,000) since
the first version's numbers were noisy and didn't make sense (more
processes sometimes looking SLOWER than fewer, which shouldn't happen for
this kind of embarrassingly-parallel work). With more documents per run:
- **Single core: about 9,700-10,000 documents/second** (a bit lower than
  the very first quick measurement, most likely because a real Gemma
  labeling job was also running on this machine at the same time competing
  for CPU -- a dedicated, idle machine would likely do somewhat better).
- **Best multi-process speed: roughly 60,000-70,000 documents/second**,
  reached around 16-32 separate processes running at once. Going past that
  (64 or 128 processes on a 128-core machine) actually got SLOWER again,
  most likely because that leaves no spare capacity for anything else
  happening on the machine (like the Gemma job also running) and processes
  start fighting each other for CPU time.
- **Honest 1.2 billion document math**: about **1.4 days on a single CPU
  core**, or about **5-8 hours using 16-32 processes on this one machine**.
  This confirms the whole point of building this classifier: it can
  realistically tag the full 1.2 billion document corpus in HOURS, compared
  to Gemma's own measured "decades" for the same job elsewhere in this
  project. That's the real, measured answer to "is this actually going to
  be fast enough," not just an assumption.
- Still have not been able to reach "Builder 2" for a side-by-side
  comparison with their embedding-based approach -- will ask if they make
  contact.

## FastText inference speed (new task from the coordinator) -- FIRST DRAFT, superseded by the cleaner numbers above

Measured directly (not assumed) using a new script,
`benchmark_fasttext_inference.py`, on 5,000 real documents pulled straight
from the actual training file:
- **Single CPU core: roughly 12,000-14,000 documents/second** for both the
  `domain` and `task_family` models. This is enormously faster than Gemma
  (which does ~1-12 documents/second depending on concurrency) -- as
  expected, since FastText is just counting word patterns and doing simple
  math, not running a large language model.
- Multi-process scaling was messy on the first two attempts -- more
  processes actually looked SLOWER than one process, which made no sense
  for embarrassingly-parallel work. Root cause (found before trusting the
  number): my benchmark was accidentally including each worker's one-time
  "load the 1.3GB model file" cost inside the timed section, and with only
  5,000 total documents split across many workers, that one-time cost
  dominated the measurement. Fixed by making every worker fully load its
  model BEFORE the clock starts. Real numbers still need a bigger, cleaner
  run (more documents, machine not shared with the relabeling job currently
  running) before I trust the multi-process number -- redoing that next
  and will report the final version here.
- This machine has 128 CPU cores.
- **Honest 1.2 billion document math, single-core**: at ~12,000-14,000
  docs/sec, that's roughly **1-1.3 days** for the WHOLE 1.2 billion document
  corpus on a single core, for one axis. That already answers the core
  question the classifier project exists to answer: yes, this is
  genuinely fast enough to be worth building, completely unlike Gemma
  (which the parent project measured at "decades" for the same corpus).
  Multi-process numbers (to get this down to hours, using all 128 cores)
  are being re-measured cleanly now -- will update with a trustworthy
  number, not the noisy first attempt.
- Have not been able to reach "Builder 2" (the other agent working on an
  embedding-model alternative) yet to compare inference speeds -- will ask
  if/when they make contact, per the instructions.

## Task 2 finished -- final clean training file built

- Relabeling recovered/fixed all 1,330 recoverable severely-truncated rows
  (1,146 first try + 184 more after a retry for transient API failures,
  same retry trick already used for the 80,962 batch).
- 11 rows had no recoverable original text -- left excluded.
- Combined this with excluding the 3,209 dropped-context rows the
  validator found (see above).
- Final result: `sft_output_sample_combined_98436_clean.jsonl` -- same
  98,436 rows kept on disk (nothing deleted), but now **95,216 are marked
  usable for training** (98,436 - 3,209 context-drop exclusions - 11
  unrecoverable-truncation exclusions). Updated `fasttext_baseline.py` to
  read from this clean file instead of the old one.

**Task 2 marked done.**

## Task 4 started -- retrain with all 3 fixes in place (truncation +
multi-label floor + dropped-context exclusion)

Ran the baseline retrain fresh on the clean data. Real result, compared to
the last "before" number in CLAUDE.md's own table (98,436 baseline-only,
before any of tonight's fixes):

| | before tonight | after tonight's fixes |
|---|---|---|
| domain F1 | 0.859 | **0.871** |
| task_family F1 | 0.808 | **0.816** |

Both went up from fixing real data-quality problems, not from any
hyperparameter change -- this matches the project's own standing lesson
("more/cleaner data is the dominant lever"). Still well short of the 95%
F1 bar, as expected -- this was a data-cleanup step, not the final push.

Now running the full Optuna hyperparameter search (50 trials/axis, 10
parallel workers, same settings CLAUDE.md documents) on top of this clean
baseline. Will report tuned numbers once it finishes, then check per-class
F1 to see which categories are still weak and need more targeted data,
same methodology as the rest of this project.

## Validator independently re-checked the dropped-context fix -- confirmed clean

The validator re-checked my work directly (not just took my word for it):
all 3,209 excluded rows are correctly marked, the counts match exactly,
zero rows in the final 95,216 usable rows have the truncation problem
anymore, and they read through my splice-fix code and confirmed it's
correctly scoped (only the 57 named files, handles the miriad
"passage_position is just a number, not real text" edge case correctly).
Marked verified-fixed on their side too, not just reported-fixed on mine.

## A third real bug found by the validator -- tool-call turns silently vanishing

The validator found a DIFFERENT real bug, also not from the audit report
file, this time specific to the "agentic" tool-use data: some tool-calling
chat formats mark an assistant's tool call with `content: None` (the actual
function name and arguments live in a separate field, not in the normal
text field). The shared production code that turns raw rows into
conversations has a line that builds message text from `content` (or a
couple of backup field names), and if all of those are empty, it just
throws the whole turn away. Net effect: the assistant's "I'm calling X
function with Y arguments" turn disappears completely, and the next thing
anyone sees is a tool result appearing out of nowhere with no explanation
of what was asked for.

I checked this myself against the real production code and real rows before
doing anything (didn't just trust the description): confirmed the exact
line responsible, and pulled real rows from all 6 affected files myself
(not just the ones in the report) to see the actual shapes. Found there
are really 2 different shapes, not one:
- 3 files store the call as an already-readable text string
  (e.g. `cooking.convert_cooking_measurements(quantity=2, from_unit="cup"...)`)
  -- easy, just use that string as-is.
- 3 files store it as a structured, code-like text blob (Python-style
  dictionary text) that needs a little parsing to pull out the function
  name and its arguments cleanly.

**What I did:** wrote a small function (`synthesize_tool_call_content` in
`sample_from_sft_output.py`) that handles both shapes, and made it run
BEFORE the production code gets a chance to drop the turn (patches the raw
conversation first, then hands it off as normal). Tested it live through
the real sampling code, not just in isolation -- pulled 50 real records from
one of the 6 files and confirmed zero turns are missing anymore. Also added
the same fix to `sample_weak_categories.py` so it doesn't only apply to one
of the two sampler scripts.

Excluded the 642 already-affected rows from tonight's training data the
same way as the context-drop rows (kept on disk, marked "don't use this for
training," not deleted). Final clean training file now has **94,574 usable
rows** (98,436 total - 3,209 context-drop - 642 tool-call-drop - 11
unrecoverable-truncation).

## A fourth finding -- NOT fixed tonight, flagged as a backlog item

The validator found something bigger: **51 raw data files (about 10 million
rows) are being completely skipped** right now, not because of a small bug,
but because the production code doesn't recognize their format at all. These
files store a whole conversation as one long piece of text using a template
style (things like `<|im_start|>user ... <|im_start|>assistant`, or
`[INST] ... [/INST]`) that the current code's pattern-matching doesn't know
how to split into separate turns. The validator opened several by hand and
confirmed they're genuine, complete, readable conversations -- this isn't
junk data being correctly skipped, it's good data with no door in for it.

I agree with the validator's own recommendation: this is real and worth
doing, but it needs an actual small parser built and carefully checked
against several different template styles across 51 different files, which
is a bigger and riskier piece of work than tonight's other fixes (each of
which was a small, narrow, one-file-list patch). Given tonight's list was
already full, I did not attempt this. **This is a real, verified, sizeable
opportunity (~10 million currently-invisible real rows) that the human
should decide when to prioritize** -- not urgent to tonight's F1 push, but
worth knowing about.

## A fifth finding -- also NOT fixed tonight, a taxonomy-mapping-choice backlog item

The validator found that 249 real documents (all the same "cybersecurity
penetration testing / CVE analysis" template) are landing in
`task_family=analysis, task_subfamily=other`, when the taxonomy tree
already has an exact-fit category for this
(`security_assessment`/`vulnerability_assessment`) that's barely used
(7-8 times total in the whole 98,436-doc set). This is the LLM mapper
(Gemma) reaching for a generic bucket instead of the specific correct one
that already exists -- not a data bug, a prompt/labeling-choice issue in
`taxonomy_map_document.py` (the shared production mapper). Fixing this
properly means adding a worked-example rule to that file's prompts and
then deciding whether to re-label already-affected documents -- a real
decision for the human/main taxonomy project, and outside what I was asked
to touch tonight (only the truncation bug there). Logging it here so it
isn't lost, not acting on it.

## Weak-category check on the freshly-cleaned data -- planning the next
sampling round

Ran a fresh per-class check against the just-retrained (clean-data)
baseline models while optuna keeps tuning in the background. Real result:
**18 of 49 domains and 24 of 53 task_families are still weak (F1 under
0.5)** -- mostly the same small-sample categories as before (veterinary,
defence, journalism, mining, etc. all still under 20 real examples each),
plus a few I hadn't seen flagged as weak before: `government_and_public_
administration`, `politics_and_civics`, `consumer_and_personal_services`,
`sales_and_marketing`, `arts_and_culture`, `telecommunications`,
`aerospace_and_aviation`, `human_resources_and_talent` (domains), and
`search_and_retrieval`, `rewriting_and_editing`, `software_design`,
`code_transformation`, `file_and_storage_operations`, `configuration_and_
deployment`, `risk_assessment`, `negotiation_and_persuasion`, and several
more (task_families). Some of these already had keyword scanners built
from an earlier round but still show very few real examples (e.g.
veterinary=9, defence=19, mining=1) -- the earlier weak-category batch got
diluted as more general data was added on top of it, matching the
project's own standing lesson that this needs to be revisited with real
volume, not assumed fixed once. Planning to expand
`keywords_weak_categories.py` with the newly-identified gaps and run
another targeted sampling pass once optuna finishes (don't want to compete
for CPU with the 10-worker tuning job that's still running).

## Getting ready for the next weak-category sampling round

Before pulling more data for the ~32 newly-identified weak categories, I
first quickly tested my planned keyword lists against real documents (same
check the original list's author did) instead of trusting them blind. Good
thing I did -- found several keywords that were too generic and would have
wasted scanning budget matching completely unrelated documents: "parliament"
matched a chemistry paper that happened to mention EU regulatory limits,
"spacecraft"/"satellite system" matched astrophysics/ecology papers instead
of aviation, "atmospheric pressure" matched a basic physics homework
problem, "editorial team" matched a medical website's staff byline, and
"social worker" matched a random list of healthcare job titles. Removed or
tightened those before running anything for real. (Worth noting: a bad
keyword here only wastes scanning budget, since the real label always comes
from Gemma reading the actual document afterward -- it can't corrupt
training data by itself, but it's still worth getting right so the scan
finds real examples instead of noise.)

Also found and fixed a real, separate problem before running anything:
`existing_content_hashes.json` (the file that stops the sampler from
picking a document that's already been used) was stale -- last updated Sep
16 07:20, before batch4's ~20,000 rows and the weak-categories batch were
even pulled. Running a new sampling pass against it as-is would have risked
picking rows that are already in the training set, faking "new" diversity.
Rebuilt it fresh from all 4 real raw sample files (136,115 unique hashes
now, up from 65,017) before doing anything else.

Now running a second, bigger weak-category sampling pass
(`sample_weak_categories_v2.log`) targeting 32 categories total (the
original 12 that are still thin, plus that many completely new ones),
CAP_PER_CATEGORY raised from 400 to 800. Running in the background while
Optuna tuning also finishes. Will label and merge once both are done.

## A sixth finding -- same pattern as #5, also backlog

Validator found another case of the same "generic bucket instead of the
specific correct category" pattern: 66 of 410 real `transformation`+"other"
documents are all the same template (translate an English sentence into
First Order Logic notation). The tree already has an exact-fit category
(`data_type_conversion`) that's only used 29 times total in the whole
corpus -- this one template alone is bigger than the category's entire
current real usage. Same as finding #5 (the CVE/security_assessment one):
this is a `taxonomy_map_document.py` prompt-choice issue, not a data bug,
needs a deliberate fix + re-label decision. Logged as backlog, not acted on
tonight.

## A seventh finding -- third example of the same pattern, still backlog

Validator found a third case: 36+ of the `transformation`+"other" documents
are StackOverflow-style "how can I improve this code" questions that should
be `code_review` (an already-used, non-dead category, 168 real uses
elsewhere) but aren't getting it consistently for this specific template
style. Same root cause class as findings #5 and #6. Logged together as one
backlog group for whoever picks up the analysis/transformation gap work --
not acted on tonight.

## Optuna tuning finished + a real gotcha I hit myself

Optuna tuning finished: domain F1 0.871 -> 0.872 tuned, task_family
0.816 -> 0.811 tuned (barely moves either way -- matches this project's own
long-standing finding that tuning helps much less than more/cleaner data).
Both numbers independently confirmed against the same file by the
validator too.

Hit the exact zombie-process gotcha this folder's own CLAUDE.md already
warned about: my "wait until the process is gone" background checks kept
saying the optuna job was still running for several minutes after it had
actually finished and printed its full results -- because the process was
a zombie (finished but not yet cleaned up by its parent), and the simple
check I was using (`ps -p <pid>`) still says "yes it exists" for a zombie.
Had to check the process's actual STATE (not just whether the PID exists)
to see it was really done. Cleaned up 4 stray background checks that would
have spun forever. Good reminder that this documented gotcha is real, not
just a hypothetical warning.

## Second weak-category sampling round finished, labeling now

`sample_weak_categories_v2` finished: **8,780 new real documents** across
the 32 target categories (some filled completely at the 800 cap --
law_enforcement, veterinary, comparison, construction, transportation,
defence, supply_chain -- others stayed thin because real supply is
genuinely scarce in this corpus, e.g. `ranking`=7, `code_review`=1,
`system_design`=1 -- consistent with earlier findings, not a scanning
problem). Labeling this batch with the real Gemma pipeline now
(`map_weak_categories_v2_full.log`). Once done: merge with the 94,574-doc
clean set, retrain baseline, check per-class F1 again, tune again if the
new categories show real movement.

## Validator independently confirmed the truncation-relabel fix too

Checked the exact document the original external audit report flagged as
"blind guessed cybersecurity" (Reasoning.jsonl:353) -- confirmed it now has
a real user question in it, and Gemma's new label for it is grounded in
that real content, not a guess. Also independently confirmed 0 of the
94,433 usable rows have zero real user turns. Both the truncation fix and
the dropped-context/tool-call fixes are now independently checked, not just
self-reported.

## User said "fix it plz" -- explicit go-ahead to act on the backlog items,
in priority order: #1 the 10M invisible rows, #2 the multi-turn bug, #3 the
3 generic-bucket mapping issues

### Item #1 -- unlocking the ~10 million invisible rows (the big one)

Found a real head start already sitting in the parent taxonomy folder (not
mine, not touched): `sft_normalize_v4.py`, a candidate improved version of
the real production file with working parsers already built for 2 of the
~5-6 real chat-template styles (ChatML and Meta's Llama-3 format). Did NOT
modify that file (it's someone else's in-progress work, and modifying it
risks stepping on it) -- instead reused its 2 working parsers by importing
them (read-only), and wrote NEW parsers of my own for the other template
styles that came up in real files: `[INST]...[/INST]` (Llama-2/Mistral
style), `### Human:`/`### Assistant:` (Alpaca/Vicuna style), and
`<|user|>`/`<|assistant|>`/`<|system|>` (Zephyr/TinyLlama style). All of
this lives in a brand new file, `chat_template_parsers.py`, in this folder
only -- DEDUP_PIPELINE was never touched.

Before trusting any of this, ran my own independent real-data scan
(`scan_chat_template_files.py`) rather than working purely from the
validator's description -- found **55 real files, ~10.55 million rows**
(close to the validator's own 51/~10M count; small difference is just
slightly different detection thresholds, not a disagreement about the
underlying fact). All 3 of the validator's own example files showed up in
my scan with the template type I'd have guessed too.

**Caught a real bug during verification, exactly the way this project asks
you to** (verify against MULTIPLE real files, not one): my first version of
the `<|user|>`/`<|assistant|>` parser scored 0/50 on a real file
(`code/smangrul__code-chat-assistant-v1.parquet`) that my own scan had
flagged as matching. Reason: that file actually uses a different, related
style (`<|prompter|>` instead of `<|user|>`, an OpenAssistant convention),
which only coincidentally shared the `<|system|>` marker with true Zephyr
files. Fixed by adding `<|prompter|>` as a recognized marker (mapped to
"user"). Re-verified across all 55 real files, 100 real rows each (5,450
total rows checked): **98.1% parsed successfully.** The one meaningful
failure (`Maths/datafreak__MATH-Llama2-train.parquet`, 0/100) turned out to
be a genuinely malformed file (its own text column is missing the closing
`[/INST]` tag) that ALSO happens to have clean, separate
`user_message`/`assistant_message` columns sitting unused right next to it
-- handled with a narrow, single-file override that uses those clean
columns directly instead of forcing a broken template through the parser.

Built `sample_chat_template_files.py` to actually pull real documents from
these 55 files using the new parsers, running each result through the same
real production `S.canonicalize_turns()`/`S.build()` every other sampler in
this project uses (so the output shape is identical, nothing reinvented
downstream). Capped at 500 rows per file (each of the 55 is treated as its
own real, distinct dataset, same "family" idea as the other sampler) so a
few giant 1M-row shards can't crowd out the other ~50 smaller ones.

**Real result: 25,749 new, previously fully-invisible real documents**,
deduped against everything already in the training pool. One file (a
"llama2format" variant of a dataset already sampled from its sibling file)
correctly yielded 0 new rows -- checked directly, its content really is
already covered under a different template rendering of the exact same
underlying data, not a bug.

Labeling this with the real Gemma pipeline next, queued to run right after
the weak-category-v2 batch currently in progress finishes (never running
two processes against gemma-4-31b at once, per this project's own standing
rule -- it crashed the backend before).

### Item #2 -- the multi-turn labeling bug, fixed carefully

Read real examples of both the "728 CONVERSATION_HISTORY" group and the
wrongly-labeled group side by side, like the coordinator asked, before
changing anything. Real finding, more subtle than it first looked:

**`input.type == "CONVERSATION_HISTORY"` is NOT a safe signal to protect on
by itself.** Checked directly: of the 728 docs with that type AND a
"multi_turn" label AND <=1 real "[USER]:" marker, only 68 are genuinely
multi-turn (real embedded "User: ... Assistant: ... User: ..." dialogue
history pasted into one turn). The other 660 are the EXACT SAME underlying
bug as the ones outside this group -- the model saw a phrase like "A
conversation between User and Assistant..." (a generic template preamble
many tool-use datasets use) or "conversation context" and wrongly assigned
BOTH `input.type=CONVERSATION_HISTORY` AND `interaction_mode=multi_turn`
from the same false pattern-match. So gating the fix on that field would
have left 660 of the real errors unfixed.

**The actual reliable signal, found by reading real examples of each
group**: does the text contain a REAL embedded exchange -- at least 2
literal "User:"/"Human:" markers (or the pipe-style "<|user|>"/"<|human|>"
variant, found while checking a broader sample) AND at least 1 "Assistant:
"/"AI:" marker, case-insensitive? A document that just mentions the WORD
"conversation" or describes "a conversation between User and Assistant" in
prose has zero literal "User:" occurrences (no colon follows the word), so
it can't accidentally pass this check -- but a real embedded transcript
always has these markers repeated with a colon or pipe-bracket, literally,
multiple times.

Also caught and fixed my own bug while building this: my first version of
the check was accidentally case-SENSITIVE (only matched "User:", not
"user:" or "USER:"), which would have missed a lot of real multi-turn docs
that use lowercase or all-caps role markers. Found this by spot-checking a
RANDOM sample of the "would be downgraded" group (not just eyeballing the
first few) and noticing genuinely multi-turn documents in there that
should have been protected.

**Final result, verified on real data**: of the 1,445 docs originally
flagged, 721 have a real embedded conversation and are correctly kept as
`multi_turn`; 724 are genuinely single-turn and get corrected. Randomly
sampled 20 from each group (not cherry-picked) and manually read every one
-- 0 false downgrades, 0 false keeps found. Added this as step "2b" in
`deterministic_fixes()` in `taxonomy_map_document.py`, with full reasoning
written inline the same way every other rule in that function documents
itself.

Since `deterministic_fixes()` is a pure function (no LLM call, just takes
the model's raw result + the text and cleans it up), I could re-run it
against ALL already-collected real data for free, with no new Gemma calls
needed -- did that (`reapply_deterministic_fixes.py`) against the clean
98,436-doc training file: exactly 753 rows got a field corrected (724 of
them the interaction_mode fix; the rest are the same fix applying to a few
already-excluded rows, harmless since they're skipped from training
anyway), 0 errors. **Item #2 done and already reflected in the current
training data**, not just fixed for future documents.

## (continued below as work progresses)
