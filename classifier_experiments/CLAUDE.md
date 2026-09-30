# Classifier Distillation Project

## Purpose
The real per-document labeler (`../taxonomy_map_document.py`, gemma-4-31b,
2 LLM calls per doc) is too slow to ever run on the eventual 1.2B-document
corpus -- at realistic production throughput this would take on the order of
decades (see the parent `CLAUDE.md`'s Round 10 section for the actual
measured numbers). This folder builds a **fast, cheap, non-LLM classifier**
(FastText) trained to approximate Gemma's own labeling decisions, so the
whole corpus can be tagged in a reasonable amount of time. Gemma's own output
is the training data and the ground truth this classifier is measured
against -- the goal is not to beat Gemma, it's to get close to it at a tiny
fraction of the cost. **Minimum acceptable bar: 95% F1**, not just "better
than before" -- this is a hard requirement, not a nice-to-have.

## Why FastText, not an embedding model
An embedding-model baseline (gte-base-en-v1.5 + a classifier head) was tried
first and worked, but FastText performed better and is far cheaper to train/
run, so the embedding-model track was deliberately deprioritized in favor of
FastText. Don't re-litigate this from scratch -- re-open the embedding-model
path only if FastText's own headroom (more data, better weak-category
coverage) is genuinely exhausted and still short of the 95% F1 bar.

## Data pipeline (how training data gets built)
1. **Sample real documents from the source corpus.** Two different sampler
   scripts, for two different sampling goals, both READ-ONLY against
   `/projects/data/datasets/translation_data/SFT/OUTPUT` (never write there)
   and both reuse the real production normalization code directly --
   `sft_schema.py`/`sft_normalize.py` from
   `/projects/data/datasets/translation_data/SFT/DEDUP_PIPELINE` -- never
   reimplemented, so a sampled record's shape matches what the real
   production `CANONICAL/` pipeline (`normalize_job.py`) would produce for
   the same document. One confirmed, deliberate gap from true production
   output: the real pipeline also fills an `extra` field (leftover raw
   parquet columns not mapped into the standard message fields); this
   project's samplers don't pass that through, so `extra` is always empty in
   our samples. Checked directly (20,000 rows), 100% empty. This has zero
   effect on labeling or training (neither ever reads `extra`, and it
   doesn't even survive into the final labeled files) but is worth knowing
   if a future check compares against true `CANONICAL/` output.
   - `sample_from_sft_output.py` -- general diversity sampling, used for the
     bulk of the training set. Groups raw shard files into "families" (by
     stripping the trailing `_\d+` shard suffix from the filename) and
     drains each family **sequentially**, one fully before moving to the
     next, before moving to the next domain. This was a deliberate fix for
     two real bugs found earlier: (1) round-robining raw shard files instead
     of families let one heavily-sharded dataset (e.g. a 169-shard
     `songlab__gnomad`) fake diversity while a domain only really touched
     6-8 distinct real datasets -- caught by the user asking "did u read the
     parquet file names man?"; (2) keeping ALL of a domain's families open
     simultaneously (round-robin across families) leaked file descriptors
     (1,000+ concurrently open files, confirmed via `/proc/PID/fd`) and
     progressively slowed the whole job down -- the user's own reaction to
     the first, over-engineered attempt at a fix was "why overcomplicate
     this so much man," which is why the actual fix is the simplest one:
     drain one family fully, close it, move to the next.
   - `sample_weak_categories.py` -- targeted sampling for categories the
     classifier measures badly on (see "Weak-category targeting" below).
     Different approach on purpose: these categories are scattered thinly
     across many different OUTPUT folders rather than concentrated in one
     (confirmed by manually pulling real examples before writing this), so
     it reads actual document TEXT (not just file/schema metadata) and
     matches against keyword lists per category (`keywords_weak_categories.
     py`, grounded in the taxonomy's own category descriptions plus real
     phrasing already seen in the data -- never guessed blind).
2. **Label the sampled documents with the real Gemma pipeline.**
   `map_batch2.py` imports `map_document` directly from
   `taxonomy_map_document.py` (never reimplements the prompts/validation
   logic) and runs it over a raw sample file. Uses a worker-pool pattern
   (`asyncio.Queue` + a fixed number of worker tasks), not
   `asyncio.gather` on a pre-built task list -- the gather-based version
   silently stalled at ~20,000-row scale with zero progress for 15+ minutes
   despite working fine at smaller scales; the worker-pool version doesn't
   have this problem. `CONCURRENCY = 192` is the real-measured safe ceiling
   for gemma-4-31b through this router for a single process (192-320
   plateaus around ~11-12 docs/sec on trivial short text; pushing to 8
   concurrent *processes* at this level crashed the gemma-4-31b backend for
   ~30-40s, so stay single-process against gemma unless that capacity
   picture changes).
3. **Combine into one training file** (`sft_output_sample_combined_*.jsonl`,
   filename suffix = row count) and retrain.

## Weak-category targeting (real methodology, not guessed)
After each retrain, check per-class precision/recall/F1 on the validation
set -- not just the overall micro F1, which is dominated by common classes
and can look fine while specific categories are at 0. Real finding: several
domains/task_families had near-zero F1 purely because of severe class
imbalance (as few as 39-198 examples vs. 26,000+ for `mathematics`) -- a
classifier can't learn a category it almost never sees. Fixed by building a
keyword-based content scanner (see `sample_weak_categories.py` above) that
targets exactly those weak categories specifically, rather than pulling more
generic diverse data and hoping it happens to cover them. **Confirmed lesson
across 3 retraining rounds: more data is the dominant lever, bigger than
hyperparameter tuning, at every stage measured so far** -- always pull more
(targeted, where possible) data before spending a lot of time squeezing more
out of Optuna trials on a fixed dataset.

## Retraining pipeline
`fasttext_baseline.py` trains the `domain` and `task_family` axes (the only
two with enough per-class volume to be meaningful right now -- subdomain and
task_subfamily are too thin). Includes a monkeypatch on `np.array` at the
top of the file to work around a real fasttext/numpy>=2.0 incompatibility
(`fasttext`'s internal `model.predict()` calls `np.array(..., copy=False)`,
which numpy>=2.0 raises on) -- this patch must stay; it's not optional
cleanup. `fasttext_optuna_parallel.py` (50 trials/axis, 10 parallel worker
processes) tunes hyperparameters and per-label prediction thresholds after
the baseline retrain. Always retrain the baseline first, then tune -- tuning
on stale data wastes trials.

## Results so far (retrain history, chronological)
| training docs | domain F1 | task_family F1 | tuned? |
|---|---|---|---|
| 15,722 | 0.793 | 0.707 | yes |
| 35,722 | 0.827 | 0.771 | yes |
| 77,234 | 0.858 | 0.803 | yes |
| 80,962 (+3,453 weak-category-targeted docs) | 0.857 | 0.803 | yes |
| 98,436 (+3,032 recovered retries, +19,223 more general docs, capped near the user's 100k target) | 0.859 (baseline only) | 0.808 (baseline only) | **no -- paused before tuning, see "Known issue" below** |

Check `fasttext_baseline_results.json` / `fasttext_optuna_*` logs in this
folder for the current numbers rather than trusting this table long-term --
same "check the live source, not a stale note" rule as the rest of this
project.

**As of 2026-09-16, the 98,436-doc round is intentionally incomplete.** The
user explicitly said "stop everything man plz" mid-cycle, right after the
baseline retrain finished but before hyperparameter tuning started, once the
truncation bug below was discovered. Do not resume this pipeline (more
sampling, tuning, or further merging) until that bug is actually fixed and
the user says to proceed again -- this was a deliberate pause, not a crash.

## Alternative-model evaluation (gpt-oss-20b vs. gemma-4-31b) -- DECIDED, closed
A newly-deployed `openai/gpt-oss-20b` model was evaluated as a possible
faster/cheaper labeler, using `map_gptoss20b_15k.py`. Reuses the real
`taxonomy_map_document.py` prompts/validation logic via import (monkeypatches
nothing in the production file -- builds its own thin `call_llm_oss`/
`map_document_oss` wrapper with the model swapped, same "explicit separate
test copy" pattern as `../taxonomy_map_document_fast.py`), relabels the same
15,722 Gemma-labeled documents, and reports field-by-field agreement against
Gemma's existing labels. **Critical gotcha confirmed before this was
usable**: see the parent `CLAUDE.md`'s Infra section on gpt-oss's hidden
reasoning tokens -- without raising `max_tokens` and setting
`"reasoning_effort": "low"`, gpt-oss-20b returned empty answers because its
own thinking ate the whole token budget.

Real measured results: gpt-oss-20b ran ~2.8x faster than gemma-4-31b per
request (8.2 docs/sec at concurrency 256 vs. gemma's 2.9 docs/sec at
concurrency 192), and agreed well with Gemma on the easy/mechanical fields
(response_behavior 99%, interaction_mode 92.5%, language ~95%) but much
worse on the fields needing real judgment (domain_subdomain 49% match,
task_subfamily 42% match, tool_category F1=0.34, constraints F1=0.10 --
badly under-detecting real tool use and constraints). Full numbers in
`gptoss20b_vs_gemma_report.json`.

**Decision (2026-09-16, made by the user): proceed with the FastText
classifier approach only, not gpt-oss-20b, for taxonomy mapping at scale.**
Even at ~2.8x faster than Gemma, gpt-oss-20b is still nowhere close to fast
enough to realistically cover 1.2B documents (single-process math: ~4.6
years at gpt-oss-20b's rate vs. ~13 years at Gemma's -- both impractical
without horizontal scaling that hasn't been sized yet), and its quality on
the nuanced fields wasn't good enough to trust as a full replacement labeler
either. This is a closed decision -- don't re-propose gpt-oss-20b as the
production labeler without new evidence (e.g. a much bigger deployment, or a
finding that changes the speed math).

## Known issue -- truncation cuts off real conversation content before Gemma
sees it (found 2026-09-16, NOT YET FIXED, work paused for this)
`map_batch2.py`'s `to_full_text()` (and the retry script's equivalent) cuts
every document down to `MAX_CHARS = 3500` before sending it to Gemma. For
documents with a long system prompt, that cutoff can happen before the real
user question ever appears in what Gemma sees. Measured directly against
the 80,962-doc training set (not guessed):
- **1,147 documents (1.4%)** have NEITHER a `[USER]` nor an `[ASSISTANT]`
  turn visible in what was sent to Gemma -- just a truncated system prompt
  (e.g. "You are a function-calling AI model...", "You are a deep thinking
  AI..."). Gemma's label for these is essentially a guess with zero real
  signal about what was actually asked. These labels should not be trusted
  and these documents should be identified and excluded (or re-labeled
  properly) before they're used in any future retrain.
- **6,222 more documents (7.7%)** have a real user question visible but the
  assistant's answer got cut off (99% of these -- 6,161/6,222 -- literally
  end mid-sentence with "...[truncated]"). Less severe since
  `SYSTEM_A_RULES` already says to classify what the user asked, not what
  the assistant produced, but worth knowing Gemma usually didn't see the
  full answer either.
**Status: found and root-caused, not yet fixed.** The user explicitly paused
all further sampling/labeling/training work until this is addressed (see
"stop everything man plz" in the 2026-09-16 status note above). The likely
fix direction (not yet implemented, needs the user's sign-off before
building): change the truncation logic to prioritize keeping the user's
question and the assistant's answer, truncating/dropping the system prompt
first if space is tight, rather than truncating in raw document order.

## Retry mechanism for previously-failed Gemma calls (real, working)
Built `retry_failed_80962.py` (2026-09-16) after finding ~4% of documents in
any large labeling batch fail (all 4 retries inside `call_llm` exhausted --
usually a transient network/timeout issue, not a content problem: checked
several real failed documents by hand, all were normal, unremarkable text).
Rather than losing that ~4% permanently, this script pulls the already-saved
`full_text` straight out of the failed records (no need to re-sample from
OUTPUT) and calls `map_document` on them again. Real result: **3,032 of
3,217 retried documents (94.2%) succeeded on the second attempt**, confirming
most failures really were transient. Reusable pattern for any future batch
with a meaningful failure count -- don't just accept the loss, retry using
the stored text first.

## Multi-label prediction quality (domain/task_family) -- mostly working, one
real gap found
FastText's `loss="ova"` (one-vs-all) is specifically built for multi-label
output -- at predict time it scores every class independently and returns
every one that clears the probability threshold, not just a single winner.
Confirmed this works in practice, not just in theory: tested the trained
`domain_fasttext_optuna_best.bin` against 10 real documents that have 2 true
domain labels -- 9/10 got the correct 2-label set back (order sometimes
differs, which doesn't matter since it's compared as a set), but **1/10 got
an empty prediction (zero labels) despite having 2 real domains** -- the
model wasn't confident enough about either label to clear the 0.5 threshold.
This is a real, open gap (tracked in the 2026-09-16 standup TODO as "fix the
multiple labels issue") -- likely a threshold-calibration problem, not a
fundamental multi-label support problem. Not yet investigated further.

Confidence-vs-accuracy is a real, useable signal (checked directly against
the validation set, not assumed): when the model is right it's usually very
confident (avg ~92% for domain, ~90% for task_family); when wrong, confidence
drops sharply (avg ~39% for domain, ~53% for task_family). Concretely, for
`domain`, requiring predictions to clear a 70-90% confidence threshold gets
you to 95-96% accuracy on the ~78-83% of documents that clear it -- already
at or above this project's 95% F1 bar on that filtered subset. `task_family`
tops out around 92.7% accuracy even at a 90% threshold -- still short of the
bar, consistent with it being the harder of the two axes throughout this
project. This is a real, practical lever for anyone wanting to deploy the
current model at scale without waiting for it to hit 95% F1 unconditionally:
trust high-confidence predictions now, route low-confidence ones to Gemma or
exclude them.

## Open TODOs (from the 2026-09-16/17 standup, not yet done)
1. Fix the multi-label under-prediction issue above (domain/task_family
   sometimes return zero labels for genuinely multi-label documents).
2. Fix the truncation issue above (mapper cuts off real user/assistant
   content before Gemma sees it).
3. Some SFT source data rows are missing the column(s) that indicate
   single-turn vs. multi-turn conversation structure -- this needs
   coordination with Bharath (a teammate, not something fixable from this
   environment alone) to actually get the source data corrected.

## Gotchas specific to this folder
- Always check a background labeling/sampling job's *actual* completion via
  a real marker in its log (e.g. grep for `"^DONE:"`) or the process's exit,
  not just "has some expected line appeared" -- a race where the log-watch
  loop starts before the target process has actually spawned can make a job
  look finished when it hasn't even started. Also always re-check with `ps`
  whether the process died silently (no Python traceback in the log) before
  assuming a "not done yet" state is still progressing -- a job's underlying
  shell can die independently of the watch script.
- When relaunching a killed/crashed labeling job, note that
  `map_batch2.py`/`map_gptoss20b_15k.py` open their output file in `"w"`
  mode (overwrite, not append/resume) -- any partial progress from the dead
  run is lost on relaunch. This is accepted as fine for now (no resume
  logic has been built) since partial progress lost so far has been small
  (low hundreds of docs out of thousands); revisit only if a crash starts
  happening late into a much larger run where the lost work would be
  significant.
