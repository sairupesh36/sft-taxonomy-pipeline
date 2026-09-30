# Overnight builder 2 — log (embedding-model re-attempt + simple baselines)

Plain English notes, dated, added as I go. I am NOT calling Gemma or touching
the labeling pipeline — I only use the already-labeled data sitting in this
folder. Working alongside a first builder (fixing FastText data bugs) and a
validator (auditing everyone). Full context read first: the parent
`../CLAUDE.md` and this folder's `CLAUDE.md`.

## 2026-09-20, start of shift

**What I'm doing and why.** The user wants three things checked, all on the
current biggest labeled dataset (`sft_output_sample_combined_98436.jsonl`,
98,436 documents labeled by Gemma):
1. Try the embedding-model idea again (turn each document into a vector with
   a pretrained model, then train a small classifier on top of the vectors),
   properly this time, on much more data than the first attempt used.
2. Also try a dead-simple method (TF-IDF word counts + logistic regression)
   as a sanity check.
3. Compare all of them fairly against FastText (the current best method) —
   same train/test split, same scoring, no cherry-picking.

**What I found by reading the files first, before writing any code:**
- The old embedding attempt in this exact folder
  (`embedding_baseline_results_*.json`) used two SMALL, fast models
  (`all-MiniLM-L6-v2` and `bge-small-en-v1.5`), on an even smaller old
  dataset (~15,700 docs) — not the `gte-base-en-v1.5` model the project's
  own notes say was tried. Both of those small models scored clearly below
  FastText's own number at that same data size (domain F1 0.74-0.77 vs
  FastText's 0.79 at 15,722 docs).
- The real `gte-base-en-v1.5` attempt turned out to be a *different* script,
  one directory up (`../build_embeddings.py`), not this folder's
  `embedding_baseline.py`. That run was started but never finished or
  scored — the log shows it stopped partway through encoding (only 37 of
  246 batches done), and no classifier was ever trained on top of it. So
  "gte-base-en-v1.5 was tried and FastText won" is not quite accurate — the
  gte-base-en-v1.5 attempt was never actually completed or measured. I'm
  finishing that measurement now, properly, on the current 98k dataset.
- Good news: the environment already has the exact library versions needed
  for `gte-base-en-v1.5`'s custom code to work (`transformers==4.44.2`,
  `sentence-transformers==3.0.1`), and the model weights are already
  downloaded in the cache. No reinstalling needed.
- The machine has 112 CPU cores and ~1TB RAM, no GPU. I benchmarked
  gte-base-en-v1.5 on this box directly rather than guessing: at
  batch_size=128 it encodes about **7.4 documents/second** using all 112
  threads (each document truncated to 3,000 characters, same limit the
  original abandoned attempt used, to keep it from crawling on the very
  longest documents). At that rate, encoding all 98,436 documents once
  takes roughly 3.5-4 hours. I'm running that now as one background job,
  saved in 5,000-document chunks on disk so a crash partway through doesn't
  lose everything already encoded.
- I'm reusing the exact same train/validation split code, random seed (42),
  and scoring formulas (micro precision/recall/F1, exact-label-set-match
  rate) that `fasttext_baseline.py` uses, so every method's number is
  measured the same way on the same rows — a fair comparison, not
  apples-to-oranges.

**In progress right now:**
- `embed_gte_base_98k.py` running in the background — turning all 98,436
  documents into gte-base-en-v1.5 vectors, chunk by chunk. ETA ~3.5-4 hours
  from 17:31 UTC start.
- `tfidf_baseline.py` running in the background — the simple TF-IDF +
  logistic regression baseline. This one is much faster (no neural network),
  should finish in well under an hour.
- Once the embeddings finish, `gte_head_baseline.py` (already written, ready
  to go) trains the same kind of small classifier on top of the frozen
  vectors and scores it the same way.

I will fill in real numbers below as each piece finishes — no numbers are
being guessed or assumed ahead of the actual runs.

## 2026-09-20, update 1 — a real bug hit and fixed in the simple TF-IDF baseline

Writing this down right away instead of waiting until everything is done, in
case the session gets cut off.

**Bug 1**: First run of `tfidf_baseline.py` crashed while training the
classifier, with `ValueError: WRITEBACKIFCOPY base is read-only`. Cause:
scikit-learn's default way of training many small classifiers at once
(one per label, ~49+63 of them) uses multiple separate PROCESSES, and to
share the big TF-IDF word-count table between those processes cheaply, it
maps the data as READ-ONLY. But the actual math library underneath
(liblinear) needs to rearrange that data in place, which isn't allowed on
read-only data — so it crashed. Real, reproduced, not a guess.

**First fix attempt**: switched from separate processes to separate THREADS
(which share memory directly, no read-only copy needed). This avoided that
specific crash, but caused a NEW one instead — the whole program crashed
outright (a segfault, exit code 139) partway into training. Root cause:
liblinear's underlying C code isn't safe to run many-at-once inside threads
of the same program.

**Actual fix, now working**: turned off the "train many labels at once"
parallelism entirely for this one step — train the ~112 per-label
classifiers one after another instead of in parallel. Slower, but this is
supposed to be the "dead simple, cheap" baseline anyway, so correctness beats
speed here. Applied the same defensive fix to `gte_head_baseline.py` (the
script that will train on top of the embedding vectors once they're ready)
before running it, so we don't hit the same crash hours from now after a
multi-hour wait.

## 2026-09-20, update 2 — background jobs currently running, with settings and expected finish time

Writing this down BEFORE waiting on them, per the "log before you wait, not
just after" instruction, so nothing is only sitting in memory.

1. **`embed_gte_base_98k.py`** (background, started 17:31 UTC, PID 161222).
   Turns all 98,436 labeled documents into gte-base-en-v1.5 vectors (768
   numbers per document). Settings: batch_size=128, all 112 CPU threads,
   documents cut to the first 3,000 characters (same limit the earlier
   abandoned attempt used, chosen for speed on the very longest documents).
   Saves progress in 5,000-document chunks to
   `gte_embed_chunks/chunk_XXXXX.npy` so a crash partway through doesn't
   lose everything — the script skips any chunk file that already exists if
   restarted. Measured real speed on this box: ~7.4 documents/second.
   **Expected total time: roughly 3.5-4 hours** (finish around 21:00-21:30
   UTC). This is by far the slowest single piece of this whole comparison.
2. **`tfidf_baseline.py`** (background, restarted with the fix above). Fits
   a word-count table (TF-IDF, up to 100,000 word/word-pair features) on the
   training rows, then one logistic regression classifier per label, single
   process, single-threaded per classifier. Much faster than the embedding
   model — expected to finish in well under an hour total.
3. **Already measured, not waiting on anything**: I benchmarked the CURRENT
   FastText model's plain prediction speed (no training, just using the
   already-saved model file) directly, for the practicality comparison the
   user asked for: **~12,575 documents/second, single process, on this same
   box.** This number matters a lot for the final "is it worth it" verdict —
   see the reasoning-so-far note below.

**Reasoning so far on practicality (not a final verdict yet, numbers aren't
all in):** gte-base-en-v1.5's ~7.4 docs/sec is roughly 1,700x slower than
FastText's ~12,575 docs/sec just for turning text into a usable form — even
before adding the classifier step on top. On the eventual 1.2 billion
documents this project needs to eventually process, that gap is the
difference between about a day (FastText, spread across enough machines) and
literally years (gte-base) for a single machine's worth of throughput. So
even if the embedding model turns out more accurate, it would need to be
dramatically more accurate to be worth that cost — this is the same kind of
"a small accuracy win isn't worth 100x the cost" check the task asked me to
apply. Will confirm the actual accuracy numbers once the embedding job
finishes before drawing a final conclusion.

I am not stopping to wait idly — next, once these finish, I'll train the
classifier head on the embeddings and write up the full side-by-side
comparison table with real numbers for all three methods.

## 2026-09-20, update 3 — first real result in: TF-IDF on `domain`

TF-IDF + logistic regression, `domain` axis, same 83,671/14,765 train/val
split as everything else:
- micro precision: 0.956
- micro recall: 0.736
- **micro F1: 0.831**
- exact-label-set match: 0.729 (10,769/14,765)
- prediction speed: **92,542 documents/second**, single process, once
  trained (this is even faster than FastText's own ~12,575 docs/sec on this
  box — TF-IDF word-lookup is very cheap once the vectorizer is fitted)

This is a genuinely useful sanity-floor result already: the dead-simple
method (no neural network, no pretrained model) gets to F1=0.831 on
`domain`, compared to FastText's current baseline of F1=0.859 (or ~0.868
with the other builder's in-progress threshold-fallback fix — that number is
moving in parallel, so treating 0.859 as the safer "official" comparison
point for now). That's close — FastText is better but not by a huge margin.

## 2026-09-20, update 4 — TF-IDF baseline fully done, both axes

Full, final TF-IDF + logistic regression numbers (`tfidf_baseline_results.json`),
same 83,671 train / 14,765 val split as FastText:

| axis | precision | recall | F1 | exact-match | predict speed (single process) |
|---|---|---|---|---|---|
| domain | 0.956 | 0.736 | **0.831** | 0.729 | 92,542 docs/sec |
| task_family | 0.892 | 0.681 | **0.772** | 0.668 | 80,448 docs/sec |

Plain-English read of this: the simple method is NOT close to a false floor
— it's a real, respectable score, just clearly behind FastText (FastText:
domain 0.859, task_family 0.808 as of the last saved baseline). The pattern
in the errors is the same shape as FastText's: very high precision (when it
says a label, it's usually right) but noticeably lower recall (it misses
real labels more often) — this is a known property of TF-IDF word-matching:
it's good at "this document uses words strongly associated with category X"
but weaker than FastText's subword/n-gram model at catching paraphrased or
less keyword-heavy documents.

Interesting side-finding, not the main point but worth recording: TF-IDF's
own prediction speed (once trained) is actually FASTER than FastText's
(80-92k docs/sec vs FastText's ~12,575 docs/sec measured earlier) — TF-IDF
lookup + a linear model is about as cheap as it gets. So TF-IDF confirms
FastText is the right choice on ACCURACY, not on speed — speed was never
the risk for either of these two, both would be plenty fast at 1.2B-document
scale. The real speed risk is the embedding-model path, still running.

Now the only piece left is the gte-base-en-v1.5 embedding result. Still
waiting on the encoding job (see settings/ETA logged in update 2) — no
change to that estimate yet.

## 2026-09-20, update 5 — embedding job first chunk done, but slower than expected (explained)

First 5,000-document chunk finished: 1,304s = only 3.83 docs/sec, about HALF
my earlier benchmark's 7.4 docs/sec. Reason, most likely: that first chunk
was encoding WHILE the TF-IDF job (update 4) was also running full-tilt on
the same machine, competing for the same 112 CPU threads — two processes
both wanting all the cores at once roughly halves each one's real speed,
which matches the numbers almost exactly. The TF-IDF job has now finished,
so I expect the NEXT chunk to speed back up toward the un-contended ~7.4
docs/sec. Will confirm with the real number rather than assume.

At the current (contended, worst-case) 3.83 docs/sec, finishing all 98,436
documents would take ~406 more minutes (~6.8 hours) from now. If it speeds
back up to ~7.4 docs/sec as expected now that TF-IDF is done, that drops to
roughly 3.5-4 hours from now instead. Updating the log again once chunk 2's
real speed is in — not guessing which one is right.

## 2026-09-20, update 6 — checkpoint at 22:59 UTC: embedding job 61% done, real speed settled around ~3.1 docs/sec (not 7.4)

Session had a brief interruption (temporary hourly rate limit, now reset per
the coordinator) — the background embedding job was completely unaffected,
it's a separate OS process that kept running the whole time. Checked it
directly rather than assuming: still alive, healthy, no errors.

Real progress as of now: **12 of 20 chunks done (60,000 / 98,436 documents,
61%)**. Real measured overall rate has settled around **~3.0-3.2 docs/sec**
— noticeably slower than my early small-sample benchmark (7.4 docs/sec).
That first benchmark was not wrong, just not representative: it was a
400-document sample and ran once in isolation, while the real chunks are
each 5,000 documents with more natural variety in length, and the very
first chunk overlapped with the TF-IDF job competing for the same CPU
cores. Even after TF-IDF finished, the sustained rate stayed around 3 docs/
sec rather than climbing back to 7.4, so I'm treating ~3.1 docs/sec as the
real, trustworthy number now rather than the smaller early sample.

**Updated ETA: ~202 minutes (~3.4 hours) remaining from 22:59 UTC, i.e.
finishing around 02:20-02:30 UTC.** No errors, no crashes, no restarts
needed. Re-armed my progress-monitoring watchers after the interruption.
Nothing else to do but let it keep running — will log again at the next
real milestone (next monitor expiry or the job finishing).

**Small process note**: my own watcher scripts (the ones that were supposed
to tell me when the embedding job finishes) died along with the session
interruption — they were attached to my own session, unlike the actual
embedding job itself (`embed_gte_base_98k.py`, PID 161222) which is a fully
separate, detached process and was never at risk. To make the "tell me when
it's done" watcher itself survive a future interruption too, I relaunched it
two ways: one tracked normally (so I get an automatic notification), and one
fully detached with `nohup` (so even if the tracked one dies again, I can
still manually check `/tmp/embed_watch.log` for "EMBED_JOB_ENDED" and know
the real status). The actual training data (the embedding vectors being
saved chunk by chunk to `gte_embed_chunks/`) was never at risk either way —
only my own "please tell me" notification layer was fragile, not the work
itself.

## 2026-09-20, update 7 — validator flagged a real fairness gap; fixed it, and
caught a second, more serious bug while fixing it

The overnight validator agent messaged me: Builder 1 and the validator found
and fixed 2 real bugs in the training data tonight (3,209 documents had real
background text silently dropped before Gemma ever saw them; 783 documents
had the AI's own tool-call turn silently deleted) and FastText has already
been retrained on the fixed file
(`sft_output_sample_combined_98436_clean.jsonl`, current numbers: domain
F1=0.871, task_family F1=0.816). My TF-IDF run compared against the OLD,
not-yet-fixed file — not a fair fight. Fixing this now, and it changed a few
of my earlier "final" numbers in this log.

**What "clean" actually does, checked directly rather than assumed**: same
98,436 rows are all still there, but 4,003 of them got switched from
`ok: true` to `ok: false` (excluded from training, not deleted), and 1,330
rows got their document text (`full_text`) AND their Gemma labels corrected
(part of a separate truncation fix Builder 1 did earlier tonight). Confirmed
by direct comparison of every one of the 98,436 lines in both files, not
assumed.

**Re-ran TF-IDF against the fixed file** (`tfidf_baseline.py`'s DATA_PATH
updated, results below).

**Second, more serious bug caught while wiring up the embedding-model
comparison to the same fixed file — this one would have silently produced
WRONG numbers if I hadn't checked**: my classifier-head script
(`gte_head_baseline.py`) matches each document's text-vector (from the
overnight embedding job) to that document's row by COUNTING POSITION in the
file, after skipping any row marked `ok: false`. That's fine as long as the
SAME rows get skipped on both sides. But the embedding job counted positions
in the OLD file (where all 98,436 rows were still `ok: true` — the "clean"
fix didn't exist yet when it started), while the classifier-head script was
about to count positions in the NEW fixed file (where 4,003 rows are now
skipped). Skipping different rows on each side means position #50,000 in
one file is NOT the same document as position #50,000 in the other, once
you're past the first skipped row — so beyond the very first excluded
document, EVERY SINGLE downstream document-to-vector pairing would have
been silently wrong (each vector paired with the wrong label). This would
not have crashed or errored — it would have just quietly trained on
garbage-paired data and reported a real-looking but meaningless score.
Caught this by checking the actual counting logic before running anything,
not after seeing a suspicious number. **Fixed**: now using each row's true,
fixed position in the file (not its position after skipping bad rows) to
look up its vector, so a document's text-vector and its label always stay
correctly paired no matter which rows get excluded on either side.

Also planned (not yet run, waiting on the main embedding job to finish
first): a small top-up job (`patch_gte_embeddings_for_relabel.py`) that
re-computes vectors for just the 1,330 documents whose text actually
changed in the fix — cheap (~1,330 documents, a few minutes) compared to
redoing the whole multi-hour job. Everything else (the other ~97,000
documents) keeps its already-computed vector since their text didn't
change.

Replying to the validator now to confirm this is handled. (Done — validator
double-checked my numbers independently and confirmed them: 4,003 flipped
rows, 1,330 changed rows, 0 uid mismatches. Also confirmed the embedding
job itself was never at risk from the position-join bug, since it already
saves an explicit uid sidecar file — only my not-yet-run classifier-head
script had the risk, and it's fixed now.)

## 2026-09-20, update 8 — TF-IDF re-run on CLEAN data: final numbers, barely moved

| axis | precision | recall | F1 | exact-match | predict speed |
|---|---|---|---|---|---|
| domain | 0.954 | 0.738 | **0.832** | 0.732 | 78,962 docs/sec |
| task_family | 0.894 | 0.677 | **0.770** | 0.663 | 70,831 docs/sec |

Barely changed from the dirty-data run (domain 0.831->0.832, task_family
0.772->0.770) — expected, since only ~4% of rows were affected. Good
confirmation the earlier numbers weren't meaningfully distorted, but this is
now the correct, fair one to quote going forward.

**Current running scoreboard (domain / task_family F1, all on the CLEAN
94,433-doc file, all same 85/15 split, same scoring):**
| method | domain F1 | task_family F1 | predict speed |
|---|---|---|---|
| FastText (current, Builder 1's) | 0.871 | 0.816 | ~12,575 docs/sec |
| TF-IDF + logistic regression | 0.832 | 0.770 | ~71-79k docs/sec |
| gte-base-en-v1.5 + logistic regression | *(waiting on embedding job)* | *(waiting)* | dominated by ~3.1 docs/sec encoding |

Embedding job status as of 23:22 UTC: 13 of 20 chunks done (65,000/98,436,
66%), real sustained rate ~3.1-3.2 docs/sec, ETA roughly 176 more minutes
(~2:55 AM UTC finish). No errors. Still on track, just slow — this is
expected and already explained (CPU-only encoding of a 137-million
parameter model is inherently much slower per document than either FastText
or TF-IDF, which is exactly the practicality question this whole comparison
is meant to answer).

