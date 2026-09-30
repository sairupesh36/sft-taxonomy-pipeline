"""
Shared prediction helper for all fasttext classifier scripts in this folder
(fasttext_baseline.py, fasttext_optuna.py -- and fasttext_optuna_parallel.py
via its import of fasttext_optuna's score()). One place to fix the
multi-label zero-prediction gap so retraining, tuning, and eval all agree.

Root cause and verification: see fix_multilabel_zero_predictions.py in this
folder. Summary: a single global probability threshold can leave a genuinely
multi-label document (e.g. 2 true domains) with NO label clearing threshold
if the model splits confidence between the two candidates. Since every real
training/eval document has >=1 true label (empty-label rows are filtered out
before training), a zero-label prediction is always wrong. Measured on the
real 98,436-doc validation split (domain_fasttext.bin / task_family_fasttext.bin,
threshold=0.5): 13.6-13.8% of val docs got zero predicted labels before this
fix.

**2026-09-20 update, checked carefully per the task's own warning ("don't
just lower the threshold blindly, verify it doesn't tank precision on
single-label docs")**: an earlier version of this fix did an UNCONDITIONAL
top-1 fallback (always guess the best single label when nothing clears
threshold). Measured its real effect split by true label count, and found a
real, non-trivial cost: on single-label val docs, precision dropped from
0.928->0.862 (domain) and 0.868->0.815 (task_family) -- because on the
zero-prediction subset, the model's top-1 guess is only right 44-47% of the
time, so guessing unconditionally trades a guaranteed miss (0 precision
contribution either way) for a coin-flip-ish guess. Overall F1 still went up
very slightly (+0.001 domain, +0.008 task_family) because recall gains
outweighed the precision loss in aggregate, but "barely positive with a real
precision cost" is not a clean win.

**Fix: added a minimum-confidence FLOOR (`min_fallback_score`, default
0.10) on the fallback guess.** If nothing clears the real `threshold`, only
fall back to the top-1 guess if its own score is at least the floor;
otherwise still return nothing (the model is too unsure even to guess).
Swept floor values 0.0-0.30 on the real validation sets: 0.10 sits right at
the best F1 for both axes (domain 0.858->0.868, task_family 0.808->0.819)
while cutting the single-label precision damage roughly in half (domain
0.928->0.909, task_family 0.868->0.845 -- vs 0.862/0.815 with no floor) and
actually IMPROVING single-label F1 over the no-fallback baseline (domain
0.874->0.882, task_family 0.826->0.835), not just avoiding harm. This is a
strictly better version of the same idea, not a different idea -- kept as
the shared default so `fasttext_baseline.py`/`fasttext_optuna.py`/
`fasttext_optuna_parallel.py` all agree.
"""


def predict_with_fallback(model, text, threshold, min_fallback_score=0.10):
    """Returns a set of predicted label strings (no '__label__' prefix).
    Falls back to the single top-scoring label when nothing clears
    `threshold`, but only if that top guess itself scores at least
    `min_fallback_score` -- below that, the model is too unsure to guess and
    an empty set is still returned. See the module docstring for why the
    floor matters (a floor of 0 measurably hurts single-label precision for
    only a negligible F1 gain over floor=0.10).
    """
    pred_labels, _ = model.predict(text, k=-1, threshold=threshold)
    if pred_labels:
        return {p.replace("__label__", "") for p in pred_labels}
    top_labels, top_scores = model.predict(text, k=1)
    if top_scores[0] >= min_fallback_score:
        return {p.replace("__label__", "") for p in top_labels}
    return set()


def predict_capped(model, text, threshold, min_fallback_score=0.10, max_labels=2):
    """Same as predict_with_fallback, but caps the result at `max_labels` (default 2), matching the
    1-2 label cap the ground-truth Gemma mapper (taxonomy_map_document.validate_result) already
    enforces. predict_with_fallback has NO cap: with loss="ova" every label gets its own
    independent score, so nothing stops several (or, on real production output, checked directly:
    occasionally 10-53) labels all clearing threshold on one document at once -- almost always a
    genuine failure, not a genuinely 10-label document. fastText's own predict() already returns
    labels sorted by descending score when k=-1, so keeping the first `max_labels` keeps the
    model's own best guesses, not an arbitrary subset. Returns a LIST (order = confidence, best
    first), not a set, since order matters here and didn't for the uncapped function's callers.
    """
    pred_labels, pred_scores = model.predict(text, k=-1, threshold=threshold)
    if pred_labels:
        return [p.replace("__label__", "") for p in pred_labels[:max_labels]]
    top_labels, top_scores = model.predict(text, k=1)
    if top_scores[0] >= min_fallback_score:
        return [p.replace("__label__", "") for p in top_labels]
    return []
