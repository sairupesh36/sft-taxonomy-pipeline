"""Full-corpus keyword scan for the weak/rare domain+task_family classes (see weak_classes.json).
Cheap raw-text prefilter (one combined regex) on every line of the CLEAN corpus (en, uncertain,
no_natural_language -- 517M rows), only json.loads + build full_text on an actual match. Caps
candidates per label (both a per-worker soft cap to bound memory, and a final global trim), and
excludes anything already present (by content hash) in the labeled sources already used for
training, so nothing gets sent to Gemma twice."""
import sys, os, glob, json, re, random, hashlib, collections, time
from multiprocessing import Pool
sys.path.insert(0, "/projects/data/datasets/code_data/sai_rupesh/taxonomy")
sys.path.insert(0, "/projects/data/datasets/code_data/sai_rupesh/taxonomy/classifier_experiments")
from keywords_weak_categories import DOMAIN_KEYWORDS, TASK_FAMILY_KEYWORDS
from keywords_weak_categories_v2 import DOMAIN_KEYWORDS_V2, TASK_FAMILY_KEYWORDS_V2
from sft_classify_domain_task import to_full_text

ROOT = "/projects/data/datasets/code_data/sai_rupesh/taxonomy/sft_43_language_wise_clean"
W = "/projects/data/datasets/code_data/sai_rupesh/taxonomy/lc_work"
CE = "/projects/data/datasets/code_data/sai_rupesh/taxonomy/classifier_experiments"
CAP_PER_WORKER = 12
GLOBAL_CAP = 350

weak = json.load(open(f"{W}/weak_classes.json"))
target_domains = set(weak["domain_master"]) - {"cross_domain"}
target_tasks = set(weak["task_family_master"])
ALL_D = {**DOMAIN_KEYWORDS, **DOMAIN_KEYWORDS_V2}
ALL_T = {**TASK_FAMILY_KEYWORDS, **TASK_FAMILY_KEYWORDS_V2}

PHRASE_TO_LABELS = collections.defaultdict(list)
for label in target_domains:
    for p in ALL_D.get(label, []):
        PHRASE_TO_LABELS[p.lower()].append(("domain", label))
for label in target_tasks:
    for p in ALL_T.get(label, []):
        PHRASE_TO_LABELS[p.lower()].append(("task_family", label))
PHRASES = list(PHRASE_TO_LABELS)
print(f"tracking {len(target_domains)} domain + {len(target_tasks)} task_family labels, "
      f"{len(PHRASES)} phrases total", flush=True)
# NOTE: a single re.compile("|".join(...)) over 620 literal alternatives was tried first and was
# catastrophically slow (benchmarked: didn't even finish 50,000 lines in 120s on a real file --
# Python's re engine does not handle a large literal alternation well, especially combined with
# re.IGNORECASE, against the long lines this corpus has). Plain `phrase in lowered_line` in a loop
# (CPython's C-optimized substring search) benchmarked at 2,575 lines/sec on a typical real file --
# ~6-20x faster -- so that's what scan_file() below actually uses.

def already_labeled_hashes():
    seen = set()
    for fn in ("sft_output_sample_combined_98436_clean.jsonl", "sft_output_sample_weak_categories_v2_mapped.jsonl"):
        p = f"{CE}/{fn}"
        if os.path.exists(p):
            for line in open(p):
                d = json.loads(line)
                if d.get("full_text"): seen.add(hashlib.md5(d["full_text"].encode()).hexdigest())
    for fn in ("labeled_A.jsonl", "labeled_B.jsonl", "labeled_T.jsonl"):
        p = f"{W}/{fn}"
        if os.path.exists(p):
            for line in open(p):
                d = json.loads(line)
                if d.get("full_text"): seen.add(hashlib.md5(d["full_text"].encode()).hexdigest())
    return seen

_SEEN = None                                                  # loaded ONCE per worker process (via Pool initializer), not once per file --
                                                                # the original per-file reload was the reason the first run never printed
                                                                # a single progress line in 11+ minutes: every one of 4928 files was paying
                                                                # the full cost of re-reading and re-hashing >100k already-labeled docs.

def _init_worker():
    global _SEEN
    _SEEN = already_labeled_hashes()

def scan_file(path):
    seen = _SEEN
    counts = collections.Counter(); out = []
    with open(path, "rb") as f:
        for line in f:
            low = line.decode("utf-8", "ignore").lower()
            matched = [ph for ph in PHRASES if ph in low]
            if not matched:
                continue
            need = [(ax, lb) for ph in matched for ax, lb in PHRASE_TO_LABELS[ph] if counts[(ax, lb)] < CAP_PER_WORKER]
            if not need:
                continue
            try:
                row = json.loads(line)
                text = to_full_text(row["messages"])
            except Exception:
                continue
            h = hashlib.md5(text.encode()).hexdigest()
            if h in seen:
                continue
            seen.add(h)
            for ax, lb in need:
                counts[(ax, lb)] += 1
            out.append({"h": h, "full_text": text, "hit_labels": [f"{ax}:{lb}" for ax, lb in need], "src_file": os.path.relpath(path, ROOT)})
    return out

if __name__ == "__main__":
    files = sorted(glob.glob(ROOT + "/*/*.jsonl"))
    files = [f for f in files if os.path.getsize(f) > 0]
    print(f"{len(files)} non-empty files to scan", flush=True)
    t0 = time.time(); by_label = collections.defaultdict(list); n_done = 0
    with Pool(96, initializer=_init_worker) as p:
        for out in p.imap_unordered(scan_file, files, chunksize=2):
            n_done += 1
            for r in out:
                for lab in r["hit_labels"]:
                    if len(by_label[lab]) < GLOBAL_CAP:
                        by_label[lab].append(r)
            if n_done % 300 == 0:
                el = time.time() - t0
                covered = sum(1 for k, v in by_label.items() if len(v) >= 3)
                print(f"[{n_done}/{len(files)}] {el/60:.1f} min | labels with >=3 candidates so far: {covered}/{len(target_domains)+len(target_tasks)}", flush=True)
    all_rows = {}
    for lab, rows in by_label.items():
        for r in rows: all_rows[r["h"]] = r
    print(f"\nDONE {time.time()-t0:.0f}s | unique candidate documents: {len(all_rows)}")
    empty = [lab for lab in [f"domain:{d}" for d in target_domains] + [f"task_family:{t}" for t in target_tasks] if len(by_label.get(lab, [])) == 0]
    print(f"labels with ZERO candidates found anywhere in the corpus: {len(empty)}"); print(" ", empty)
    for lab in sorted(by_label, key=lambda k: len(by_label[k])):
        print(f"  {lab:42s} {len(by_label[lab]):>4d} candidates")
    json.dump(list(all_rows.values()), open(f"{W}/weak_candidates.json", "w"), ensure_ascii=False)
    print(f"saved {len(all_rows)} candidate docs -> lc_work/weak_candidates.json")
