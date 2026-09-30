"""
Independent scan (read-only, never writes to OUTPUT) for the "no_adapter"
files that actually DO hold real, complete conversations -- just rendered
as one long chat-template string in a single text-like column, a format
sft_normalize.py's plan()/to_messages() doesn't recognize at all (mode=None,
whole file skipped). Cross-checked against the validator agent's
independent finding of the same thing (51 files, ~10M rows) rather than
built blind.

Detects 4 real template flavors by literal marker text (checked against
real file content before trusting each pattern, not guessed):
  - chatml:     <|im_start|>role ... <|im_end|>
  - llama3:     <|start_header_id|>role<|end_header_id|> ... <|eot_id|>
  - inst:       [INST] ... [/INST]  (Llama-2/Mistral instruct style)
  - human_asst: ### Human: ... ### Assistant: ...  (Alpaca/Vicuna style)
  - zephyr:     <|system|>/<|user|>/<|assistant|>  (Zephyr/TinyLlama style)
"""
import os
import re
import sys
import glob
import json

sys.path.insert(0, "/projects/data/datasets/translation_data/SFT/DEDUP_PIPELINE")
import sft_normalize as N
import pyarrow.parquet as pq

OUTPUT_DIR = "/projects/data/datasets/translation_data/SFT/OUTPUT"

MARKERS = {
    "chatml": re.compile(r"<\|im_start\|>\s*(system|user|assistant)", re.I),
    "llama3": re.compile(r"<\|start_header_id\|>\s*(system|user|assistant)", re.I),
    "inst": re.compile(r"\[INST\].*?\[/INST\]", re.S),
    "human_asst": re.compile(r"###\s*Human\s*:.*?###\s*Assistant\s*:", re.S | re.I),
    "zephyr": re.compile(r"<\|(?:system|user|assistant)\|>", re.I),
}

# text-like columns worth checking (same "unrecognized single free-text column" shape)
TEXT_COL_CANDIDATES = ["text", "conversation", "content", "transcript", "chat", "dialogue"]


def detect_template(sample_texts):
    counts = {k: 0 for k in MARKERS}
    for t in sample_texts:
        if not isinstance(t, str):
            continue
        for name, pat in MARKERS.items():
            if pat.search(t):
                counts[name] += 1
    # require at least 2/N samples matching to call it real (not one fluke row)
    hits = {k: v for k, v in counts.items() if v >= 2}
    if not hits:
        return None
    return max(hits, key=hits.get)


def scan_file(path):
    try:
        pf = pq.ParquetFile(path)
        cols = list(pf.schema_arrow.names)
    except Exception:
        return None
    p = N.plan(cols)
    if p["mode"] is not None:
        pf.close()
        return None  # already recognized, not one of the invisible ones

    text_col = next((c for c in TEXT_COL_CANDIDATES if c in cols), None)
    if text_col is None:
        pf.close()
        return None

    samples = []
    try:
        for batch in pf.iter_batches(batch_size=20):
            rows = batch.to_pylist()
            for r in rows:
                v = r.get(text_col)
                if v:
                    samples.append(v)
            break
    except Exception:
        pf.close()
        return None
    finally:
        pf.close()

    if not samples:
        return None
    template = detect_template(samples)
    if template is None:
        return None
    return {"path": path, "text_col": text_col, "template": template, "n_cols": len(cols)}


def main():
    domains = sorted([d for d in os.listdir(OUTPUT_DIR) if os.path.isdir(os.path.join(OUTPUT_DIR, d))])
    found = []
    n_scanned = 0
    for d in domains:
        files = glob.glob(os.path.join(OUTPUT_DIR, d, "*.parquet"))
        for f in files:
            n_scanned += 1
            result = scan_file(f)
            if result:
                result["domain"] = d
                found.append(result)
                print(f"  FOUND [{result['template']:12s}] {os.path.relpath(f, OUTPUT_DIR)}", flush=True)
        if n_scanned % 200 == 0:
            print(f"...scanned {n_scanned} files so far, {len(found)} matches", flush=True)

    print(f"\nDONE: scanned {n_scanned} files, found {len(found)} real chat-template files")
    by_template = {}
    for r in found:
        by_template.setdefault(r["template"], []).append(r["path"])
    for tmpl, paths in by_template.items():
        print(f"  {tmpl}: {len(paths)} files")

    with open("chat_template_files_found.json", "w") as f:
        json.dump(found, f, indent=2)
    print("wrote chat_template_files_found.json")


if __name__ == "__main__":
    main()
