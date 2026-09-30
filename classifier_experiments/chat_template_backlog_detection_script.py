import json, pyarrow.parquet as pq

d = json.load(open('/projects/data/datasets/code_data/sai_rupesh/taxonomy/output_column_inventory.json'))
no_adapter = [x for x in d if x['mode'] is None]
text_only = [x for x in no_adapter if any(c.lower()=='text' for c in x['cols'])]

# Ordered so more-specific markers are checked first where formats could overlap.
MARKER_STYLES = [
    ("chatml", ["<|im_start|>"]),
    ("llama3_header", ["<|start_header_id|>"]),
    ("zephyr_style", ["<|user|>", "<|assistant|>", "<|system|>"]),
    ("llama2_inst", ["[INST]", "[/INST]"]),
    ("markdown_human_assistant", ["### Human:", "### Assistant:", "### human:", "### assistant:"]),
]

results = []
for x in text_only:
    path = x['path']
    try:
        pf = pq.ParquetFile(path)
        tbl = pf.read_row_group(0) if pf.num_row_groups else pf.read()
        n = min(3, tbl.num_rows)
        sample_rows = tbl.slice(0, n).to_pylist()
        matched_style = None
        for r in sample_rows:
            t = str(r.get('text') or r.get('Text') or '')
            for style, markers in MARKER_STYLES:
                if any(m in t for m in markers):
                    matched_style = style
                    break
            if matched_style:
                break
        if matched_style:
            results.append({"path": path, "rows": x['rows'], "template_style": matched_style})
    except Exception as e:
        pass

print(json.dumps(results, indent=1))
print(f"TOTAL: {len(results)} files, {sum(r['rows'] for r in results)} rows", flush=True)
from collections import Counter
print(Counter(r['template_style'] for r in results))
