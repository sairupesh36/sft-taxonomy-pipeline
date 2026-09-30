# SFT clean data — stats

Data: `sft_43_language_wise_clean`. Built by `sft_clean_filter.py` from the raw
`sft_43_language_wise` folder. All numbers below come from actually scanning
the finished output (`lc_work/final_verify.py`, `lc_work/turn_shape_scan.py`,
`lc_work/turn_shape_no_tool.py`, `lc_work/tools_scan.py`, `_filter_report.json`),
not estimates.

## 1. Overall

| | Rows |
|---|---|
| Read in | 562,568,787 |
| Kept | 521,757,429 |
| Removed | 40,811,358 |
| **Kept %** | **92.75%** |

## 2. Why rows were removed (top reasons)

| Reason | Rows | What it means |
|---|---:|---|
| exact_duplicate | 21,247,570 | The exact same line appeared before |
| missing_context_placeholder | 14,000,581 | Question points at an `<image>`/`<DNA>`/etc. tag with nothing real behind it |
| prompt_missing_payload | 1,904,801 | Says "answer from the following text:" but the text is missing |
| think_unbalanced | 709,244 | A `<think>` tag opened but never closed, or the reverse |
| control_or_replacement_char | 463,530 | Broken characters (`�`, control bytes) |
| special_token_in_text | 379,118 | A chat-template control string (`<\|im_start\|>` etc.) leaked into plain text |
| nested_conversation_in_prompt | 377,981 | A whole chat pasted inside one message |
| think_no_answer | 357,989 | Model thought but never gave a final answer |
| tool_result_without_call | 323,185 | A tool result with no matching tool call |
| repetition_loop | 273,871 | Text stuck repeating itself |
| think_junk | 208,880 | Thinking block has only dots/punctuation |
| tool_response_missing | 117,164 | Tool was called but never got a result |
| placeholder_answer | 87,692 | Answer is just a placeholder |
| think_empty | 85,178 | Nothing between the think tags |
| missing_context_cutoff | 59,877 | Question cuts off mid-sentence and the answer says so |
| single_char_run | 41,157 | Same character repeated 25+ times |
| seq_invalid | 38,036 | Roles out of order, or the row doesn't end on a plain answer |
| multimodal_image_part | 28,583 | An image part (text-only data, so dropped) |
| dot_run | 26,290 | Long run of dots |
| tool_call_unbalanced | 24,789 | Broken `<tool_call>` tags |
| answer_equals_question | 22,936 | Answer just repeats the question |
| too_long | 14,023 | Over 200,000 characters |
| tool_call_dropped | 10,749 | Assistant says it's calling a function, but the call is missing |
| tool_call_unparseable | 4,731 | Tool-call JSON couldn't be read |
| empty_content | 2,669 | Empty message |
| tool_mismatch | 729 | Tool result doesn't match its call |
| internal_error | 5 | Unexpected error |

## 3. What was repaired, not removed

| Repair | Rows |
|---|---:|
| hermes_tool_calls_converted | 2,411,938 |
| code_tool_call_converted | 1,695,369 |
| python_tool_schema_attached | 1,663,151 |
| answer_tag_unwrapped | 1,725,202 |
| special_token_end_stripped | 1,545,357 |
| duplicate_think_tag_collapsed | 3,113,089 |
| repr_parts_joined | 826,835 |
| nested_think_repaired | 642,577 |
| empty_system_dropped | 536,208 |
| repr_conversation_unwrapped | 331,189 |
| pii_email_replaced | 295,382 |
| pii_ip_replaced | 56,947 |

## 4. Conversation shape (521.76M rows, every row checked)

| Shape | Rows | Share |
|---|---:|---:|
| Single turn (1 question, 1 answer) | 512,761,071 | 98.28%* |
| Multi-turn (2+ back-and-forth user turns) | 6,741,835 | 1.29%* |
| Has at least one tool call | 2,254,523 | 0.43% |

\*percentages of the 519,502,906 rows that have no tool call.

Inside the multi-turn rows: 4.36M have 2 user turns, 1.19M have 3, 0.41M have
4, 0.79M have 5 or more.

Tool-calling rows are not a separate shape — they sit inside either bucket:
488,647 are single-turn (one question, then tool calls, then the answer);
about 1.77M are multi-turn (several user turns, tools called somewhere in
between).

## 5. Tool-calling rows (2,254,523 rows, 0.43% of the data)

| | Rows |
|---|---:|
| Have a `tools` schema field | 2,015,333 |
| No `tools` schema field, but every called name appears in the system/user text | 294,475 |
| No `tools` schema field, and it's a `python` code-execution call | 1,663,151 → now has a schema attached |
| No `tools` schema field, and the called tool is genuinely undefined anywhere | 8,122 |

Most common undefined tool names (each below the 3-instance pattern bar,
scattered across many different tasks): `respond_user`, `sql_query`,
`get_descriptions`, `get_table_info`, `get_stock_price`, `get_movie_details`,
`calculator`, `convert_currency`, `search_recipes`.

## 6. Format check (every one of the 521.76M rows scanned, not a sample)

| Check | Result |
|---|---:|
| Strict OpenAI format violations | 5 rows |
| Rows with a leftover `<tool_call>` text tag | 2 |
| Rows with a leftover missing-context cut-off | 2 |
| Rows with an empty `<think></think>` | 1 |

That is 5 bad rows out of 521,757,429 — all in the `en` folder.

## 7. Chat-template test (real rows, all 34 language folders)

Tested on Qwen2.5, Qwen3, Llama-3.1, GLM-4.5, GLM-4.6, GLM-4-9b-chat.

| | Result |
|---|---|
| General rows (2,698 sampled across every folder) | 100% render correctly on every model |
| Assistant part splits out cleanly (needed for loss masking) | 100% |
| Chat-template control tokens leaking into visible content | 0% |
| Tool-calling rows, `arguments` as JSON string | GLM-4.5/4.6 fail to render (need `arguments` as a dict) |
| Tool-calling rows, `arguments` converted to a dict | 100% render on every model tested |
| Llama-3.1 only: rows with 2+ tool calls in one turn | fails (Llama-3.1's own template limit, not a data problem) |

## 8. Where the data lives (by language, top rows)

| Folder | Rows | Share | Size |
|---|---:|---:|---:|
| en | 320,885,870 | 61.50% | 874.6 GB |
| no_natural_language | 144,384,821 | 27.67% | 20.6 GB |
| uncertain | 52,158,176 | 10.00% | 33.9 GB |
| zh | 1,396,422 | 0.27% | 5.1 GB |
| hi | 761,918 | 0.15% | 1.3 GB |
| vi | 685,641 | 0.13% | 0.6 GB |
| kn | 353,756 | 0.07% | 0.6 GB |
| fr | 233,539 | 0.04% | 0.5 GB |
| es | 201,639 | 0.04% | 0.5 GB |
| pt | 184,500 | 0.04% | 0.3 GB |
| ar | 130,537 | 0.03% | 0.2 GB |
| bn | 55,548 | 0.01% | 0.1 GB |
| + 22 more small language folders | | | |

`no_natural_language` and `uncertain` together are 37.67% of all rows — these
are mostly code, data tables, and DNA/protein sequences, which fastText's
language detector can't confidently tag with a human-language code, not junk.

## 9. Known repetitive source (not a bug, just worth knowing before training)

Protein-family generation prompts (`[Generate by superfamily]\n\nSuperfamily=<...>`)
repeat hundreds of thousands of times with different real answers. Top 15
prompts alone account for 4.38M rows. Consider capping repeats per exact
prompt when you build your training mix.

## 10. Two things your training code must still do

1. Convert tool-call `arguments` from a JSON string to a dict
   (`json.loads(arguments)`) — GLM breaks otherwise, Qwen2.5 double-encodes.
2. Set a maximum sequence length and compute loss on assistant tokens only.

## 11. Not covered by any filter here

- Near-duplicate removal (deliberately not done — no global dedup, per project decision)
- Benchmark contamination
- Question/answer mismatches
- Personal names and addresses (only emails and public IPs are handled)
