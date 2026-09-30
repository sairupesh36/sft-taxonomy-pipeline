# hf-agentic-data -- final stats

- files: 69
- rows: 36,327,623
- size: 256.0 GB
- rows with a `tools` list: 844,975
- total tool calls: 2,406,444

## Stage 1 -- convert
- source rows read: 37,286,501, rows written: 36,329,213
- datasets converted: 38, skipped: 40

## Stage 4 -- cleanup
- {'unchanged': 36327297, 'trimmed': 326}

## Stage 3 -- final audit
- rows audited: 36,327,623
- violations: {'null_in_args': 8695, 'tool_without_call': 3972}

## Stage 5 -- GLM template check
- sampled: 1380, fully OK: 1380
- problems: {}

## Per file

| file | rows | GB | rows with tools | avg msgs | tool calls |
|---|---:|---:|---:|---:|---:|
| HuggingFaceTB__cosmopedia_1.jsonl | 1,000,000 | 6.29 | 0 | 2.0 | 0 |
| HuggingFaceTB__cosmopedia_10.jsonl | 1,000,000 | 5.62 | 0 | 2.0 | 0 |
| HuggingFaceTB__cosmopedia_11.jsonl | 1,000,000 | 5.63 | 0 | 2.0 | 0 |
| HuggingFaceTB__cosmopedia_12.jsonl | 1,000,000 | 5.63 | 0 | 2.0 | 0 |
| HuggingFaceTB__cosmopedia_13.jsonl | 1,000,000 | 5.62 | 0 | 2.0 | 0 |
| HuggingFaceTB__cosmopedia_14.jsonl | 1,000,000 | 5.13 | 0 | 2.0 | 0 |
| HuggingFaceTB__cosmopedia_15.jsonl | 1,000,000 | 4.58 | 0 | 2.0 | 0 |
| HuggingFaceTB__cosmopedia_16.jsonl | 1,000,000 | 4.50 | 0 | 2.0 | 0 |
| HuggingFaceTB__cosmopedia_17.jsonl | 1,000,000 | 4.33 | 0 | 2.0 | 0 |
| HuggingFaceTB__cosmopedia_18.jsonl | 1,000,000 | 4.33 | 0 | 2.0 | 0 |
| HuggingFaceTB__cosmopedia_19.jsonl | 1,000,000 | 4.33 | 0 | 2.0 | 0 |
| HuggingFaceTB__cosmopedia_2.jsonl | 1,000,000 | 5.64 | 0 | 2.0 | 0 |
| HuggingFaceTB__cosmopedia_20.jsonl | 1,000,000 | 4.33 | 0 | 2.0 | 0 |
| HuggingFaceTB__cosmopedia_21.jsonl | 1,000,000 | 4.90 | 0 | 2.0 | 0 |
| HuggingFaceTB__cosmopedia_22.jsonl | 1,000,000 | 5.74 | 0 | 2.0 | 0 |
| HuggingFaceTB__cosmopedia_23.jsonl | 1,000,000 | 5.74 | 0 | 2.0 | 0 |
| HuggingFaceTB__cosmopedia_24.jsonl | 1,000,000 | 5.74 | 0 | 2.0 | 0 |
| HuggingFaceTB__cosmopedia_25.jsonl | 1,000,000 | 5.73 | 0 | 2.0 | 0 |
| HuggingFaceTB__cosmopedia_26.jsonl | 1,000,000 | 5.74 | 0 | 2.0 | 0 |
| HuggingFaceTB__cosmopedia_27.jsonl | 1,000,000 | 5.73 | 0 | 2.0 | 0 |
| HuggingFaceTB__cosmopedia_28.jsonl | 1,000,000 | 5.73 | 0 | 2.0 | 0 |
| HuggingFaceTB__cosmopedia_29.jsonl | 1,000,000 | 5.73 | 0 | 2.0 | 0 |
| HuggingFaceTB__cosmopedia_3.jsonl | 1,000,000 | 5.63 | 0 | 2.0 | 0 |
| HuggingFaceTB__cosmopedia_30.jsonl | 1,000,000 | 5.74 | 0 | 2.0 | 0 |
| HuggingFaceTB__cosmopedia_31.jsonl | 1,000,000 | 5.71 | 0 | 2.0 | 0 |
| HuggingFaceTB__cosmopedia_4.jsonl | 1,000,000 | 5.63 | 0 | 2.0 | 0 |
| HuggingFaceTB__cosmopedia_5.jsonl | 1,000,000 | 5.63 | 0 | 2.0 | 0 |
| HuggingFaceTB__cosmopedia_6.jsonl | 1,000,000 | 5.62 | 0 | 2.0 | 0 |
| HuggingFaceTB__cosmopedia_7.jsonl | 1,000,000 | 5.63 | 0 | 2.0 | 0 |
| HuggingFaceTB__cosmopedia_8.jsonl | 1,000,000 | 5.63 | 0 | 2.0 | 0 |
| HuggingFaceTB__cosmopedia_9.jsonl | 1,000,000 | 5.63 | 0 | 2.0 | 0 |
| allenai__WildChat-1M.jsonl | 837,988 | 5.29 | 0 | 4.7 | 0 |
| WaltonFuture__agentic-sft-new.jsonl | 640,354 | 43.90 | 0 | 30.9 | 0 |
| allenai__WildChat.jsonl | 529,426 | 2.67 | 0 | 4.7 | 0 |
| HuggingFaceH4__ultrachat_200k.jsonl | 515,281 | 2.60 | 0 | 5.2 | 0 |
| Agent-Ark__Toucan-1.5M_1.jsonl | 513,384 | 12.28 | 513,384 | 10.9 | 1,489,693 |
| allenai__SciRIFF.jsonl | 432,740 | 3.22 | 0 | 2.0 | 0 |
| camel-ai__code.jsonl | 367,006 | 0.40 | 0 | 2.0 | 0 |
| Agent-Ark__Toucan-1.5M_2.jsonl | 236,839 | 4.12 | 236,839 | 9.8 | 675,441 |
| deepmind__aqua_rat.jsonl | 195,950 | 0.10 | 0 | 2.0 | 0 |
| allenai__ultrafeedback_binarized_cleaned.jsonl | 186,053 | 0.39 | 0 | 2.0 | 0 |
| ibm__duorc.jsonl | 170,666 | 1.00 | 0 | 2.0 | 0 |
| glaiveai__glaive-function-calling-v2.jsonl | 112,960 | 0.25 | 78,362 | 6.5 | 83,827 |
| ise-uiuc__Magicoder-Evol-Instruct-110K.jsonl | 111,183 | 0.26 | 0 | 2.0 | 0 |
| ise-uiuc__Magicoder-OSS-Instruct-75K.jsonl | 75,197 | 0.18 | 0 | 2.0 | 0 |
| Kwai-Klear__SWE-smith-mini_swe_agent_plus-trajectories-66k.jsonl | 65,994 | 4.90 | 0 | 69.6 | 0 |
| HuggingFaceTB__cosmopedia_32.jsonl | 64,744 | 0.35 | 0 | 2.0 | 0 |
| PersonalAILab__AFM-CodeAgent-SFT-Dataset.jsonl | 59,939 | 0.57 | 0 | 2.0 | 0 |
| Solaris99__AgentBank.jsonl | 53,205 | 0.12 | 0 | 9.2 | 0 |
| internlm__Agent-FLAN.jsonl | 34,440 | 0.21 | 0 | 16.6 | 0 |
| HuggingFaceH4__no_robots.jsonl | 20,000 | 0.03 | 0 | 2.4 | 0 |
| AgentGym__AgentTraj-L.jsonl | 14,485 | 0.09 | 0 | 17.6 | 0 |
| Nanbeige__ToolMind.jsonl | 12,882 | 0.24 | 12,882 | 16.3 | 66,416 |
| EleutherAI__hendrycks_math.jsonl | 12,000 | 0.01 | 0 | 2.0 | 0 |
| deepmind__code_contests.jsonl | 11,377 | 0.04 | 0 | 2.0 | 0 |
| Lite-Coder__LiteCoder-Terminal-SFT.jsonl | 11,255 | 1.19 | 0 | 52.7 | 0 |
| groundhogLLM__ACC-dataset.jsonl | 10,802 | 2.69 | 0 | 2.0 | 0 |
| PersonalAILab__AFM-MHQA-Agent-SFT-Dataset.jsonl | 8,826 | 0.10 | 0 | 3.0 | 0 |
| argilla__dpo-mix-7k.jsonl | 7,500 | 0.02 | 0 | 3.3 | 0 |
| TIGER-Lab__SWE-Next-SFT-Trajectories.jsonl | 3,692 | 0.22 | 0 | 55.5 | 0 |
| R2E-Gym__R2EGym-SFT-Trajectories.jsonl | 3,231 | 0.16 | 0 | 33.2 | 0 |
| Lite-Coder__LiteCoder-SFT-Terminal-preview.jsonl | 1,880 | 0.18 | 0 | 43.9 | 0 |
| Exgentic__agent-llm-traces.jsonl | 1,780 | 0.29 | 1,157 | 45.0 | 32,184 |
| FinGPT__fingpt-forecaster-dow30-202305-202405.jsonl | 1,530 | 0.01 | 0 | 3.0 | 0 |
| TIGER-Lab__SWE-QA-Pro-SFT-Trajectories.jsonl | 1,000 | 0.07 | 1,000 | 48.9 | 22,963 |
| agent-data__misc-merged-claude-code-traces-v1.jsonl | 895 | 0.22 | 860 | 62.0 | 27,141 |
| SWE-Gym__OpenHands-Sampled-Trajectories.jsonl | 491 | 0.03 | 491 | 39.9 | 8,779 |
| AlicanKiraz0__Agentic-Chain-of-Thought-Coding-SFT-Dataset.jsonl | 429 | 0.01 | 0 | 3.0 | 0 |
| Jarrodbarnes__tau2-sft-v4-dataset.jsonl | 219 | 0.00 | 0 | 5.3 | 0 |

## Skipped datasets

- **AgentGym__AgentGym-RL-Data-ID** -- item ids only
- **BlueZeros__AgentEHR-Bench** -- EHR tables, no conversations
- **BytedTsinghua-SIA__CUDA-Agent-Ops-6K** -- op specs + reference code, no prompt/response
- **DeepNLP__Coding-Agent-Github-2025-Feb** -- repo listing metadata
- **DeepNLP__ai-agent-teacher** -- web listing metadata
- **DeepNLP__mcp-servers** -- web listing metadata
- **Hcompany__WebClick** -- image grounding data
- **HuggingFaceH4__MATH-500** -- benchmark (test set)
- **HuggingFaceH4__mt_bench_prompts** -- benchmark (test set)
- **HuggingFaceH4__ultrafeedback_binarized** -- same UltraFeedback prompts as allenai__ultrafeedback_binarized_cleaned (kept)
- **II-Vietnam__Agentic-Multi-SWE-RL** -- RL environment specs, no trajectories
- **JasperHaozhe__AgentGen-Bench** -- benchmark manifest (no conversations)
- **Lakera__b3-agent-security-benchmark-weak** -- benchmark (security eval)
- **McGill-NLP__WebLINX** -- split index only (1 row)
- **Multi-Agent-LLMs__DEBATE** -- multi-agent debate logs, no user/assistant structure
- **PresageLabs__NewsBench** -- news articles, no conversations
- **Qwen__AgentWorldBench** -- benchmark (test set)
- **R2E-Gym__R2E-Gym-Subset** -- environment specs, no trajectories
- **SWE-Gym__SWE-Gym-Raw** -- issues/patches, no trajectories
- **Salesforce__CRMArena** -- benchmark (test set)
- **ScaleAI__MCP-Atlas** -- benchmark (test set)
- **Snowflake__AgentWorldModel-1K** -- environment specs, no trajectories
- **Srijan-Chakraborty__GLM-5.2-Agent-Distilled** -- broken: tool results follow the user turn with no assistant tool call before them (315/318 rows) -- the calls are missing from the source
- **THU-KEG__AgentIF** -- benchmark (test set)
- **agentsea__wave-ui-25k** -- image grounding data
- **ai-safety-institute__AgentHarm** -- benchmark (harmful-behaviour eval, 6 rows)
- **argilla__ultrafeedback-binarized-preferences-cleaned** -- same UltraFeedback prompts as allenai__ultrafeedback_binarized_cleaned (kept)
- **armand0e__gpt-5.5-agent** -- 2 sessions of raw Codex event logs
- **artillerywu__DeepResearch-9K** -- broken: every tool observation is replaced by a copy of the question (12,596/12,974 rows)
- **automatelab__mcp-servers-tool-catalog** -- tool catalog, no conversations
- **cais__mmlu** -- benchmark (test set)
- **cua-lite__UI-Genie-Agent** -- image-based GUI agent data
- **data-agents__jupyter-agent-dataset** -- corrupt parquet (footer missing)
- **data-for-agents__insta-150k-v3** -- task + planned steps only, no executed trajectory
- **derek-thomas__ScienceQA** -- image QA (and a benchmark)
- **disco-eth__AgentsNet** -- graph definitions
- **eth-sri__agentbench** -- benchmark (PR metadata, no trajectories)
- **google-research-datasets__mbpp** -- benchmark (test set)
- **hkust-nlp__agentboard** -- benchmark (goals only, no trajectories)
- **inference-net__HALO-Gemini-3-Flash-AppWorld** -- 57 traces in flattened OTel attributes; too few to justify a bespoke parser
