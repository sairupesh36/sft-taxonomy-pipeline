# Agentic SFT Datasets: Multi-Source 5k Capped Manifest
**Completed At**: 2026-09-08 17:29:23 UTC
**Storage Directory**: `/projects/data/datasets/code_data/sai_rupesh/taxonomy/agentic_data`
**Total Samples Downloaded**: **69,511 rows**
**Total Execution Time**: **1.5 minutes**

| # | Sub-Source Name | Original Mixture | Rows | File Size (MB) | Trace Type / Description |
|---|---|---|:---:|:---:|---|
| 1 | `agent_flan_agent_instruct_react` | `internlm/Agent-FLAN` | 1,731 | 0.51 | Multi-step ReAct action trajectories on AgentInstruct seeds |
| 2 | `agent_flan_agent_instruct_tflan` | `internlm/Agent-FLAN` | 1,731 | 0.7 | Conversational tool execution flows |
| 3 | `agent_flan_toolbench_react_10p` | `internlm/Agent-FLAN` | 2,288 | 3.25 | ReAct tool execution over real-world RapidAPI REST tools |
| 4 | `agent_flan_toolbench_tflan_60p` | `internlm/Agent-FLAN` | 5,000 | 6.98 | Multi-turn API execution dialogues with arguments |
| 5 | `agent_flan_toolbench_tflan_cot` | `internlm/Agent-FLAN` | 5,000 | 6.98 | Chain-of-thought tool argument reasoning and selection |
| 6 | `agent_flan_toolbench_instruct_3k` | `internlm/Agent-FLAN` | 3,000 | 0.9 | Precise instruction-to-API mapping (full 3k rows) |
| 7 | `agent_flan_toolbench_negative` | `internlm/Agent-FLAN` | 761 | 0.15 | Negative agent samples (learning when NOT to call tools) |
| 8 | `toolace_complex_tools` | `Team-ACE/ToolACE` | 5,000 | 3.38 | BFCL #1 rank nested tool calling & error recovery |
| 9 | `glaive_function_calling` | `glaiveai/glaive-function-calling-v2` | 0 | 0.0 | Parallel and sequential function calling dialogue turns |
| 10 | `nemotron_interactive_agent` | `nvidia/Nemotron-SFT-Agentic-v2` | 5,000 | 24.06 | Interactive multi-turn goal decomposition agent traces |
| 11 | `nemotron_search_agent` | `nvidia/Nemotron-SFT-Agentic-v2` | 5,000 | 122.38 | Multi-hop search & information retrieval agent traces |
| 12 | `nemotron_tool_calling` | `nvidia/Nemotron-SFT-Agentic-v2` | 5,000 | 24.67 | Single & multi-turn tool calling with observation reasoning |
| 13 | `agentinstruct_tool_use` | `microsoft/orca-agentinstruct-1M-v1` | 5,000 | 6.22 | AgentInstruct synthetic API tool usage flows |
| 14 | `agentinstruct_webagent_flow` | `microsoft/orca-agentinstruct-1M-v1` | 5,000 | 5.85 | Autonomous web browsing and DOM element action flows |
| 15 | `agentinstruct_code_agent` | `microsoft/orca-agentinstruct-1M-v1` | 5,000 | 8.45 | Code generation, debugging, refactoring, and test writing |
| 16 | `agentinstruct_rag_agent` | `microsoft/orca-agentinstruct-1M-v1` | 5,000 | 12.92 | Multi-hop retrieval-augmented generation and verification |
| 17 | `agentinstruct_analytical_reasoning` | `microsoft/orca-agentinstruct-1M-v1` | 5,000 | 4.53 | Goal decomposition, planning, and multi-step deduction |
| 18 | `swe_bench_software_engineering_agent` | `princeton-nlp/SWE-bench` | 5,000 | 13.36 | Real GitHub issue resolution via terminal, git diff, & tests |
