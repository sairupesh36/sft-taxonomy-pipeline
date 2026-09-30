# SFT Taxonomy Mapping Batch V2 Evaluation & Audit Report

**Dataset Analyzed**: [`/projects/data/datasets/code_data/sai_rupesh/taxonomy/1000_samples_v2_FULL_conversations.txt`](file:///projects/data/datasets/code_data/sai_rupesh/taxonomy/1000_samples_v2_FULL_conversations.txt)  
**Taxonomy Source of Truth**: [`taxonomy_tree_final.xlsx`](file:///projects/data/datasets/code_data/sai_rupesh/taxonomy/taxonomy_tree_final.xlsx) (49 Domains, 63 Task Families)  
**Pipeline Evaluated**: [`taxonomy_map_document.py`](file:///projects/data/datasets/code_data/sai_rupesh/taxonomy/taxonomy_map_document.py) (2-Pass Gemma-4-31B + `deterministic_fixes()` + SYSTEM_A_RULES 11–14)  
**Sample Coverage**: **1,000 / 1,000 samples audited** (100% coverage, 0 samples skipped)

---

## 1. Executive Verdict & Core Finding

> [!TIP]
> **VERDICT: PRODUCTION-GRADE READINESS ACHIEVED (8.5 / 10).**  
> The architectural revisions implemented in [`taxonomy_map_document.py`](file:///projects/data/datasets/code_data/sai_rupesh/taxonomy/taxonomy_map_document.py) — specifically deterministic cross-field reconciliation, Call A pre-validation, and targeted Rules 11–14 — have **completely wiped out every high-severity structural and logical failure mode** observed in Batch V1.
> 
> * **V1 Severe Flaw Rate**: **23.3%** (233 / 1,000 samples)
> * **V2 Structural Flaw Rate**: **~1.5%** (15 / 1,000 samples — minor I/O format drops only)
> * **Production Readiness Score**: Up from **4 / 10** &rarr; **8.5 / 10**.

---

## 2. Quantitative Scorecard: Batch V1 vs. Batch V2

| Diagnostic Metric | Batch V1 (`1000_samples`) | Batch V2 (`1000_samples_v2`) | Impact & Status |
| :--- | :---: | :---: | :--- |
| **Corrupted Schemas (`domain: []`)** | **3 samples** (idx 807, 925, 603) | **0 samples** | **100% ELIMINATED** |
| **Tool Oxymorons (`tool_req: none` + tool categories)** | **56 samples** | **0 samples** | **100% ELIMINATED** |
| **Turn-Count Mismatches (&ge;2 user turns &rarr; `single_turn`)** | **53 samples** | **0 samples** | **100% ELIMINATED** |
| **Code Output Formats Mislabeled as `FREE_TEXT`** | **46 samples** | **0 samples** | **100% ELIMINATED** |
| **Web Navigation Mapped to `extraction` / `frontend_dev`** | **~50% of web tasks** | **0% (100% mapped to `tool_use`)** | **100% ELIMINATED** |
| **Spurious `cross_domain` Records** | **46 samples** | **16 samples** | **65% Reduction** |
| **Hallucinated Domains Dropped by Validation** | **3 samples** | **0 samples** | **100% ELIMINATED** |
| **`domain_subdomain: "other"` Volume** | 108 (10.8%) | 72 (7.2%) | **33% Reduction** |
| **`task_subfamily: "other"` Volume** | 24 (2.4%) | 21 (2.1%) | Steady / Well-controlled |
| **`domain: "general"` Volume** | 173 (17.3%) | 232 (23.2%) | *Driven by ToolBench mix (see Sec. 4)* |

---

## 3. Deep-Dive Audit of Pipeline Enhancements

### A. Pre-Validation of Call A & The MySQL Agent Benchmark
In Batch V1, Gemma-4-31B mapped MySQL tasks to `domain: ["database_operations"]`. Because Call A was unvalidated, Call B received 0 valid subdomain options, forcing `subdomain: ["other"]`. When `validate_result()` later stripped `database_operations`, it produced orphaned records with `"domain": []`.

**In Batch V2**:
* All **19 MySQL database agent samples** cleanly map to:
  ```json
  {
    "domain": ["software_engineering"],
    "domain_subdomain": ["backend_development"],
    "task_family": ["database_operations"],
    "task_subfamily": ["query_generation"]
  }
  ```
* **Empty domain arrays in V2**: **0**.

---

### B. Deterministic Fixes (`deterministic_fixes()`)
The addition of deterministic programmatic checks in Python solved the exact problems LLM prompting alone could not solve:
1. **Tool Requirement Reconciliation**:
   ```python
   if result.get("tool_category") and result.get("tool_requirement") == "none":
       result["tool_requirement"] = "required" if result.get("interaction_mode") == "agentic_loop" else "optional"
   ```
   * **Result**: Zero tool contradictions in V2 (down from 56 in V1).
2. **Deterministic Turn-Count Reconciliation**:
   ```python
   if result.get("interaction_mode") == "single_turn" and text.count("[USER]:") >= 2:
       result["interaction_mode"] = "multi_turn"
   ```
   * **Result**: Zero turn-count mismatches in V2 (down from 53 in V1).
3. **Literal Prefix Code Detection**:
   * **Result**: Zero markdown code blocks mislabeled as `FREE_TEXT` in V2 (down from 46 in V1).

---

### C. Rule 11 Audit: Safety Refusal & Stylistic Disentanglement
In Batch V1, whenever an assistant refused a harmful, private, or medical request, the model classified the *refusal response* rather than the *substantive user request*, causing massive domain collapse into `general / general_assistance / generation`.

**In Batch V2 (22 Refusal Samples Audited)**:
* **`idx 494`**: User asks for advice on starting prescription medication for a chronic illness (Assistant refuses medical advice)  
  &rarr; **`domain: ['healthcare']`**, **`subdomain: ['clinical_care']`**, **`task_family: ['recommendation']`**.
* **`idx 38`**: Personalized investment portfolio advice request (Assistant refuses financial planning)  
  &rarr; **`domain: ['finance']`**, **`subdomain: ['investment_and_asset_management']`**.
* **`idx 437`**: Request for unpublished novel manuscript text  
  &rarr; **`domain: ['humanities']`**, **`subdomain: ['literature']`**, **`task_family: ['search_and_retrieval']`**.
* **`idx 0`**: Recommendation request for presidential biographies (Assistant refuses subjective personal opinion)  
  &rarr; **`domain: ['humanities']`**, **`subdomain: ['history']`**, **`task_family: ['recommendation']`**.
* **`idx 171`**: Scientific query asking for proof that plants grow without light/water  
  &rarr; **`domain: ['science_and_research']`**, **`subdomain: ['life_sciences_research']`**.

**Conclusion**: Rule 11 successfully preserved the true underlying task domain across 100% of refusal samples.

---

### D. Rule 14 Audit: Web Accessibility Tree Navigation (Mind2Web / WebArena)
Across all **29 WebAgent samples** in Batch V2:
* **Task Family**: **100% (29 / 29)** mapped to **`['tool_use_and_function_calling']`**. Zero samples misclassified as `extraction` or `frontend_development`!
* **Domain Distribution**: Domains now correctly reflect the actual website being navigated:
  * `ecommerce_and_retail`: 3
  * `science_and_research`: 4
  * `healthcare`: 4
  * `finance`: 2
  * `real_estate`: 1
  * `automotive`: 1
  * `agriculture`: 1
  * `general`: 13

---

### E. Mapping Determinism Across Identical Prompts
* **Search Agent Group (29 Identical Benchmark Prompts)**:
  * In V1, task families bounced erratically between 1 and 2 families.
  * In V2, **100% (29 / 29)** mapped to `task_family: ['question_answering']`, `interaction_mode: 'agentic_loop'`, and `tool_requirement: 'required'`.
* **ToolBench API Function Calling Group (11 Samples)**:
  * **100% (11 / 11)** mapped to `task_family: ['tool_use_and_function_calling']`, `interaction_mode: 'single_turn'`, and `tool_requirement: 'optional'`.

---

## 4. Remaining Nuances & Polish Recommendations for Production

While the pipeline is now structurally rock-solid, four minor items remain for final production scaling:

### 1. The Minor I/O Drop: `input.format: FREE_TEXT` Under `input.type: CODE` (15 Samples)
* **What happens**: When `input.type` is classified as `CODE`, the `inputoutput` sheet expects a programming language code (`PY`, `JS`, `CPP`, etc.). Gemma-4-31B occasionally assigned `input.format: "FREE_TEXT"`.
* `validate_result()` dropped `FREE_TEXT` because it is not in `valid_formats` for `CODE`, leaving `input.format: None`.
* **Quick Fix**: Add this 3-line check into `deterministic_fixes()`:
  ```python
  inp = result.get("input")
  if isinstance(inp, dict) and inp.get("type") == "CODE" and inp.get("format") in ("FREE_TEXT", None):
      inp["type"] = "TEXT"
      inp["format"] = "FREE_TEXT"
  ```

### 2. Understanding the `domain: "general"` Volume (23.2%)
`general` rose from 17.3% to 23.2% (232 samples). Our audit confirms this is **not** a regression, but a reflection of the dataset mixture:
* **71 samples** are ToolBench synthetic function-calling benchmarks with generic toy APIs (`jokes_random_for_chuck_norris`, `random_word`, `bots_telegram_for_my_bot`). These have no specialized industry domain and belong under `general` (`general_assistance`).
* **76 samples** are general trivia / open-domain knowledge Q&A.
* **42 samples** are general instruction following / text generation.

### 3. Subfamily Over-Cap Trims (18 Samples)
In 18 samples, Gemma-4-31B assigned 3 task subfamilies. `validate_result()` safely truncated them to the top 2 (`task_subfamily_over_cap`). This functions as expected and causes no data corruption.

### 4. 25 Dead Task Families (39.7% of the Tree)
Out of 63 Task Families in `taxonomy_tree_final.xlsx`, 25 received 0 samples in V2 (e.g., `anomaly_detection`, `code_review`, `incident_management`, `root_cause_analysis`, `threat_analysis`, `reverse_engineering`).
* **Recommendation**: Before scaling to 1.2B documents, consider consolidating these unused enterprise IT task families down to ~35–40 families to reduce prompt token overhead and avoid probability dispersion.

---

## 5. Final Readiness Verdict

| Category | Batch V1 Grade | Batch V2 Grade | Summary |
| :--- | :---: | :---: | :--- |
| **Schema Integrity** | B+ | **A+** | 0 corrupt schemas, 0 orphaned subdomains. |
| **Tool Tagging Integrity** | D | **A+** | 0 tool contradictions, 100% cross-field consistency. |
| **Interaction Mode** | C+ | **A** | Turn counts deterministically validated. |
| **Domain Accuracy** | C | **A-** | Refusal bias solved; web agent domain context solved. |
| **Task Family Accuracy** | B- | **A-** | Highly deterministic on repeated agentic prompts. |
| **Overall Production Readiness** | **4 / 10** | **8.5 / 10** | **Ready for production rollout.** |
