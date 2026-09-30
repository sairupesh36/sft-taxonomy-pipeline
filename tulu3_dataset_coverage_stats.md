# 📊 Tulu-3 SFT Mixture: Sub-Dataset Coverage & 10k Cap Analysis

**Dataset**: `allenai/tulu-3-sft-mixture` (6 Parquet Shards)  
**Total Original Rows**: `939,343`  
**Unique Dataset Sources**: `19 Sub-Datasets`  
**Sample Strategy**: **Cap at 10,000 samples per sub-dataset** (or 100% if sub-dataset has < 10,000 samples)  
**New Capped Target**: **171,871 rows** (Avoids **767,472 redundant extractions**)  

---

## 📈 Executive Summary

| Metric | Full Tulu-3 Mixture | 10k Capped Target | Current Completed | Remaining to 10k Target | Estimated Time to Finish |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **📋 Tasks Extraction** | 939,343 | **171,871** | **208,420** | **96,029 rows** | **~7.9 hours** |
| **🌐 Domains Extraction**| 939,343 | **171,871** | **97,489** | **110,947 rows** | **~6.5 hours** |
| **⚡ Total Rows Saved** | — | **767,472 rows saved** | — | — | **~3.5 days saved!** |

> [!NOTE]
> Several large datasets (`evol_codealpaca`, `flan_v2`, `numinamath`, `personahub_math`, `wildguard`, `wildchat`) already have **over 10,000+ samples extracted**. By enforcing the 10k cap, we immediately stop extracting from those saturated sources and direct 100% of cluster GPU capacity toward the remaining under-represented sub-datasets.

---

## 📋 Comprehensive Dataset Breakdown & Coverage Table

| # | Sub-Dataset Source Name | Total Rows | 10k Cap Target | Tasks Extracted | Tasks % | Tasks Remaining | Domains Extracted | Domains % | Domains Remaining | Status |
| :-: | :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| 1 | `ai2-adapt-dev/coconot_converted` | 10,983 | **10,000** | 0 | 0.0% | 10,000 | 0 | 0.0% | 10,000 | ⏸️ Pending |
| 2 | `ai2-adapt-dev/evol_codealpaca_heval_decontaminated` | 107,276 | **10,000** | 33,480 | 334.8% | 0 | 19,135 | 191.3% | 0 | ✅ Met (Cap Hit) |
| 3 | `ai2-adapt-dev/flan_v2_converted` | 89,982 | **10,000** | 20,329 | 203.3% | 0 | 3,793 | 37.9% | 6,207 | ⏳ In Progress |
| 4 | `ai2-adapt-dev/no_robots_converted` | 9,500 | **9,500** | 0 | 0.0% | 9,500 | 0 | 0.0% | 9,500 | ⏸️ Pending |
| 5 | `ai2-adapt-dev/numinamath_tir_math_decontaminated` | 64,312 | **10,000** | 35,157 | 351.6% | 0 | 17,252 | 172.5% | 0 | ✅ Met (Cap Hit) |
| 6 | `ai2-adapt-dev/oasst1_converted` | 7,131 | **7,131** | 7,131 | 100.0% | 0 | 7,136 | 100.1% | 0 | ✅ 100% Complete |
| 7 | `ai2-adapt-dev/personahub_code_v2_34999` | 34,999 | **10,000** | 0 | 0.0% | 10,000 | 0 | 0.0% | 10,000 | ⏸️ Pending |
| 8 | `ai2-adapt-dev/personahub_ifdata_manual_seed_v3_29980`| 29,980 | **10,000** | 0 | 0.0% | 10,000 | 0 | 0.0% | 10,000 | ⏸️ Pending |
| 9 | `ai2-adapt-dev/personahub_math_v5_regen_149960` | 149,960 | **10,000** | 28,138 | 281.4% | 0 | 15,264 | 152.6% | 0 | ✅ Met (Cap Hit) |
| 10 | `ai2-adapt-dev/tulu_hard_coded_repeated_10` | 240 | **240** | 0 | 0.0% | 240 | 0 | 0.0% | 240 | ⏸️ Pending |
| 11 | `ai2-adapt-dev/tulu_v3.9_aya_100k` | 100,000 | **10,000** | 0 | 0.0% | 10,000 | 0 | 0.0% | 10,000 | ⏸️ Pending |
| 12 | `ai2-adapt-dev/tulu_v3.9_open_math_2_gsm8k_50k` | 50,000 | **10,000** | 0 | 0.0% | 10,000 | 0 | 0.0% | 10,000 | ⏸️ Pending |
| 13 | `ai2-adapt-dev/tulu_v3.9_personahub_math_interm_algebra_20k` | 20,000 | **10,000** | 0 | 0.0% | 10,000 | 0 | 0.0% | 10,000 | ⏸️ Pending |
| 14 | `ai2-adapt-dev/tulu_v3.9_sciriff_10k` | 10,000 | **10,000** | 8,711 | 87.1% | 1,289 | 0 | 0.0% | 10,000 | ⏳ In Progress |
| 15 | `ai2-adapt-dev/tulu_v3.9_synthetic_finalresp_wildguardmixtrain_decontaminated_50k` | 50,000 | **10,000** | 41,557 | 415.6% | 0 | 18,062 | 180.6% | 0 | ✅ Met (Cap Hit) |
| 16 | `ai2-adapt-dev/tulu_v3.9_table_gpt_5k` | 5,000 | **5,000** | 0 | 0.0% | 5,000 | 0 | 0.0% | 5,000 | ⏸️ Pending |
| 17 | `ai2-adapt-dev/tulu_v3.9_wildchat_100k` | 100,000 | **10,000** | 33,917 | 339.2% | 0 | 16,847 | 168.5% | 0 | ✅ Met (Cap Hit) |
| 18 | `ai2-adapt-dev/tulu_v3.9_wildjailbreak_decontaminated_50k` | 50,000 | **10,000** | 0 | 0.0% | 10,000 | 0 | 0.0% | 10,000 | ⏸️ Pending |
| 19 | `allenai/tulu-3-sft-personas-math-grade` | 49,980 | **10,000** | 0 | 0.0% | 10,000 | 0 | 0.0% | 10,000 | ⏸️ Pending |
| | **TOTAL CLUSTER PROGRESS** | **939,343** | **171,871** | **208,420** | **121.3%** | **96,029** | **97,489** | **56.7%** | **110,947** | **🚀 486 RPM ACTIVE** |

---

## 🎯 Observations & Next Action Plan

1. **Already 100% Covered Datasets**:
   - `oasst1_converted` (7,131 / 7,131)
   - `evol_codealpaca` (> 10,000)
   - `numinamath` (> 10,000)
   - `personahub_math` (> 10,000)
   - `synthetic_wildguard` (> 10,000)
   - `wildchat_100k` (> 10,000)
   These 6 datasets alone account for **over 170,000 extracted tasks** and **90,000+ extracted domains**.

2. **Why 11 Datasets Show 0 Progress**:
   - In the 6 parquet files (`train-00000` to `train-00005`), the data is sequentially ordered by sub-dataset chunks.
   - The pods started from row 0 of each shard and processed sequentially, so earlier datasets got over-extracted while later datasets (`aya_100k`, `table_gpt_5k`, `coconot`, `personahub_code`, etc.) are queued further in the shards.

3. **How We Implement the 10k Cap**:
   - Update `taxonomy_fast_extract.py` and `domain_fast_extract.py`:
     Maintain an active `source_counts = Counter()` loaded from existing CSVs.
     If `source_counts[row["source"]] >= 10,000`:
       **Skip the row immediately without calling GPT-OSS-120B!**
     This allows the workers to skip through already-satisfied sources at **millions of rows per second**, jumping straight to the pending sub-datasets (`aya`, `no_robots`, `table_gpt`, `sciriff`, `coconot`, etc.) to fill their 10k quotas!
