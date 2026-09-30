"""Rough domain of an SFT source, from its dataset / file / folder NAME only.

Used by sft_domain_stats.py (user ask, 2026-09-25: "domain-wise stats from the
parquet names"). This is a name-based guess, not a per-row classifier: a
source called "NuminaMath" is counted as Math even if a few of its rows are
not. Rules are checked in order; the first match wins. Anything no rule
matches is "Other / Mixed" and is listed in the report so it can be checked.

    source_of(dataset, rel_path) -> short source name for a final .jsonl file
    domain_of(name)              -> one of DOMAINS
"""
import re

DOMAINS = [
    "Math", "Code & Software Engineering", "Science & STEM", "Medical & Health",
    "Finance & Business", "Law & Policy", "Cybersecurity", "Safety & Alignment",
    "Agentic & Tool Use", "Reasoning & Logic", "Data Analysis & Visualization",
    "Translation", "Translated Code & Math (Indic)", "NLP Tasks (QA / Summarization / Classification)",
    "Educational / Textbook", "General Chat & Instructions", "Other / Mixed",
]

# (regex on the lower-cased name, domain). ORDER MATTERS: first match wins.
RULES = [
    # --- raw sources that are not chat data at all (only in the 'before' side, never in the final data)
    (r"flame-moe|spec_cpu|branch_traces|atari|frogger", "Other / Mixed"),
    # --- whole families first
    (r"gpt_translations_|sarvam_subset|gpt_oss_subset", "Translated Code & Math (Indic)"),
    (r"data_analysis|visuali[sz]ation|data_viz|viz_analysis|openmle", "Data Analysis & Visualization"),
    (r"cosmopedia", "Educational / Textbook"),
    # xP3 tasks (dataset sft_datasets_sft): name is "xP3::<task>"
    (r"^xp3::.*(flores|wmt|tatoeba|translat|opus|mt_|_mt\b|ted_talks)", "Translation"),
    (r"^xp3::.*(code|mbpp|humaneval|apps_|conala|codecomplex|codecontests)", "Code & Software Engineering"),
    (r"^xp3::.*(math|gsm|aqua|mathqa)", "Math"),
    (r"^xp3::|bigscience_xp3", "NLP Tasks (QA / Summarization / Classification)"),
    # aya_collection / Updesh (dataset sft_datasets_sft): name is "aya::<subset>" / "updesh::<task>". Their names contain
    # "translated_" because the DATA was machine-translated into 100+ languages -- the TASK is summarization, QA,
    # dialogue, ... not translation (checked on real rows 2026-09-25). So the domain comes from the task part only.
    (r"^aya::(translated_|templated_)?(thai_scb|thai_usembassy)", "Translation"),
    (r"^aya::(translated_|templated_)?(flan_cot|piqa|xcsqa)", "Reasoning & Logic"),
    (r"^aya::(translated_|templated_)?(scirepeval)", "Science & STEM"),
    (r"^aya::(translated_|templated_)?(soda|dolly|aya_dataset|.*instruct|joke|.*poems|.*jokes|.*riddles|.*stories|"
     r"telugu_food|tamil_thirukkural)", "General Chat & Instructions"),
    (r"^aya::", "NLP Tasks (QA / Summarization / Classification)"),
    (r"^updesh::translation", "Translation"),
    (r"^updesh::(math|fermi)", "Math"),
    (r"^updesh::(brain_teaser|.*reasoning|fs_cot_flow)", "Reasoning & Logic"),
    (r"^updesh::(dialog_gen|creative_writing)", "General Chat & Instructions"),
    (r"^updesh::", "NLP Tasks (QA / Summarization / Classification)"),
    # exact-name fixes found by reading the assignments (2026-09-25)
    (r"dyve_plus|theoremqa|ape210k|nl4opt", "Math"),
    (r"infinibyte|nl2sql|nl-sql|nl2shell|nlp2linux|nl2fix|\bswe_(repair|localization|testgen)", "Code & Software Engineering"),
    (r"jee|neet|entrance_exam|agi_eval", "Science & STEM"),
    (r"deepseek-r1-distilled|deepseek_r1_distilled|deepseek-r1-traces|openthoughts", "Reasoning & Logic"),
    (r"samsum|xlsum|multi_news|coqa|topiocqa|condaqa|msmarco|paraphrase|e2e_nlg|reading_for_understanding", "NLP Tasks (QA / Summarization / Classification)"),
    # --- agents / tools before code (SWE agents are agentic)
    (r"tool_use|tool-use|tool_calling|tool-calling|function-calling|function_calling|glaive-function|toolmind|"
     r"agent|swe-gym|swe-smith|swesmith|r2e-?gym|openhands|swe-next|swe-qa|terminal|tbench|terminus|"
     r"sandboxes|mcp_traces|web-search|mind2web|nnetnav|tau2|tau-?bench|taubench|gaia|synatra|"
     r"codeact|hermes-agent|clingen|taskmaster|freelancer|nl2bash|bash_textbook|open-swe|trajector|"
     r"toucan|agentbank|agenttraj|agent-flan|retail", "Agentic & Tool Use"),
    # --- domains
    (r"medic|clinical|health|pubmed|medmcqa|medreason|cure-bench|biomed", "Medical & Health"),
    (r"financ|fingpt|stock|econom|business", "Finance & Business"),
    (r"law|legal|policy", "Law & Policy"),
    (r"vulnerab|cve|malware|phish|cyber|security", "Cybersecurity"),
    (r"safety|harmless|red.?team|jailbreak|ultrafeedback|dpo", "Safety & Alignment"),
    (r"math|aime|gsm8k|numina|algebra|geometry|aqua_rat|deepscaler|(?<![a-z])limo|s1k|countdown|proof|"
     r"omni_math|prism-math|mathinstruct|metamath|orca-math|hendrycks_math", "Math"),
    (r"code|coder|codeforces|code_contests|code-contests|kodcode|opencoder|magicoder|leetcode|"
     r"software_engineering|programming|python|py_code|debug|defects4j|inferredbugs|staqc|"
     r"stackexchange-(codereview|overflow|superuser|unix|tezos)|glaive-code|compiler|"
     r"scientific-coding|evol_instruct|evol-instruct|package_instruct|mceval|educational_instruct|"
     r"realuser_instruct|largescale_diverse|filtered_infinity", "Code & Software Engineering"),
    (r"biolog|genom|chemi|material|physic|science|scientific|stem|scp_|sciriff|qasper|pdf_science|"
     r"engineering_systems|rqa|infinibyte|protein|molec", "Science & STEM"),
    (r"logic|puzzle|reason|sft_to_reasoning|bespoke|stratos|sky-t1|r1_traces|steps__s1|multi-subject", "Reasoning & Logic"),
    (r"aya_dataset", "General Chat & Instructions"),      # human-written multilingual prompts, not translation
    (r"translat|multilingual", "Translation"),
    (r"summar|duorc|squad|wiki_?qa|wikitable|yourbench|trec|classif", "NLP Tasks (QA / Summarization / Classification)"),
    (r"tulu|wildchat|lmsys|openhermes|openorca|infinity-instruct|infinity_instruct|flan|ultrachat|"
     r"no_robots|capybara|sharegpt|gpt-4o|chat|general|instruction-following|cascade_rm_training|"
     r"\bnl\b|^nl/|final_general|dpo-mix|traces$|gpt_4\.1", "General Chat & Instructions"),
]
_RULES = [(re.compile(p), d) for p, d in RULES]

# hf_domain source folders -> domain (folder name IS the domain there); generic folders fall back to RULES
HF_FOLDERS = {
    "Maths": "Math", "Biology_and_Genomics": "Science & STEM", "code": "Code & Software Engineering",
    "Reasoning": "Reasoning & Logic", "Software_Engineering": "Code & Software Engineering",
    "Finance": "Finance & Business", "Clinical_Decision_Support": "Medical & Health",
    "Law_Policy_Reasoning": "Law & Policy", "Tool_Use": "Agentic & Tool Use",
    "Material_Sciences": "Science & STEM", "Vulnerability": "Cybersecurity",
    "Scientific_Literature_Unders": "Science & STEM", "Safety": "Safety & Alignment",
    "Engineering_Systems_Fault": "Science & STEM", "Computational_Chemistry": "Science & STEM",
    "code_pipeline_team2": "Code & Software Engineering", "sft_to_reasoning": "Reasoning & Logic",
}


def domain_of(name):
    n = name.lower()
    for rx, d in _RULES:
        if rx.search(n):
            return d
    return "Other / Mixed"


def hf_source_domain(src_rel):
    """domain of an hf_domain SOURCE parquet (path relative to SFT/OUTPUT)."""
    top = src_rel.split("/")[0] if "/" in src_rel else ""
    if top in HF_FOLDERS:
        return HF_FOLDERS[top]
    return domain_of(src_rel)


def source_of(dataset, rel):
    """short source name of a FINAL file (path relative to the dataset folder)."""
    p = rel.split("/")
    stem = re.sub(r"(_sft)?(_\d{4})?\.jsonl(\.gz)?$", "", p[-1])
    if dataset.startswith("sft_posttraining"):
        parent = "/".join(p[:-1])
        # files directly under cascade_sft_stageN are their own source (general, science, tool_calling ...)
        if re.search(r"cascade_sft_stage\d$", parent) or parent.endswith("cascade_RM_Training"):
            return f"{parent}/{stem}"
        return parent or stem
    if dataset == "sft_datasets_sft_openai_ready":
        m = re.match(r"bigscience_xP3__([^_]+)__xp3_(.*)", p[0])
        if m:
            return f"xP3::{m.group(2)}"
        m = re.match(r"CohereForAI_aya_collection__(.+?)__", p[0])
        if m:
            return f"aya::{m.group(1)}"
        m = re.match(r"microsoft_Updesh_beta__(.+?)__", p[0])
        if m:
            return f"updesh::{m.group(1)}"
        return re.split(r"__", p[0])[0]
    if len(p) > 1:
        return "/".join(p[:-1]).replace("by_dataset/", "")
    return stem
