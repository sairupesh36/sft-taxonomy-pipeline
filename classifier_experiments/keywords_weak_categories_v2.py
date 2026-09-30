"""
Round-2 additions to keywords_weak_categories.py, written when the domain/task_family fastText
classifiers were found (2026-09-22, after the user pointed out only micro-F1 had ever been
computed) to score macro-F1 as low as 0.42 on a real 1000-doc unbiased test set, with many classes
at a flat 0.000 F1 -- traced to real training-data scarcity (some classes have single-digit real
labeled examples across every source used so far). This file adds keyword search terms for the
weak classes NOT already covered by the original DOMAIN_KEYWORDS/TASK_FAMILY_KEYWORDS, grounded in
each category's own official taxonomy_tree_final.xlsx description (see lc_work/weak_classes.json
for the full evidence: per-class F1 and real training counts that produced this target list).
Kept as a SEPARATE file (not edited into the original) so the original's own history/comments stay
intact; a caller should merge MERGED_DOMAIN_KEYWORDS/MERGED_TASK_FAMILY_KEYWORDS from both files.

Same caution as the original file's own 2026-09-20 note: multi-word/specific phrases for
generic-sounding categories (optimization, transformation, decision_support, ...) to avoid
matching incidental uses of the plain word.

`cross_domain` is deliberately NOT included here: it's "tasks combining multiple domains or
intentionally domain-independent" -- a structural property of a document, not a topic, so there is
no keyword that identifies it. Left as a known gap for this round, not a keyword-search target.
"""

DOMAIN_KEYWORDS_V2 = {
    "agriculture": [
        "crop yield", "crop production", "irrigation schedule", "soil fertility",
        "pesticide application", "farm management", "agricultural technology", "livestock feed",
        "harvest season", "crop rotation", "agronomy", "fertilizer application",
    ],
    "cybersecurity": [
        "cyber threat", "malware analysis", "phishing attack", "data breach response",
        "network security", "firewall configuration", "intrusion detection", "ransomware",
        "security incident", "zero-day exploit", "endpoint protection", "identity and access management",
    ],
    "data_and_information_management": [
        "data governance", "master data management", "data pipeline", "database schema design",
        "data privacy policy", "data warehouse", "etl pipeline", "data lineage",
        "records management", "data quality issue",
    ],
    "ecommerce_and_retail": [
        "online marketplace", "product catalog", "shopping cart", "point of sale system",
        "inventory management for retail", "retail analytics", "merchandising strategy",
        "checkout flow", "online storefront", "product listing",
    ],
    "energy_and_utilities": [
        "power grid", "renewable energy project", "oil and gas", "utility company",
        "energy trading", "smart meter", "electricity tariff", "solar power plant",
        "wind farm", "grid infrastructure",
    ],
    "environment_and_sustainability": [
        "climate policy", "carbon footprint", "sustainability report", "waste management plan",
        "conservation effort", "environmental impact assessment", "renewable resource",
        "biodiversity conservation", "recycling program", "greenhouse gas emissions",
    ],
    "hospitality_and_food_services": [
        "hotel booking", "restaurant menu", "catering service", "food service industry",
        "hospitality management", "guest check-in", "banquet event", "room reservation",
        "culinary", "dining experience",
    ],
    "information_technology": [
        "it support ticket", "network infrastructure", "cloud infrastructure setup",
        "enterprise it system", "it asset management", "help desk", "server maintenance",
        "it department", "system administrator",
    ],
    "insurance": [
        "insurance policy", "insurance claim", "underwriting process", "actuarial analysis",
        "insurance premium", "policyholder", "claims adjuster", "insurance risk management",
        "reinsurance",
    ],
    "manufacturing": [
        "production line", "quality control inspection", "manufacturing process",
        "plant maintenance", "assembly line", "manufacturing automation",
        "production planning schedule", "factory floor",
    ],
    "real_estate": [
        "residential property", "commercial real estate", "property management company",
        "real estate investment", "property valuation", "mortgage application",
        "real estate listing", "lease agreement for property",
    ],
    "religion_and_spirituality": [
        "religious practice", "theological", "comparative religion", "religious organization",
        "spiritual practice", "scripture interpretation", "religious ritual", "faith community",
    ],
    "sports_and_recreation": [
        "professional sports team", "sports analytics", "fitness training program",
        "sports medicine", "sports broadcasting", "recreational activity", "athletic performance",
        "sports league",
    ],
}

TASK_FAMILY_KEYWORDS_V2 = {
    "anomaly_detection": [
        "detect anomalies in", "identify unusual patterns", "flag suspicious activity in",
        "outlier detection", "detect abnormal behavior", "unexpected pattern in this data",
    ],
    "code_debugging": [
        "debug this code", "fix this bug", "why is this code failing", "identify the error in this code",
        "this code throws an error", "trace this bug", "fix the defect in",
    ],
    "compilers_and_systems": [
        "write a compiler", "build a parser for", "lexer and parser", "abstract syntax tree",
        "language tooling", "low-level system execution", "bytecode interpreter",
        "compiler optimization pass",
    ],
    "compliance_assessment": [
        "assess compliance with", "regulatory compliance check", "adherence to the standard",
        "compliance audit of", "policy compliance review", "meets the regulatory requirement",
    ],
    "content_moderation": [
        "moderate this content", "content moderation policy", "flag this content as",
        "review this post for policy violations", "is this content appropriate",
        "content policy violation",
    ],
    "decision_support": [
        "help me decide between", "evaluate these alternatives", "which option should i choose",
        "decision matrix for", "weigh the pros and cons of", "support this decision with evidence",
    ],
    "explanation_and_tutoring": [
        "explain this concept", "help me understand", "walk me through how",
        "teach me how to", "explain step by step why", "tutor me on",
    ],
    "fact_checking_and_claim_verification": [
        "fact-check this claim", "verify this claim", "is this statement true",
        "check the accuracy of this claim", "verify against the source",
        "confirm whether this fact is correct",
    ],
    "knowledge_management": [
        "organize this knowledge base", "create a knowledge article", "maintain the wiki for",
        "structured knowledge repository", "retrieve this documented knowledge",
        "update the knowledge base with",
    ],
    "localization": [
        "localize this content for", "adapt this text for the region", "translate and localize",
        "cultural adaptation of", "locale-specific formatting", "localize the app for",
    ],
    "optimization": [
        "optimize this for", "find the optimal solution to", "minimize the cost of",
        "maximize the efficiency of", "best possible solution given the constraints",
        "optimization problem",
    ],
    "prediction_and_forecasting": [
        "predict the future value of", "forecast the trend of", "predict what will happen",
        "sales forecast for", "predictive model for", "forecast the demand for",
    ],
    "recommendation": [
        "recommend a", "what would you recommend for", "suggest the best option for",
        "give me a recommendation on", "recommend a course of action",
        "which one would you recommend",
    ],
    "reconciliation": [
        "reconcile these two accounts", "reconcile the ledger", "resolve the discrepancy between these statements",
        "bank reconciliation", "reconcile the numbers in", "match these financial records",
    ],
    "reporting_and_documentation": [
        "write a report on", "prepare the documentation for", "create a status report",
        "document this process", "write up the findings in a report", "produce a summary report of",
    ],
    "reverse_engineering": [
        "reverse engineer this", "recover the source from this binary", "infer the protocol used by",
        "reconstruct the original design from", "decompile this", "analyze this binary to understand",
    ],
    "root_cause_analysis": [
        "root cause analysis of", "5 whys analysis", "trace this failure back to its root cause",
        "postmortem for this incident", "underlying cause of this system failure",
    ],
    "selection_and_filtering": [
        "filter out the items that", "select only the ones that", "exclude any that do not meet",
        "narrow down this list to", "which of these meet the criteria",
    ],
    "software_testing": [
        "write test cases for", "unit test for this function", "test plan for",
        "how do i test this code", "write a test suite for", "identify edge cases to test",
    ],
    "threat_analysis": [
        "threat actor analysis", "analyze this attack vector", "indicators of compromise",
        "threat intelligence report", "analyze this vulnerability's impact",
        "attack behavior analysis",
    ],
    "transformation": [
        "convert this data into", "transform this format into", "map this structure to",
        "convert the following into", "reformat this into",
    ],
    "validation_and_verification": [
        "validate that this output meets", "verify this satisfies the requirement",
        "check whether this is valid", "validation check for", "does this meet the specification",
    ],
    "workflow_automation": [
        "automate this workflow", "set up an automated pipeline for", "automate this recurring task",
        "workflow automation for", "build an automation that",
    ],
}
