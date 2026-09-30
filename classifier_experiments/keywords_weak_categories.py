"""
Keyword sets for the categories the classifier is currently weakest on (measured
directly from real per-class F1 on the validation set -- these are genuinely
rare topics, 39-198 total examples out of 77,234, and they're scattered thinly
across many different OUTPUT folders rather than concentrated in one, confirmed
by pulling real examples of each before writing these lists. Grounded in the
taxonomy's own official description of each category, plus real example phrasing
already seen in the existing data -- not guessed blind.
"""

DOMAIN_KEYWORDS = {
    "veterinary_and_animal_welfare": [
        "veterinar", "animal welfare", "livestock", "feline", "canine", "fiv", "felv",
        "shelter animal", "wildlife conservation", "pet health", "zoonotic", "foal",
        "equine", "vet clinic", "animal shelter", "spay", "neuter",
    ],
    "automotive": [
        "automotive", "car engine", "vehicle engineering", "dealership", "fleet vehicle",
        "connected vehicle", "car manufactur", "engine block", "transmission fluid",
        "tire pressure", "brake pad", "car maintenance", "auto repair", "carburetor",
        "combustion engine",
    ],
    "construction_and_infrastructure": [
        "construction project", "civil infrastructure", "building code", "contractor",
        "architect", "damp course", "scaffolding", "construction site", "blueprint",
        "load-bearing", "concrete slab", "building permit", "civil engineer",
    ],
    "supply_chain_and_logistics": [
        "supply chain", "bill of lading", "warehousing", "inventory management",
        "freight", "demand planning", "shipment", "consignee", "shipper", "logistics",
        "procurement", "purchase order", "cargo manifest", "delivery route",
    ],
    "defence_and_national_security": [
        "national security", "military operation", "homeland security", "counterterrorism",
        "intelligence agency", "armed forces", "defence strategy", "naval", "army",
        "strategic security", "war strategy", "combat", "military intelligence",
    ],
    "transportation_and_mobility": [
        "public transit", "ride sharing", "traffic management", "transportation planning",
        "maritime transport", "traffic sign", "highway", "railway", "airline route",
        "commute", "transit system", "road safety",
    ],
    "law_enforcement_and_criminal_justice": [
        "law enforcement", "criminal investigation", "police department", "courts",
        "corrections facility", "forensics", "public safety", "criminal justice",
        "identity theft", "illegal activity", "arrest", "prosecut", "detective",
    ],
    # 2026-09-20 additions -- still weak (F1<0.5) on the freshly-cleaned data,
    # not previously covered by any scanner. Grounded in each category's own
    # official taxonomy_tree_final.xlsx description, kept multi-word/specific
    # where the plain word would be too generic (e.g. "market" alone would
    # false-positive on "financial market"; "sales pitch"/"lead generation"
    # are specific enough not to).
    "government_and_public_administration": [
        "public administration", "government policy", "government procurement",
        "public sector", "municipal government", "government agency", "civil service",
        "public services department", "government budget", "tax policy",
    ],
    "politics_and_civics": [
        "political campaign", "election result", "legislative bill", "civic engagement",
        "diplomatic relations", "political analysis", "voter turnout", "member of parliament",
        "foreign policy", "political party",
    ],
    "consumer_and_personal_services": [
        "personal care service", "home cleaning service", "consumer financial advice",
        "personal shopper", "lifestyle concierge", "home repair service", "moving service",
        "personal stylist", "consumer complaint",
    ],
    "sales_and_marketing": [
        "sales pitch", "marketing campaign", "lead generation", "brand strategy",
        "advertising copy", "market research report", "crm pipeline", "sales funnel",
        "customer acquisition", "digital marketing strategy",
    ],
    "arts_and_culture": [
        "visual arts", "performing arts", "museum exhibit", "cultural heritage",
        "art restoration", "gallery curator", "craft technique", "sculpture",
        "art preservation", "theater production",
    ],
    "telecommunications": [
        "telecom network", "mobile network operator", "telecom billing", "network provisioning",
        "cellular tower", "5g network", "telecom infrastructure", "voip",
        "broadband provider", "sim card",
    ],
    "aerospace_and_aviation": [
        "aircraft maintenance", "air traffic control", "airline operations",
        "aviation safety", "flight schedule", "aerospace engineering", "airline pilot",
        "runway", "avionics",
    ],
    "human_resources_and_talent": [
        "employee recruitment", "performance review", "compensation package", "workforce planning",
        "employee relations", "talent acquisition", "hr policy", "onboarding process",
        "job candidate", "employee benefits",
    ],
    "weather_and_climate_services": [
        "weather forecast", "climate model", "meteorolog", "storm warning",
        "precipitation forecast", "weather station", "hurricane tracking", "climate data",
        "disaster weather", "weather advisory",
    ],
    "journalism_and_publishing": [
        "news article", "fact-check", "press release",
        "news reporting", "publishing house", "newsroom", "journalist interview",
        "breaking news", "op-ed piece",
    ],
    "social_services_and_nonprofit": [
        "nonprofit organization", "charity fundraising", "disaster relief effort",
        "community outreach", "volunteer coordination", "social work referral", "advocacy campaign",
        "grant application", "homeless shelter", "food bank",
    ],
    "mining_and_natural_resources": [
        "mineral exploration", "mine safety", "ore extraction", "natural resource management",
        "mining operation", "quarry", "drilling site", "mineral deposit",
        "coal mine", "resource extraction",
    ],
}

TASK_FAMILY_KEYWORDS = {
    "ranking": [
        "rank the following", "rank these", "order from best to worst", "prioritize the",
        "sort by importance", "which is more important", "rank in order",
    ],
    "code_review": [
        "review this code", "code review", "identify bugs in this code", "critique this code",
        "is this code secure", "code quality", "refactor for readability", "pull request review",
    ],
    "system_design": [
        "design a system", "system architecture", "design a scalable", "distributed system design",
        "design the infrastructure", "high-level design", "system design interview",
    ],
    "legal_and_regulatory_reasoning": [
        "legal reasoning", "regulatory compliance", "interpret this contract", "under the law",
        "legal authority", "statute", "regulation requires", "contract clause",
    ],
    "comparison": [
        "compare and contrast", "what are the differences between", "compare the following",
        "similarities and differences", "which is better,", "versus", " vs ",
    ],
    # 2026-09-20 additions -- grounded in each task_family's own official
    # description. Kept as multi-word/specific phrases for the generic-word
    # categories (estimation, monitoring, normalization, visualization) to
    # avoid matching incidental uses of the plain word -- same caution this
    # project already applied to "ranking" (rank the following, not just
    # "rank") and documented as a real risk in the parent CLAUDE.md's Round 7
    # evidence-quality-contamination note.
    "search_and_retrieval": [
        "search for information on", "retrieve the document", "find relevant results",
        "look up information about", "information retrieval system", "search query",
        "retrieve records matching",
    ],
    "rewriting_and_editing": [
        "rewrite this", "edit this paragraph", "improve the wording", "proofread this",
        "revise this text", "restructure this document", "correct the grammar in",
        "polish this draft",
    ],
    "software_design": [
        "design the interface for", "software architecture diagram", "design a class structure",
        "design the api for", "component design", "design pattern for", "design the data model",
    ],
    "code_transformation": [
        "refactor this code", "migrate this code to", "convert this code from",
        "port this code to", "rewrite this function in", "modernize this codebase",
        "transpile",
    ],
    "file_and_storage_operations": [
        "read this file", "write to a file", "move the file to", "delete the file",
        "file system operation", "manage storage for", "persist this data to disk",
        "backup this directory",
    ],
    "configuration_and_deployment": [
        "deploy this application", "configure the server", "set up the deployment pipeline",
        "package this application", "release this build", "configuration file for",
        "ci/cd pipeline",
    ],
    "risk_assessment": [
        "assess the risk of", "risk score for", "identify potential risks",
        "risk mitigation plan", "evaluate the risk", "prioritize these risks",
        "risk exposure",
    ],
    "negotiation_and_persuasion": [
        "negotiate a deal", "write a persuasive", "negotiation strategy", "counter-offer",
        "convince the reader", "persuasive argument for", "negotiate the terms",
    ],
    "simulation_and_scenario_analysis": [
        "simulate what would happen if", "run a simulation of", "scenario analysis for",
        "model this hypothetical", "what-if scenario", "simulate the outcome of",
    ],
    "visualization": [
        "create a chart showing", "create a dashboard for", "visualize this data as",
        "plot this data", "create a graph of", "design a visualization for",
    ],
    "estimation": [
        "estimate the cost of", "estimate the probability of", "roughly how much would",
        "give a rough estimate", "estimate the number of", "ballpark figure for",
    ],
    "data_management": [
        "organize this dataset", "data governance policy", "validate this dataset",
        "maintain the database", "data cleaning pipeline", "manage this data",
    ],
    "requirements_analysis": [
        "gather the requirements for", "analyze these requirements", "clarify the requirements",
        "prioritize these requirements", "requirements specification for",
    ],
    "security_assessment": [
        "assess this system for vulnerabilities", "penetration testing report",
        "vulnerability assessment of", "security weakness in", "cve vulnerability analysis",
        "security audit of",
    ],
    "research_and_scientific": [
        "design an experiment to", "formulate a hypothesis", "reproduce this experiment",
        "research methodology for", "scientific investigation of", "experimental design for",
    ],
    "monitoring": [
        "monitor this metric", "track the status of", "set up monitoring for",
        "alert when this changes", "monitor system health", "track these events over time",
    ],
    "forensics_and_investigation": [
        "forensic investigation of", "reconstruct the events of", "digital forensics analysis",
        "evidence analysis for", "forensic examination of",
    ],
    "normalization": [
        "normalize this data", "standardize these values", "normalize the formatting of",
        "standardize the terminology", "normalize these names",
    ],
    "matching_and_resolution": [
        "match these records", "entity resolution for", "deduplicate these records",
        "resolve these duplicate entries", "record linkage between",
    ],
    "incident_management": [
        "incident response plan", "triage this incident", "coordinate the incident response",
        "resolve this outage", "declare an incident", "incident postmortem",
    ],
}
