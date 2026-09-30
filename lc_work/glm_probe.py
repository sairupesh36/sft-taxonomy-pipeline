import os, sys, json
os.environ["HF_HOME"] = "/projects/data/datasets/code_data/sai_rupesh/taxonomy/classifier_experiments/hf_model_cache"
sys.path.insert(0, "/projects/data/datasets/code_data/sai_rupesh/taxonomy/pylibs")
from transformers import AutoTokenizer
for name in ["zai-org/GLM-4.5", "zai-org/GLM-4.5-Air", "zai-org/GLM-4.6", "THUDM/GLM-4-32B-0414", "THUDM/glm-4-9b-chat-hf", "THUDM/glm-4-9b-chat"]:
    try:
        t = AutoTokenizer.from_pretrained(name, trust_remote_code=False)
        print("LOADED", name, "| chat_template:", bool(t.chat_template), "| len", len(t.chat_template or ""), flush=True)
    except Exception as e:
        print("FAILED", name, "->", str(e).replace("\n", " ")[:110], flush=True)
