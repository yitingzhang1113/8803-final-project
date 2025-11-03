import os
from dotenv import load_dotenv

print("[hf_env_check] Loading .env…")
load_dotenv()

token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if token:
    masked = token[:6] + "…" + token[-4:] if len(token) > 10 else "(set)"
    print(f"[hf_env_check] Token loaded: {masked}")
else:
    print("[hf_env_check] Token not found in environment (.env/secrets).")

try:
    from huggingface_hub import InferenceClient
    model = os.getenv("HF_MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.3")
    print(f"[hf_env_check] Probing model: {model}")
    client = InferenceClient(model=model, token=token)
    out = client.text_generation("Hello, what is 2+2?", max_new_tokens=16, stream=False, return_full_text=False)
    print("[hf_env_check] Inference ok:", repr(out[:120]))
except Exception as e:
    print("[hf_env_check] Inference failed:", e)
