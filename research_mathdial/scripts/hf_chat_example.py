#!/usr/bin/env python3
import os
import sys
import argparse
from huggingface_hub import InferenceClient


def main():
    parser = argparse.ArgumentParser(description="Minimal chat example via huggingface_hub InferenceClient")
    parser.add_argument("--provider", choices=["hf", "hf-inference", "together"], default="hf", help="Which provider backend to use: Hugging Face Inference API (hf) or Together.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", help="Model ID to use.")
    parser.add_argument("--question", default="What is the capital of France?", help="User question to ask the model.")
    args = parser.parse_args()

    provider = args.provider
    model = args.model

    if provider in ("hf", "hf-inference"):
        # Accept both common env var names and HF CLI login fallback
        api_key = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if api_key:
            client = InferenceClient(api_key=api_key)
        else:
            # Try without explicit key to use cached login if available
            try:
                client = InferenceClient()
            except Exception:
                print("[ERROR] Missing HF token.", file=sys.stderr)
                print("Set one of: export HF_TOKEN=hf_***  or  export HUGGINGFACEHUB_API_TOKEN=hf_***", file=sys.stderr)
                print("Alternatively: huggingface-cli login", file=sys.stderr)
                sys.exit(1)
    elif provider == "together":
        api_key = os.getenv("TOGETHER_API_KEY")
        if not api_key:
            print("[ERROR] Missing TOGETHER_API_KEY environment variable for Together provider.", file=sys.stderr)
            print("Set it with: export TOGETHER_API_KEY=***", file=sys.stderr)
            sys.exit(1)
        client = InferenceClient(provider="together", api_key=api_key)
    else:
        print(f"[ERROR] Unknown provider: {provider}", file=sys.stderr)
        sys.exit(1)

    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": args.question}],
        temperature=0.2,
        max_tokens=256,
    )

    # Print the assistant message content (OpenAI-style response object)
    msg = completion.choices[0].message
    # Some providers return dict-like objects; normalize to string when possible
    if isinstance(msg, dict):
        content = msg.get("content", str(msg))
    else:
        content = getattr(msg, "content", str(msg))

    print(content)


if __name__ == "__main__":
    main()
