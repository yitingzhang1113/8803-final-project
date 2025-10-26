from typing import List, Optional

from huggingface_hub import InferenceClient


class HFTextGenerator:
    """Thin wrapper around Hugging Face serverless inference API.

    Uses text-generation interface with optional stop sequences.
    """

    def __init__(
        self,
        model: str = "eth-nlped/MathDial-SFT-Qwen2.5-1.5B-Instruct",
        hf_token: Optional[str] = None,
        fallback_model: Optional[str] = None,
        fallback_models: Optional[List[str]] = None,
    ):
        self.model = model
        self.token = hf_token
        # Build a fallback chain: explicit list -> single fallback -> sensible defaults
        self.fallback_models: List[str] = []
        if fallback_models:
            self.fallback_models.extend([m for m in fallback_models if m])
        if fallback_model:
            self.fallback_models.append(fallback_model)
        if not self.fallback_models:
            # Conservative defaults that are commonly available on HF Inference
            self.fallback_models = [
                "mistralai/Mistral-7B-Instruct-v0.3",
                "microsoft/Phi-3.5-mini-instruct",
                "google/gemma-2-2b-it",
            ]
        self.client = InferenceClient(model=model, token=hf_token)

    def generate(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
    ) -> str:
        """Generate text using serverless Inference with a robust streaming path.

        We use streaming mode unconditionally to avoid StopIteration quirks from some providers.
        """
        try:
            # 1) Try streaming API (preferred)
            chunks: List[str] = []
            final_text: Optional[str] = None
            try:
                for ev in self.client.text_generation(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    stop=stop or None,
                    return_full_text=False,
                    stream=True,
                    details=True,
                ):
                    if hasattr(ev, "token") and getattr(ev.token, "text", None):
                        chunks.append(ev.token.text)
                    elif hasattr(ev, "generated_text") and ev.generated_text:
                        final_text = ev.generated_text
                    elif isinstance(ev, str):
                        chunks.append(ev)
            except StopIteration:
                pass

            text = final_text if final_text is not None else "".join(chunks)

            # 2) If empty, try non-streaming call (some backends only support non-stream)
            if not text:
                try:
                    non_stream_text = self.client.text_generation(
                        prompt,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        stop=stop or None,
                        return_full_text=False,
                        stream=False,
                    )
                    if isinstance(non_stream_text, str):
                        text = non_stream_text
                    elif hasattr(non_stream_text, "generated_text"):
                        text = getattr(non_stream_text, "generated_text") or ""
                except Exception:
                    # swallow here; we'll try fallback model next if configured
                    text = text or ""

            # 2b) If still empty, try primary again without stop constraints (some models mishandle stop at start)
            if not text:
                try:
                    non_stream_text = self.client.text_generation(
                        prompt,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        return_full_text=False,
                        stream=False,
                    )
                    if isinstance(non_stream_text, str):
                        text = non_stream_text
                    elif hasattr(non_stream_text, "generated_text"):
                        text = getattr(non_stream_text, "generated_text") or ""
                except Exception:
                    text = text or ""

            # 2c) If still empty, try primary via chat_completion (some models support chat only)
            if not text:
                try:
                    cc = self.client.chat_completion(
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=max_new_tokens,
                        temperature=temperature,
                        stop=stop or None,
                    )
                    # Extract first choice content if present
                    if hasattr(cc, "choices") and cc.choices:
                        choice = cc.choices[0]
                        # ChatCompletionChoice may have .message or .delta
                        content = None
                        if hasattr(choice, "message") and getattr(choice.message, "content", None):
                            content = choice.message.content
                        elif hasattr(choice, "delta") and getattr(choice.delta, "content", None):
                            content = choice.delta.content
                        if isinstance(content, str):
                            text = content
                except Exception:
                    text = text or ""

            # 3) If still empty, iterate through fallback model chain (non-stream)
            if not text:
                for fb_model in self.fallback_models:
                    if not fb_model or fb_model == self.model:
                        continue
                    try:
                        fb_client = InferenceClient(model=fb_model, token=self.token)
                        non_stream_text = fb_client.text_generation(
                            prompt,
                            max_new_tokens=max_new_tokens,
                            temperature=temperature,
                            stop=stop or None,
                            return_full_text=False,
                            stream=False,
                        )
                        if isinstance(non_stream_text, str):
                            text = non_stream_text
                        elif hasattr(non_stream_text, "generated_text"):
                            text = getattr(non_stream_text, "generated_text") or ""
                        if text:
                            break
                    except Exception:
                        # try next fallback
                        # Try chat_completion on this fallback
                        try:
                            cc = fb_client.chat_completion(
                                messages=[{"role": "user", "content": prompt}],
                                max_tokens=max_new_tokens,
                                temperature=temperature,
                                stop=stop or None,
                            )
                            if hasattr(cc, "choices") and cc.choices:
                                choice = cc.choices[0]
                                content = None
                                if hasattr(choice, "message") and getattr(choice.message, "content", None):
                                    content = choice.message.content
                                elif hasattr(choice, "delta") and getattr(choice.delta, "content", None):
                                    content = choice.delta.content
                                if isinstance(content, str) and content:
                                    text = content
                                    break
                        except Exception:
                            continue

            if not text:
                raise RuntimeError(
                    "HF Inference returned empty response. Model may not be available for serverless inference."
                )

            return (text or "").strip()
        except Exception as e:
            raise RuntimeError(
                "HF Inference call failed. If you didn't provide a Hugging Face token, "
                "please add one in the sidebar or set HUGGINGFACEHUB_API_TOKEN. Original error: "
                f"{repr(e)}"
            )
