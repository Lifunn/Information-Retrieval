"""
LLM Client — provider abstraction layer for Auto-HKG.

Supported providers: OpenAI, Groq, Anthropic, Google Gemini, Ollama, Unsloth (local).

Provider selection guide:
  - groq      : Recommended for cloud inference. Free tier available at console.groq.com.
                Supports large models (Llama 3.3 70B) with fast inference speed.
  - openai    : Reliable for production use. Requires paid API key.
  - anthropic : High-quality instruction following. Requires paid API key.
  - gemini    : Google's model. Free tier available via Google AI Studio.
  - ollama    : Local inference via Ollama server. No API key required.
                Requires Ollama to be running at localhost:11434.
  - unsloth   : Local inference via Unsloth + HuggingFace. No API key required.
                Optimized for NVIDIA GPU (T4 16GB or A100 40/80GB).
                Automatically selects dtype and context length based on GPU capability.
"""

import os
import json
import torch


# ---------------------------------------------------------------------------
# Default model per provider
# ---------------------------------------------------------------------------

DEFAULT_MODELS = {
    "groq":      "llama-3.3-70b-versatile",
    "openai":    "gpt-4o-mini",
    "anthropic": "claude-haiku-4-5-20251001",
    "gemini":    "gemini-2.0-flash",
    "ollama":    "llama3.2",
    # Default for unsloth is set to Qwen3-14B 4-bit, suitable for A100-40GB.
    # If running on T4 (16GB), override with a smaller alias such as 'qwen2.5-7b'.
    "unsloth":   "unsloth/Qwen3-14B-bnb-4bit",
}


# ---------------------------------------------------------------------------
# Unsloth model registry
#
# All models listed here are available on HuggingFace under the 'unsloth' org.
# Entries suffixed with '-bnb-4bit' are 4-bit quantized (NF4 via bitsandbytes).
# Entries without the suffix are loaded in full precision (bfloat16 on A100).
#
# VRAM estimates are approximate and assume no other processes occupy GPU memory.
# ---------------------------------------------------------------------------

UNSLOTH_MODELS = {
    # -- Qwen 2.5 series (strong JSON instruction following) ------------------
    "qwen2.5-7b":       "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",    # ~5 GB  | T4 / A100
    "qwen2.5-14b":      "unsloth/Qwen2.5-14B-Instruct-bnb-4bit",   # ~9 GB  | T4 / A100
    "qwen2.5-32b":      "unsloth/Qwen2.5-32B-Instruct-bnb-4bit",   # ~20 GB | A100-40GB

    # -- Qwen 3 series (latest, best multilingual + reasoning) ----------------
    "qwen3-8b":         "unsloth/Qwen3-8B-bnb-4bit",               # ~6 GB  | T4 / A100
    "qwen3-14b":        "unsloth/Qwen3-14B-bnb-4bit",              # ~10 GB | A100-40GB  [recommended]
    "qwen3-32b":        "unsloth/Qwen3-32B-bnb-4bit",              # ~20 GB | A100-40GB
    "qwen3-14b-bf16":   "unsloth/Qwen3-14B",                       # ~28 GB | A100-40GB (full bf16)
    "qwen3-32b-bf16":   "unsloth/Qwen3-32B",                       # ~65 GB | A100-80GB only

    # -- Llama 3 series -------------------------------------------------------
    "llama3.2-3b":      "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",  # ~3 GB  | T4 / A100
    "llama3.1-8b":      "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit", # ~6 GB | T4 / A100

    # -- Gemma 3 series -------------------------------------------------------
    "gemma3-4b":        "unsloth/gemma-3-4b-it-bnb-4bit",          # ~4 GB  | T4 / A100
    "gemma3-12b":       "unsloth/gemma-3-12b-it-bnb-4bit",         # ~8 GB  | T4 / A100

    # -- DeepSeek R1 distilled ------------------------------------------------
    "deepseek-r1-7b":   "unsloth/DeepSeek-R1-Distill-Qwen-7B-bnb-4bit",  # ~5 GB | T4 / A100
}


# ---------------------------------------------------------------------------
# LLMClient
# ---------------------------------------------------------------------------

class LLMClient:
    """
    Unified interface for multiple LLM providers.

    Usage:
        client = LLMClient(provider="groq")
        response = client.complete("Your prompt here")

    For Unsloth local inference:
        client = LLMClient(provider="unsloth", model="qwen3-14b")
        response = client.complete("Your prompt here")

    The model parameter accepts either a shorthand alias from UNSLOTH_MODELS
    or a full HuggingFace repository ID (e.g. 'unsloth/Qwen3-14B-bnb-4bit').
    """

    def __init__(self, provider: str = "groq", model: str = None):
        self.provider = provider.lower()
        self.model = model or DEFAULT_MODELS[self.provider]
        self._client = self._init_client()

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------

    def _init_client(self):
        if self.provider == "groq":
            from groq import Groq
            return Groq(api_key=os.environ["GROQ_API_KEY"])

        elif self.provider == "openai":
            from openai import OpenAI
            return OpenAI(api_key=os.environ["OPENAI_API_KEY"])

        elif self.provider == "anthropic":
            import anthropic
            return anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

        elif self.provider == "gemini":
            import google.generativeai as genai
            genai.configure(api_key=os.environ["GEMINI_API_KEY"])
            return genai.GenerativeModel(self.model)

        elif self.provider == "ollama":
            self._ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
            return None

        elif self.provider == "unsloth":
            return self._init_unsloth()

        else:
            raise ValueError(f"Unknown provider: '{self.provider}'. "
                             f"Valid options: groq, openai, anthropic, gemini, ollama, unsloth.")

    def _init_unsloth(self):
        """
        Load a HuggingFace model using Unsloth for optimized local inference.

        Behavior:
        - If the GPU supports bfloat16 (A100, H100, RTX 30xx+), dtype is set to
          torch.bfloat16 for better numerical stability and range than float16.
        - If the GPU does not support bfloat16 (T4, V100), dtype falls back to
          torch.float16 automatically.
        - max_seq_length is set to 4096 on bfloat16-capable GPUs (A100) and 2048
          on float16-only GPUs (T4) to stay within safe VRAM limits.
        - load_in_4bit is enabled only for models with 'bnb-4bit' in their repo ID.
          Full-precision models (e.g. qwen3-14b-bf16) are loaded without quantization.

        First-time model loading downloads the model weights from HuggingFace Hub
        and caches them locally. This may take 3 to 10 minutes depending on model
        size and network speed. Subsequent loads use the local cache.
        """
        try:
            from unsloth import FastLanguageModel
        except ImportError:
            raise ImportError(
                "Unsloth is not installed.\n"
                "Run: pip install \"unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git\""
            )

        # Resolve shorthand alias to full HuggingFace repo ID
        model_id = UNSLOTH_MODELS.get(self.model, self.model)
        self.model = model_id

        # Determine optimal dtype based on hardware capability
        bf16_supported = torch.cuda.is_bf16_supported()
        dtype = torch.bfloat16 if bf16_supported else torch.float16

        # Use 4-bit quantization only for bnb-4bit models
        use_4bit = "bnb-4bit" in model_id

        # A100 (bf16 capable) can safely handle longer context windows
        max_seq_length = 4096 if bf16_supported else 2048

        print(f"[Unsloth] Model         : {model_id}")
        print(f"[Unsloth] dtype         : {dtype}")
        print(f"[Unsloth] load_in_4bit  : {use_4bit}")
        print(f"[Unsloth] max_seq_length: {max_seq_length}")
        print("[Unsloth] Loading model weights. First run may take 3-10 minutes...")

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_id,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=use_4bit,
        )
        FastLanguageModel.for_inference(model)

        print("[Unsloth] Model loaded successfully.")
        return (model, tokenizer)

    # -------------------------------------------------------------------------
    # Inference
    # -------------------------------------------------------------------------

    def complete(self, prompt: str) -> str:
        """
        Send a prompt to the configured provider and return the response as a string.

        Temperature is set to 0.1 across all providers to maximize output consistency
        and reduce hallucination in structured JSON extraction tasks.
        """
        if self.provider in ("groq", "openai"):
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1024,
            )
            return resp.choices[0].message.content.strip()

        elif self.provider == "anthropic":
            resp = self._client.messages.create(
                model=self.model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text.strip()

        elif self.provider == "gemini":
            resp = self._client.generate_content(
                prompt,
                generation_config={"temperature": 0.1, "max_output_tokens": 1024}
            )
            return resp.text.strip()

        elif self.provider == "ollama":
            import requests
            resp = requests.post(
                f"{self._ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1}
                },
                timeout=120
            )
            resp.raise_for_status()
            return resp.json()["response"].strip()

        elif self.provider == "unsloth":
            return self._complete_unsloth(prompt)

    def _complete_unsloth(self, prompt: str) -> str:
        """
        Run inference using the locally loaded Unsloth model.

        The prompt is formatted using the model's chat template before tokenization.
        Only newly generated tokens are decoded — the prompt tokens are excluded
        from the output to return a clean response string.

        max_new_tokens is set to 1024 to accommodate detailed JSON extractions
        that may include multiple concepts, prerequisites, and successors.
        use_cache=True enables Unsloth's KV-cache optimization for faster decoding.
        """
        import torch

        model, tokenizer = self._client

        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(text, return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )

        # Decode only the newly generated tokens, excluding the input prompt
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        result = tokenizer.decode(new_tokens, skip_special_tokens=True)
        return result.strip()
