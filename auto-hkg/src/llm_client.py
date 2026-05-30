"""
LLM Client — provider abstraction layer for Auto-HKG.

Supported providers: OpenAI, Groq, Anthropic, Google Gemini, Ollama, Unsloth (local).
"""

import os
import torch


DEFAULT_MODELS = {
    "groq":      "llama-3.3-70b-versatile",
    "openai":    "gpt-4o-mini",
    "anthropic": "claude-haiku-4-5-20251001",
    "gemini":    "gemini-2.0-flash",
    "ollama":    "llama3.2",
    "unsloth":   "unsloth/gemma-4-9b-it-bnb-4bit",
}


class LLMClient:
    def __init__(self, provider: str = "groq", model: str = None):
        self.provider = provider.lower()
        self.model = model or DEFAULT_MODELS[self.provider]
        self._client = self._init_client()

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
            raise ValueError(
                f"Unknown provider: '{self.provider}'. "
                f"Valid options: groq, openai, anthropic, gemini, ollama, unsloth."
            )

    def _init_unsloth(self):
        try:
            from unsloth import FastLanguageModel
        except ImportError:
            raise ImportError(
                "Unsloth is not installed.\n"
                "Run: pip install \"unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git\""
            )

        model_id = self.model

        bf16_supported = torch.cuda.is_bf16_supported()
        dtype = torch.bfloat16 if bf16_supported else torch.float16
        use_4bit = "bnb-4bit" in model_id
        max_seq_length = 4096 if bf16_supported else 2048

        print(f"[Unsloth] Model         : {model_id}")
        print(f"[Unsloth] dtype         : {dtype}")
        print(f"[Unsloth] load_in_4bit  : {use_4bit}")
        print(f"[Unsloth] max_seq_length: {max_seq_length}")
        print("[Unsloth] Loading model weights. First run may take 3-10 minutes...")

        model, processor = FastLanguageModel.from_pretrained(
            model_name=model_id,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=use_4bit,
        )
        FastLanguageModel.for_inference(model)

        print("[Unsloth] Model loaded successfully.")
        # Deteksi apakah ini multimodal processor (Gemma 4) atau tokenizer biasa
        self._is_multimodal = hasattr(processor, 'apply_chat_template') and hasattr(processor, 'decode')
        print(f"[Unsloth] Processor type: {'Multimodal Processor' if self._is_multimodal else 'Tokenizer'}")
        return (model, processor)

    def complete(self, prompt: str) -> str:
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
        model, processor = self._client

        # Gemma 4 pakai Processor multimodal — format input berbeda dari tokenizer biasa.
        # Content harus dalam format list of dicts dengan "type" dan "text".
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        ).to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
            )

        # Decode hanya token baru (potong bagian prompt)
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        result = processor.decode(new_tokens, skip_special_tokens=True).strip()

        if not result:
            raise ValueError("Model returned an empty response. Check VRAM or try a smaller model.")

        return result
