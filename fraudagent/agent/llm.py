"""LLM access through any OpenAI-compatible endpoint (Gemini, Groq, OpenAI, Ollama).

The LLM plans which tool to use next, synthesises evidence into a summary, and writes the SAR
narrative and the description of undocumented patterns. It never picks actions or routes;
those come from the policy engine. With no key configured the agent still runs, using
deterministic templates, and reports 0 tokens.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from fraudagent.config import settings

PROVIDERS = {
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/", "gemini-2.5-flash"),
    "groq": ("https://api.groq.com/openai/v1", "llama-3.3-70b-versatile"),
    "openai": ("https://api.openai.com/v1", "gpt-4o-mini"),
    "ollama": ("http://localhost:11434/v1", "llama3.1"),
}


def _detect_provider() -> str:
    if settings.llm_provider:
        return settings.llm_provider
    import os
    for env, name in (("GEMINI_API_KEY", "gemini"), ("GROQ_API_KEY", "groq"), ("OPENAI_API_KEY", "openai")):
        if os.getenv(env):
            return name
    return ""


@dataclass
class LLM:
    provider: str = ""
    model: str = ""
    tokens: int = 0
    enabled: bool = False

    def __post_init__(self) -> None:
        self.provider = self.provider or _detect_provider()
        if not self.provider or (self.provider != "ollama" and not settings.llm_api_key):
            return
        from openai import OpenAI
        base, default_model = PROVIDERS.get(self.provider, PROVIDERS["openai"])
        self.model = self.model or settings.llm_model or default_model
        self._client = OpenAI(base_url=base, api_key=settings.llm_api_key or "ollama", timeout=60, max_retries=2)
        self.enabled = True

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.model}" if self.enabled else "templates (no LLM key)"

    def complete(self, system: str, user: str, max_tokens: int = 700, json_mode: bool = False) -> str | None:
        if not self.enabled:
            return None
        try:
            kw = {"response_format": {"type": "json_object"}} if json_mode else {}
            r = self._client.chat.completions.create(
                model=self.model, temperature=0.2, max_tokens=max_tokens,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}], **kw)
            if r.usage:
                self.tokens += int(r.usage.total_tokens or 0)
            return (r.choices[0].message.content or "").strip()
        except Exception as e:  # network / quota: fall back to templates, never fail the case
            self.last_error = str(e)[:200]
            return None

    def complete_json(self, system: str, user: str, max_tokens: int = 700) -> dict | None:
        out = self.complete(system, user, max_tokens, json_mode=True)
        if not out:
            return None
        m = re.search(r"\{.*\}", out, re.S)
        try:
            return json.loads(m.group(0) if m else out)
        except (json.JSONDecodeError, AttributeError):
            return None
