"""LLM access through any OpenAI-compatible endpoint (Gemini, Groq, OpenAI, Ollama).

The LLM plans which tool to use next, synthesises evidence into a summary, and writes the SAR
narrative and the description of undocumented patterns. It never picks actions or routes;
those come from the policy engine. With no key configured the agent still runs, using
deterministic templates, and reports 0 tokens.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass

from fraudagent.config import settings

PROVIDERS = {
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/", "gemini-3.5-flash-lite"),
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
    calls: int = 0
    min_interval: float = 4.2   # free tiers allow ~15 requests/minute; stay under it
    _last: float = 0.0

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
        kw = {"response_format": {"type": "json_object"}} if json_mode else {}
        for attempt in range(3):
            wait = self.min_interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            try:
                r = self._client.chat.completions.create(
                    model=self.model, temperature=0.2, max_tokens=max_tokens,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}], **kw)
                self.calls += 1
                if r.usage:
                    self.tokens += int(r.usage.total_tokens or 0)
                return (r.choices[0].message.content or "").strip() or None
            except Exception as e:  # rate limit: back off and retry; anything else: fall back to templates
                self.last_error = str(e)[:300]
                m = re.search(r"retry in ([\d.]+)s", self.last_error)
                if "429" in self.last_error and attempt < 2:
                    time.sleep(float(m.group(1)) + 1 if m else 30)
                    continue
                return None
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
