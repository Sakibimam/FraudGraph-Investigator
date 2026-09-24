"""LLM access through OpenAI-compatible endpoints, with a fallback chain.

The LLM re-ranks graph tools, synthesises evidence into summaries, and writes SAR narratives and
descriptions of undocumented patterns. It never picks actions or routes; those come from the
policy engine.

Backends are tried in order: a hosted model (Gemini / Groq / OpenAI, if a key is set) and then a
local model served by Ollama (default qwen2.5:3b). When the hosted free tier runs out of quota the
agent keeps going on the local model. With neither available, deterministic templates are used.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field

import httpx

from fraudagent.config import settings

PROVIDERS = {
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/", "gemini-3.5-flash-lite", "GEMINI_API_KEY"),
    "groq": ("https://api.groq.com/openai/v1", "llama-3.3-70b-versatile", "GROQ_API_KEY"),
    "openai": ("https://api.openai.com/v1", "gpt-4o-mini", "OPENAI_API_KEY"),
}
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")


@dataclass
class _Backend:
    name: str
    model: str
    client: object
    min_interval: float
    dead: bool = False
    last: float = 0.0


def _ollama_up() -> bool:
    try:
        tags = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=2).json().get("models", [])
        return any(m.get("name", "") == OLLAMA_MODEL or m.get("model", "") == OLLAMA_MODEL for m in tags)
    except (httpx.HTTPError, ValueError):
        return False


@dataclass
class LLM:
    tokens: int = 0
    calls: int = 0
    backends: list[_Backend] = field(default_factory=list)
    used: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        from openai import OpenAI
        only = settings.llm_provider  # optional: force one provider (e.g. "ollama")
        for name, (base, model, env) in PROVIDERS.items():
            key = os.getenv(env)
            if key and only in ("", name):
                self.backends.append(_Backend(name, settings.llm_model or model,
                                              OpenAI(base_url=base, api_key=key, timeout=60, max_retries=0), 4.2))
        if os.getenv("USE_LOCAL_LLM", "1") == "1" and only in ("", "ollama") and _ollama_up():
            self.backends.append(_Backend("ollama", OLLAMA_MODEL,
                                          OpenAI(base_url=f"{OLLAMA_URL}/v1", api_key="ollama", timeout=180,
                                                 max_retries=0), 0.0))

    @property
    def enabled(self) -> bool:
        return any(not b.dead for b in self.backends)

    @property
    def label(self) -> str:
        live = [f"{b.name}:{b.model}" for b in self.backends if not b.dead]
        return " -> ".join(live) if live else "templates (no LLM available)"

    def complete(self, system: str, user: str, max_tokens: int = 700, json_mode: bool = False) -> str | None:
        for b in self.backends:
            if b.dead:
                continue
            out = self._try(b, system, user, max_tokens, json_mode)
            if out:
                self.used[b.name] = self.used.get(b.name, 0) + 1
                return out
        return None

    def _try(self, b: _Backend, system: str, user: str, max_tokens: int, json_mode: bool) -> str | None:
        kw = {"response_format": {"type": "json_object"}} if json_mode else {}
        for attempt in range(2):
            wait = b.min_interval - (time.monotonic() - b.last)
            if wait > 0:
                time.sleep(wait)
            b.last = time.monotonic()
            try:
                r = b.client.chat.completions.create(
                    model=b.model, temperature=0.2, max_tokens=max_tokens,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}], **kw)
                self.calls += 1
                if r.usage:
                    self.tokens += int(r.usage.total_tokens or 0)
                return (r.choices[0].message.content or "").strip() or None
            except Exception as e:  # never fail a case because of the LLM
                msg = str(e)
                self.last_error = f"{b.name}: {msg[:300]}"
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                    m = re.search(r"retry in ([\d.]+)s", msg)
                    delay = float(m.group(1)) + 1 if m else 30
                    if attempt == 0 and delay <= 40 and "PerDay" not in msg:
                        time.sleep(delay)
                        continue
                    b.dead = True  # quota gone: the next backend takes over for the rest of the run
                elif any(code in msg for code in ("401", "403", "404")):
                    b.dead = True
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
