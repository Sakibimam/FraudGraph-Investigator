"""Runtime settings, read from the environment (.env is loaded if present)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    tg_host: str = os.getenv("TG_HOST", "http://localhost:14240")
    tg_graph: str = os.getenv("TG_GRAPH", "FraudGraph")
    tg_user: str = os.getenv("TG_USERNAME", "tigergraph")
    tg_password: str = os.getenv("TG_PASSWORD", "tigergraph")
    tg_secret: str = os.getenv("TG_SECRET", "")
    use_mcp: bool = os.getenv("USE_TIGERGRAPH_MCP", "1") == "1"
    graph_backend: str = os.getenv("GRAPH_BACKEND", "tigergraph")  # tigergraph | mcp | local

    # Any OpenAI-compatible endpoint works: Gemini, Groq, OpenAI, a local Ollama.
    llm_provider: str = os.getenv("LLM_PROVIDER", "")
    llm_model: str = os.getenv("LLM_MODEL", "")
    llm_api_key: str = field(default_factory=lambda: (
        os.getenv("LLM_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("GROQ_API_KEY")
        or os.getenv("OPENAI_API_KEY") or ""))

    data_dir: Path = ROOT / "data"
    cache_dir: Path = ROOT / "data" / "cache"
    knowledge_dir: Path = ROOT / "knowledge"
    cases_dir: Path = ROOT / "cases"


settings = Settings()
