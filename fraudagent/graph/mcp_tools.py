"""Graph tools served through the TigerGraph MCP server.

The agent talks to `tigergraph-mcp` (https://github.com/tigergraph/tigergraph-mcp) over stdio
and calls `tigergraph__run_installed_query` for every investigation tool, so the same GSQL
queries are available to any MCP-capable agent (Claude, LangGraph, Copilot) as well.
A single MCP session is kept open on a background event loop for the lifetime of the tools.
"""
from __future__ import annotations

import asyncio
import json
import re
import shutil
import sys
import threading
from pathlib import Path
from urllib.parse import urlparse

from fraudagent.config import settings
from fraudagent.graph.client import GraphError
from fraudagent.graph.tools import TigerGraphTools


class _MCPSession:
    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._err: BaseException | None = None
        threading.Thread(target=self._run, daemon=True).start()
        self._ready.wait(60)
        if self._err:
            raise self._err

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._main())

    async def _main(self) -> None:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import get_default_environment, stdio_client
        u = urlparse(settings.tg_host)
        env = {"TG_HOST": f"{u.scheme}://{u.hostname}", "TG_GRAPHNAME": settings.tg_graph,
               "TG_USERNAME": settings.tg_user, "TG_PASSWORD": settings.tg_password,
               "TG_GS_PORT": str(u.port or 14240), "TG_RESTPP_PORT": "9000"}
        exe = shutil.which("tigergraph-mcp") or str(Path(sys.executable).parent / "tigergraph-mcp")
        try:
            params = StdioServerParameters(command=exe, args=[], env={**get_default_environment(), **env})
            async with stdio_client(params) as (r, w):
                async with ClientSession(r, w) as s:
                    await s.initialize()
                    self.session = s
                    self.tool_names = [t.name for t in (await s.list_tools()).tools]
                    self._stop = asyncio.Event()
                    self._ready.set()
                    await self._stop.wait()
        except BaseException as e:  # surface startup errors to the caller
            self._err = e
            self._ready.set()

    def call(self, tool: str, args: dict) -> dict:
        fut = asyncio.run_coroutine_threadsafe(self.session.call_tool(tool, arguments=args), self.loop)
        res = fut.result(timeout=180)
        text = "".join(getattr(c, "text", "") for c in res.content)
        m = re.search(r"```json\s*(.*?)```", text, re.S)
        payload = json.loads(m.group(1) if m else text)
        if not payload.get("success", True):
            raise GraphError(f"{tool}: {payload.get('error') or payload.get('summary')}")
        return payload


class MCPTools(TigerGraphTools):
    backend = "tigergraph-mcp"
    _session: _MCPSession | None = None

    def __init__(self) -> None:
        super().__init__()
        if MCPTools._session is None:
            MCPTools._session = _MCPSession()
        self.mcp = MCPTools._session

    def _q(self, name: str, **p) -> list[dict]:
        out = self.mcp.call("tigergraph__run_installed_query",
                            {"graph_name": settings.tg_graph, "query_name": name, "params": p})
        data = out.get("data")
        if isinstance(data, dict):
            data = data.get("result") or data.get("results") or data.get("output") or data
        return data if isinstance(data, list) else [data]
