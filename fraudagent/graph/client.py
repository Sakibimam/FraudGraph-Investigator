"""Thin TigerGraph REST client used by the agent's tools.

Every call is counted and timed so each investigation can report exactly how many graph
and retrieval calls it made (the answer file's `tool_calls`).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from fraudagent.config import settings


class GraphError(RuntimeError):
    pass


@dataclass
class CallLog:
    name: str
    params: dict
    ms: float
    ok: bool


@dataclass
class TigerGraph:
    host: str = settings.tg_host
    graph: str = settings.tg_graph
    user: str = settings.tg_user
    password: str = settings.tg_password
    timeout: float = 120.0
    calls: list[CallLog] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._http = httpx.Client(base_url=self.host, auth=(self.user, self.password), timeout=self.timeout)

    # -- low level ---------------------------------------------------------------------
    def _request(self, method: str, path: str, name: str, params: dict | None = None,
                 body: Any = None) -> Any:
        t0 = time.perf_counter()
        ok = False
        try:
            r = self._http.request(method, path, params=params,
                                   content=json.dumps(body) if body is not None else None)
            data = r.json()
            if r.status_code >= 400 or data.get("error"):
                raise GraphError(f"{name}: {data.get('message', r.text)[:300]}")
            ok = True
            return data
        finally:
            self.calls.append(CallLog(name, params or {}, (time.perf_counter() - t0) * 1000, ok))

    def ping(self) -> bool:
        try:
            return self._http.get("/api/ping", timeout=5).status_code == 200
        except httpx.HTTPError:
            return False

    # -- queries -----------------------------------------------------------------------
    def query(self, name: str, **params: Any) -> list[dict]:
        """Run an installed GSQL query. List params are sent as repeated keys."""
        if any(isinstance(v, list) for v in params.values()):
            data = self._request("POST", f"/restpp/query/{self.graph}/{name}", name, body=params)
        else:
            data = self._request("GET", f"/restpp/query/{self.graph}/{name}", name, params=params)
        return data.get("results", [])

    def upsert(self, vertices: dict | None = None, edges: dict | None = None, name: str = "upsert") -> dict:
        body = {"vertices": vertices or {}, "edges": edges or {}}
        return self._request("POST", f"/restpp/graph/{self.graph}", name, body=body)

    def get_vertex(self, vtype: str, vid: str) -> dict | None:
        try:
            data = self._request("GET", f"/restpp/graph/{self.graph}/vertices/{vtype}/{vid}", f"get_{vtype}")
        except GraphError:
            return None
        res = data.get("results") or []
        return res[0] if res else None

    def count(self, vtype: str) -> int:
        data = self._request("GET", f"/restpp/graph/{self.graph}/vertices/{vtype}", "count",
                             params={"count_only": "true"})
        return int(data["results"][0]["count"])

    def reset_calls(self) -> None:
        self.calls.clear()
