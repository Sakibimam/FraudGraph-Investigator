"""Analyst API: case queue, live investigations (SSE), evidence graph and the approval workflow.

    uvicorn fraudagent.api.server:app --port 8000
"""
from __future__ import annotations

import json
import queue
import threading
from datetime import datetime
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from fraudagent.agent.investigator import Investigator, Trigger
from fraudagent.agent.llm import LLM
from fraudagent.agent.policy import AUTO_ACTIONS
from fraudagent.config import ROOT, settings
from fraudagent.graph.tools import make_tools

TRACES = ROOT / "runs" / "traces"
APPROVALS = ROOT / "runs" / "approvals.json"
UI = ROOT / "ui"

app = FastAPI(title="FraudGraph Investigator")
_tools = make_tools(settings.graph_backend)
_llm = LLM()
_lock = threading.Lock()


def _pack() -> pd.DataFrame:
    return pd.read_csv(settings.data_dir / "case_pack.csv", dtype={"flagged_txn_id": str})


def _trace(case_id: str) -> dict | None:
    p = TRACES / f"{case_id}.json"
    return json.loads(p.read_text()) if p.exists() else None


def _approvals() -> list[dict]:
    return json.loads(APPROVALS.read_text()) if APPROVALS.exists() else []


@app.get("/api/health")
def health() -> dict:
    return {"graph_backend": _tools.backend, "llm": _llm.label, "time": datetime.now().isoformat(timespec="seconds")}


@app.get("/api/cases")
def cases() -> list[dict]:
    out = []
    for r in _pack().itertuples():
        t = _trace(r.case_id) or {}
        c = t.get("case", {})
        out.append({"case_id": r.case_id, "opened_at": r.opened_at, "trigger_type": r.trigger_type,
                    "card_id": r.card_id, "flagged_txn_id": r.flagged_txn_id, "trigger_text": r.trigger_text,
                    "risk_score": None if pd.isna(r.risk_score) else r.risk_score,
                    "verdict": c.get("verdict"), "probability": c.get("fraud_probability"),
                    "status": c.get("status"), "pattern": c.get("pattern"), "exposure": c.get("exposure_usd"),
                    "sar": t.get("sar", {}).get("file"),
                    "pending": sum(1 for a in t.get("next_best_actions", {}).get("final", []) if a["route"] != "auto")})
    return out


@app.get("/api/cases/{case_id}")
def case(case_id: str) -> dict:
    t = _trace(case_id)
    if not t:
        raise HTTPException(404, "not investigated yet")
    t["approvals"] = [a for a in _approvals() if a["case_id"] == case_id]
    return t


def _trigger(case_id: str) -> Trigger:
    p = _pack()
    r = p[p.case_id == case_id]
    if r.empty:
        raise HTTPException(404, "unknown case")
    r = r.iloc[0]
    return Trigger(r.case_id, r.opened_at, r.trigger_type, r.trigger_text, str(r.flagged_txn_id), r.card_id,
                   r.customer_id, None if pd.isna(r.risk_score) else float(r.risk_score))


@app.get("/api/cases/{case_id}/investigate")
def investigate(case_id: str) -> StreamingResponse:
    """Run the agent now and stream each step as a server-sent event."""
    trig = _trigger(case_id)
    q: queue.Queue = queue.Queue()

    def work() -> None:
        with _lock:
            try:
                agent = Investigator(_tools, _llm, on_event=q.put)
                result = agent.run(trig)
                TRACES.mkdir(parents=True, exist_ok=True)
                (TRACES / f"{case_id}.json").write_text(json.dumps(
                    {k: v for k, v in result.items() if k != "embedding"}, indent=2, default=str))
                q.put({"step": "done", "detail": "Investigation complete"})
            except Exception as e:  # report to the UI instead of dropping the stream
                q.put({"step": "error", "detail": str(e)[:300]})
            q.put(None)

    threading.Thread(target=work, daemon=True).start()

    def stream():
        while (ev := q.get()) is not None:
            yield f"data: {json.dumps(ev, default=str)}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/api/cases/{case_id}/graph")
def evidence_graph(case_id: str) -> dict:
    """Nodes/edges for the evidence view, built from the investigation record."""
    t = _trace(case_id)
    if not t:
        raise HTTPException(404, "not investigated yet")
    c, trig = t["case"], t["trigger"]
    nodes, edges = {}, []

    def node(i, label, group, title=""):
        nodes.setdefault(i, {"id": i, "label": label, "group": group, "title": title or label})

    card = t["card_id"]
    node(card.split("-")[0], card.split("-")[0], "customer")
    node(card, card, "card")
    edges.append({"from": card.split("-")[0], "to": card, "label": "OWNS"})
    ids = set(c["affected_txn_ids"]) | {trig["flagged_txn_id"]}
    for tx in ids:
        node(tx, tx, "txn_flag" if tx == trig["flagged_txn_id"] else "txn")
        edges.append({"from": card, "to": tx, "label": "MADE"})
    for d in c["connected_device_profiles"][:3]:
        node(d, d.split(" | ")[0] or "device", "device", d)
        for tx in ids:
            edges.append({"from": tx, "to": d, "label": "FROM_DEVICE", "dashes": True})
    for k in c["connected_card_ids"][:25]:
        node(k, k, "card_linked")
        if c["connected_device_profiles"]:
            edges.append({"from": k, "to": c["connected_device_profiles"][0], "label": "USED"})
    for s in c["similar_prior_cases"][:6]:
        node(s, s, "closed_case")
        edges.append({"from": t["graph_case_id"], "to": s, "label": "SIMILAR_TO", "dashes": True})
    node(t["graph_case_id"], t["graph_case_id"], "case", c["summary"])
    edges.append({"from": t["graph_case_id"], "to": card, "label": "INV_ON_CARD"})
    return {"nodes": list(nodes.values()), "edges": edges}


class Decision(BaseModel):
    action: str
    decision: str          # approve | reject
    role: str              # L1 | L2
    note: str = ""


@app.post("/api/cases/{case_id}/approve")
def approve(case_id: str, d: Decision) -> dict:
    """Human-in-the-loop: L1/L2 actions wait here; the route must match the approver's role."""
    t = _trace(case_id)
    if not t:
        raise HTTPException(404, "not investigated yet")
    rec = next((a for a in t["next_best_actions"]["final"] if a["action"] == d.action), None)
    if rec is None:
        raise HTTPException(400, f"{d.action} is not a recommended action for {case_id}")
    if rec["route"] == "auto":
        raise HTTPException(400, f"{d.action} is an auto action; the agent executes it without approval")
    if rec["route"] == "L2" and d.role != "L2":
        raise HTTPException(403, f"{d.action} requires a fraud manager (L2); {d.role} cannot approve it")
    log = _approvals()
    entry = {"case_id": case_id, "action": d.action, "route": rec["route"], "decision": d.decision,
             "role": d.role, "note": d.note, "at": datetime.now().isoformat(timespec="seconds"),
             "executed": d.decision == "approve"}
    log.append(entry)
    APPROVALS.parent.mkdir(parents=True, exist_ok=True)
    APPROVALS.write_text(json.dumps(log, indent=2))
    return entry


@app.get("/api/cases/{case_id}/execution")
def execution(case_id: str) -> list[dict]:
    """What the agent executed itself (auto) vs what is waiting for a human."""
    t = _trace(case_id)
    if not t:
        raise HTTPException(404, "not investigated yet")
    done = {(a["action"]) for a in _approvals() if a["case_id"] == case_id and a["executed"]}
    return [{**a, "state": "executed (simulated API)" if a["action"] in AUTO_ACTIONS else
             ("approved + executed" if a["action"] in done else f"awaiting {a['route']} approval")}
            for a in t["next_best_actions"]["final"]]


@app.get("/api/monitor")
def monitor() -> list[dict]:
    p = ROOT / "monitor" / "alerts.json"
    return json.loads(p.read_text()) if p.exists() else []


@app.get("/")
def index() -> FileResponse:
    return FileResponse(UI / "index.html")


app.mount("/ui", StaticFiles(directory=UI), name="ui")
