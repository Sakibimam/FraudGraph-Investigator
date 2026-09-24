"""The agent's graph tools.

Each tool is a named, documented function the planner can call. `TigerGraphTools` runs them
as installed GSQL queries (the production path, also exposed through TigerGraph MCP).
`LocalTools` answers the same questions from the parquet cache the loader writes; it exists
so the detectors can be unit-tested and the UI can be developed without a running database.
Both return the same shapes.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from fraudagent.config import settings
from fraudagent.graph.client import TigerGraph

TS_FMT = "%Y-%m-%d %H:%M:%S"

TOOL_DOCS = {
    "txn_detail": "Attributes of one transaction: amount, product, channel, region, device, email, risk and case-model score.",
    "card_window": "Transactions on a card within N hours of a moment, with device/region/email context.",
    "card_baseline": "The card's behaviour before a moment: regions, products, channels, devices, emails, amounts.",
    "device_neighbors": "Other cards that used the same device profile in a window, with proxy/new-device flags and case history.",
    "device_ring": "Windowed connected component of cards linked through shared (non-generic) device profiles.",
    "region_cluster": "Cards transacting in a billing region in a window and whether the region is new to each card.",
    "recurring_matches": "Earlier charges on the card with the same amount and product (recurring-charge check, R7).",
    "card_case_history": "Closed cases and agent cases on this card, the customer's other cards, and as a connected card.",
    "device_case_links": "Closed cases on any card that used this device profile.",
    "similar_closed_cases": "Vector search over closed-case profiles (TigerVector) for similar past investigations.",
    "search_docs": "Vector search over policy, typologies and regulatory guidance (GraphRAG).",
}


@dataclass
class ToolCall:
    tool: str
    args: dict
    ms: float
    summary: str = ""


class GraphTools(ABC):
    backend = "abstract"

    def __init__(self) -> None:
        self.log: list[ToolCall] = []

    def _record(self, tool: str, args: dict, t0: float, summary: str = "") -> None:
        self.log.append(ToolCall(tool, {k: v for k, v in args.items() if k != "qv"}, (time.perf_counter() - t0) * 1000, summary))

    def call(self, tool: str, **args: Any) -> Any:
        t0 = time.perf_counter()
        out = getattr(self, f"_{tool}")(**args)
        self._record(tool, args, t0, _summarise(out))
        return out

    # each backend implements the underscore methods
    @abstractmethod
    def _txn_detail(self, txn_id: str) -> dict: ...
    @abstractmethod
    def _card_window(self, card: str, center_ts: str, hours_before: int = 48, hours_after: int = 48) -> list[dict]: ...
    @abstractmethod
    def _card_baseline(self, card: str, before_ts: str, lookback_days: int = 150) -> dict: ...
    @abstractmethod
    def _device_neighbors(self, device: str, center_ts: str, days: int = 30) -> dict: ...
    @abstractmethod
    def _device_ring(self, seed: str, center_ts: str, days: int = 30, hops: int = 2, max_device_cards: int = 60) -> dict: ...
    @abstractmethod
    def _region_cluster(self, region: str, center_ts: str, hours: int = 72, channel: str = "in_person") -> list[dict]: ...
    @abstractmethod
    def _recurring_matches(self, card: str, before_ts: str, amt: float, product: str, tolerance: float = 0.02) -> list[dict]: ...
    @abstractmethod
    def _card_case_history(self, card: str) -> list[dict]: ...
    @abstractmethod
    def _device_case_links(self, device: str) -> list[dict]: ...
    @abstractmethod
    def _similar_closed_cases(self, qv: list[float], k: int = 8) -> list[dict]: ...
    @abstractmethod
    def _search_docs(self, qv: list[float], k: int = 5) -> list[dict]: ...

    def write_case(self, case: dict) -> bool:
        return False


def _summarise(out: Any) -> str:
    if isinstance(out, list):
        return f"{len(out)} rows"
    if isinstance(out, dict):
        return ", ".join(f"{k}={len(v) if isinstance(v, (list, dict, set)) else v}" for k, v in list(out.items())[:5])
    return str(out)[:80]


# ------------------------------------------------------------------------------------------
# TigerGraph backend
# ------------------------------------------------------------------------------------------
class TigerGraphTools(GraphTools):
    backend = "tigergraph"

    def __init__(self, tg: TigerGraph | None = None) -> None:
        super().__init__()
        self.tg = tg or TigerGraph()

    def _q(self, name: str, **p) -> list[dict]:
        return self.tg.query(name, **p)

    def _txn_detail(self, txn_id: str) -> dict:
        r = self._q("txn_detail", txn=str(txn_id))
        return r[0]["txn"] if r else {}

    def _card_window(self, card, center_ts, hours_before=48, hours_after=48):
        r = self._q("card_window", card=card, center_ts=center_ts, hours_before=hours_before, hours_after=hours_after)
        return sorted(r[0]["txns"], key=lambda x: x["ts"]) if r else []

    def _card_baseline(self, card, before_ts, lookback_days=150):
        r = self._q("card_baseline", card=card, before_ts=before_ts, lookback_days=lookback_days)
        return r[0] if r else {}

    def _device_neighbors(self, device, center_ts, days=30):
        r = self._q("device_neighbors", device=device, center_ts=center_ts, days=days)
        return r[0] if r else {}

    def _device_ring(self, seed, center_ts, days=30, hops=2, max_device_cards=60):
        r = self._q("device_ring", seed=seed, center_ts=center_ts, days=days, hops=hops, max_device_cards=max_device_cards)
        return r[0] if r else {"cards": [], "devices": []}

    def _region_cluster(self, region, center_ts, hours=72, channel="in_person"):
        r = self._q("region_cluster", region=region, center_ts=center_ts, hours=hours, channel=channel)
        return r[0]["cards"] if r else []

    def _recurring_matches(self, card, before_ts, amt, product, tolerance=0.02):
        r = self._q("recurring_matches", card=card, before_ts=before_ts, amt=amt, product=product, tolerance=tolerance)
        return r[0]["matches"] if r else []

    def _card_case_history(self, card):
        r = self._q("card_case_history", card=card)
        return r[0]["cases"] if r else []

    def _device_case_links(self, device):
        r = self._q("device_case_links", device=device)
        return r[0]["links"] if r else []

    def _similar_closed_cases(self, qv, k=8):
        r = self._q("similar_closed_cases", qv=qv, k=k)
        if not r:
            return []
        dist = r[1].get("distances", {}) if len(r) > 1 else {}
        out = []
        for v in r[0]["cases"]:
            a = v["attributes"]
            out.append({"case_id": v["v_id"], "outcome": a.get("R.outcome"), "pattern": a.get("R.pattern"),
                        "exposure": a.get("R.exposure"), "actions": a.get("R.actions"),
                        "report_filed": a.get("R.report_filed"), "notes": a.get("R.notes"),
                        "distance": dist.get(v["v_id"])})
        return sorted(out, key=lambda x: x["distance"] if x["distance"] is not None else 1)

    def _search_docs(self, qv, k=5):
        r = self._q("search_docs", qv=qv, k=k)
        if not r:
            return []
        dist = r[1].get("distances", {}) if len(r) > 1 else {}
        out = [{"id": v["v_id"], "source": v["attributes"].get("R.source"), "section": v["attributes"].get("R.section"),
                "text": v["attributes"].get("R.text"), "distance": dist.get(v["v_id"])} for v in r[0]["docs"]]
        return sorted(out, key=lambda x: x["distance"] if x["distance"] is not None else 1)

    def write_case(self, case: dict) -> bool:
        from fraudagent.memory.case_store import case_to_graph_payload
        vertices, edges = case_to_graph_payload(case)
        t0 = time.perf_counter()
        self.tg.upsert(vertices, edges, name="write_case")
        self._record("write_case", {"case": case["graph_case_id"]}, t0, "upserted")
        return True


# ------------------------------------------------------------------------------------------
# Local mirror (tests / UI development without a database)
# ------------------------------------------------------------------------------------------
class LocalTools(GraphTools):
    backend = "local-mirror"
    _tx: pd.DataFrame | None = None

    def __init__(self) -> None:
        super().__init__()
        if LocalTools._tx is None:
            tx = pd.read_parquet(settings.cache_dir / "txn_enriched.parquet")
            sc = pd.read_parquet(settings.cache_dir / "case_model_scores.parquet")
            tx = tx.merge(sc, on="TransactionID", how="left")
            tx["ts"] = pd.to_datetime(tx["ts"])
            tx["tid"] = tx["TransactionID"].astype(str)
            LocalTools._tx = tx
            closed = pd.read_csv(settings.data_dir / "closed_cases_history.csv")
            LocalTools._closed = closed
            prof = pd.read_parquet(settings.cache_dir / "closed_case_profiles.parquet")
            LocalTools._profiles = prof
            LocalTools._docs = None
        self.tx = LocalTools._tx
        self.closed = LocalTools._closed
        self.agent_cases: list[dict] = []

    @staticmethod
    def _row(r) -> dict:
        g = lambda k: "" if pd.isna(getattr(r, k)) else getattr(r, k)  # noqa: E731
        return {"id": r.tid, "ts": r.ts.strftime(TS_FMT), "amt": float(r.TransactionAmt), "product": g("ProductCD"),
                "channel": r.channel, "risk": float(r.risk_score), "region": g("region"),
                "p_email": g("P_emaildomain"), "r_email": g("R_emaildomain"), "device": g("device_id"),
                "dev_status": g("id_15"), "proxy_type": g("id_23"), "m4": g("M4"), "m6": g("M6"),
                "cm_score": float(r.case_model_prob), "card": r.card_id}

    def _txn_detail(self, txn_id):
        r = self.tx[self.tx.tid == str(txn_id)]
        return self._row(next(r.itertuples())) if len(r) else {}

    def _card_window(self, card, center_ts, hours_before=48, hours_after=48):
        c = pd.Timestamp(center_ts)
        w = self.tx[(self.tx.card_id == card) & (self.tx.ts >= c - timedelta(hours=hours_before)) & (self.tx.ts <= c + timedelta(hours=hours_after))]
        return [self._row(r) for r in w.sort_values("ts").itertuples()]

    def _card_baseline(self, card, before_ts, lookback_days=150):
        c = pd.Timestamp(before_ts)
        h = self.tx[(self.tx.card_id == card) & (self.tx.ts < c) & (self.tx.ts >= c - timedelta(days=lookback_days))]
        vc = lambda s: {str(k): int(v) for k, v in s[s != ""].value_counts().items()} if len(s) else {}  # noqa: E731
        return {"n_txns": len(h), "total_amt": float(h.TransactionAmt.sum()), "amounts": h.TransactionAmt.round(2).tolist(),
                "regions": vc(h.region.fillna("")), "products": vc(h.ProductCD.fillna("")), "channels": vc(h.channel),
                "emails": vc(h.P_emaildomain.fillna("")), "devices": vc(h.device_id.fillna("")),
                "first_seen": h.ts.min().strftime(TS_FMT) if len(h) else "", "last_seen": h.ts.max().strftime(TS_FMT) if len(h) else ""}

    def _device_neighbors(self, device, center_ts, days=30):
        c = pd.Timestamp(center_ts)
        d = self.tx[self.tx.device_id == device]
        w = d[(d.ts >= c - timedelta(days=days)) & (d.ts <= c + timedelta(days=days))]
        uses = [{"card": r.card_id, "txn": r.tid, "ts": r.ts.strftime(TS_FMT), "amt": float(r.TransactionAmt),
                 "product": r.ProductCD, "risk": float(r.risk_score), "dev_status": "" if pd.isna(r.id_15) else r.id_15,
                 "proxy_type": "" if pd.isna(r.id_23) else r.id_23, "region": r.region} for r in w.itertuples()]
        cards = sorted(set(w.card_id))
        cc = self.closed[self.closed.card_id.isin(cards)]
        return {"uses": uses, "cards": cards,
                "card_cases": {r.card_id: f"{r.case_id}:{r.outcome}:{r.pattern}" for r in cc.itertuples()},
                "all_time_txns": len(d), "all_time_cards": d.card_id.nunique()}

    def _device_ring(self, seed, center_ts, days=30, hops=2, max_device_cards=60):
        c = pd.Timestamp(center_ts)
        w = self.tx[(self.tx.ts >= c - timedelta(days=days)) & (self.tx.ts <= c + timedelta(days=days)) & self.tx.device_id.ne("") & self.tx.device_id.notna()]
        fan = w.groupby("device_id").card_id.nunique()
        ok = set(fan[fan <= max_device_cards].index)
        cards, devices, frontier = {seed}, set(), {seed}
        for _ in range(hops):
            devs = set(w[w.card_id.isin(frontier) & w.device_id.isin(ok)].device_id)
            devices |= devs
            nxt = set(w[w.device_id.isin(devs)].card_id) - cards
            cards |= nxt
            frontier = nxt
            if not frontier:
                break
        return {"cards": sorted(cards), "devices": sorted(devices)}

    def _region_cluster(self, region, center_ts, hours=72, channel="in_person"):
        c = pd.Timestamp(center_ts)
        lo = c - timedelta(hours=hours)
        w = self.tx[(self.tx.region == region) & (self.tx.channel == channel) & (self.tx.ts >= lo) & (self.tx.ts <= c + timedelta(hours=hours))]
        prior = self.tx[self.tx.card_id.isin(set(w.card_id)) & (self.tx.ts < lo)]
        pr = prior.groupby("card_id").agg(total=("tid", "size"), in_region=("region", lambda s: int((s == region).sum())))
        g = w.groupby("card_id").agg(n=("tid", "size"), amt=("TransactionAmt", "sum"), mr=("risk_score", "max"))
        return [{"card": k, "n_in_window": int(r.n), "amt_in_window": float(r.amt),
                 "prior_in_region": int(pr.in_region.get(k, 0)), "prior_total": int(pr.total.get(k, 0)),
                 "max_risk": float(r.mr)} for k, r in g.iterrows()]

    def _recurring_matches(self, card, before_ts, amt, product, tolerance=0.02):
        c = pd.Timestamp(before_ts)
        m = self.tx[(self.tx.card_id == card) & (self.tx.ts < c) & (self.tx.ProductCD == product) & ((self.tx.TransactionAmt - amt).abs() <= amt * tolerance)]
        return [{"id": r.tid, "ts": r.ts.strftime(TS_FMT), "amt": float(r.TransactionAmt),
                 "p_email": "" if pd.isna(r.P_emaildomain) else r.P_emaildomain, "channel": r.channel} for r in m.itertuples()]

    def _card_case_history(self, card):
        cust = card.split("-")[0]
        out = []
        for r in self.closed[self.closed.customer_id == cust].itertuples():
            out.append({"case_id": r.case_id, "card": r.card_id, "relation": "same_card" if r.card_id == card else "same_customer",
                        "outcome": r.outcome, "pattern": r.pattern, "exposure": float(r.exposure_usd),
                        "opened_at": r.opened_at[:10], "notes": r.analyst_notes})
        for r in self.closed[self.closed.connected_card_ids.fillna("").str.contains(card)].itertuples():
            out.append({"case_id": r.case_id, "card": card, "relation": "connected_card", "outcome": r.outcome,
                        "pattern": r.pattern, "exposure": float(r.exposure_usd), "opened_at": r.opened_at[:10], "notes": r.analyst_notes})
        for c in self.agent_cases:
            if c["card_id"].split("-")[0] == cust:
                out.append({"case_id": c["graph_case_id"], "card": c["card_id"], "relation": "agent_case", "outcome": c["status"],
                            "pattern": c["pattern"], "exposure": c["exposure_usd"], "opened_at": c["opened_at"][:10], "notes": c["summary"]})
        return out

    def _device_case_links(self, device):
        tids = set(self.tx[self.tx.device_id == device].TransactionID.astype(str))
        out = []
        for r in self.closed.itertuples():
            if tids.intersection(str(r.txn_ids).split("|")):
                out.append({"case_id": r.case_id, "card": r.card_id, "outcome": r.outcome, "pattern": r.pattern, "notes": r.analyst_notes})
        return out

    def _similar_closed_cases(self, qv, k=8):
        from fraudagent.memory.case_store import local_vectors
        ids, mat = local_vectors("closed")
        sims = mat @ np.asarray(qv, dtype="float32")
        top = np.argsort(-sims)[:k]
        c = self.closed.set_index("case_id")
        return [{"case_id": ids[i], "outcome": c.at[ids[i], "outcome"], "pattern": c.at[ids[i], "pattern"],
                 "exposure": float(c.at[ids[i], "exposure_usd"]), "actions": c.at[ids[i], "actions_taken"],
                 "report_filed": c.at[ids[i], "report_filed"] == "Yes", "notes": c.at[ids[i], "analyst_notes"],
                 "distance": float(1 - sims[i])} for i in top]

    def _search_docs(self, qv, k=5):
        from fraudagent.memory.case_store import local_vectors, load_doc_chunks
        ids, mat = local_vectors("docs")
        sims = mat @ np.asarray(qv, dtype="float32")
        chunks = {c["id"]: c for c in load_doc_chunks()}
        return [{**chunks[ids[i]], "distance": float(1 - sims[i])} for i in np.argsort(-sims)[:k]]

    def write_case(self, case: dict) -> bool:
        self.agent_cases.append(case)
        return False


def make_tools(prefer: str = "tigergraph") -> GraphTools:
    if prefer == "mcp" or (prefer == "tigergraph" and settings.use_mcp):
        from fraudagent.graph.mcp_tools import MCPTools
        return MCPTools()
    if prefer == "tigergraph":
        tg = TigerGraph()
        if tg.ping():
            return TigerGraphTools(tg)
    return LocalTools()
