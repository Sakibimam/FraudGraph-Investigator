"""Load the prepared CSVs into TigerGraph through RESTPP upserts, in batches.

Equivalent to the GSQL loading job in gsql/load.gsql, but it only needs the REST endpoint,
so it also works against a remote Savanna workspace (set TG_HOST / TG_USERNAME / TG_PASSWORD).

    python scripts/load_graph_rest.py            # everything
    python scripts/load_graph_rest.py txn made   # selected steps
"""
from __future__ import annotations

import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fraudagent.config import settings  # noqa: E402
from fraudagent.graph.client import TigerGraph  # noqa: E402

G = settings.data_dir / "graph"
BATCH = 4000


def vals(row: dict, skip: str = "id") -> dict:
    return {k: {"value": v} for k, v in row.items() if k != skip}


def push_vertices(tg: TigerGraph, vtype: str, df: pd.DataFrame, id_col: str = "id") -> None:
    recs = df.to_dict("records")
    t0 = time.time()
    for i in range(0, len(recs), BATCH):
        chunk = recs[i:i + BATCH]
        tg.upsert({vtype: {str(r[id_col]): vals(r, id_col) for r in chunk}}, name=f"load_{vtype}")
    print(f"  {vtype:18} {len(recs):>8,} vertices  {time.time() - t0:6.1f}s")


def push_edges(tg: TigerGraph, src_type: str, etype: str, dst_type: str, df: pd.DataFrame,
               attrs: list[str] | None = None) -> None:
    rows = df.values.tolist()
    t0 = time.time()
    for i in range(0, len(rows), BATCH):
        body: dict = defaultdict(dict)
        for r in rows[i:i + BATCH]:
            a = {k: {"value": v} for k, v in zip(attrs or [], r[2:])}
            body[str(r[0])].setdefault(etype, {}).setdefault(dst_type, {})[str(r[1])] = a
        tg.upsert(edges={src_type: dict(body)}, name=f"load_{etype}")
    print(f"  {etype:18} {len(rows):>8,} edges     {time.time() - t0:6.1f}s")


def main(steps: set[str]) -> None:
    tg = TigerGraph(timeout=900)
    run = lambda s: not steps or s in steps  # noqa: E731
    rd = lambda f, **kw: pd.read_csv(G / f, keep_default_na=False, **kw)  # noqa: E731

    if run("txn"):
        tx = rd("txn.csv", dtype={"id": str, "addr1": str, "addr2": str})
        scores = pd.read_parquet(settings.cache_dir / "case_model_scores.parquet")
        scores["id"] = scores["TransactionID"].astype(str)
        tx = tx.merge(scores[["id", "case_model_prob"]].rename(columns={"case_model_prob": "cm_score"}), on="id", how="left")
        tx["cm_score"] = tx["cm_score"].fillna(0.0)
        push_vertices(tg, "Txn", tx)
    if run("card"):
        push_vertices(tg, "Card", rd("card.csv", dtype={"home_region": str}))
        push_vertices(tg, "Customer", rd("customer.csv"))
        push_edges(tg, "Customer", "OWNS", "Card", rd("owns.csv"))
    if run("device"):
        push_vertices(tg, "DeviceProfile", rd("device.csv"))
    if run("made"):
        push_edges(tg, "Card", "MADE", "Txn", rd("made.csv", dtype=str))
    if run("from_device"):
        push_edges(tg, "Txn", "FROM_DEVICE", "DeviceProfile", rd("from_device.csv", dtype=str))
    if run("email"):
        e = rd("email.csv", dtype=str)
        push_vertices(tg, "EmailDomain", pd.DataFrame({"id": e.iloc[:, 1].unique()}))
        push_edges(tg, "Txn", "PURCHASER_EMAIL", "EmailDomain", e)
    if run("region"):
        r = rd("region.csv", dtype=str)
        push_vertices(tg, "BillingRegion", pd.DataFrame({"id": r.iloc[:, 1].unique()}))
        push_edges(tg, "Txn", "BILLED_IN", "BillingRegion", r)
    if run("next"):
        push_edges(tg, "Txn", "NEXT", "Txn", rd("next.csv", dtype={"src": str, "dst": str}), attrs=["gap_s"])
    if run("cases"):
        cc = rd("closed_case.csv")
        cc["report_filed"] = cc["report_filed"].astype(str).str.lower() == "true"
        push_vertices(tg, "ClosedCase", cc)
        push_edges(tg, "ClosedCase", "ON_CARD", "Card", rd("on_card.csv", dtype=str))
        push_edges(tg, "ClosedCase", "CASE_PATTERN", "FraudPattern", rd("case_pattern.csv", dtype=str))
        push_edges(tg, "ClosedCase", "INVOLVES", "Txn", rd("involves.csv", dtype=str))
        push_edges(tg, "ClosedCase", "CONNECTED_TO", "Card", rd("connected.csv", dtype=str))


if __name__ == "__main__":
    main(set(sys.argv[1:]))
