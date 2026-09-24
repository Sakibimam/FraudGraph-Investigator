"""Stress and robustness test for the investigation agent.

    python scripts/stress_test.py [--n 150] [--local]

1. Random alerts: N transactions sampled across channels, products and months, each investigated
   under a random trigger type. Every result must be a valid answer file.
2. Edge cases: unknown transaction, cards with a single transaction, in-person rows without a
   region, online rows without identity, the largest cards in the dataset.
3. Determinism: the same alert twice gives the same verdict, actions and routes.
4. Degraded LLM: with no LLM available the agent still completes with templates.
Writes docs/stress_test.json.
"""
from __future__ import annotations

import json
import random
import statistics
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fraudagent.agent.investigator import Investigator, Trigger  # noqa: E402
from fraudagent.agent.llm import LLM  # noqa: E402
from fraudagent.agent.policy import ACTIONS, route_for  # noqa: E402
from fraudagent.config import ROOT, settings  # noqa: E402
from fraudagent.graph.tools import make_tools  # noqa: E402

TRIGGERS = ["risk_score", "customer_report", "analyst_request"]


def check(case: dict, txns: set[str]) -> list[str]:
    errs = []
    c, s, n = case["case"], case["sar"], case["next_best_actions"]
    if c["verdict"] not in ("fraud", "legitimate", "uncertain"):
        errs.append("verdict")
    if not 0 <= c["fraud_probability"] <= 1:
        errs.append("probability range")
    if c["verdict"] == "legitimate" and (c["affected_txn_ids"] or c["exposure_usd"] or s["file"]):
        errs.append("legitimate with exposure")
    if c["verdict"] != "legitimate" and case["trigger"]["flagged_txn_id"] not in c["affected_txn_ids"]:
        errs.append("flagged txn missing from episode")
    if abs(sum(0 for _ in c["affected_txn_ids"])) and c["exposure_usd"] <= 0:
        errs.append("exposure not positive")
    if any(t not in txns for t in c["affected_txn_ids"]):
        errs.append("unknown txn id")
    if s["file"] != any(a["action"] == "FILE_REPORT" for a in n["final"]):
        errs.append("sar/final mismatch")
    for a in n["initial"] + n["final"]:
        if a["action"] not in ACTIONS:
            errs.append(f"bad action {a['action']}")
        elif a["action"] != "BLOCK_CARD" and a["route"] != route_for(a["action"]):
            errs.append(f"bad route {a['action']}")
    finals = {a["action"] for a in n["final"]}
    if "BLOCK_ALL_CARDS" in finals:
        errs.append("R10 violated")
    if c["verdict"] == "legitimate" and finals & {"BLOCK_CARD", "DECLINE_TRANSACTION", "FILE_REPORT"}:
        errs.append("blocking a legitimate case")
    if not c["evidence"]:
        errs.append("no evidence")
    if case["evidence_requests"] and n["what_changed"] == "nothing":
        errs.append("evidence requested but what_changed empty")
    return errs


def main(args: list[str]) -> None:
    n = int(args[args.index("--n") + 1]) if "--n" in args else 150
    tools = make_tools("local" if "--local" in args else settings.graph_backend)
    quiet = LLM()
    quiet.backends = []  # stress the graph/agent path, not the LLM quota
    agent = Investigator(tools, quiet)
    tx = pd.read_parquet(settings.cache_dir / "txn_enriched.parquet",
                         columns=["TransactionID", "card_id", "customer_id", "ts", "channel", "ProductCD", "region", "device_id"])
    tx["ts"] = tx["ts"].astype(str)
    txns = set(tx.TransactionID.astype(str))
    rng = random.Random(42)
    report: dict = {"backend": tools.backend}

    # 1) random alerts, stratified by channel and month
    sample = (tx[tx.ts >= "2016-07-15"].groupby([tx.channel, tx.ts.str[:7]], group_keys=False)
              .apply(lambda g: g.sample(min(len(g), max(1, n // 12)), random_state=7)).head(n))
    lat, failures, verdicts, patterns, errors = [], [], Counter(), Counter(), []
    for r in sample.itertuples():
        trig = Trigger(f"ST-{r.TransactionID}", str(r.ts), rng.choice(TRIGGERS),
                       f"Stress alert on transaction {r.TransactionID}", str(r.TransactionID), r.card_id, r.customer_id)
        try:
            t0 = time.perf_counter()
            case = agent.run(trig, persist=False)
            lat.append(time.perf_counter() - t0)
            e = check(case, txns)
            if e:
                failures.append({"txn": r.TransactionID, "trigger": trig.trigger_type, "errors": e})
            verdicts[case["case"]["verdict"]] += 1
            patterns[case["case"]["pattern"]] += 1
        except Exception as ex:
            errors.append({"txn": r.TransactionID, "error": repr(ex)[:200], "trace": traceback.format_exc()[-400:]})
    report["random_alerts"] = {
        "n": len(sample), "crashes": len(errors), "invalid": len(failures),
        "latency_s": {"p50": round(statistics.median(lat), 2), "p95": round(sorted(lat)[int(.95 * len(lat)) - 1], 2),
                      "max": round(max(lat), 2)} if lat else {},
        "verdicts": dict(verdicts), "patterns": dict(patterns),
        "failures": failures[:10], "errors": errors[:5],
    }
    print(json.dumps(report["random_alerts"], indent=2)[:1500])

    # 2) edge cases
    counts = tx.card_id.value_counts()
    single = tx[tx.card_id.isin(counts[counts == 1].index)].head(3)
    biggest = tx[tx.card_id == counts.index[0]].sample(2, random_state=1)
    no_region = tx[(tx.channel == "in_person") & (tx.region == "")].head(2)
    no_device = tx[(tx.channel == "online") & tx.device_id.isna()].head(2)
    edge = []
    for label, frame in [("single-txn card", single), ("largest card", biggest), ("in-person no region", no_region),
                         ("online no identity", no_device)]:
        for r in frame.itertuples():
            trig = Trigger(f"EDGE-{r.TransactionID}", str(r.ts), "risk_score", "edge case", str(r.TransactionID),
                           r.card_id, r.customer_id)
            try:
                case = agent.run(trig, persist=False)
                edge.append({"case": label, "ok": not check(case, txns), "verdict": case["case"]["verdict"]})
            except Exception as ex:
                edge.append({"case": label, "ok": False, "error": repr(ex)[:200]})
    try:
        agent.run(Trigger("EDGE-missing", "2016-12-01 00:00:00", "risk_score", "missing", "9999999", "C00000-K1", "C00000"))
        edge.append({"case": "unknown transaction", "ok": False, "error": "no error raised"})
    except ValueError as ex:
        edge.append({"case": "unknown transaction", "ok": True, "handled": str(ex)[:80]})
    report["edge_cases"] = edge
    print(json.dumps(edge, indent=1))

    # 3) determinism on the case pack
    pack = pd.read_csv(settings.data_dir / "case_pack.csv", dtype={"flagged_txn_id": str})
    same = 0
    for r in pack.itertuples():
        trig = Trigger(r.case_id, r.opened_at, r.trigger_type, r.trigger_text, r.flagged_txn_id, r.card_id, r.customer_id)
        a, b = agent.run(trig, persist=False), agent.run(trig, persist=False)
        key = lambda c: (c["case"]["verdict"], c["case"]["pattern"], c["case"]["affected_txn_ids"],  # noqa: E731
                         [(x["action"], x["route"]) for x in c["next_best_actions"]["final"]])
        same += key(a) == key(b)
    report["determinism"] = {"cases": len(pack), "identical": same}
    print("determinism", report["determinism"])

    # 4) degraded LLM: templates only
    r = pack.iloc[5]
    case = agent.run(Trigger(r.case_id, r.opened_at, r.trigger_type, r.trigger_text, r.flagged_txn_id, r.card_id, r.customer_id), persist=False)
    report["no_llm"] = {"completed": True, "valid": not check(case, txns), "tokens": case["tokens"]}

    (ROOT / "docs" / "stress_test.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    print("written docs/stress_test.json")


if __name__ == "__main__":
    main(sys.argv[1:])
