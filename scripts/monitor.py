"""Autonomous monitoring of the exam period (Nov-Dec), beyond the 20 case-pack alerts.

Sweeps the whole period for the two coordinated typologies the closed cases could not name
(sub-$500 structuring bursts and anonymous-proxy device rings), raises an alert for each hit,
and lets the same agent investigate it end to end. Output goes to monitor/ (one answer file per
alert plus alerts.json), kept apart from the scored cases/ folder.

    python scripts/monitor.py [--local] [--limit 15]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fraudagent.agent.detectors import is_generic_device  # noqa: E402
from fraudagent.agent.investigator import Investigator, Trigger  # noqa: E402
from fraudagent.agent.llm import LLM  # noqa: E402
from fraudagent.config import ROOT, settings  # noqa: E402
from fraudagent.graph.tools import make_tools  # noqa: E402

OUT = ROOT / "monitor"


def sweep() -> list[dict]:
    tx = pd.read_parquet(settings.cache_dir / "txn_enriched.parquet",
                         columns=["TransactionID", "card_id", "customer_id", "ts", "TransactionAmt", "channel",
                                  "device_id", "id_23"])
    tx["ts"] = pd.to_datetime(tx["ts"])
    tx = tx[tx["ts"] >= "2016-11-01"].sort_values("ts")
    pack = set(pd.read_csv(settings.data_dir / "case_pack.csv")["card_id"])
    alerts: list[dict] = []

    # 1) structuring: >=3 online purchases in [$400, $500) on one card within 90 minutes
    band = tx[(tx["channel"] == "online") & tx["TransactionAmt"].between(400, 499.99)]
    for card, g in band.groupby("card_id"):
        times = g["ts"].tolist()
        for i in range(len(times) - 2):
            if times[i + 2] - times[i] <= pd.Timedelta(minutes=90):
                last = g.iloc[min(i + 3, len(g) - 1)] if i + 3 < len(g) and times[i + 3] - times[i] <= pd.Timedelta(minutes=90) else g.iloc[i + 2]
                alerts.append({"kind": "structuring", "card_id": card, "customer_id": card.split("-")[0],
                               "txn": str(last["TransactionID"]), "ts": str(last["ts"]),
                               "detail": f"{card}: 3+ online purchases of $400-$500 within 90 minutes"})
                break

    # 2) device rings: a specific handset behind anonymous proxies on 5+ cards in 30 days
    anon = tx[tx["id_23"].fillna("").str.contains("ANONYMOUS|HIDDEN") & tx["device_id"].fillna("").ne("")]
    for dev, g in anon.groupby("device_id"):
        if g["card_id"].nunique() < 5 or is_generic_device(dev, 0):
            continue
        for card, cg in g.groupby("card_id"):
            last = cg.iloc[-1]
            alerts.append({"kind": "device_ring", "card_id": card, "customer_id": card.split("-")[0],
                           "txn": str(last["TransactionID"]), "ts": str(last["ts"]),
                           "detail": f"{card} used device '{dev}' shared by {g['card_id'].nunique()} cards behind anonymous proxies"})
    return [a for a in alerts if a["card_id"] not in pack]


def main(args: list[str]) -> None:
    limit = int(args[args.index("--limit") + 1]) if "--limit" in args else 15
    alerts = sweep()
    print(f"sweep found {len(alerts)} alerts outside the case pack")
    OUT.mkdir(exist_ok=True)
    agent = Investigator(make_tools("local" if "--local" in args else settings.graph_backend), LLM())
    results = []
    by_kind: dict[str, int] = {}
    for i, a in enumerate(alerts):
        if by_kind.get(a["kind"], 0) >= limit:
            continue
        by_kind[a["kind"]] = by_kind.get(a["kind"], 0) + 1
        cid = f"MON-{i + 1:03d}"
        trig = Trigger(cid, a["ts"], "analyst_request", f"Monitoring alert ({a['kind']}): {a['detail']}. "
                       f"Review transaction {a['txn']} on card {a['card_id']}.", a["txn"], a["card_id"], a["customer_id"])
        case = agent.run(trig)
        (OUT / f"{cid}.json").write_text(json.dumps({k: v for k, v in case.items() if k not in ("embedding",)},
                                                    indent=2, default=str) + "\n")
        c = case["case"]
        results.append({"case_id": cid, **a, "verdict": c["verdict"], "probability": c["fraud_probability"],
                        "pattern": c["pattern"], "exposure_usd": c["exposure_usd"], "sar": case["sar"]["file"]})
        print(f"{cid} {a['kind']:12} {a['card_id']} -> {c['verdict']} p={c['fraud_probability']} ${c['exposure_usd']}")
    (OUT / "alerts.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
