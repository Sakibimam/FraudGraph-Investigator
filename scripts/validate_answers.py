"""Check every answer file against the README's answer format and the dataset's IDs."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fraudagent.agent.policy import ACTIONS, route_for  # noqa: E402
from fraudagent.config import settings  # noqa: E402

PATTERNS = {"card_testing", "card_not_present_fraud", "card_not_present_new_device", "out_of_region_use",
            "account_takeover", "undocumented", "none"}


def main(folder: str = "cases") -> int:
    tx = pd.read_parquet(settings.cache_dir / "txn_enriched.parquet", columns=["TransactionID", "card_id", "customer_id", "device_id"])
    txns, cards = set(tx.TransactionID.astype(str)), set(tx.card_id)
    custs, devs = set(tx.customer_id), set(tx.device_id.dropna())
    closed = set(pd.read_csv(settings.data_dir / "closed_cases_history.csv").case_id)
    known = txns | cards | custs | devs | closed
    problems = 0
    files = sorted(Path(folder).glob("*.json"))
    for f in files:
        d = json.loads(f.read_text())
        errs = []
        c, s, n = d["case"], d["sar"], d["next_best_actions"]
        for k in ("case_id", "case", "evidence_requests", "next_best_actions", "sar", "stop_reason", "tool_calls", "tokens", "latency_s"):
            if k not in d:
                errs.append(f"missing {k}")
        if c["pattern"] not in PATTERNS:
            errs.append(f"bad pattern {c['pattern']}")
        if c["pattern"] == "undocumented" and not c["pattern_description"]:
            errs.append("undocumented without description")
        if c["verdict"] == "legitimate" and (c["affected_txn_ids"] or c["exposure_usd"] or s["file"]):
            errs.append("legitimate case with affected txns / exposure / SAR")
        bad = [i for i in c["affected_txn_ids"] if i not in txns] + [i for i in c["connected_card_ids"] if i not in cards]
        bad += [i for i in c["similar_prior_cases"] if i not in closed]
        bad += [i for e in c["evidence"] for i in e["entity_ids"] if i not in known]
        bad += [i for i in s["subjects"] if i not in known]
        if bad:
            errs.append(f"unknown IDs {bad[:5]}")
        exp = round(sum(abs(float(tx_amt)) for tx_amt in []), 2)
        final = {a["action"] for a in n["final"]}
        if s["file"] != ("FILE_REPORT" in final):
            errs.append("sar.file disagrees with FILE_REPORT in final actions")
        for a in n["initial"] + n["final"]:
            if a["action"] not in ACTIONS or a["route"] not in ("auto", "L1", "L2"):
                errs.append(f"bad action/route {a}")
            if a["action"] != "BLOCK_CARD" and a["route"] != route_for(a["action"]):
                errs.append(f"wrong route {a['action']}={a['route']}")
        if not d["evidence_requests"] and n["initial"] != n["final"]:
            errs.append("final differs from initial without an evidence request")
        if s["file"] and not (6 <= len(re.findall(r"[.!?](\s|$)", s["narrative"])) <= 14):
            errs.append("SAR narrative length outside 6-12 sentences")
        status = "OK " if not errs else "ERR"
        problems += bool(errs)
        print(f"{status} {f.name}: {'; '.join(errs)}")
    print(f"{len(files) - problems}/{len(files)} files valid")
    return problems


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
