"""Run the agent on the case pack and write one answer file per case to cases/.

    python scripts/run_benchmark.py                 # all 20 cases
    python scripts/run_benchmark.py HHG-006 HHG-014 # selected cases
    python scripts/run_benchmark.py --local         # use the local mirror instead of TigerGraph
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fraudagent.agent.investigator import Investigator, Trigger  # noqa: E402
from fraudagent.agent.llm import LLM  # noqa: E402
from fraudagent.config import settings  # noqa: E402
from fraudagent.graph.tools import make_tools  # noqa: E402

SCORED_KEYS = ["case_id", "case", "evidence_requests", "next_best_actions", "sar", "stop_reason",
               "tool_calls", "tokens", "latency_s"]


def load_pack() -> list[Trigger]:
    pack = pd.read_csv(settings.data_dir / "case_pack.csv", dtype={"flagged_txn_id": str})
    return [Trigger(r.case_id, r.opened_at, r.trigger_type, r.trigger_text, str(r.flagged_txn_id), r.card_id,
                    r.customer_id, None if pd.isna(r.risk_score) else float(r.risk_score)) for r in pack.itertuples()]


def main(args: list[str]) -> None:
    tools = make_tools("local" if "--local" in args else "tigergraph")
    llm = LLM()
    print(f"graph backend: {tools.backend} | llm: {llm.label}")
    agent = Investigator(tools, llm)
    want = {a for a in args if a.startswith("HHG-")}
    out_dir = settings.cases_dir
    trace_dir = settings.cases_dir.parent / "runs" / "traces"
    out_dir.mkdir(exist_ok=True)
    trace_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for trig in load_pack():
        if want and trig.case_id not in want:
            continue
        case = agent.run(trig)
        answer = {k: case[k] for k in SCORED_KEYS}
        (out_dir / f"{trig.case_id}.json").write_text(json.dumps(answer, indent=2) + "\n")
        (trace_dir / f"{trig.case_id}.json").write_text(json.dumps(
            {k: v for k, v in case.items() if k != "embedding"}, indent=2, default=str) + "\n")
        c = case["case"]
        rows.append((trig.case_id, c["verdict"], c["fraud_probability"], c["pattern"], c["exposure_usd"],
                     case["sar"]["file"], " > ".join(a["action"] for a in case["next_best_actions"]["final"])))
        print(f"{trig.case_id} {c['verdict']:10} p={c['fraud_probability']:.2f} {c['pattern']:28} "
              f"${c['exposure_usd']:>9,.2f} sar={case['sar']['file']!s:5} calls={case['tool_calls']:>2} "
              f"| {rows[-1][-1]}")


if __name__ == "__main__":
    main(sys.argv[1:])
