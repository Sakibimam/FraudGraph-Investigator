"""Back-test the agent on closed cases from October (not used to train the case model).

Each closed case is replayed as a risk-score alert on its first transaction; the agent's verdict
and pattern are compared with the analysts' recorded outcome. The simulator is switched to
"no customer contact" so the verdict rests on graph evidence alone.

    python scripts/backtest.py [--n 200] [--local]
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fraudagent.agent.investigator import Investigator, Trigger  # noqa: E402
from fraudagent.agent.llm import LLM  # noqa: E402
from fraudagent.config import ROOT, settings  # noqa: E402
from fraudagent.graph.tools import make_tools  # noqa: E402


def main(args: list[str]) -> None:
    n = int(args[args.index("--n") + 1]) if "--n" in args else 200
    cc = pd.read_csv(settings.data_dir / "closed_cases_history.csv")
    oct_ = cc[cc["opened_at"].str.startswith("2016-10")]
    sample = pd.concat([oct_[oct_.outcome == "cleared"].sample(n // 2, random_state=1),
                        oct_[oct_.outcome == "confirmed_fraud"].sample(n // 2, random_state=1)])
    quiet = LLM()
    quiet.backends = []  # the back-test measures graph evidence only
    agent = Investigator(make_tools("local" if "--local" in args else settings.graph_backend), quiet)
    rows = []
    for r in sample.itertuples():
        tx = str(r.first_fraud_txn_id) if isinstance(r.first_fraud_txn_id, (int, float)) and r.first_fraud_txn_id == r.first_fraud_txn_id else str(r.txn_ids).split("|")[0]
        tx = tx.split(".")[0]
        trig = Trigger(r.case_id, r.opened_at, "risk_score", f"Replay of {r.case_id}", tx, r.card_id, r.customer_id)
        out = agent.run(trig, persist=False)
        p0 = next(e["probability"] for e in out["timeline"] if e["step"] == "assess")
        rows.append({"case": r.case_id, "truth": r.outcome, "pattern": r.pattern, "p_graph": p0,
                     "verdict_graph": "fraud" if p0 >= 0.5 else "legitimate", "agent_pattern": out["case"]["pattern"]})
    df = pd.DataFrame(rows)
    y = df.truth == "confirmed_fraud"
    pred = df.verdict_graph == "fraud"
    from sklearn.metrics import roc_auc_score
    res = {
        "n": len(df), "accuracy": round((y == pred).mean(), 3),
        "fraud_recall": round((pred & y).sum() / y.sum(), 3), "legit_specificity": round((~pred & ~y).sum() / (~y).sum(), 3),
        "auc_agent_probability": round(roc_auc_score(y, df.p_graph), 3),
        "pattern_match_on_confirmed": round((df[y].pattern == df[y].agent_pattern).mean(), 3),
        "pattern_confusion": {f"{a}->{b}": c for (a, b), c in Counter(zip(df[y].pattern, df[y].agent_pattern, strict=True)).most_common(12)},
    }
    print(json.dumps(res, indent=2))
    (ROOT / "docs" / "backtest_october.json").write_text(json.dumps(res, indent=2) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
