"""Learn from case memory: a transaction-level fraud model trained on the bank's closed cases.

The risk score attached to every transaction is noisy (most alerts above 0.7 are cleared,
and plenty of confirmed fraud scores low). The closed cases are the only place the truth is
written down, so we use them as labels:

  positive  = a transaction inside a confirmed-fraud closed case (Jul-Oct)
  negative  = everything else in Jul-Oct, with cleared-alert transactions up-weighted
              because they are the hard negatives the analysts already looked at

Train on Jul-Sep, calibrate on October (isotonic), then score every transaction.
The score becomes one piece of evidence for the agent, never a verdict on its own.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow.csv as pacsv
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data"
CACHE = ROOT / "data" / "cache"
DROP = {"TransactionID", "TransactionDT", "ts", "customer_id", "month"}


def load() -> pd.DataFrame:
    tx = pacsv.read_csv(RAW / "transactions.csv").to_pandas()
    ident = pacsv.read_csv(RAW / "identity.csv").to_pandas()
    df = tx.merge(ident, on="TransactionID", how="left")
    df["ts"] = pd.to_datetime(df["ts"])
    df["month"] = df["ts"].dt.month
    df["hour"] = df["ts"].dt.hour
    for c in df.columns:
        if c not in DROP and not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].astype("category").cat.codes.replace(-1, np.nan)
    return df


def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    df = load()
    closed = pd.read_csv(RAW / "closed_cases_history.csv")
    ids = lambda frame: {int(x) for s in frame["txn_ids"] for x in str(s).split("|") if x}  # noqa: E731
    fraud_ids = ids(closed[closed["outcome"] == "confirmed_fraud"])
    cleared_ids = ids(closed[closed["outcome"] == "cleared"])

    features = [c for c in df.columns if c not in DROP]
    lab = df[df["month"] <= 10]
    y = lab["TransactionID"].isin(fraud_ids).astype(int)
    w = np.where(lab["TransactionID"].isin(cleared_ids), 5.0, 1.0)
    train, calib = lab["month"] <= 9, lab["month"] == 10

    model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=63,
                                           random_state=7)
    model.fit(lab.loc[train, features].astype("float32"), y[train], sample_weight=w[train])
    raw_cal = model.predict_proba(lab.loc[calib, features].astype("float32"))[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip").fit(raw_cal, y[calib])

    metrics = {
        "holdout_month": "2016-10",
        "auc_model": round(roc_auc_score(y[calib], raw_cal), 4),
        "ap_model": round(average_precision_score(y[calib], raw_cal), 4),
        "auc_bank_risk_score": round(roc_auc_score(y[calib], lab.loc[calib, "risk_score"]), 4),
        "ap_bank_risk_score": round(average_precision_score(y[calib], lab.loc[calib, "risk_score"]), 4),
        "n_train": int(train.sum()), "n_positive_train": int(y[train].sum()),
    }
    case_rows = calib & lab["TransactionID"].isin(fraud_ids | cleared_ids)
    if case_rows.any():
        p_case = model.predict_proba(lab.loc[case_rows, features].astype("float32"))[:, 1]
        metrics["auc_confirmed_vs_cleared_model"] = round(roc_auc_score(y[case_rows], p_case), 4)
        metrics["auc_confirmed_vs_cleared_bank"] = round(
            roc_auc_score(y[case_rows], lab.loc[case_rows, "risk_score"]), 4)
    print(json.dumps(metrics, indent=2))

    raw_all = model.predict_proba(df[features].astype("float32"))[:, 1]
    scores = pd.DataFrame({
        "TransactionID": df["TransactionID"],
        "case_model_raw": raw_all.round(5),
        "case_model_prob": iso.predict(raw_all).round(5),
    })
    scores.to_parquet(CACHE / "case_model_scores.parquet", index=False)
    joblib.dump({"model": model, "iso": iso, "features": features}, CACHE / "case_model.joblib")
    (ROOT / "docs" / "case_model_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")


if __name__ == "__main__":
    main()
