"""Turn the HHGOA_IEEE CSVs into vertex/edge files for the TigerGraph loading jobs.

Outputs go to data/graph/. The raw files are large (708 MB), so we read them once
with pyarrow, keep only the columns the graph needs, and write compact CSVs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.csv as pacsv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fraudagent.graph.profiles import case_profile_text, device_profile_id  # noqa: E402

RAW = ROOT / "data"
OUT = ROOT / "data" / "graph"
CACHE = ROOT / "data" / "cache"

TXN_COLS = [
    "TransactionID", "TransactionDT", "TransactionAmt", "ProductCD", "card4", "card6",
    "addr1", "addr2", "dist1", "P_emaildomain", "R_emaildomain", "M1", "M4", "M6",
    "C1", "C13", "D1", "D15", "customer_id", "ts", "channel", "risk_score",
]
ID_COLS = ["TransactionID", "id_15", "id_23", "id_30", "id_31", "id_33", "DeviceType", "DeviceInfo"]


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    opts = pacsv.ConvertOptions(include_columns=TXN_COLS, strings_can_be_null=True)
    tx = pacsv.read_csv(RAW / "transactions.csv", convert_options=opts).to_pandas()
    ident = pacsv.read_csv(
        RAW / "identity.csv", convert_options=pacsv.ConvertOptions(include_columns=ID_COLS)
    ).to_pandas()
    closed = pd.read_csv(RAW / "closed_cases_history.csv")
    pack = pd.read_csv(RAW / "case_pack.csv")
    return tx, ident, closed, pack


def assign_card_ids(tx: pd.DataFrame, closed: pd.DataFrame, pack: pd.DataFrame) -> pd.Series:
    """Rebuild the dataset's card IDs (C01234-K1).

    Cards are not a column in the data. Checking the 1,900+ labelled (customer, txn, card)
    triples in the closed cases and case pack shows a card is the customer's card network
    (card4, missing network as its own card), numbered from the most recently first-seen
    network. That rule reproduces 99.3% of labels; the remaining ones are taken straight
    from the labels, so every card ID we emit matches the dataset where the dataset says.
    """
    net = tx["card4"].fillna("~")
    first_seen = tx.groupby([tx["customer_id"], net])["TransactionDT"].min().rename("first").reset_index()
    first_seen.columns = ["customer_id", "net", "first"]
    first_seen = first_seen.sort_values(["customer_id", "first"], ascending=[True, False])
    first_seen["k"] = first_seen.groupby("customer_id").cumcount() + 1
    first_seen["card_id"] = first_seen["customer_id"] + "-K" + first_seen["k"].astype(str)

    labels = pd.concat([
        closed.assign(tx=closed["txn_ids"].str.split("|")).explode("tx")[["card_id", "tx"]],
        pack[["card_id", "flagged_txn_id"]].rename(columns={"flagged_txn_id": "tx"}),
    ])
    labels["tx"] = labels["tx"].astype(np.int64)
    lab = labels.merge(
        tx[["TransactionID", "customer_id"]].assign(net=net.values),
        left_on="tx", right_on="TransactionID",
    )
    voted = (lab.groupby(["customer_id", "net"])["card_id"]
             .agg(lambda s: s.value_counts().index[0]).rename("label").reset_index())
    m = first_seen.merge(voted, on=["customer_id", "net"], how="left")
    m["card_id"] = m["label"].fillna(m["card_id"])
    key = pd.MultiIndex.from_frame(pd.DataFrame({"customer_id": tx["customer_id"], "net": net}))
    lookup = m.set_index(["customer_id", "net"])["card_id"]
    return pd.Series(lookup.reindex(key).values, index=tx.index)


def s(v) -> str:
    return "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)


def region_code(v) -> str:
    return "" if pd.isna(v) else str(int(v))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    tx, ident, closed, pack = load_raw()
    print(f"loaded {len(tx):,} txns, {len(ident):,} identity rows, {len(closed):,} closed cases")

    tx["card_id"] = assign_card_ids(tx, closed, pack)
    tx = tx.merge(ident, on="TransactionID", how="left")
    tx["device_id"] = [
        device_profile_id(a, b, c, d) if isinstance(a, str) or isinstance(c, str) else ""
        for a, b, c, d in zip(tx["DeviceInfo"], tx["id_30"], tx["id_31"], tx["id_33"], strict=True)
    ]
    tx["region"] = tx["addr1"].map(region_code)
    tx = tx.sort_values(["card_id", "TransactionDT"]).reset_index(drop=True)

    # --- vertices ---------------------------------------------------------------------
    txv = pd.DataFrame({
        "id": tx["TransactionID"].astype(str), "ts": tx["ts"], "dt": tx["TransactionDT"],
        "amt": tx["TransactionAmt"].round(2), "product": tx["ProductCD"], "channel": tx["channel"],
        "risk": tx["risk_score"], "addr1": tx["region"], "addr2": tx["addr2"].map(region_code),
        "dist1": tx["dist1"].fillna(-1), "p_email": tx["P_emaildomain"].fillna(""),
        "r_email": tx["R_emaildomain"].fillna(""), "m1": tx["M1"].fillna(""),
        "m4": tx["M4"].fillna(""), "m6": tx["M6"].fillna(""),
        "c1": tx["C1"].fillna(-1), "c13": tx["C13"].fillna(-1), "d1": tx["D1"].fillna(-1),
        "d15": tx["D15"].fillna(-1), "dev_status": tx["id_15"].fillna(""),
        "proxy_type": tx["id_23"].fillna(""), "device_type": tx["DeviceType"].fillna(""),
    })
    txv.to_csv(OUT / "txn.csv", index=False)

    g = tx.groupby("card_id")
    cards = pd.DataFrame({
        "id": g.size().index,
        "network": g["card4"].agg(lambda x: s(x.dropna().iloc[0]) if x.notna().any() else ""),
        "card_type": g["card6"].agg(lambda x: s(x.dropna().iloc[0]) if x.notna().any() else ""),
        "n_txns": g.size().values,
        "home_region": g["region"].agg(lambda x: x[x != ""].mode().iloc[0] if (x != "").any() else ""),
        "median_amt": g["TransactionAmt"].median().round(2).values,
        "online_share": g["channel"].agg(lambda x: round((x == "online").mean(), 3)),
    })
    cards.to_csv(OUT / "card.csv", index=False)
    cust = tx.groupby("customer_id")["card_id"].nunique().reset_index()
    cust.columns = ["id", "n_cards"]
    cust.to_csv(OUT / "customer.csv", index=False)
    tx[["customer_id", "card_id"]].drop_duplicates().to_csv(OUT / "owns.csv", index=False)

    dev = tx[tx["device_id"] != ""].drop_duplicates("device_id")
    pd.DataFrame({
        "id": dev["device_id"], "device_info": dev["DeviceInfo"].fillna(""),
        "os": dev["id_30"].fillna(""), "browser": dev["id_31"].fillna(""), "screen": dev["id_33"].fillna(""),
    }).to_csv(OUT / "device.csv", index=False)

    # --- edges ------------------------------------------------------------------------
    tx[["card_id", "TransactionID"]].to_csv(OUT / "made.csv", index=False)
    tx.loc[tx["device_id"] != "", ["TransactionID", "device_id"]].to_csv(OUT / "from_device.csv", index=False)
    tx.loc[tx["P_emaildomain"].notna(), ["TransactionID", "P_emaildomain"]].to_csv(OUT / "email.csv", index=False)
    tx.loc[tx["region"] != "", ["TransactionID", "region"]].to_csv(OUT / "region.csv", index=False)
    same_card = tx["card_id"].eq(tx["card_id"].shift(-1))
    nxt = pd.DataFrame({
        "src": tx["TransactionID"], "dst": tx["TransactionID"].shift(-1),
        "gap": tx["TransactionDT"].shift(-1) - tx["TransactionDT"],
    })[same_card]
    nxt.astype({"dst": np.int64, "gap": np.int64}).to_csv(OUT / "next.csv", index=False)

    # --- case memory --------------------------------------------------------------------
    by_id = tx.set_index("TransactionID")
    rows, involves, conn, profiles = [], [], [], []
    for r in closed.itertuples():
        ids = [int(x) for x in str(r.txn_ids).split("|") if x]
        sub = by_id.loc[by_id.index.intersection(ids)]
        rows.append({
            "id": r.case_id, "opened_at": r.opened_at, "closed_at": r.closed_at, "outcome": r.outcome,
            "pattern": r.pattern, "n_txns": r.n_txns, "exposure": r.exposure_usd,
            "actions": r.actions_taken, "report_filed": r.report_filed == "Yes",
            "notes": str(r.analyst_notes).replace("\n", " "), "card": r.card_id,
        })
        involves += [(r.case_id, i) for i in ids]
        if isinstance(r.connected_card_ids, str):
            conn += [(r.case_id, c) for c in r.connected_card_ids.split("|")]
        profiles.append({"id": r.case_id, "text": case_profile_text(sub.reset_index(), r.analyst_notes)})
    cc = pd.DataFrame(rows)
    cc.drop(columns=["card"]).to_csv(OUT / "closed_case.csv", index=False)
    cc[["id", "card"]].to_csv(OUT / "on_card.csv", index=False)
    cc[["id", "pattern"]].to_csv(OUT / "case_pattern.csv", index=False)
    pd.DataFrame(involves, columns=["case", "txn"]).to_csv(OUT / "involves.csv", index=False)
    pd.DataFrame(conn, columns=["case", "card"]).to_csv(OUT / "connected.csv", index=False)
    pd.DataFrame(profiles).to_parquet(CACHE / "closed_case_profiles.parquet", index=False)

    # Local analytical cache used by offline tests and the batch benchmark runner.
    tx.to_parquet(CACHE / "txn_enriched.parquet", index=False)
    print(f"cards={len(cards):,} devices={len(dev):,} next_edges={len(nxt):,} closed_cases={len(cc):,}")


if __name__ == "__main__":
    main()
