"""Shared helpers that turn raw rows into graph identifiers and retrieval text."""
from __future__ import annotations

import math

import pandas as pd


def _clean(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    return str(v).strip()


def device_profile_id(device_info, os, browser, screen) -> str:
    """Device profile = DeviceInfo + OS + browser + screen, as the answer format expects."""
    return " | ".join(_clean(x) for x in (device_info, os, browser, screen))


def amount_bucket(amt: float) -> str:
    for edge, name in ((5, "tiny"), (30, "small"), (100, "medium"), (500, "large"), (1000, "xlarge")):
        if amt < edge:
            return f"amt_{name}"
    return "amt_huge"


def case_profile_text(rows: pd.DataFrame, notes: str = "") -> str:
    """Compact, tokenised description of an episode, used to embed cases for similarity search.

    Both historical closed cases and new investigations go through this function, so the
    vector search compares like with like: channel mix, amounts, device and proxy flags,
    product codes and the narrative.
    """
    tokens: list[str] = []
    if len(rows):
        tokens += [f"channel_{c}" for c in rows["channel"].dropna().unique()]
        tokens += [f"product_{p}" for p in rows["ProductCD"].dropna().unique()]
        tokens += sorted({amount_bucket(a) for a in rows["TransactionAmt"]})
        n = len(rows)
        tokens.append("single_txn" if n == 1 else "few_txns" if n <= 4 else "many_txns")
        if "id_15" in rows and (rows["id_15"] == "New").any():
            tokens.append("device_new")
        if "id_23" in rows and rows["id_23"].notna().any():
            tokens += [f"proxy_{str(p).split(':')[-1].lower()}" for p in rows["id_23"].dropna().unique()]
        if rows["TransactionAmt"].between(400, 500).sum() >= 3:
            tokens.append("sub_threshold_cluster")
        if (rows["TransactionAmt"] < 10).sum() >= 3:
            tokens.append("micro_auth_run")
    return " ".join(tokens) + " " + _clean(notes)
