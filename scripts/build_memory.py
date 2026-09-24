"""Build GraphRAG memory: fit the embedder, embed documents and closed cases,
cache the vectors locally and (optionally) push them into TigerGraph vector attributes.

    python scripts/build_memory.py            # local cache only
    python scripts/build_memory.py --push     # also upsert DocChunk / ClosedCase.emb / FraudPattern
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fraudagent.config import settings  # noqa: E402
from fraudagent.memory.case_store import (  # noqa: E402
    EMBEDDER_PATH,
    VEC_PATH,
    load_doc_chunks,
)
from fraudagent.memory.embed import Embedder  # noqa: E402

PATTERNS = {
    "card_testing": ("Card testing", True, "Three or more tiny online authorisations, then a larger purchase."),
    "card_not_present_fraud": ("Card-not-present fraud", True, "Card number used online without the card; burst of 2-4 unusual purchases in 48h."),
    "card_not_present_new_device": ("CNP fraud from a new device", True, "CNP fraud from a device marked New for the account, sometimes behind a proxy."),
    "out_of_region_use": ("Out-of-region use", True, "Card-present purchases in a billing region with no history while home activity continues."),
    "account_takeover": ("Account takeover", True, "Mixed-channel activity with device and match-flag anomalies; stolen credentials."),
    "undocumented": ("Undocumented pattern", False, "Coordinated or repeated abuse that fits none of the known patterns."),
    "none": ("No fraud", False, "Alert cleared as legitimate."),
}


def main(push: bool) -> None:
    chunks = load_doc_chunks()
    prof = pd.read_parquet(settings.cache_dir / "closed_case_profiles.parquet")
    corpus = [c["section"] + " " + c["text"] for c in chunks] + prof["text"].tolist()
    emb = Embedder.fit(corpus)
    emb.save(EMBEDDER_PATH)
    doc_vecs = emb.encode([c["section"] + " " + c["text"] for c in chunks])
    case_vecs = emb.encode(prof["text"].tolist())
    np.savez(VEC_PATH, docs_ids=np.array([c["id"] for c in chunks]), docs_vecs=doc_vecs,
             closed_ids=prof["id"].to_numpy(), closed_vecs=case_vecs)
    print(f"embedded {len(chunks)} doc chunks and {len(prof)} closed cases -> {VEC_PATH}")

    if not push:
        return
    from fraudagent.graph.client import TigerGraph
    tg = TigerGraph(timeout=600)
    vec = lambda v: [round(float(x), 6) for x in v]  # noqa: E731
    tg.upsert({"DocChunk": {c["id"]: {"source": {"value": c["source"]}, "section": {"value": c["section"]},
                                      "text": {"value": c["text"]}, "emb": {"value": vec(v)}}
                            for c, v in zip(chunks, doc_vecs, strict=True)},
               "FraudPattern": {k: {"name": {"value": n}, "documented": {"value": d}, "description": {"value": desc}}
                                for k, (n, d, desc) in PATTERNS.items()}})
    ids = prof["id"].tolist()
    for i in range(0, len(ids), 500):
        tg.upsert({"ClosedCase": {cid: {"emb": {"value": vec(v)}} for cid, v in zip(ids[i:i + 500], case_vecs[i:i + 500], strict=True)}})
    print("pushed DocChunk, FraudPattern and ClosedCase.emb to TigerGraph")


if __name__ == "__main__":
    main("--push" in sys.argv)
