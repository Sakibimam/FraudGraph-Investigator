"""Case memory: embeddings for closed cases and documents, and writing agent cases to the graph."""
from __future__ import annotations

import json
from datetime import datetime
from functools import lru_cache

import numpy as np

from fraudagent.config import settings
from fraudagent.memory.embed import Embedder
from fraudagent.memory.knowledge import chunk_markdown

EMBEDDER_PATH = settings.cache_dir / "embedder.joblib"
VEC_PATH = settings.cache_dir / "vectors.npz"


def load_doc_chunks() -> list[dict]:
    chunks: list[dict] = []
    for p in sorted(settings.knowledge_dir.glob("*.md")):
        chunks += chunk_markdown(p)
    return chunks


@lru_cache(maxsize=1)
def embedder() -> Embedder:
    return Embedder.load(EMBEDDER_PATH)


@lru_cache(maxsize=2)
def local_vectors(kind: str) -> tuple[list[str], np.ndarray]:
    z = np.load(VEC_PATH, allow_pickle=True)
    return list(z[f"{kind}_ids"]), z[f"{kind}_vecs"]


def _ts(s: str) -> str:
    return s.replace("T", " ")[:19]


def case_to_graph_payload(case: dict) -> tuple[dict, dict]:
    """Answer-file case -> TigerGraph upsert payload (InvestigationCase + edges)."""
    cid = case["graph_case_id"]
    c = case["case"]
    attrs = {
        "source_case": case["case_id"], "opened_at": _ts(case["opened_at"]),
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "status": c["status"],
        "verdict": c["verdict"], "pattern": c["pattern"], "probability": c["fraud_probability"],
        "exposure": c["exposure_usd"], "sar_filed": case["sar"]["file"], "summary": c["summary"],
        "final_actions": "|".join(a["action"] for a in case["next_best_actions"]["final"]),
        "decision_log": json.dumps(case.get("timeline", []))[:60000],
    }
    if case.get("embedding"):
        attrs["emb"] = case["embedding"]
    vertices = {"InvestigationCase": {cid: {k: {"value": v} for k, v in attrs.items()}}}
    e_inv = {t: {} for t in c["affected_txn_ids"]}
    edges = {"InvestigationCase": {cid: {
        "INV_INVOLVES": {"Txn": e_inv},
        "INV_ON_CARD": {"Card": {case["card_id"]: {}}},
        "INV_CONNECTED": {"Card": {k: {} for k in c["connected_card_ids"]}},
        "INV_DEVICE": {"DeviceProfile": {d: {} for d in c["connected_device_profiles"]}},
        "INV_PATTERN": {"FraudPattern": {c["pattern"]: {}}},
        "SIMILAR_TO": {"ClosedCase": {s: {"score": {"value": 1.0}} for s in c["similar_prior_cases"]}},
    }}}
    return vertices, edges
