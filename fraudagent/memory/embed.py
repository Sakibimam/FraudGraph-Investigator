"""Local text embedder for GraphRAG (TF-IDF + truncated SVD, 128 dims).

A hosted embedding API would work too, but this keeps the whole pipeline free, offline and
reproducible: the same vectors are written into TigerGraph's vector attributes and used to
embed queries at investigation time.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

DIM = 128


class Embedder:
    def __init__(self, pipeline):
        self._p = pipeline

    @classmethod
    def fit(cls, corpus: list[str]) -> "Embedder":
        pipe = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, token_pattern=r"[A-Za-z_][A-Za-z_0-9]+"),
            TruncatedSVD(n_components=DIM, random_state=7),
            Normalizer(copy=False),
        )
        pipe.fit(corpus)
        return cls(pipe)

    def encode(self, texts: list[str]) -> np.ndarray:
        return self._p.transform(texts).astype("float32")

    def encode_one(self, text: str) -> list[float]:
        return [round(float(x), 6) for x in self.encode([text])[0]]

    def save(self, path: Path) -> None:
        joblib.dump(self._p, path)

    @classmethod
    def load(cls, path: Path) -> "Embedder":
        return cls(joblib.load(path))
