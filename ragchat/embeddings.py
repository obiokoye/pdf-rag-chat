"""Embedders turn text into vectors so similar meanings end up close together.

- "local": all-MiniLM-L6-v2 via Chroma's built-in ONNX model. Runs on your
  machine, free, works with every LLM provider. Downloads ~80 MB the first time.
- "hash": a tiny keyword-hashing embedder with no download and no network.
  It matches words, not meanings, so it's used for demo mode and the tests.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

_STOP = set(
    "a an and are as at be by can do does for from get has have how i if in is it its me my of on or our "
    "so that the their there they this to up us was we what when where which who will with you your".split()
)


class Embedder(Protocol):
    name: str
    min_score: float  # below this similarity a chunk counts as "not relevant"

    def embed(self, texts: list[str]) -> list[list[float]]: ...


def _tokens(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.lower())
    out = []
    for w in words:
        if w in _STOP:
            continue
        if len(w) > 4 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]  # crude plural folding: "credits" -> "credit"
        out.append(w)
    return out


class HashEmbedder:
    min_score = 0.10

    def __init__(self, dims: int = 4096):
        self.dims = dims
        self.name = f"hash{dims}"

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dims
        tokens = _tokens(text)
        features = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
        for feature in features:
            h = int(hashlib.md5(feature.encode()).hexdigest(), 16)  # md5: stable across runs, not for security
            vec[h % self.dims] += 1.0
        vec = [math.log1p(v) for v in vec]  # dampen repeated words
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]


class LocalEmbedder:
    name = "minilm"
    min_score = 0.20

    def __init__(self) -> None:
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

        self._fn = DefaultEmbeddingFunction()

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [list(map(float, v)) for v in self._fn(texts)]


def make_embedder(kind: str) -> Embedder:
    kind = kind.strip().lower()
    if kind == "hash":
        return HashEmbedder()
    if kind == "local":
        return LocalEmbedder()
    raise SystemExit(f"Unknown EMBEDDINGS '{kind}'. Use 'local' or 'hash'.")
