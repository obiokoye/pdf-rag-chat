"""A thin wrapper around a Chroma collection: add chunks, search, list documents."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.config import Settings

from .documents import Chunk
from .embeddings import Embedder


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float  # cosine similarity, 1.0 = identical


@dataclass(frozen=True)
class DocumentInfo:
    name: str
    pages: int
    chunks: int


class VectorStore:
    def __init__(self, embedder: Embedder, path: str | Path | None = None):
        settings = Settings(anonymized_telemetry=False)
        self._client = (
            chromadb.PersistentClient(path=str(path), settings=settings)
            if path
            else chromadb.EphemeralClient(settings=settings)
        )
        self.embedder = embedder
        # One collection per embedder: vectors from different models can't be compared.
        self._name = f"pdf-chunks-{embedder.name}"
        self._col = self._client.get_or_create_collection(
            self._name, embedding_function=None, metadata={"hnsw:space": "cosine"}
        )

    def add(self, chunks: list[Chunk]) -> int:
        for start in range(0, len(chunks), 64):
            batch = chunks[start : start + 64]
            self._col.upsert(
                ids=[c.id for c in batch],
                embeddings=self.embedder.embed([c.text for c in batch]),  # type: ignore[arg-type]
                documents=[c.text for c in batch],
                metadatas=[{"doc": c.doc, "page": c.page} for c in batch],
            )
        return len(chunks)

    def remove(self, doc: str) -> None:
        self._col.delete(where={"doc": doc})

    def search(self, query: str, k: int = 4) -> list[Hit]:
        if self._col.count() == 0:
            return []
        res = self._col.query(
            query_embeddings=self.embedder.embed([query]),  # type: ignore[arg-type]
            n_results=min(k, self._col.count()),
        )
        hits = []
        for cid, text, meta, dist in zip(
            res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]  # type: ignore[index]
        ):
            chunk = Chunk(cid, str(meta["doc"]), int(meta["page"]), text)  # type: ignore[arg-type]
            hits.append(Hit(chunk, round(1.0 - float(dist), 4)))
        return hits

    def documents(self) -> list[DocumentInfo]:
        metas = self._col.get(include=["metadatas"])["metadatas"] or []
        chunk_counts = Counter(str(m["doc"]) for m in metas)
        pages: dict[str, set[int]] = {}
        for m in metas:
            pages.setdefault(str(m["doc"]), set()).add(int(m["page"]))  # type: ignore[arg-type]
        return [DocumentInfo(name, len(pages[name]), chunk_counts[name]) for name in sorted(chunk_counts)]

    def count(self) -> int:
        return self._col.count()
