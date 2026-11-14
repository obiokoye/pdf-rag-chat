from __future__ import annotations

import math

from ragchat.documents import chunk_pages, read_pdf
from ragchat.embeddings import HashEmbedder
from ragchat.library import Library
from ragchat.store import VectorStore

from .conftest import AGREEMENT, HANDBOOK


def test_hash_embedder_is_deterministic_and_normalised() -> None:
    e = HashEmbedder()
    a, b = e.embed(["Annual leave carry over"])[0], e.embed(["Annual leave carry over"])[0]
    assert a == b
    assert math.isclose(math.sqrt(sum(v * v for v in a)), 1.0, rel_tol=1e-9)


def test_search_finds_the_right_page(store: VectorStore) -> None:
    for pdf in (HANDBOOK, AGREEMENT):
        store.add(chunk_pages(read_pdf(pdf)))
    top = store.search("How many days of annual leave can I carry over?", k=3)[0]
    assert (top.chunk.doc, top.chunk.page) == ("brightline-employee-handbook.pdf", 3)
    top = store.search("How long are backups kept?", k=3)[0]
    assert (top.chunk.doc, top.chunk.page) == ("nimbushost-service-agreement.pdf", 3)


def test_scores_are_sorted_best_first(store: VectorStore) -> None:
    store.add(chunk_pages(read_pdf(AGREEMENT)))
    scores = [h.score for h in store.search("service credit uptime", k=4)]
    assert scores == sorted(scores, reverse=True)


def test_reingesting_does_not_duplicate(library: Library) -> None:
    before = library.store.count()
    library.ingest(HANDBOOK)
    assert library.store.count() == before


def test_documents_and_remove(library: Library) -> None:
    docs = {d.name: d for d in library.store.documents()}
    assert docs["brightline-employee-handbook.pdf"].pages == 5
    assert docs["nimbushost-service-agreement.pdf"].pages == 4
    library.store.remove("nimbushost-service-agreement.pdf")
    assert [d.name for d in library.store.documents()] == ["brightline-employee-handbook.pdf"]


def test_empty_store_returns_nothing(store: VectorStore) -> None:
    assert store.search("anything") == []
