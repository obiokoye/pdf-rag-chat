from __future__ import annotations

from pathlib import Path

import pytest

from ragchat.embeddings import HashEmbedder
from ragchat.library import SAMPLES_DIR, Library
from ragchat.providers import LLMSetup
from ragchat.demo import DemoLLM
from ragchat.store import VectorStore

HANDBOOK = SAMPLES_DIR / "brightline-employee-handbook.pdf"
AGREEMENT = SAMPLES_DIR / "nimbushost-service-agreement.pdf"


@pytest.fixture(autouse=True)
def _no_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ["GEMINI_API_KEY", "GROQ_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "DEMO_MODE", "LLM_PROVIDER", "EMBEDDINGS"]:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def store(tmp_path: Path) -> VectorStore:
    # A persistent store in a temp folder keeps every test isolated.
    return VectorStore(HashEmbedder(), tmp_path / "chroma")


@pytest.fixture
def library(tmp_path: Path, store: VectorStore) -> Library:
    lib = Library(store, tmp_path / "pdfs", LLMSetup(DemoLLM(), "demo-scripted", "demo", demo=True, demo_reason="tests"))
    lib.ingest_samples_if_empty()
    return lib
