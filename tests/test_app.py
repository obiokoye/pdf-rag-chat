from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app import create_app
from ragchat.library import Library
from ragchat.providers import make_client

from .conftest import AGREEMENT

SID = "test-session-1234"


@pytest.fixture
def client(library: Library) -> TestClient:
    return TestClient(create_app(library))


def test_status_reports_demo_mode(client: TestClient) -> None:
    body = client.get("/api/status").json()
    assert body["demo"] is True and body["embedder"].startswith("hash")


def test_samples_are_indexed_on_start(client: TestClient) -> None:
    names = [d["name"] for d in client.get("/api/documents").json()]
    assert names == ["brightline-employee-handbook.pdf", "nimbushost-service-agreement.pdf"]


def test_chat_returns_verified_citations(client: TestClient) -> None:
    body = client.post("/api/chat", json={"session_id": SID, "message": "What uptime does NimbusHost promise?"}).json()
    assert body["grounded"] is True
    assert body["citations"][0]["doc"] == "nimbushost-service-agreement.pdf"
    assert body["citations"][0]["page"] == 2
    assert "99.9%" in body["answer"]


def test_upload_indexes_a_new_pdf(client: TestClient) -> None:
    files = {"file": ("../../hosting terms.pdf", AGREEMENT.read_bytes(), "application/pdf")}
    body = client.post("/api/documents", files=files).json()
    assert body == {"name": "hosting-terms.pdf", "pages": 4, "chunks": body["chunks"]}
    assert client.get("/files/hosting-terms.pdf").headers["content-type"] == "application/pdf"


def test_upload_rejects_non_pdf(client: TestClient) -> None:
    res = client.post("/api/documents", files={"file": ("notes.pdf", b"hello", "application/pdf")})
    assert res.status_code == 422


def test_unknown_file_is_404(client: TestClient) -> None:
    assert client.get("/files/missing.pdf").status_code == 404


def test_bad_session_id_is_rejected(client: TestClient) -> None:
    assert client.post("/api/chat", json={"session_id": "../x", "message": "hi"}).status_code == 422


def test_index_page_is_served(client: TestClient) -> None:
    assert "PDF RAG Chat" in client.get("/").text


def test_demo_mode_without_any_key() -> None:
    setup = make_client()
    assert setup.demo and setup.demo_reason == "no GEMINI_API_KEY set"


def test_demo_mode_flag_wins_over_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    monkeypatch.setenv("DEMO_MODE", "true")
    assert make_client().demo_reason == "DEMO_MODE=true"


def test_unknown_provider_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "nope")
    with pytest.raises(SystemExit):
        make_client()
