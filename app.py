"""FastAPI web app: upload PDFs, chat with them, click a citation to open the page.

    python main.py serve            # or: uvicorn app:create_app --factory --reload
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ragchat import RAGChat, UnreadablePDF
from ragchat.library import Library, open_library

STATIC = Path(__file__).resolve().parent / "static"


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=8, max_length=64, pattern=r"^[A-Za-z0-9-]+$")
    message: str = Field(min_length=1, max_length=2000)


def create_app(library: Library | None = None) -> FastAPI:
    if library is None:
        load_dotenv()
        library = open_library()
    library.ingest_samples_if_empty()
    lib = library
    sessions: OrderedDict[str, RAGChat] = OrderedDict()
    app = FastAPI(title="PDF RAG Chat")

    def session(session_id: str) -> RAGChat:
        if session_id not in sessions:
            sessions[session_id] = RAGChat(lib.store, lib.llm.client, lib.llm.model)
            if len(sessions) > 200:  # keep memory bounded for a public demo
                sessions.popitem(last=False)
        sessions.move_to_end(session_id)
        return sessions[session_id]

    @app.get("/api/status")
    def status() -> dict[str, object]:
        return {
            "demo": lib.llm.demo,
            "demo_reason": lib.llm.demo_reason,
            "model": lib.llm.model,
            "provider": lib.llm.provider,
            "embedder": lib.store.embedder.name,
        }

    @app.get("/api/documents")
    def documents() -> list[dict[str, object]]:
        return [asdict(d) for d in lib.store.documents()]

    @app.post("/api/documents")
    async def upload(file: UploadFile) -> dict[str, object]:
        data = await file.read()
        try:
            path = lib.save_upload(file.filename or "document.pdf", data)
            pages, chunks = lib.ingest(path)
        except (ValueError, UnreadablePDF) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"name": path.name, "pages": pages, "chunks": chunks}

    @app.get("/files/{name}")
    def pdf_file(name: str) -> FileResponse:
        path = lib.file_path(name)
        if path is None:
            raise HTTPException(status_code=404, detail="no such PDF")
        return FileResponse(path, media_type="application/pdf")

    @app.post("/api/chat")
    def chat(req: ChatRequest) -> dict[str, object]:
        answer = session(req.session_id).ask(req.message)
        return {
            "answer": answer.text,
            "grounded": answer.grounded,
            "called_llm": answer.called_llm,
            "citations": [asdict(c) for c in answer.citations],
            "sources": [
                {"doc": h.chunk.doc, "page": h.chunk.page, "score": h.score} for h in answer.sources
            ],
        }

    @app.post("/api/reset")
    def reset(req: ChatRequest) -> dict[str, bool]:
        session(req.session_id).reset()
        return {"ok": True}

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app

