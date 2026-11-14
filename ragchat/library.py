"""The PDF library: where uploaded files live, plus ingest into the vector store.

Shared by the web app and the CLI so both behave the same way.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from .documents import chunk_pages, read_pdf
from .embeddings import make_embedder
from .providers import LLMSetup, make_client
from .store import VectorStore

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"
MAX_PDF_BYTES = 20 * 1024 * 1024


def safe_filename(name: str) -> str:
    """Strip any path and odd characters, so an upload can't write outside the library."""
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", Path(name).name).strip(".-") or "document"
    return stem if stem.lower().endswith(".pdf") else f"{stem}.pdf"


@dataclass
class Library:
    store: VectorStore
    files_dir: Path
    llm: LLMSetup

    def ingest(self, path: Path) -> tuple[int, int]:
        """Read, chunk and index one PDF. Returns (pages, chunks)."""
        pages = read_pdf(path, name=path.name)
        chunks = chunk_pages(pages)
        self.store.remove(path.name)  # replace an older copy of the same file
        self.store.add(chunks)
        return len(pages), len(chunks)

    def save_upload(self, filename: str, data: bytes) -> Path:
        if len(data) > MAX_PDF_BYTES:
            raise ValueError("PDF is larger than 20 MB")
        if not data.startswith(b"%PDF"):
            raise ValueError("that file is not a PDF")
        self.files_dir.mkdir(parents=True, exist_ok=True)
        target = self.files_dir / safe_filename(filename)
        target.write_bytes(data)
        return target

    def file_path(self, name: str) -> Path | None:
        for folder in (self.files_dir, SAMPLES_DIR):
            candidate = folder / safe_filename(name)
            if candidate.is_file():
                return candidate
        return None

    def ingest_samples_if_empty(self) -> None:
        if self.store.count() == 0:
            for pdf in sorted(SAMPLES_DIR.glob("*.pdf")):
                self.ingest(pdf)


def open_library(data_dir: str | Path | None = None, persist: bool = True) -> Library:
    """Wire up the LLM client, embedder and Chroma store from environment settings."""
    llm = make_client()
    kind = os.getenv("EMBEDDINGS") or ("hash" if llm.demo else "local")
    embedder = make_embedder(kind)
    root = Path(data_dir or os.getenv("DATA_DIR", "data"))
    store = VectorStore(embedder, root / "chroma" if persist else None)
    return Library(store, root / "pdfs", llm)
