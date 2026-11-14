"""Read PDFs page by page and split each page into overlapping chunks.

Every chunk remembers which file and which page it came from. That metadata
is what lets the chat answer say "handbook.pdf, page 3" later on.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from pypdf import PdfReader


class UnreadablePDF(Exception):
    pass


@dataclass(frozen=True)
class Page:
    doc: str  # file name, e.g. "brightline-employee-handbook.pdf"
    number: int  # 1-based, as a human would say it
    text: str


@dataclass(frozen=True)
class Chunk:
    id: str
    doc: str
    page: int
    text: str


def _clean(text: str) -> str:
    text = text.replace("­", "")  # soft hyphens
    text = re.sub(r"-\n(\w)", r"\1", text)  # re-join words hyphenated across lines
    return re.sub(r"\s+", " ", text).strip()


def read_pdf(source: Path | BinaryIO, name: str | None = None) -> list[Page]:
    """Extract the text of every page. Pages with no text layer are skipped."""
    doc_name = name or Path(getattr(source, "name", "document.pdf")).name
    try:
        reader = PdfReader(source)
        pages = [Page(doc_name, i, _clean(p.extract_text() or "")) for i, p in enumerate(reader.pages, start=1)]
    except Exception as exc:  # pypdf raises many different error types
        raise UnreadablePDF(f"could not read {doc_name}: {exc}") from exc
    pages = [p for p in pages if p.text]
    if not pages:
        raise UnreadablePDF(f"{doc_name} has no text layer (scanned PDFs need OCR first)")
    return pages


def chunk_pages(pages: list[Page], size: int = 80, overlap: int = 20) -> list[Chunk]:
    """Split each page into chunks of about `size` words, overlapping by `overlap` words.

    Chunks never cross a page boundary, so each one maps to exactly one page.
    The overlap keeps a sentence that straddles two chunks findable from both.
    """
    if not 0 <= overlap < size:
        raise ValueError("overlap must be smaller than size")
    chunks: list[Chunk] = []
    step = size - overlap
    for page in pages:
        words = page.text.split()
        for start in range(0, max(len(words) - overlap, 1), step):
            text = " ".join(words[start : start + size])
            # Stable id: re-ingesting the same file replaces its chunks instead of duplicating them.
            digest = hashlib.sha1(f"{page.doc}|{page.number}|{start}|{text}".encode()).hexdigest()[:16]
            chunks.append(Chunk(digest, page.doc, page.number, text))
    return chunks
