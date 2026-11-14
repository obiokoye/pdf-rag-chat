from __future__ import annotations

import io

import pytest

from ragchat.documents import Page, UnreadablePDF, chunk_pages, read_pdf
from ragchat.library import safe_filename

from .conftest import AGREEMENT, HANDBOOK


def test_reads_every_page_with_numbers() -> None:
    pages = read_pdf(HANDBOOK)
    assert [p.number for p in pages] == [1, 2, 3, 4, 5]
    assert pages[0].doc == "brightline-employee-handbook.pdf"
    assert "20 days of paid annual leave" in pages[2].text
    assert "\n" not in pages[2].text  # whitespace is normalised


def test_reads_from_a_file_object_with_a_given_name() -> None:
    pages = read_pdf(io.BytesIO(AGREEMENT.read_bytes()), name="contract.pdf")
    assert len(pages) == 4 and pages[1].doc == "contract.pdf"
    assert "99.9% monthly uptime" in pages[1].text


def test_garbage_is_rejected() -> None:
    with pytest.raises(UnreadablePDF):
        read_pdf(io.BytesIO(b"definitely not a pdf"), name="x.pdf")


def test_chunks_overlap_and_stay_on_their_page() -> None:
    words = " ".join(f"w{i}" for i in range(200))
    chunks = chunk_pages([Page("a.pdf", 1, words), Page("a.pdf", 2, "short page")], size=80, overlap=20)
    first, second = chunks[0].text.split(), chunks[1].text.split()
    assert len(first) == 80
    assert first[-20:] == second[:20]  # overlap
    assert chunks[-1].page == 2 and chunks[-1].text == "short page"
    assert all(c.page == 1 for c in chunks[:-1])


def test_chunk_ids_are_stable_and_unique() -> None:
    pages = read_pdf(HANDBOOK)
    a, b = chunk_pages(pages), chunk_pages(pages)
    assert [c.id for c in a] == [c.id for c in b]
    assert len({c.id for c in a}) == len(a)


def test_overlap_must_be_smaller_than_size() -> None:
    with pytest.raises(ValueError):
        chunk_pages([Page("a.pdf", 1, "x")], size=10, overlap=10)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("../../etc/passwd", "passwd.pdf"),
        ("My Report (final).PDF", "My-Report-final-.PDF"),
        ("..\\..\\evil.pdf", "evil.pdf"),
        ("", "document.pdf"),
    ],
)
def test_safe_filename(raw: str, expected: str) -> None:
    assert safe_filename(raw) == expected
