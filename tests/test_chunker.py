from itertools import pairwise

import pytest

from app.ingestion.chunker import chunk_documents, clean_text
from app.ingestion.models import Document


def test_clean_text_preserves_structure() -> None:
    assert clean_text(" a\x00  b\r\n\r\n\r\nc\t d ") == "a b\n\nc d"


@pytest.mark.parametrize("overlap", [0, 1, 15, 79])
def test_complete_coverage_exact_overlap_and_limits(overlap: int) -> None:
    text = clean_text("Heading\n\n" + "Alpha sentence. Next sentence with details.\n" * 30)
    chunks = chunk_documents([Document(text, {"doc_id": "a", "page": 2})], 80, overlap)
    reconstructed = chunks[0].text
    for previous, current in pairwise(chunks):
        if overlap:
            assert previous.text[-overlap:] == current.text[:overlap]
        reconstructed += current.text[overlap:]
    assert reconstructed == text
    assert all(0 < len(chunk.text) <= 80 for chunk in chunks)
    assert all(chunk.metadata["page"] == 2 for chunk in chunks)
    assert [chunk.metadata["chunk_index"] for chunk in chunks] == list(range(len(chunks)))
    for chunk in chunks:
        assert chunk.text == text[chunk.metadata["char_start"] : chunk.metadata["char_end"]]


def test_long_word_splits_and_small_final_chunk() -> None:
    chunks = chunk_documents([Document("a" * 201)], 100, 0)
    assert [len(chunk.text) for chunk in chunks] == [100, 100, 1]


def test_chunks_do_not_cross_source_units() -> None:
    chunks = chunk_documents(
        [
            Document("Page one", {"doc_id": "a", "page": 1}),
            Document("Page two", {"doc_id": "a", "page": 2}),
            Document("Other doc", {"doc_id": "b", "page": 1}),
        ]
    )
    assert [chunk.metadata["chunk_index"] for chunk in chunks] == [0, 1, 0]
    assert [chunk.metadata["page"] for chunk in chunks] == [1, 2, 1]


def test_empty_documents() -> None:
    assert chunk_documents([Document(" \n\t")]) == []


@pytest.mark.parametrize("size,overlap", [(0, 0), (10, -1), (10, 10), (10, 11)])
def test_invalid_parameters(size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        chunk_documents([], size, overlap)
