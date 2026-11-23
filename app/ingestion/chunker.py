"""Recursive boundary selection with exact character overlap and source offsets."""

import hashlib
import re
from collections.abc import Iterable

from app.ingestion.models import Chunk, Document


def clean_text(text: str) -> str:
    """Normalize whitespace without flattening paragraphs, rows, or Markdown tables."""
    text = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[^\S\n]+", " ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _boundary(text: str, start: int, limit: int, minimum: int) -> int:
    """Prefer paragraph, newline, sentence, then word boundaries; otherwise hard cut."""
    if limit >= len(text):
        return len(text)
    for separator in ("\n\n", "\n", ". ", " "):
        found = text.rfind(separator, start + minimum, limit)
        if found >= 0 and found + len(separator) <= limit:
            return found + len(separator)
    return limit


def chunk_documents(
    documents: Iterable[Document],
    chunk_size: int = 1000,
    overlap: int = 150,
) -> list[Chunk]:
    """Split each source unit independently, preserving page/sheet/section metadata.

    Offsets refer to the cleaned source unit. Chunk indices are per doc_id, globally
    across its units. The overlap is exact, including whitespace, and every source
    character is covered. Long words are cut to guarantee bounded chunk size.
    """
    if chunk_size < 1 or not 0 <= overlap < chunk_size:
        raise ValueError("Require chunk_size > 0 and 0 <= overlap < chunk_size")
    chunks: list[Chunk] = []
    indices: dict[str, int] = {}
    for unit_index, document in enumerate(documents):
        text = clean_text(document.text)
        if not text:
            continue
        doc_id = str(document.metadata.get("doc_id", "document"))
        start = 0
        while start < len(text):
            # Stay beyond overlap to guarantee forward progress.
            minimum = max(overlap + 1, chunk_size // 2)
            end = _boundary(text, start, min(start + chunk_size, len(text)), minimum)
            fragment = text[start:end]
            index = indices.get(doc_id, 0)
            metadata = {
                **document.metadata,
                "chunk_index": index,
                "unit_index": unit_index,
                "char_start": start,
                "char_end": end,
                "chunk_hash": hashlib.sha256(fragment.encode("utf-8")).hexdigest(),
            }
            chunks.append(Chunk(text=fragment, metadata=metadata))
            indices[doc_id] = index + 1
            if end == len(text):
                break
            start = end - overlap
    return chunks
