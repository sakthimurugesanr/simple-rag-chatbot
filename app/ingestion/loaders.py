"""PDF, DOCX, TXT/MD, XLS/XLSX and CSV loaders with common source metadata."""

import io
import math
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from charset_normalizer import from_bytes
from loguru import logger

from app.config import Settings, get_settings
from app.exceptions import (
    DocumentParseError,
    EmptyDocumentError,
    IngestionError,
    OCRUnavailableError,
)
from app.ingestion.chunker import clean_text
from app.ingestion.models import Document
from app.utils.security import content_hash, validate_file

Loader = Callable[[Path, Settings], list[Document]]


def decode_text(raw: bytes) -> str:
    """Prefer explicit BOM/UTF-8 and use charset-normalizer for legacy encodings."""
    if raw.startswith((b"\xff\xfe\x00\x00", b"\x00\x00\xfe\xff")):
        return raw.decode("utf-32")
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16")
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        result = from_bytes(raw).best()
        if result is None:
            raise DocumentParseError("Text encoding could not be detected") from None
        return str(result)


def load_text(path: Path, settings: Settings) -> list[Document]:
    """Keep Markdown headings and paragraph breaks as readable source text."""
    return [Document(decode_text(path.read_bytes()))]


def _ocr_page(path: Path, page: int, settings: Settings) -> str:
    try:
        import pytesseract
        from pdf2image import convert_from_path
    except ImportError as exc:
        raise OCRUnavailableError("Install requirements-ocr.txt to enable PDF OCR") from exc
    try:
        if settings.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd
        images = convert_from_path(
            str(path),
            first_page=page,
            last_page=page,
            dpi=settings.ocr_dpi,
            poppler_path=settings.poppler_path,
            timeout=settings.ocr_timeout_seconds,
            thread_count=1,
        )
        try:
            return "\n".join(
                pytesseract.image_to_string(
                    image,
                    lang=settings.ocr_language,
                    timeout=settings.ocr_timeout_seconds,
                )
                for image in images
            )
        finally:
            for image in images:
                image.close()
    except Exception as exc:
        raise OCRUnavailableError(
            f"OCR failed on page {page}; check Tesseract, Poppler and language installation"
        ) from exc


def _remove_repeated_margins(pages: list[Document]) -> list[Document]:
    """Remove identical first/last lines appearing on >=75% of at least 3 pages."""
    if len(pages) < 3:
        return pages
    lines = [clean_text(page.text).splitlines() for page in pages]
    first = Counter(row[0] for row in lines if len(row) >= 3)
    last = Counter(row[-1] for row in lines if len(row) >= 3)
    threshold = max(3, math.ceil(len(pages) * 0.75))
    output = []
    for page, row in zip(pages, lines):
        if len(row) >= 3:
            drop_first = first[row[0]] >= threshold
            drop_last = last[row[-1]] >= threshold
            row = row[int(drop_first) : len(row) - int(drop_last)]
        output.append(Document("\n".join(row), page.metadata))
    return output


def load_pdf(path: Path, settings: Settings) -> list[Document]:
    """One source unit per page; OCR only pages without extractable text."""
    from pypdf import PdfReader

    pages: list[Document] = []
    with path.open("rb") as stream:
        reader = PdfReader(stream)
        if reader.is_encrypted and not reader.decrypt(""):
            raise DocumentParseError("Password-protected PDFs are unsupported")
        if len(reader.pages) > settings.max_pdf_pages:
            raise DocumentParseError("PDF page count exceeds configured limit")
        total_chars = 0
        for number, page in enumerate(reader.pages, 1):
            text = page.extract_text() or ""
            used_ocr = False
            if not clean_text(text) and settings.ocr_enabled:
                text = _ocr_page(path, number, settings)
                used_ocr = True
            if not clean_text(text):
                logger.warning(
                    "PDF page {} has no readable text; OCR enabled={}", number, settings.ocr_enabled
                )
            total_chars += len(text)
            if total_chars > settings.max_extracted_chars:
                raise DocumentParseError("Extracted text exceeds configured character limit")
            pages.append(Document(text, {"page": number, "ocr": used_ocr}))
    return _remove_repeated_margins(pages)


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return clean_text(str(value)).replace("\n", " / ")


def _headers(values: Sequence[Any]) -> list[str]:
    """Give blank and duplicate columns unambiguous names."""
    output: list[str] = []
    used: set[str] = set()
    for index, value in enumerate(values, 1):
        base = _cell(value) or f"Column {index}"
        name = base
        occurrence = 2
        while name in used:
            name = f"{base} ({occurrence})"
            occurrence += 1
        used.add(name)
        output.append(name)
    return output


def _table_blocks(
    rows: Iterable[Sequence[Any]],
    sheet: str,
    settings: Settings,
) -> list[Document]:
    """First nonempty row is the header; preserve physical 1-based row numbers."""
    headers: list[str] | None = None
    block: list[str] = []
    output: list[Document] = []
    row_start = row_end = header_row = 0
    characters = 0

    def flush() -> None:
        if block:
            output.append(
                Document(
                    f"Sheet: {sheet}\n" + "\n".join(block),
                    {
                        "sheet": sheet,
                        "row_start": row_start,
                        "row_end": row_end,
                        "header_row": header_row,
                    },
                )
            )
            block.clear()

    for row_number, row in enumerate(rows, 1):
        if row_number > settings.max_table_rows:
            raise DocumentParseError("Sheet exceeds configured row limit")
        values = [_cell(value) for value in row]
        if not any(values):
            continue
        if headers is None:
            # Ignore trailing unused columns reported by spreadsheet formatting.
            while values and not values[-1]:
                values.pop()
            headers = _headers(values)
            header_row = row_number
            continue
        while len(headers) < len(values):
            candidate = f"Column {len(headers) + 1}"
            while candidate in headers:
                candidate += " (extra)"
            headers.append(candidate)
        values += [""] * (len(headers) - len(values))
        rendered = f"Row {row_number}: " + "; ".join(
            f"{header}: {value or '(empty)'}" for header, value in zip(headers, values)
        )
        characters += len(rendered)
        if characters > settings.max_extracted_chars:
            raise DocumentParseError("Table text exceeds configured character limit")
        if not block:
            row_start = row_number
        row_end = row_number
        block.append(rendered)
        if len(block) >= settings.table_rows_per_block:
            flush()
    flush()
    if headers is not None and not output:
        output.append(
            Document(
                f"Sheet: {sheet}\nColumns: " + "; ".join(headers) + "\n(No data rows)",
                {
                    "sheet": sheet,
                    "row_start": header_row,
                    "row_end": header_row,
                    "header_row": header_row,
                },
            )
        )
    return output


def load_docx(path: Path, settings: Settings) -> list[Document]:
    """Keep paragraphs and tables in document order, with nearest heading metadata."""
    from docx import Document as WordDocument
    from docx.table import Table

    document = WordDocument(str(path))
    output: list[Document] = []
    paragraphs: list[str] = []
    section = "Document"
    table_index = 0

    def flush() -> None:
        if paragraphs:
            output.append(Document("\n\n".join(paragraphs), {"section": section}))
            paragraphs.clear()

    for element in document.iter_inner_content():
        if isinstance(element, Table):
            flush()
            table_index += 1
            blocks = _table_blocks(
                ([cell.text for cell in row.cells] for row in element.rows),
                f"Table {table_index}",
                settings,
            )
            output.extend(
                Document(item.text, {**item.metadata, "section": section}) for item in blocks
            )
        else:
            text = element.text.strip()
            if not text:
                continue
            style_name = element.style.name if element.style else ""
            if style_name.startswith("Heading") or style_name == "Title":
                flush()
                section = text
            paragraphs.append(text)
    flush()
    return output


def load_xlsx(path: Path, settings: Settings) -> list[Document]:
    """Read each sheet lazily; use cached formula values without executing formulas."""
    from openpyxl import load_workbook

    workbook = load_workbook(path, read_only=True, data_only=True, keep_links=False)
    output: list[Document] = []
    try:
        for sheet in workbook.worksheets:
            output.extend(_table_blocks(sheet.iter_rows(values_only=True), sheet.title, settings))
    finally:
        workbook.close()
    return output


def load_xls(path: Path, settings: Settings) -> list[Document]:
    """Parse legacy BIFF XLS, converting Excel date serials to ISO dates."""
    import xlrd

    workbook = xlrd.open_workbook(str(path), on_demand=True)
    output: list[Document] = []
    try:
        for sheet in workbook.sheets():

            def rows(current_sheet: Any = sheet) -> Iterable[list[Any]]:
                for row_number in range(current_sheet.nrows):
                    yield [
                        xlrd.xldate_as_datetime(cell.value, workbook.datemode)
                        if cell.ctype == xlrd.XL_CELL_DATE
                        else cell.value
                        for cell in current_sheet.row(row_number)
                    ]

            output.extend(_table_blocks(rows(), sheet.name, settings))
    finally:
        workbook.release_resources()
    return output


def load_csv(path: Path, settings: Settings) -> list[Document]:
    """Use pandas CSV parsing; preserve header labels, empty cells and record numbers."""
    import pandas as pd

    text = decode_text(path.read_bytes())
    try:
        frames = pd.read_csv(
            io.StringIO(text),
            header=None,
            dtype=str,
            keep_default_na=False,
            skip_blank_lines=False,
            chunksize=1000,
        )
        return _table_blocks(
            (tuple(row) for frame in frames for row in frame.itertuples(index=False, name=None)),
            "CSV",
            settings,
        )
    except pd.errors.EmptyDataError:
        return []


LOADERS: dict[str, Loader] = {
    ".pdf": load_pdf,
    ".docx": load_docx,
    ".txt": load_text,
    ".md": load_text,
    ".xlsx": load_xlsx,
    ".xls": load_xls,
    ".csv": load_csv,
}


def load_file(
    path: str | Path,
    settings: Settings | None = None,
    *,
    mime_type: str | None = None,
    source_name: str | None = None,
) -> list[Document]:
    """Validate and extract a file, assigning content identity and UTC ingestion time.

    Hash identity prepares collection-level deduplication in step 2. This function
    does not persist a manifest or vectors and does not claim to deduplicate uploads.
    """
    settings = settings or get_settings()
    path = Path(path)
    suffix = validate_file(path, settings, mime_type)
    digest = content_hash(path)
    metadata = {
        "doc_id": digest,
        "content_hash": digest,
        "source": source_name or path.name,
        "file_type": suffix.lstrip("."),
        "uploaded_at": datetime.now(UTC).isoformat(),
    }
    try:
        units = LOADERS[suffix](path, settings)
    except IngestionError:
        raise
    except Exception as exc:
        raise DocumentParseError(
            f"Could not parse {suffix} file; it may be damaged or encrypted"
        ) from exc
    output: list[Document] = []
    total = 0
    for unit in units:
        text = clean_text(unit.text)
        if not text:
            continue
        total += len(text)
        if total > settings.max_extracted_chars:
            raise DocumentParseError("Extracted text exceeds configured character limit")
        output.append(Document(text, {**metadata, **unit.metadata}))
    if not output:
        hint = " Enable RAG_OCR_ENABLED for scanned PDFs." if suffix == ".pdf" else ""
        raise EmptyDocumentError("The document contains no readable text." + hint)
    logger.info("Extracted {} source units ({} characters, type={})", len(output), total, suffix)
    return output
