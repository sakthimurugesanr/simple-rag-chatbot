from pathlib import Path

import pytest

from app.config import Settings
from app.exceptions import DocumentParseError, EmptyDocumentError, FileValidationError
from app.ingestion.loaders import decode_text, load_file


@pytest.mark.parametrize(
    "name,expected",
    [
        ("notes.txt", "24 days"),
        ("handbook.md", "# Leave"),
        ("policy.pdf", "24 days"),
        ("policy.docx", "24 days"),
        ("products.xlsx", "Price INR: 2500"),
        ("products.xls", "Price INR: 2500"),
        ("products.csv", "Price INR: 2500"),
    ],
)
def test_each_actual_file_format(
    samples: Path, settings: Settings, name: str, expected: str
) -> None:
    units = load_file(samples / name, settings)
    assert expected in "\n".join(unit.text for unit in units)
    for unit in units:
        assert unit.metadata["source"] == name
        assert len(unit.metadata["doc_id"]) == 64
        assert unit.metadata["uploaded_at"].endswith("+00:00")


def test_pdf_page_metadata_and_repeated_margins(samples: Path, settings: Settings) -> None:
    units = load_file(samples / "policy.pdf", settings)
    assert [unit.metadata["page"] for unit in units] == [1, 2, 3]
    assert all("Confidential" not in unit.text for unit in units)
    assert all("Northstar internal handbook" not in unit.text for unit in units)
    assert all("24 days" in unit.text for unit in units)


def test_docx_section_and_table_order(samples: Path, settings: Settings) -> None:
    units = load_file(samples / "policy.docx", settings)
    assert units[0].metadata["section"] == "Annual leave"
    assert "Product: Atlas keyboard" in units[1].text
    assert units[1].metadata["section"] == "Annual leave"
    assert units[2].metadata["section"] == "Remote work"


def test_xlsx_sheet_and_row_ranges(samples: Path, settings: Settings) -> None:
    settings.table_rows_per_block = 1
    units = load_file(samples / "products.xlsx", settings)
    assert [
        (unit.metadata["sheet"], unit.metadata["row_start"], unit.metadata["row_end"])
        for unit in units
    ] == [("Products", 2, 2), ("Products", 3, 3), ("People", 2, 2)]
    assert all(unit.metadata["header_row"] == 1 for unit in units)


def test_csv_quoted_values_and_duplicate_headers(tmp_path: Path, settings: Settings) -> None:
    path = tmp_path / "test.csv"
    path.write_text('Name,Name,\n"A, B",second,third\n', encoding="utf-8")
    text = load_file(path, settings)[0].text
    assert "Name: A, B; Name (2): second; Column 3: third" in text


@pytest.mark.parametrize("encoding", ["utf-8-sig", "utf-16", "utf-32"])
def test_bom_encodings(tmp_path: Path, settings: Settings, encoding: str) -> None:
    path = tmp_path / "unicode.txt"
    path.write_bytes("Café — 東京".encode(encoding))
    assert load_file(path, settings)[0].text == "Café — 東京"


def test_encoding_detection_legacy() -> None:
    raw = ("François a commandé un café. L'été à Paris est agréable. " * 20).encode("cp1252")
    assert "café" in decode_text(raw)


def test_duplicate_bytes_have_same_identity(
    samples: Path, tmp_path: Path, settings: Settings
) -> None:
    first = samples / "notes.txt"
    second = tmp_path / "renamed.txt"
    second.write_bytes(first.read_bytes())
    assert (
        load_file(first, settings)[0].metadata["doc_id"]
        == load_file(
            second,
            settings,
        )[0].metadata["doc_id"]
    )


def test_empty_scanned_pdf_has_actionable_error(samples: Path, settings: Settings) -> None:
    with pytest.raises(EmptyDocumentError, match="OCR"):
        load_file(samples / "scanned-placeholder.pdf", settings)


def test_ocr_flag_routes_only_empty_page(
    samples: Path, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []

    def fake_ocr(path: Path, page: int, config: Settings) -> str:
        calls.append(page)
        return "Scanned policy: 24 days leave."

    monkeypatch.setattr("app.ingestion.loaders._ocr_page", fake_ocr)
    settings.ocr_enabled = True
    units = load_file(samples / "scanned-placeholder.pdf", settings)
    assert units[0].metadata["ocr"] is True
    assert calls == [1]
    load_file(samples / "policy.pdf", settings)
    assert calls == [1]


def test_character_limit(samples: Path, settings: Settings) -> None:
    settings.max_extracted_chars = 10
    with pytest.raises(DocumentParseError, match="character limit"):
        load_file(samples / "notes.txt", settings)


def test_csv_row_limit(samples: Path, settings: Settings) -> None:
    settings.max_table_rows = 2
    with pytest.raises(DocumentParseError, match="row limit"):
        load_file(samples / "products.csv", settings)


def test_pdf_page_limit(samples: Path, settings: Settings) -> None:
    settings.max_pdf_pages = 1
    with pytest.raises(DocumentParseError, match="page count"):
        load_file(samples / "policy.pdf", settings)


def test_corrupt_pdf_error_is_safe(tmp_path: Path, settings: Settings) -> None:
    path = tmp_path / "damaged.pdf"
    path.write_bytes(b"%PDF-1.7\nnot a real document")
    with pytest.raises(DocumentParseError, match="damaged"):
        load_file(path, settings)


def test_whitespace_only_file(tmp_path: Path, settings: Settings) -> None:
    path = tmp_path / "empty.md"
    path.write_text(" \n\t\n", encoding="utf-8")
    with pytest.raises(EmptyDocumentError):
        load_file(path, settings)


def test_binary_text_rejected(tmp_path: Path, settings: Settings) -> None:
    path = tmp_path / "binary.txt"
    path.write_bytes(b"\x89PNG\x00binary")
    with pytest.raises(FileValidationError, match="Binary"):
        load_file(path, settings)
