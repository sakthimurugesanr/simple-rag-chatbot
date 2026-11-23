"""Validate untrusted files before parsing or saving them."""

import hashlib
import re
import zipfile
from pathlib import Path
from uuid import uuid4

from app.config import Settings
from app.exceptions import FileValidationError

MIME_TYPES: dict[str, set[str]] = {
    ".pdf": {"application/pdf"},
    ".docx": {"application/vnd.openxmlformats-officedocument.wordprocessingml.document"},
    ".xlsx": {"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"},
    ".xls": {"application/vnd.ms-excel", "application/x-ole-storage"},
    ".csv": {"text/csv", "application/csv", "application/vnd.ms-excel", "text/plain"},
    ".txt": {"text/plain"},
    ".md": {"text/markdown", "text/x-markdown", "text/plain"},
}
GENERIC_MIME = {"application/octet-stream", "binary/octet-stream"}


def content_hash(path: Path) -> str:
    """Hash raw bytes in bounded blocks for stable document identity."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sanitized_filename(name: str) -> str:
    """Return a unique, portable basename, discarding directories and device names."""
    basename = name.replace("\\", "/").rsplit("/", 1)[-1]
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(basename).stem).strip("_")[:80] or "document"
    return f"{uuid4().hex}_{stem}{Path(basename).suffix.lower()}"


def validate_file(path: Path, settings: Settings, mime_type: str | None = None) -> str:
    """Check size, MIME/extension and signatures, including expanded OOXML size.

    A missing or generic MIME is accepted only with the same signature checks.
    This is format validation, not a malware scanner.
    """
    suffix = path.suffix.lower()
    if suffix not in MIME_TYPES:
        raise FileValidationError(f"Unsupported extension: {suffix or '(none)'}")
    if not path.is_file() or path.is_symlink():
        raise FileValidationError("Input must be a regular, non-symlink file")
    size = path.stat().st_size
    if size == 0:
        raise FileValidationError("The file is empty")
    if size > settings.max_upload_mb * 1024 * 1024:
        raise FileValidationError(f"File exceeds {settings.max_upload_mb} MB upload limit")
    if mime_type:
        declared = mime_type.split(";", 1)[0].strip().lower()
        if declared not in MIME_TYPES[suffix] | GENERIC_MIME:
            raise FileValidationError("Declared MIME type does not match the file extension")
    with path.open("rb") as stream:
        signature = stream.read(8192)
    if suffix == ".pdf" and not signature.startswith(b"%PDF-"):
        raise FileValidationError("Invalid PDF signature")
    if suffix == ".xls" and not signature.startswith(bytes.fromhex("D0CF11E0A1B11AE1")):
        raise FileValidationError("Invalid legacy XLS signature")
    if suffix in {".docx", ".xlsx"}:
        try:
            with zipfile.ZipFile(path) as archive:
                entries = archive.infolist()
                if len(entries) > 10_000:
                    raise FileValidationError("Office archive contains too many members")
                if sum(item.file_size for item in entries) > settings.max_archive_mb * 1024**2:
                    raise FileValidationError("Expanded Office archive exceeds safety limit")
                names = {item.filename for item in entries}
                required = "word/document.xml" if suffix == ".docx" else "xl/workbook.xml"
                if required not in names or "[Content_Types].xml" not in names:
                    raise FileValidationError("Office archive does not match its extension")
                if any(item.flag_bits & 1 for item in entries):
                    raise FileValidationError("Encrypted Office archives are unsupported")
        except zipfile.BadZipFile as exc:
            raise FileValidationError("Invalid Office ZIP archive") from exc
    if suffix in {".txt", ".md", ".csv"}:
        # UTF-16/32 text legitimately contains NULs; other NUL-heavy files are binary.
        has_bom = signature.startswith((b"\xff\xfe", b"\xfe\xff", b"\x00\x00\xfe\xff"))
        if b"\x00" in signature and not has_bom:
            raise FileValidationError("Binary content is not supported as text")
    return suffix


def mask_secrets(value: str) -> str:
    """Redact common key formats and authorization headers in diagnostic strings."""
    value = re.sub(r"(?i)(bearer\s+)\S+", r"\1[REDACTED]", value)
    value = re.sub(r"(?i)((?:api[_-]?key|token|secret)\s*[:=]\s*)[^\s,;]+", r"\1[REDACTED]", value)
    return re.sub(r"\b(?:sk-|gsk_|AIza)[A-Za-z0-9_-]+", "[REDACTED]", value)
