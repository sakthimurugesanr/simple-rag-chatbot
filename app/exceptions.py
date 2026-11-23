"""Safe public errors; parser exceptions are retained as causes, not displayed."""


class IngestionError(Exception):
    """Base class for document ingestion failures."""


class FileValidationError(IngestionError):
    """Unsupported type, invalid signature, archive or size limit violation."""


class DocumentParseError(IngestionError):
    """A valid-looking file could not be parsed."""


class EmptyDocumentError(IngestionError):
    """No readable text was extracted."""


class OCRUnavailableError(IngestionError):
    """OCR dependencies are unavailable or OCR failed."""
