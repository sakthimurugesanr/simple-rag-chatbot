"""Console-only logging with explicit secret masking; never logs document text."""

import sys
from typing import Any

from loguru import logger

from app.utils.security import mask_secrets


def _redact(record: dict[str, Any]) -> bool:
    record["message"] = mask_secrets(record["message"])
    return True


def configure_logging(level: str = "INFO") -> None:
    """Keep diagnostics on stderr so CLI JSON on stdout remains valid."""
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        filter=_redact,
        backtrace=False,
        diagnose=False,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    )
