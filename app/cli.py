"""Offline inspection CLI: python -m app.cli samples/handbook.md --preview."""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from pydantic import ValidationError

from app.config import Settings
from app.exceptions import IngestionError
from app.ingestion.chunker import chunk_documents
from app.ingestion.loaders import load_file
from app.utils.logging import configure_logging


def main(argv: list[str] | None = None) -> int:
    """Emit extraction/chunk JSON to stdout or a user-specified output file."""
    parser = argparse.ArgumentParser(description="Validate, extract and chunk local documents")
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--overlap", type=int)
    parser.add_argument(
        "--ocr", action="store_true", help="Enable OCR fallback for empty PDF pages"
    )
    parser.add_argument(
        "--preview", action="store_true", help="Show summaries rather than full JSON"
    )
    parser.add_argument("--output", type=Path, help="Write full JSON to this path")
    args = parser.parse_args(argv)
    overrides: dict[str, object] = {}
    if args.chunk_size is not None:
        overrides["chunk_size"] = args.chunk_size
    if args.overlap is not None:
        overrides["chunk_overlap"] = args.overlap
    if args.ocr:
        overrides["ocr_enabled"] = True
    try:
        settings = Settings(**overrides)
        configure_logging(settings.log_level)
        documents = [unit for file in args.files for unit in load_file(file, settings)]
        chunks = chunk_documents(documents, settings.chunk_size, settings.chunk_overlap)
        payload = {
            "documents": [asdict(unit) for unit in documents],
            "chunks": [asdict(chunk) for chunk in chunks],
        }
        encoded = json.dumps(payload, ensure_ascii=False, indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded + "\n", encoding="utf-8")
        if args.preview:
            print(f"Extracted {len(documents)} source units into {len(chunks)} chunks.")
            for chunk in chunks:
                print(
                    f"\n[{chunk.metadata['source']} | chunk {chunk.metadata['chunk_index']}]"
                    f"\n{chunk.text[:250]}"
                )
        elif not args.output:
            print(encoded)
        return 0
    except (IngestionError, ValidationError, OSError) as exc:
        # ValidationError's full text may contain input values; never print settings inputs.
        message = (
            "Invalid settings; check chunk size/overlap and .env values"
            if isinstance(
                exc,
                ValidationError,
            )
            else str(exc)
        )
        print(f"Error: {message}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
