import json
import zipfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.cli import main
from app.config import PROJECT_ROOT, Settings
from app.exceptions import FileValidationError
from app.utils.security import mask_secrets, sanitized_filename, validate_file


def test_settings_defaults_and_relative_path() -> None:
    settings = Settings(_env_file=None)
    assert settings.chunk_size == 1000
    assert settings.chunk_overlap == 150
    assert settings.data_dir == PROJECT_ROOT / "data"
    assert settings.embedding_provider == "local"
    assert settings.llm_provider == "openrouter"


def test_environment_overrides_dotenv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env = tmp_path / ".env"
    env.write_text("RAG_CHUNK_SIZE=500\nRAG_CHUNK_OVERLAP=50\n", encoding="utf-8")
    monkeypatch.setenv("RAG_CHUNK_SIZE", "600")
    assert Settings(_env_file=env).chunk_size == 600
    assert Settings(_env_file=env, chunk_size=700).chunk_size == 700


def test_invalid_chunk_config() -> None:
    with pytest.raises(ValidationError, match="smaller"):
        Settings(_env_file=None, chunk_size=100, chunk_overlap=100)


def test_secrets_masked() -> None:
    settings = Settings(_env_file=None, openai_api_key="my-private-value")
    assert "my-private-value" not in repr(settings)
    assert "sk-123456789" not in mask_secrets("key=sk-123456789")
    assert "abc" not in mask_secrets("Authorization: Bearer abc")


def test_unique_sanitized_name() -> None:
    first = sanitized_filename("../../CON bad file.PDF")
    assert first.endswith("_CON_bad_file.pdf")
    assert "/" not in first and "\\" not in first
    assert first != sanitized_filename("../../CON bad file.PDF")


def test_extension_mime_and_signature(samples: Path, settings: Settings, tmp_path: Path) -> None:
    with pytest.raises(FileValidationError, match="MIME"):
        validate_file(samples / "notes.txt", settings, "image/png")
    assert validate_file(samples / "policy.pdf", settings, "application/octet-stream") == ".pdf"
    spoof = tmp_path / "spoof.pdf"
    spoof.write_text("not PDF", encoding="utf-8")
    with pytest.raises(FileValidationError, match="signature"):
        validate_file(spoof, settings)
    wrong = tmp_path / "wrong.xlsx"
    wrong.write_bytes((samples / "policy.docx").read_bytes())
    with pytest.raises(FileValidationError, match="extension"):
        validate_file(wrong, settings)


def test_size_limit(tmp_path: Path, settings: Settings) -> None:
    path = tmp_path / "large.txt"
    path.write_bytes(b"a" * (1024 * 1024 + 1))
    settings.max_upload_mb = 1
    with pytest.raises(FileValidationError, match="upload limit"):
        validate_file(path, settings)


def test_expanded_archive_limit(tmp_path: Path, settings: Settings) -> None:
    path = tmp_path / "bomb.docx"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("word/document.xml", "a" * (1024 * 1024 + 1))
        archive.writestr("[Content_Types].xml", "content")
    settings.max_archive_mb = 1
    with pytest.raises(FileValidationError, match="Expanded"):
        validate_file(path, settings)


def test_unsupported_and_empty_file(tmp_path: Path, settings: Settings) -> None:
    path = tmp_path / "bad.exe"
    path.write_bytes(b"bad")
    with pytest.raises(FileValidationError, match="extension"):
        validate_file(path, settings)
    empty = tmp_path / "empty.txt"
    empty.touch()
    with pytest.raises(FileValidationError, match="empty"):
        validate_file(empty, settings)


def test_cli_json_output(samples: Path, tmp_path: Path) -> None:
    output = tmp_path / "output" / "chunks.json"
    assert main([str(samples / "policy.docx"), "--output", str(output)]) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["chunks"]
    assert "Annual leave" in payload["documents"][0]["text"]


def test_cli_invalid_settings_is_nonzero(samples: Path) -> None:
    assert main([str(samples / "notes.txt"), "--chunk-size", "50", "--overlap", "100"]) == 1


def test_cli_missing_file_is_nonzero(tmp_path: Path) -> None:
    assert main([str(tmp_path / "missing.txt")]) == 1
