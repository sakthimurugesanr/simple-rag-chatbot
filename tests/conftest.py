"""Tiny generated files exercise actual parsers, not parser mocks."""

from pathlib import Path

import pytest

from app.config import Settings
from scripts.generate_samples import generate_samples


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(_env_file=None, data_dir=tmp_path / "data")


@pytest.fixture(scope="session")
def samples(tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = tmp_path_factory.mktemp("samples")
    generate_samples(directory)
    (directory / "notes.txt").write_text("Annual leave is 24 days.\nOwner: Mira.", encoding="utf-8")
    (directory / "handbook.md").write_text("# Leave\n\nAnnual leave is 24 days.", encoding="utf-8")
    (directory / "products.csv").write_text(
        'Product,Price INR,Stock\nAtlas keyboard,2500,12\n"Mouse, wireless",1200,25\n',
        encoding="utf-8",
    )
    return directory
