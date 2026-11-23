"""Common parser and chunk interfaces, independent of the vector database."""

from dataclasses import dataclass, field

MetadataValue = str | int | float | bool


@dataclass(frozen=True)
class Document:
    """One page, section or table block with flat, vector-store-safe metadata."""

    text: str
    metadata: dict[str, MetadataValue] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    """One context fragment and its source coordinates."""

    text: str
    metadata: dict[str, MetadataValue]
