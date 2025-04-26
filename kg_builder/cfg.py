from dataclasses import field

from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class SplitConfig:
    chunk_char_size: int = 4096
    chunk_char_overlap: int = 128
    separators: list[str] = field(default_factory=lambda: ["\n\n", "\n", "\t", ".", " ", ""])
