from __future__ import annotations

import re
from pathlib import Path


SLUG_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
WINDOWS_RESERVED_NAMES = {
    "aux",
    "con",
    "nul",
    "prn",
    *(f"com{index}" for index in range(1, 10)),
    *(f"lpt{index}" for index in range(1, 10)),
}


def validate_slug(value: str, label: str = "identifier") -> str:
    base_name = value.split(".", 1)[0].lower()
    if value in {".", ".."} or value.endswith(".") or base_name in WINDOWS_RESERVED_NAMES or not SLUG_PATTERN.fullmatch(value):
        raise ValueError(
            f"Invalid {label}. Use 1-128 letters, numbers, dots, underscores, or hyphens; "
            "the first character must be a letter or number."
        )
    return value


def contained_path(root: Path, *parts: str | Path) -> Path:
    root_path = root.resolve()
    candidate = root_path.joinpath(*parts).resolve()
    if not candidate.is_relative_to(root_path):
        raise ValueError(f"Path escapes configured root: {candidate}")
    return candidate
