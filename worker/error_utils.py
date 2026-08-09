from __future__ import annotations

import re


_EXCEPTION_LINE = re.compile(r"^[\w.]+(?:Error|Exception):\s*")
_TRACEBACK_PREFIXES = (
    "Traceback ",
    "Original Traceback",
    "During handling of the above exception",
    "The above exception was the direct cause",
    "File \"",
)


def concise_error(error: BaseException | str, max_chars: int = 800) -> str:
    """Return the useful final exception line without exposing a full traceback."""
    lines = [line.strip() for line in str(error).splitlines() if line.strip()]
    useful = [
        line
        for line in lines
        if not line.startswith(_TRACEBACK_PREFIXES) and not set(line) <= {"^", "~", " "}
    ]
    message = useful[-1] if useful else (lines[-1] if lines else type(error).__name__)
    if isinstance(error, BaseException) and not _EXCEPTION_LINE.match(message):
        message = f"{type(error).__name__}: {message}"
    if len(message) > max_chars:
        return message[: max_chars - 3].rstrip() + "..."
    return message
