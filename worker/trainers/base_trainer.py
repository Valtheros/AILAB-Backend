from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class BaseTrainer(ABC):
    @abstractmethod
    def train(self, config: dict, log_path: Path | None = None) -> dict:
        pass

    @abstractmethod
    def validate_config(self, config: dict) -> None:
        pass

    def _write_log(self, log_path: Path | None, message: str) -> None:
        print(message, flush=True)
        if log_path:
            with open(log_path, "a", encoding="utf-8") as file:
                file.write(message + "\n")
