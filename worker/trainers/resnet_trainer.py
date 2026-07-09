from __future__ import annotations

from pathlib import Path

from .base_trainer import BaseTrainer
from .classification_common import train_classifier


class ResNetTrainer(BaseTrainer):
    def validate_config(self, config: dict) -> None:
        if not config.get("dataset_path"):
            raise ValueError("ResNetTrainer requires dataset_path")

    def train(self, config: dict, log_path: Path | None = None) -> dict:
        self.validate_config(config)
        return train_classifier(config, "resnet", log_path, self._write_log)
