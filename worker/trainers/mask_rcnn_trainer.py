from __future__ import annotations

from pathlib import Path

from .base_trainer import BaseTrainer
from .detection_common import train_detection_model


class MaskRCNNTrainer(BaseTrainer):
    def validate_config(self, config: dict) -> None:
        if not config.get("dataset_path"):
            raise ValueError("MaskRCNNTrainer requires dataset_path")

    def train(self, config: dict, log_path: Path | None = None) -> dict:
        self.validate_config(config)
        return train_detection_model(config, "mask_rcnn", log_path, self._write_log)
