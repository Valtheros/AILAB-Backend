from pathlib import Path
from typing import Optional

from .base_trainer import BaseTrainer


class YOLOTrainer(BaseTrainer):
    """
    Trainer สำหรับ Ultralytics YOLO (v8, v9, v10, v11)
    รองรับ: object detection, segmentation, pose estimation, classification
    """

    SUPPORTED_TASKS = {"detect", "segment", "pose", "classify"}

    def validate_config(self, config: dict) -> None:
        if not config.get("data_yaml_path"):
            raise ValueError("YOLOTrainer requires 'data_yaml_path' in config")
        if not config.get("model_name"):
            raise ValueError("YOLOTrainer requires 'model_name' in config (e.g. 'yolo11n')")
        task = config.get("task", "detect")
        if task not in self.SUPPORTED_TASKS:
            raise ValueError(f"Unsupported task '{task}'. Must be one of {self.SUPPORTED_TASKS}")

    def train(self, config: dict, log_path: Optional[Path] = None) -> dict:
        from ultralytics import YOLO

        self.validate_config(config)

        model_name   = config["model_name"]          # e.g. "yolo11n", "yolo11s"
        data_yaml    = config["data_yaml_path"]       # absolute path inside container
        epochs       = config.get("epochs", 10)
        batch_size   = config.get("batch_size", 16)
        project_name = config.get("project_name", "train_run")
        extra_args   = config.get("extra_args", {})

        self._write_log(log_path, f"[YOLOTrainer] Starting: model={model_name}, epochs={epochs}")
        self._write_log(log_path, f"[YOLOTrainer] Data: {data_yaml}")

        # โหลด model (ถ้ามี .pt cache อยู่แล้วจะไม่ download ซ้ำ)
        model = YOLO(f"{model_name}.pt")

        results = model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch_size,
            project="/app/runs",
            name=project_name,
            exist_ok=True,      # ไม่ error ถ้า project_name ซ้ำ
            **extra_args,
        )

        results_dir = str(results.save_dir) if results and hasattr(results, "save_dir") else None
        self._write_log(log_path, f"[YOLOTrainer] Done! Results saved to: {results_dir}")

        return {
            "status": "completed",
            "model_type": "yolo",
            "project_name": project_name,
            "results_dir": results_dir,
        }
