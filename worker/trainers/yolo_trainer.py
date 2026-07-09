from __future__ import annotations

from pathlib import Path

from .base_trainer import BaseTrainer
from .trainer_utils import runs_root


class YOLOTrainer(BaseTrainer):
    SUPPORTED_TASKS = {"detect", "segment", "pose", "classify"}

    def validate_config(self, config: dict) -> None:
        if not config.get("data_yaml_path"):
            raise ValueError("YOLOTrainer requires data_yaml_path")
        if not config.get("model_name"):
            raise ValueError("YOLOTrainer requires model_name, for example yolo11n")
        task = config.get("task", "detect")
        if task not in self.SUPPORTED_TASKS:
            raise ValueError(f"Unsupported YOLO task '{task}'. Must be one of {sorted(self.SUPPORTED_TASKS)}")

    def train(self, config: dict, log_path: Path | None = None) -> dict:
        from ultralytics import YOLO

        self.validate_config(config)

        model_name = str(config["model_name"]).removesuffix(".pt")
        data_yaml = config["data_yaml_path"]
        epochs = int(config.get("epochs", 10))
        batch_size = int(config.get("batch_size", 16))
        project_name = config.get("project_name", "train_run")
        extra_args = dict(config.get("extra_args", {}))
        extra_args.pop("model_size", None)
        extra_args.pop("epochs", None)
        extra_args.pop("batch_size", None)

        self._write_log(log_path, f"[YOLOTrainer] model={model_name}, epochs={epochs}, batch={batch_size}")
        self._write_log(log_path, f"[YOLOTrainer] data={data_yaml}")

        model = YOLO(f"{model_name}.pt")
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch_size,
            project=str(runs_root()),
            name=project_name,
            exist_ok=True,
            task=config.get("task", "detect"),
            **extra_args,
        )

        fallback_dir = runs_root() / project_name
        results_dir = str(results.save_dir) if results and hasattr(results, "save_dir") else str(fallback_dir)
        self._write_log(log_path, f"[YOLOTrainer] results={results_dir}")

        return {
            "status": "completed",
            "task_type": config.get("task_type", "object_detection"),
            "model_type": "yolo",
            "project_name": project_name,
            "results_dir": results_dir,
        }
