from __future__ import annotations

import json
import traceback
from pathlib import Path

from rq import get_current_job
from security_utils import contained_path, validate_slug


def _build_registry():
    from trainers.deeplabv3plus_trainer import DeepLabV3PlusTrainer
    from trainers.efficientnet_trainer import EfficientNetTrainer
    from trainers.faster_rcnn_trainer import FasterRCNNTrainer
    from trainers.mask_rcnn_trainer import MaskRCNNTrainer
    from trainers.paddleocr_trainer import PaddleOCRTrainer
    from trainers.resnet_trainer import ResNetTrainer
    from trainers.tesseract_trainer import TesseractTrainer
    from trainers.yolo_trainer import YOLOTrainer

    return {
        "yolo": YOLOTrainer,
        "resnet": ResNetTrainer,
        "efficientnet": EfficientNetTrainer,
        "deeplabv3plus": DeepLabV3PlusTrainer,
        "mask_rcnn": MaskRCNNTrainer,
        "paddleocr": PaddleOCRTrainer,
        "tesseract": TesseractTrainer,
        "faster_rcnn": FasterRCNNTrainer,
    }


def run_training(config: dict) -> dict:
    job = get_current_job()
    runs_dir = Path("/app/runs")
    runs_dir.mkdir(parents=True, exist_ok=True)

    project_name = config.get("project_name", "train_run")
    validate_slug(project_name, "project name")
    log_dir = contained_path(runs_dir, project_name)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "train.log"
    (log_dir / "job_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    def log(message: str) -> None:
        print(message, flush=True)
        with open(log_path, "a", encoding="utf-8") as file:
            file.write(message + "\n")
        if job:
            job.meta["log_path"] = str(log_path)
            job.meta["project_name"] = project_name
            job.save_meta()

    model_type = config.get("model_type", "yolo").lower()
    task_type = config.get("task_type", "object_detection")
    log(f"[Worker] Job ID: {job.id if job else 'N/A'}")
    log(f"[Worker] Task: {task_type}")
    log(f"[Worker] Model type: {model_type}")
    log(f"[Worker] Project: {project_name}")

    registry = _build_registry()
    trainer_class = registry.get(model_type)
    if trainer_class is None:
        available = ", ".join(sorted(registry))
        raise ValueError(f"Unknown model_type '{model_type}'. Available: {available}")

    try:
        trainer = trainer_class()
        log(f"[Worker] Starting training with {trainer_class.__name__}")
        result = trainer.train(config=config, log_path=log_path)
        log(f"[Worker] Training completed: {result}")
        return result
    except Exception:
        error_detail = traceback.format_exc()
        log(f"[Worker] FAILED:\n{error_detail}")
        raise
