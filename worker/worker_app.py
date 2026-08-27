from __future__ import annotations

import json
import os
import time
import traceback
from pathlib import Path

from error_utils import concise_error
from rq import get_current_job
from security_utils import contained_path, validate_slug
from trainers.trainer_utils import runs_root


def _persist_status(
    job_id: str | None,
    status: str,
    error_detail: str | None = None,
    run_id: str | None = None,
) -> None:
    database_url = os.getenv("DATABASE_URL", "")
    if not database_url or (not job_id and not run_id):
        return
    for attempt in range(3):
        try:
            import psycopg

            with psycopg.connect(database_url, connect_timeout=3) as connection:
                connection.execute(
                    "update training_tasks set status = %s, error_detail = %s, updated_at = now(), "
                    "finished_at = case when %s in ('completed','failed','stopped','cancelled') then now() else finished_at end "
                    "where (%s::uuid is not null and id = %s::uuid) or (%s::text is not null and rq_job_id = %s)",
                    (status, error_detail, status, run_id, run_id, job_id, job_id),
                )
            return
        except Exception as exc:
            if attempt == 2:
                print(f"[Worker] Could not persist run status after 3 attempts: {exc}", flush=True)
            else:
                time.sleep(0.25 * (attempt + 1))


def _build_registry():
    from trainers.deeplabv3plus_trainer import DeepLabV3PlusTrainer
    from trainers.efficientnet_trainer import EfficientNetTrainer
    from trainers.faster_rcnn_trainer import FasterRCNNTrainer
    from trainers.mask_rcnn_trainer import MaskRCNNTrainer
    from trainers.resnet_trainer import ResNetTrainer
    from trainers.yolo_trainer import YOLOTrainer

    return {
        "yolo": YOLOTrainer,
        "resnet": ResNetTrainer,
        "efficientnet": EfficientNetTrainer,
        "deeplabv3plus": DeepLabV3PlusTrainer,
        "mask_rcnn": MaskRCNNTrainer,
        "faster_rcnn": FasterRCNNTrainer,
    }


def run_training(config: dict) -> dict:
    job = get_current_job()
    runs_dir = runs_root()
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
    metadata = config.get("dataset_metadata") or {}
    actionable_issues = list(metadata.get("errors", [])) + [
        warning
        for warning in metadata.get("warnings", [])
        if not str(warning).startswith("Upload inspection sampled")
    ]
    for warning in actionable_issues[:20]:
        log(f"[Dataset warning] {warning}")

    run_id = str(config.get("run_id") or "") or None
    try:
        _persist_status(job.id if job else config.get("job_id"), "running", run_id=run_id)
        log("[Worker] Loading training dependencies and model. The first run can take a while while model weights are prepared.")
        registry = _build_registry()
        trainer_class = registry.get(model_type)
        if trainer_class is None:
            available = ", ".join(sorted(registry))
            raise ValueError(f"Unknown model_type '{model_type}'. Available: {available}")
        trainer = trainer_class()
        log(f"[Worker] Starting training with {trainer_class.__name__}")
        result = trainer.train(config=config, log_path=log_path)
        _persist_status(job.id if job else config.get("job_id"), "completed", run_id=run_id)
        log(f"[Worker] Training completed: {result}")
        return result
    except Exception as exc:
        error_detail = concise_error(exc)
        technical_detail = traceback.format_exc()
        try:
            (log_dir / "error.log").write_text(technical_detail[-65536:], encoding="utf-8")
        except OSError as write_error:
            print(f"[Worker] Could not save error.log: {write_error}", flush=True)
        _persist_status(job.id if job else config.get("job_id"), "failed", error_detail, run_id=run_id)
        log(f"[Worker] FAILED: {error_detail}")
        log("[Worker] Full technical traceback saved to error.log.")
        raise
