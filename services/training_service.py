from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

import yaml
from redis import Redis
from rq import Queue
from rq.job import Job, JobStatus

from dataset_utils import find_dataset_yaml, inspect_dataset
from model_catalog import get_model


OCR_MODEL_TYPES = {"paddleocr", "tesseract"}


class TrainingService:
    def __init__(self):
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        try:
            self.redis = Redis.from_url(redis_url)
            self.redis.ping()
            self.queues = {
                "cv_training": Queue("cv_training", connection=self.redis),
                "ocr_training": Queue("ocr_training", connection=self.redis),
            }
            print(f"[TrainingService] Connected to Redis at {redis_url}")
        except Exception as exc:
            print(f"[TrainingService] ERROR: Cannot connect to Redis: {exc}")
            self.redis = None
            self.queues = {}

        self.base_dir = Path("/app") if Path("/app").exists() else Path(os.getcwd()).absolute()
        self.dataset_dir = self.base_dir / "dataset"
        self.runs_dir = self.base_dir / "runs"
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def _ensure_redis(self) -> None:
        if self.redis is None or not self.queues:
            raise RuntimeError("Redis is not connected. Make sure the Redis service is running.")

    def _queue_name_for_model(self, model_type: str) -> str:
        return "ocr_training" if model_type in OCR_MODEL_TYPES else "cv_training"

    def _find_dataset_path(self, dataset_name: str | None = None) -> Path:
        if dataset_name:
            dataset_path = self.dataset_dir / dataset_name
            if not dataset_path.exists() or not dataset_path.is_dir():
                raise FileNotFoundError(f"Dataset '{dataset_name}' was not found in {self.dataset_dir}.")
            return dataset_path

        candidates = [path for path in self.dataset_dir.iterdir() if path.is_dir()]
        if not candidates:
            raise FileNotFoundError(f"No datasets found in {self.dataset_dir}. Please upload a dataset first.")
        return sorted(candidates, key=lambda path: path.stat().st_ctime, reverse=True)[0]

    def _assert_dataset_matches_model(self, dataset_path: Path, model_type: str, task_type: str) -> dict[str, Any]:
        metadata = inspect_dataset(dataset_path)
        model_entry = get_model(model_type)
        expected_formats = set(model_entry.get("dataset_formats", [])) if model_entry else set()
        detected_formats = set(metadata["formats"])

        if expected_formats and not expected_formats.intersection(detected_formats):
            raise ValueError(
                f"Dataset '{dataset_path.name}' is not compatible with {model_type}. "
                f"Detected formats: {sorted(detected_formats) or ['none']}. "
                f"Expected one of: {sorted(expected_formats)}."
            )
        if task_type not in metadata["tasks"] and expected_formats:
            raise ValueError(
                f"Dataset '{dataset_path.name}' does not advertise task '{task_type}'. "
                f"Detected tasks: {metadata['tasks'] or ['none']}."
            )
        return metadata

    def _create_worker_yaml(self, original_yaml_path: Path, dataset_folder: Path) -> str:
        config = yaml.safe_load(original_yaml_path.read_text(encoding="utf-8")) or {}
        config["path"] = str(dataset_folder)

        path_map = {
            "train": ["train", "training"],
            "val": ["valid", "val", "validation"],
            "test": ["test"],
        }
        for yaml_key, folder_candidates in path_map.items():
            if yaml_key in config:
                for candidate in folder_candidates:
                    if (dataset_folder / candidate).exists():
                        config[yaml_key] = f"{candidate}/images"
                        break

        worker_yaml_path = dataset_folder / "data_worker.yaml"
        worker_yaml_path.write_text(yaml.dump(config, default_flow_style=False), encoding="utf-8")
        return str(worker_yaml_path)

    def start_training_container(
        self,
        task_type: str,
        model_type: str,
        model_name: str,
        epochs: int,
        batch_size: int,
        project_name: str,
        dataset_name: str | None = None,
        extra_args: dict[str, Any] | None = None,
    ) -> str:
        self._ensure_redis()

        dataset_path = self._find_dataset_path(dataset_name)
        dataset_metadata = self._assert_dataset_matches_model(dataset_path, model_type, task_type)
        data_yaml_path = None
        original_yaml = find_dataset_yaml(dataset_path)
        if original_yaml:
            data_yaml_path = self._create_worker_yaml(original_yaml, dataset_path)

        job_config = {
            "task_type": task_type,
            "model_type": model_type,
            "model_name": model_name,
            "epochs": epochs,
            "batch_size": batch_size,
            "project_name": project_name,
            "dataset_name": dataset_path.name,
            "dataset_path": str(dataset_path),
            "data_yaml_path": data_yaml_path,
            "dataset_metadata": dataset_metadata,
            "extra_args": extra_args or {},
        }

        queue_name = self._queue_name_for_model(model_type)
        queue = self.queues[queue_name]
        job = queue.enqueue(
            "worker_app.run_training",
            job_config,
            job_timeout="24h",
            result_ttl=86400,
            failure_ttl=86400,
        )

        job.meta["project_name"] = project_name
        job.meta["model_type"] = model_type
        job.meta["task_type"] = task_type
        job.meta["queue"] = queue_name
        job.save_meta()

        print(f"[TrainingService] Job enqueued: {job.id} ({queue_name}: {task_type}/{model_type}/{model_name})")
        return job.id

    def get_container_status(self, job_id: str) -> str:
        if not self.redis:
            return "redis_connection_error"
        try:
            job = Job.fetch(job_id, connection=self.redis)
            status = job.get_status()
            status_str = status.value if isinstance(status, JobStatus) else str(status)
            status_map = {
                "queued": "queued",
                "started": "running",
                "finished": "exited",
                "failed": "failed",
                "stopped": "stopped",
                "canceled": "stopped",
                "deferred": "queued",
                "scheduled": "queued",
            }
            return status_map.get(status_str, f"unknown ({status_str})")
        except Exception:
            return "not_found"

    def get_container_logs(self, job_id: str) -> str:
        if not self.redis:
            return "Redis connection error: Is Redis running?"
        try:
            job = Job.fetch(job_id, connection=self.redis)
            log_path_str = job.meta.get("log_path")
            if log_path_str:
                log_path = Path(log_path_str)
                if log_path.exists():
                    return log_path.read_text(encoding="utf-8")

            status = job.get_status()
            status_str = status.value if isinstance(status, JobStatus) else str(status)
            if status_str == "failed" and job.exc_info:
                return f"[Job Failed]\n{job.exc_info}"
            return "No logs available yet. Training may not have started."
        except Exception as exc:
            return f"Error fetching logs: {exc}"

    def get_training_metrics(self, project_name: str) -> list[dict[str, str]]:
        results_path = self.runs_dir / project_name / "results.csv"
        if not results_path.exists():
            return []
        try:
            with open(results_path, "r", encoding="utf-8") as file:
                reader = csv.DictReader(file)
                rows = list(reader)
            return [{(key or "").strip(): (value or "").strip() for key, value in row.items()} for row in rows]
        except Exception as exc:
            print(f"[TrainingService] Error reading metrics: {exc}")
            return []

    def stop_training_container(self, job_id: str) -> str:
        self._ensure_redis()
        try:
            job = Job.fetch(job_id, connection=self.redis)
            job.cancel()
            print(f"[TrainingService] Job {job_id} cancelled")
            return "cancelled"
        except Exception as exc:
            raise RuntimeError(f"Could not stop job {job_id}: {exc}")

    def list_runs(self) -> list[dict[str, Any]]:
        runs = []
        for project_dir in sorted(self.runs_dir.iterdir(), key=lambda path: path.stat().st_mtime, reverse=True):
            if not project_dir.is_dir():
                continue
            config_path = project_dir / "job_config.json"
            config: dict[str, Any] = {}
            if config_path.exists():
                try:
                    config = json.loads(config_path.read_text(encoding="utf-8"))
                except Exception:
                    config = {}

            files = []
            for path in project_dir.rglob("*"):
                if path.is_file():
                    files.append(
                        {
                            "path": str(path.relative_to(project_dir)).replace("\\", "/"),
                            "name": path.name,
                            "size": path.stat().st_size,
                        }
                    )

            latest_metrics = None
            metrics = self.get_training_metrics(project_dir.name)
            if metrics:
                latest_metrics = metrics[-1]

            runs.append(
                {
                    "project_name": project_dir.name,
                    "createdAt": project_dir.stat().st_ctime,
                    "updatedAt": project_dir.stat().st_mtime,
                    "task_type": config.get("task_type"),
                    "model_type": config.get("model_type"),
                    "model_name": config.get("model_name"),
                    "dataset_name": config.get("dataset_name"),
                    "epochs": config.get("epochs"),
                    "files": files,
                    "latest_metrics": latest_metrics,
                }
            )
        return runs
