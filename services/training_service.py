from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

import yaml
from redis import Redis
from rq import Queue
from rq.command import send_stop_job_command
from rq.job import Job, JobStatus

from dataset_utils import find_dataset_yaml, inspect_dataset
from model_catalog import get_model
from security_utils import contained_path, validate_slug
from settings import DATASET_DIR, REDIS_URL, RUNS_DIR, ensure_runtime_dirs


OCR_MODEL_TYPES = {"paddleocr", "tesseract"}


class TrainingService:
    def __init__(self):
        try:
            self.redis = Redis.from_url(REDIS_URL)
            self.redis.ping()
            self.queues = {
                "cv_training": Queue("cv_training", connection=self.redis),
                "ocr_training": Queue("ocr_training", connection=self.redis),
            }
            print(f"[TrainingService] Connected to Redis at {REDIS_URL}")
        except Exception as exc:
            print(f"[TrainingService] ERROR: Cannot connect to Redis: {exc}")
            self.redis = None
            self.queues = {}

        ensure_runtime_dirs()
        self.dataset_dir = DATASET_DIR
        self.runs_dir = RUNS_DIR

    def _ensure_redis(self) -> None:
        if self.redis is None or not self.queues:
            raise RuntimeError("Redis is not connected. Make sure the Redis service is running.")

    def _queue_name_for_model(self, model_type: str) -> str:
        return "ocr_training" if model_type in OCR_MODEL_TYPES else "cv_training"

    def _dataset_meta_path(self, dataset_path: Path) -> Path:
        return dataset_path / ".ailab_dataset.json"

    def _dataset_owner(self, dataset_path: Path) -> str | None:
        try:
            metadata = json.loads(self._dataset_meta_path(dataset_path).read_text(encoding="utf-8"))
        except Exception:
            return None
        if not isinstance(metadata, dict):
            return None
        owner = metadata.get("created_by")
        return str(owner) if owner else None

    def _is_dataset_visible(self, dataset_path: Path, owner_id: str | None = None) -> bool:
        dataset_owner = self._dataset_owner(dataset_path)
        return not owner_id or not dataset_owner or dataset_owner == owner_id

    def _assert_dataset_visible(self, dataset_path: Path, owner_id: str | None = None) -> None:
        if not self._is_dataset_visible(dataset_path, owner_id):
            raise FileNotFoundError(f"Dataset '{dataset_path.name}' was not found in {self.dataset_dir}.")

    def _find_dataset_path(self, dataset_name: str | None = None, owner_id: str | None = None) -> Path:
        if dataset_name:
            validate_slug(dataset_name, "dataset name")
            dataset_path = contained_path(self.dataset_dir, dataset_name)
            if not dataset_path.exists() or not dataset_path.is_dir():
                raise FileNotFoundError(f"Dataset '{dataset_name}' was not found in {self.dataset_dir}.")
            self._assert_dataset_visible(dataset_path, owner_id)
            return dataset_path

        candidates = [
            path
            for path in self.dataset_dir.iterdir()
            if path.is_dir() and not path.name.startswith(".") and self._is_dataset_visible(path, owner_id)
        ]
        if not candidates:
            raise FileNotFoundError(f"No datasets found in {self.dataset_dir}. Please upload a dataset first.")
        return sorted(candidates, key=lambda path: path.stat().st_ctime, reverse=True)[0]

    def _assert_dataset_matches_model(
        self,
        dataset_path: Path,
        model_type: str,
        task_type: str,
        extra_args: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        metadata = inspect_dataset(dataset_path)
        model_entry = get_model(model_type)
        expected_formats = set(model_entry.get("dataset_formats", [])) if model_entry else set()
        detected_formats = set(metadata["formats"])
        extra_args = extra_args or {}

        if model_entry and task_type != model_entry["task_type"]:
            raise ValueError(f"Model '{model_type}' belongs to task '{model_entry['task_type']}', not '{task_type}'.")
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
        if model_type == "paddleocr":
            requested_task = str(extra_args.get("ocr_task", "rec"))
            available_tasks = set(metadata.get("paddleocr_tasks", []))
            if available_tasks and requested_task not in available_tasks:
                raise ValueError(
                    f"Dataset '{dataset_path.name}' has PaddleOCR labels for {sorted(available_tasks)}, "
                    f"but the selected PaddleOCR task is '{requested_task}'."
                )
        if model_type == "mask_rcnn" and metadata.get("coco", {}).get("mask_annotations", 0) <= 0:
            raise ValueError(f"Dataset '{dataset_path.name}' has no COCO instance masks for Mask R-CNN.")
        if model_type == "faster_rcnn" and not ({"yolo_detection", "coco_instances"}.intersection(detected_formats)):
            raise ValueError(f"Dataset '{dataset_path.name}' has no bounding-box annotations for Faster R-CNN.")
        return metadata

    def _find_latest_compatible_dataset_path(
        self,
        model_type: str,
        task_type: str,
        owner_id: str | None = None,
        extra_args: dict[str, Any] | None = None,
    ) -> Path:
        candidates = [path for path in self.dataset_dir.iterdir() if path.is_dir() and not path.name.startswith(".")]
        for candidate in sorted(candidates, key=lambda path: path.stat().st_ctime, reverse=True):
            if not self._is_dataset_visible(candidate, owner_id):
                continue
            try:
                self._assert_dataset_matches_model(candidate, model_type, task_type, extra_args=extra_args)
                return candidate
            except ValueError:
                continue
        raise FileNotFoundError(f"No uploaded dataset is compatible with {model_type}. Please upload or select one first.")

    def _normalize_yolo_yaml_path(self, dataset_folder: Path, yaml_key: str, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"YOLO YAML '{yaml_key}' paths must be non-empty text")

        normalized = value.strip().replace("\\", "/")
        candidates = [normalized]
        stripped = normalized
        while stripped.startswith("../"):
            stripped = stripped[3:]
            candidates.append(stripped)
        if stripped.startswith("./"):
            candidates.append(stripped[2:])

        seen: set[str] = set()
        for candidate_value in candidates:
            if candidate_value in seen:
                continue
            seen.add(candidate_value)
            try:
                candidate = contained_path(dataset_folder, candidate_value)
            except ValueError:
                continue
            if candidate.exists():
                return candidate_value

        raise ValueError(f"YOLO YAML '{yaml_key}' path was not found inside the dataset: {value}")

    def _create_worker_yaml(self, original_yaml_path: Path, dataset_folder: Path) -> str:
        config = yaml.safe_load(original_yaml_path.read_text(encoding="utf-8")) or {}
        if not isinstance(config, dict):
            raise ValueError("YOLO YAML root must be an object")
        if not isinstance(config.get("names"), (dict, list)) or not config["names"]:
            raise ValueError("YOLO YAML requires a non-empty names list or mapping")
        config["path"] = str(dataset_folder)
        for yaml_key in ("train", "val", "test"):
            value = config.get(yaml_key)
            if value is None:
                continue
            if isinstance(value, list):
                config[yaml_key] = [self._normalize_yolo_yaml_path(dataset_folder, yaml_key, item) for item in value]
            else:
                config[yaml_key] = self._normalize_yolo_yaml_path(dataset_folder, yaml_key, value)
        if "train" not in config:
            raise ValueError("YOLO YAML requires a train path")

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
        owner_id: str | None = None,
        owner_email: str | None = None,
    ) -> str:
        self._ensure_redis()
        validate_slug(project_name, "project name")
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        project_dir = contained_path(self.runs_dir, project_name)
        if project_dir.exists():
            raise FileExistsError(f"Run '{project_name}' already exists.")

        dataset_path = (
            self._find_dataset_path(dataset_name, owner_id=owner_id)
            if dataset_name
            else self._find_latest_compatible_dataset_path(
                model_type,
                task_type,
                owner_id=owner_id,
                extra_args=extra_args or {},
            )
        )
        dataset_metadata = self._assert_dataset_matches_model(
            dataset_path,
            model_type,
            task_type,
            extra_args=extra_args or {},
        )
        data_yaml_path = None
        uses_yolo_dataset = model_type == "yolo" or (
            model_type == "faster_rcnn" and "yolo_detection" in dataset_metadata["formats"]
        )
        if uses_yolo_dataset:
            original_yaml = find_dataset_yaml(dataset_path)
            if original_yaml is None:
                raise ValueError(f"Dataset '{dataset_path.name}' is missing data.yaml for YOLO-style labels.")
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
            "created_by": owner_id,
            "created_by_email": owner_email,
        }
        if model_type == "yolo":
            job_config["task"] = "detect"

        queue_name = self._queue_name_for_model(model_type)
        queue = self.queues[queue_name]
        reserved_project_dir = False
        try:
            project_dir.mkdir(parents=False, exist_ok=False)
            reserved_project_dir = True
            (project_dir / "job_config.json").write_text(json.dumps(job_config, indent=2, ensure_ascii=False), encoding="utf-8")

            job = queue.enqueue(
                "worker_app.run_training",
                job_config,
                job_timeout="24h",
                result_ttl=86400,
                failure_ttl=86400,
            )
        except FileExistsError as exc:
            raise FileExistsError(f"Run '{project_name}' already exists.") from exc
        except Exception:
            if reserved_project_dir:
                shutil.rmtree(project_dir, ignore_errors=True)
            raise

        job.meta["project_name"] = project_name
        job.meta["model_type"] = model_type
        job.meta["task_type"] = task_type
        job.meta["queue"] = queue_name
        if owner_id:
            job.meta["created_by"] = owner_id
        if owner_email:
            job.meta["created_by_email"] = owner_email
        job.save_meta()

        print(f"[TrainingService] Job enqueued: {job.id} ({queue_name}: {task_type}/{model_type}/{model_name})")
        return job.id

    def _run_owner(self, project_name: str) -> str | None:
        try:
            validate_slug(project_name, "project name")
            config_path = contained_path(self.runs_dir, project_name, "job_config.json")
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if not isinstance(config, dict):
            return None
        owner = config.get("created_by")
        return str(owner) if owner else None

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
                log_path = contained_path(self.runs_dir, Path(log_path_str))
                if log_path.exists():
                    return log_path.read_text(encoding="utf-8")

            status = job.get_status()
            status_str = status.value if isinstance(status, JobStatus) else str(status)
            if status_str == "failed" and job.exc_info:
                return f"[Job Failed]\n{job.exc_info}"
            return "No logs available yet. Training may not have started."
        except Exception as exc:
            return f"Error fetching logs: {exc}"

    def get_job_owner(self, job_id: str) -> str | None:
        if not self.redis:
            return None
        try:
            job = Job.fetch(job_id, connection=self.redis)
            owner = job.meta.get("created_by")
            return str(owner) if owner else None
        except Exception:
            return None

    def _job_log_path(self, job_id: str) -> Path | None:
        if not self.redis:
            return None
        job = Job.fetch(job_id, connection=self.redis)
        log_path_str = job.meta.get("log_path")
        if not log_path_str:
            return None
        log_path = contained_path(self.runs_dir, Path(log_path_str))
        return log_path if log_path.is_file() else None

    def read_job_log_chunk(self, job_id: str, offset: int = 0) -> tuple[str, int, bool]:
        if not self.redis:
            return "", offset, False
        try:
            log_path = self._job_log_path(job_id)
            if log_path is None:
                return "", offset, False
            size = log_path.stat().st_size
            replace = size < offset
            start = 0 if replace else offset
            with open(log_path, "rb") as file:
                file.seek(start)
                return file.read().decode("utf-8", errors="replace"), size, replace
        except Exception:
            return "", offset, False

    def get_training_metrics(self, project_name: str) -> list[dict[str, str]]:
        validate_slug(project_name, "project name")
        results_path = contained_path(self.runs_dir, project_name, "results.csv")
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

    def get_job_snapshot(self, job_id: str) -> dict[str, Any]:
        status = self.get_container_status(job_id)
        if status == "not_found":
            return {"job_id": job_id, "status": status, "project_name": None, "logs": "", "metrics": []}
        project_name = None
        if self.redis:
            try:
                job = Job.fetch(job_id, connection=self.redis)
                project_name = job.meta.get("project_name")
            except Exception:
                pass
        metrics = self.get_training_metrics(project_name) if project_name else []
        log_path = None
        try:
            log_path = self._job_log_path(job_id)
        except Exception:
            pass
        return {
            "job_id": job_id,
            "status": status,
            "project_name": project_name,
            "logs": self.get_container_logs(job_id),
            "log_offset": log_path.stat().st_size if log_path else 0,
            "metrics": metrics,
        }

    def stop_training_container(self, job_id: str) -> str:
        self._ensure_redis()
        try:
            job = Job.fetch(job_id, connection=self.redis)
            status = job.get_status()
            status_str = status.value if isinstance(status, JobStatus) else str(status)
            if status_str == "started":
                send_stop_job_command(self.redis, job_id)
                print(f"[TrainingService] Stop command sent for running job {job_id}")
                return "stopping"
            job.cancel()
            print(f"[TrainingService] Job {job_id} cancelled")
            return "cancelled"
        except Exception as exc:
            raise RuntimeError(f"Could not stop job {job_id}: {exc}")

    def list_runs(self, owner_id: str | None = None) -> list[dict[str, Any]]:
        runs = []
        for project_dir in sorted(self.runs_dir.iterdir(), key=lambda path: path.stat().st_mtime, reverse=True):
            if not project_dir.is_dir() or project_dir.name.startswith("."):
                continue
            config_path = project_dir / "job_config.json"
            config: dict[str, Any] = {}
            if config_path.exists():
                try:
                    config = json.loads(config_path.read_text(encoding="utf-8"))
                except Exception:
                    config = {}
            run_owner = config.get("created_by")
            if owner_id and run_owner and run_owner != owner_id:
                continue

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
                    "created_by": run_owner,
                    "epochs": config.get("epochs"),
                    "files": files,
                    "latest_metrics": latest_metrics,
                }
            )
        return runs
