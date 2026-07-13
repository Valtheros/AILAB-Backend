from __future__ import annotations

import csv
import json
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import yaml
from redis import Redis
from rq import Queue
from rq.command import send_stop_job_command
from rq.job import Job, JobStatus

try:
    from rq.exceptions import NoSuchJobError
except ImportError:  # Lightweight unit-test stubs do not expose rq.exceptions.
    class NoSuchJobError(Exception):
        pass

from dataset_utils import compatible_models_for_metadata, inspect_dataset, prepare_dataset_for_model, read_yaml_limited
from dataset_storage import registered_storage_path
from model_catalog import get_model
from security_utils import contained_path, validate_slug
from settings import DATASET_DIR, REDIS_URL, RUNS_DIR, ensure_runtime_dirs
from resource_repository import resource_repository


OCR_MODEL_TYPES = {"paddleocr", "tesseract"}
MAX_LOG_RESPONSE_BYTES = 1024 * 1024
MAX_RUN_FILES_RETURNED = 1000


class TrainingService:
    def __init__(self):
        self.redis = None
        self.queues = {}
        self._redis_lock = threading.Lock()
        self._next_redis_attempt = 0.0
        self._redis_backoff = 1.0
        ensure_runtime_dirs()
        self.dataset_dir = DATASET_DIR
        self.runs_dir = RUNS_DIR
        if self._connect_redis():
            self._reconcile_queued_runs()

    def _reconcile_queued_runs(self) -> None:
        if not self.redis or not resource_repository.enabled:
            return
        for run in resource_repository.list_queued_runs():
            job_id = str(run.get("rq_job_id") or "")
            if not job_id:
                resource_repository.update_run_status_by_id(
                    run["id"], "failed", "Queued run had no reserved Redis job ID after backend restart."
                )
                continue
            try:
                Job.fetch(job_id, connection=self.redis)
            except NoSuchJobError:
                resource_repository.update_run_status_by_id(
                    run["id"], "failed", "Reserved Redis job was not found after backend restart."
                )
            except Exception as exc:
                print(f"[TrainingService] Could not reconcile queued run {run['run_slug']}: {exc}")

    def _connect_redis(self) -> bool:
        now = time.monotonic()
        if self.redis is not None and self.queues:
            return True
        if now < self._next_redis_attempt:
            return False
        with self._redis_lock:
            if self.redis is not None and self.queues:
                return True
            try:
                connection = Redis.from_url(REDIS_URL, socket_connect_timeout=2, socket_timeout=5)
                connection.ping()
                self.redis = connection
                self.queues = {
                    "cv_training": Queue("cv_training", connection=connection),
                    "ocr_training": Queue("ocr_training", connection=connection),
                }
                self._redis_backoff = 1.0
                self._next_redis_attempt = 0.0
                print(f"[TrainingService] Connected to Redis at {REDIS_URL}")
                return True
            except Exception as exc:
                self.redis = None
                self.queues = {}
                self._next_redis_attempt = time.monotonic() + self._redis_backoff
                self._redis_backoff = min(self._redis_backoff * 2, 30.0)
                print(f"[TrainingService] Redis unavailable; retrying later: {exc}")
                return False

    def _ensure_redis(self) -> None:
        if not self._connect_redis():
            raise RuntimeError("Redis is not connected. The backend will retry automatically.")

    def _invalidate_redis(self) -> None:
        self.redis = None
        self.queues = {}
        self._next_redis_attempt = 0.0

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
        if resource_repository.enabled:
            return bool(owner_id and resource_repository.get_dataset(owner_id, dataset_path.name))
        dataset_owner = self._dataset_owner(dataset_path)
        return bool(owner_id and dataset_owner and dataset_owner == owner_id)

    def _assert_dataset_visible(self, dataset_path: Path, owner_id: str | None = None) -> None:
        if not self._is_dataset_visible(dataset_path, owner_id):
            raise FileNotFoundError(f"Dataset '{dataset_path.name}' was not found in {self.dataset_dir}.")

    def _find_dataset_path(self, dataset_name: str | None = None, owner_id: str | None = None) -> Path:
        if dataset_name:
            validate_slug(dataset_name, "dataset name")
            record = resource_repository.get_dataset(owner_id, dataset_name) if owner_id and resource_repository.enabled else None
            dataset_path = registered_storage_path(self.dataset_dir, record["storage_path"]) if record else contained_path(self.dataset_dir, dataset_name)
            if not dataset_path.exists() or not dataset_path.is_dir():
                raise FileNotFoundError(f"Dataset '{dataset_name}' was not found in {self.dataset_dir}.")
            self._assert_dataset_visible(dataset_path, owner_id)
            return dataset_path

        if resource_repository.enabled and owner_id:
            candidates = [registered_storage_path(self.dataset_dir, row["storage_path"]) for row in resource_repository.list_datasets(owner_id)]
        else:
            candidates = [path for path in self.dataset_dir.iterdir() if path.is_dir() and not path.name.startswith(".") and self._is_dataset_visible(path, owner_id)]
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
        detected_formats = set(metadata["formats"])
        model_entry = get_model(model_type)
        extra_args = extra_args or {}

        if model_entry and task_type != model_entry["task_type"]:
            raise ValueError(f"Model '{model_type}' belongs to task '{model_entry['task_type']}', not '{task_type}'.")
        if model_entry:
            compatibility = compatible_models_for_metadata(
                metadata,
                {"tasks": [{"id": model_entry["task_type"], "models": [model_entry]}]},
            )[0]
            if not compatibility.get("ready"):
                raise ValueError(
                    f"Dataset '{dataset_path.name}' is not compatible with {model_type}. "
                    f"{compatibility.get('reason', 'No compatibility rule matched.')} "
                    f"Detected formats: {sorted(metadata['formats']) or ['none']}."
                )
        if model_entry and task_type not in metadata["tasks"]:
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
        if model_type == "mask_rcnn":
            coco = metadata.get("coco", {})
            if (
                coco.get("mask_annotations", 0) <= 0
                or coco.get("invalid_segmentations", 0)
                or coco.get("missing_segmentations", 0)
                or coco.get("errors")
            ):
                raise ValueError(
                    f"Dataset '{dataset_path.name}' does not have a valid instance mask for every COCO box."
                )
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
        if resource_repository.enabled and owner_id:
            candidates = [registered_storage_path(self.dataset_dir, row["storage_path"]) for row in resource_repository.list_datasets(owner_id)]
        else:
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
        config = read_yaml_limited(original_yaml_path) or {}
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
        resource_plan: dict[str, Any] | None = None,
        owner_id: str | None = None,
        owner_email: str | None = None,
    ) -> str:
        self._ensure_redis()
        if not owner_id:
            raise PermissionError("Authenticated user identity is required to start training.")
        validate_slug(project_name, "project name")
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        project_dir = contained_path(self.runs_dir, project_name)
        if project_dir.exists():
            raise FileExistsError(f"Run '{project_name}' already exists.")

        selected_name = dataset_name
        if not selected_name:
            selected_name = self._find_latest_compatible_dataset_path(
                model_type, task_type, owner_id=owner_id, extra_args=extra_args or {}
            ).name
        planned_job_id = uuid.uuid4().hex
        reserved_project_dir = False
        run_record: dict[str, Any] | None = None
        try:
            with resource_repository.dataset_guard(owner_id, selected_name) as connection:
                dataset_record = resource_repository.get_dataset(owner_id, selected_name, connection)
                dataset_path = self._find_dataset_path(selected_name, owner_id=owner_id)
                if resource_repository.enabled and not dataset_record:
                    raise FileNotFoundError(f"Dataset '{selected_name}' is not registered. Run the resource backfill first.")
                dataset_record = dataset_record or {"id": selected_name, "slug": selected_name}
                self._assert_dataset_matches_model(dataset_path, model_type, task_type, extra_args=extra_args or {})
                prepared_dataset = prepare_dataset_for_model(dataset_path, model_type, extra_args=extra_args or {})
                dataset_metadata = prepared_dataset["metadata"]
                worker_dataset_path = Path(prepared_dataset["dataset_path"])
                data_yaml_path = prepared_dataset.get("data_yaml_path")
                job_config = {
                    "job_id": planned_job_id,
                    "task_type": task_type, "model_type": model_type, "model_name": model_name,
                    "epochs": epochs, "batch_size": batch_size, "project_name": project_name,
                    "dataset_name": dataset_path.name, "dataset_path": str(worker_dataset_path),
                    "source_dataset_path": str(dataset_path), "data_yaml_path": data_yaml_path,
                    "dataset_metadata": dataset_metadata, "dataset_export": prepared_dataset.get("export"),
                    "extra_args": extra_args or {}, "resource_plan": resource_plan or {},
                    "created_by": owner_id, "created_by_email": owner_email,
                }
                if model_type == "yolo":
                    job_config["task"] = "detect"
                queue_name = self._queue_name_for_model(model_type)
                queue = self.queues[queue_name]
                run_record = resource_repository.create_run(
                    owner_id=owner_id, owner_email=owner_email, dataset=dataset_record, project_name=project_name,
                    task_type=task_type, model_type=model_type, model_name=model_name, params=extra_args or {},
                    storage_path=project_dir, job_id=planned_job_id, connection=connection,
                )
                project_dir.mkdir(parents=False, exist_ok=False)
                reserved_project_dir = True
                job_config["run_id"] = str(run_record.get("id"))
                job_config["dataset_id"] = str(dataset_record.get("id"))
                (project_dir / "job_config.json").write_text(json.dumps(job_config, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception:
            if reserved_project_dir:
                shutil.rmtree(project_dir, ignore_errors=True)
            raise

        job_meta = {
            "project_name": project_name,
            "model_type": model_type,
            "task_type": task_type,
            "queue": queue_name,
            "created_by": owner_id,
        }
        if owner_email:
            job_meta["created_by_email"] = owner_email
        try:
            job = queue.enqueue(
                "worker_app.run_training",
                job_config,
                job_id=planned_job_id,
                meta=job_meta,
                job_timeout="24h",
                result_ttl=86400,
                failure_ttl=86400,
            )
        except Exception as exc:
            resource_repository.update_run_status_by_id(run_record.get("id"), "failed", f"Redis enqueue failed: {exc}")
            raise RuntimeError(f"Could not enqueue training job: {exc}") from exc

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
        if not self._connect_redis():
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
            mapped = status_map.get(status_str, f"unknown ({status_str})")
            persisted = {"running": "running", "exited": "completed", "stopped": "stopped", "failed": "failed", "queued": "queued"}.get(mapped)
            if persisted:
                resource_repository.update_run_status(job_id, persisted, job.exc_info if persisted == "failed" else None)
            return mapped
        except NoSuchJobError:
            return "not_found"
        except Exception as exc:
            print(f"[TrainingService] Redis status lookup failed: {exc}")
            self._invalidate_redis()
            return "redis_connection_error"

    def get_container_logs(self, job_id: str) -> str:
        if not self.redis:
            return "Redis connection error: Is Redis running?"
        try:
            job = Job.fetch(job_id, connection=self.redis)
            log_path_str = job.meta.get("log_path")
            if log_path_str:
                log_path = contained_path(self.runs_dir, Path(log_path_str))
                if log_path.exists():
                    size = log_path.stat().st_size
                    with open(log_path, "rb") as file:
                        file.seek(max(0, size - MAX_LOG_RESPONSE_BYTES))
                        text = file.read(MAX_LOG_RESPONSE_BYTES).decode("utf-8", errors="replace")
                    return ("[Earlier log output omitted]\n" if size > MAX_LOG_RESPONSE_BYTES else "") + text

            status = job.get_status()
            status_str = status.value if isinstance(status, JobStatus) else str(status)
            if status_str == "failed" and job.exc_info:
                return f"[Job Failed]\n{job.exc_info}"
            return "No logs available yet. Training may not have started."
        except Exception as exc:
            return f"Error fetching logs: {exc}"

    def get_job_owner(self, job_id: str) -> str | None:
        if resource_repository.enabled:
            return resource_repository.get_run_owner_by_job(job_id)
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

    def read_job_log_chunk(self, job_id: str, offset: int = 0, max_bytes: int = 256 * 1024) -> tuple[str, int, bool]:
        if not self.redis:
            return "", offset, False
        try:
            log_path = self._job_log_path(job_id)
            if log_path is None:
                return "", offset, False
            size = log_path.stat().st_size
            replace = size < offset
            start = 0 if replace else offset
            if offset < 0:
                start = max(0, size - max_bytes)
                replace = True
            with open(log_path, "rb") as file:
                file.seek(start)
                data = file.read(max_bytes)
                return data.decode("utf-8", errors="replace"), start + len(data), replace
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
            "logs": "",
            "log_offset": 0,
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
                resource_repository.update_run_status(job_id, "stopping")
                print(f"[TrainingService] Stop command sent for running job {job_id}")
                return "stopping"
            job.cancel()
            resource_repository.update_run_status(job_id, "cancelled")
            print(f"[TrainingService] Job {job_id} cancelled")
            return "cancelled"
        except Exception as exc:
            raise RuntimeError(f"Could not stop job {job_id}: {exc}")

    def list_runs(self, owner_id: str | None = None) -> list[dict[str, Any]]:
        if not owner_id:
            return []
        runs_by_slug: dict[str, dict[str, Any]] = {}
        registered = {row["run_slug"]: row for row in resource_repository.list_runs(owner_id)} if resource_repository.enabled else None
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
            record = registered.get(project_dir.name) if registered is not None else None
            run_owner = record.get("owner_user_id") if record else config.get("created_by")
            if registered is not None and record is None:
                continue
            if run_owner != owner_id:
                continue

            files = []
            files_truncated = False
            for path in project_dir.rglob("*"):
                if path.is_file():
                    if len(files) >= MAX_RUN_FILES_RETURNED:
                        files_truncated = True
                        break
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

            runs_by_slug[project_dir.name] = {
                    "id": str(record.get("id")) if record else project_dir.name,
                    "project_name": project_dir.name,
                    "createdAt": project_dir.stat().st_ctime,
                    "updatedAt": project_dir.stat().st_mtime,
                    "task_type": config.get("task_type"),
                    "model_type": config.get("model_type"),
                    "model_name": config.get("model_name"),
                    "dataset_name": record.get("dataset_slug") if record else config.get("dataset_name"),
                    "created_by": run_owner,
                    "epochs": config.get("epochs"),
                    "files": files,
                    "files_truncated": files_truncated,
                    "latest_metrics": latest_metrics,
                    "status": record.get("status") if record else None,
                    "job_id": record.get("rq_job_id") if record else None,
                }
        if registered is not None:
            for run_slug, record in registered.items():
                if run_slug in runs_by_slug:
                    continue
                runs_by_slug[run_slug] = {
                    "id": str(record.get("id")),
                    "project_name": run_slug,
                    "createdAt": record.get("created_at"),
                    "updatedAt": record.get("updated_at"),
                    "task_type": record.get("task_type"),
                    "model_type": record.get("model_type"),
                    "model_name": record.get("model_name"),
                    "dataset_name": record.get("dataset_slug"),
                    "created_by": record.get("owner_user_id"),
                    "epochs": (record.get("params") or {}).get("epochs"),
                    "files": [],
                    "files_truncated": False,
                    "latest_metrics": None,
                    "status": record.get("status"),
                    "job_id": record.get("rq_job_id"),
                }
        return sorted(runs_by_slug.values(), key=lambda run: str(run.get("updatedAt") or ""), reverse=True)
