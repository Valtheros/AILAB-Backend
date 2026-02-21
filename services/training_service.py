"""
training_service.py — Backend Service Layer

แทนที่จะรัน Docker container ใหม่ทุกครั้ง เราส่ง job ไปยัง long-running
Worker Container ผ่าน Redis Queue แทน — dependencies โหลดไว้แล้วใน worker image
"""

import csv
import os
from pathlib import Path
from typing import Optional

import yaml
from redis import Redis
from rq import Queue
from rq.job import Job, JobStatus


class TrainingService:
    def __init__(self):
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        try:
            self.redis = Redis.from_url(redis_url)
            self.redis.ping()  # ตรวจสอบ connection
            self.queue = Queue("cv_training", connection=self.redis)
            print(f"[TrainingService] Connected to Redis at {redis_url}")
        except Exception as e:
            print(f"[TrainingService] ERROR: Cannot connect to Redis: {e}")
            self.redis = None
            self.queue = None

        # Path ภายใน container (ต้องตรงกับ volumes ใน docker-compose.yml)
        self.base_dir      = Path("/app")
        self.dataset_dir   = self.base_dir / "dataset"
        self.runs_dir      = self.base_dir / "runs"
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def _ensure_redis(self):
        if self.redis is None or self.queue is None:
            raise RuntimeError(
                "Redis is not connected. "
                "Make sure the Redis service is running (docker compose up redis)."
            )

    # ── Dataset Helpers ───────────────────────────────────────────────────────

    def _find_dataset_path(self):
        """ค้นหา directory ที่มี data.yaml อยู่ใน dataset folder"""
        for root, dirs, files in os.walk(self.dataset_dir):
            if "data.yaml" in files:
                return Path(root), Path(root) / "data.yaml"
        return None, None

    def _create_worker_yaml(self, original_yaml_path: Path, dataset_folder: Path) -> str:
        """
        สร้าง data_worker.yaml ที่ปรับ paths ให้ตรงกับ mount point ใน container
        dataset_folder คือ path จริงของ dataset เช่น /app/dataset/parking_lot.v1i.yolov11
        ซึ่งใช้ได้ทั้งบน backend และ worker container เพราะ mount volume เดียวกัน
        """
        with open(original_yaml_path, "r") as f:
            config = yaml.safe_load(f)

        # ── ใช้ dataset_folder path โดยตรง ───────────────────────────────────
        # dataset_folder = /app/dataset/parking_lot.v1i.yolov11  (ใน container)
        # ทั้ง backend และ worker mount volume เดียวกัน → path เดียวกัน
        config["path"] = str(dataset_folder)

        # ── ปรับ train/val/test ให้ชี้ถูก folder ────────────────────────────
        path_map = {
            "train": ["train"],
            "val":   ["valid", "val"],
            "test":  ["test"],
        }
        for yaml_key, folder_candidates in path_map.items():
            if yaml_key in config:
                for candidate in folder_candidates:
                    if (dataset_folder / candidate).exists():
                        config[yaml_key] = f"{candidate}/images"
                        break

        worker_yaml_path = dataset_folder / "data_worker.yaml"
        with open(worker_yaml_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        return str(worker_yaml_path)

    # ── Public API (ใช้แทน start_training_container เดิม) ────────────────────

    def start_training_container(
        self,
        model_name: str,
        epochs: int,
        batch_size: int,
        project_name: str,
        extra_args: Optional[dict] = None,
        model_type: str = "yolo",   # เพิ่ม parameter นี้สำหรับ model อื่นนอกจาก YOLO
    ) -> str:
        """
        ส่ง training job ไปยัง Redis Queue
        Returns job ID (ใช้แทน container_id เดิม — API ยังคงเหมือนเดิม)
        """
        self._ensure_redis()

        dataset_folder, original_yaml_path = self._find_dataset_path()
        if not dataset_folder:
            raise FileNotFoundError(
                f"Cannot find 'data.yaml' in {self.dataset_dir}. "
                "Please upload a dataset first."
            )

        yaml_path = self._create_worker_yaml(original_yaml_path, dataset_folder)

        job_config = {
            "model_type":      model_type,
            "model_name":      model_name,
            "epochs":          epochs,
            "batch_size":      batch_size,
            "project_name":    project_name,
            "data_yaml_path":  yaml_path,
            "extra_args":      extra_args or {},
        }

        job = self.queue.enqueue(
            "worker_app.run_training",  # function ใน worker container
            job_config,
            job_timeout="24h",          # training อาจนานหลายชั่วโมง
            result_ttl=86400,           # เก็บผลลัพธ์ไว้ 24 ชั่วโมง
            failure_ttl=86400,
        )

        print(f"[TrainingService] Job enqueued: {job.id} ({model_type}/{model_name})")
        return job.id

    def get_container_status(self, job_id: str) -> str:
        """ดูสถานะ job — map RQ status → สถานะที่ frontend เข้าใจ"""
        if not self.redis:
            return "redis_connection_error"
        try:
            job = Job.fetch(job_id, connection=self.redis)
            status = job.get_status()

            # รองรับทั้ง RQ เวอร์ชันเก่า (string) และใหม่ (enum)
            status_str = status.value if isinstance(status, JobStatus) else str(status)

            status_map = {
                "queued":    "queued",
                "started":   "running",
                "finished":  "exited",      # สำเร็จ
                "failed":    "failed",
                "stopped":   "stopped",
                "canceled":  "stopped",
                "deferred":  "queued",
                "scheduled": "queued",
            }
            return status_map.get(status_str, f"unknown ({status_str})")
        except Exception:
            return "not_found"

    def get_container_logs(self, job_id: str) -> str:
        """
        ดึง log ของ training job
        อ่านจาก log file ที่ worker เขียนไว้ใน /app/runs/<project_name>/train.log
        """
        if not self.redis:
            return "Redis connection error: Is Redis running?"
        try:
            job = Job.fetch(job_id, connection=self.redis)
            log_path_str = job.meta.get("log_path")

            if log_path_str:
                log_path = Path(log_path_str)
                if log_path.exists():
                    return log_path.read_text(encoding="utf-8")

            # ถ้า job failed ดึง stack trace มาแสดงด้วย
            status = job.get_status()
            status_str = status.value if isinstance(status, JobStatus) else str(status)
            if status_str == "failed" and job.exc_info:
                return f"[Job Failed]\n{job.exc_info}"

            return "No logs available yet. Training may not have started."
        except Exception as e:
            return f"Error fetching logs: {e}"

    def get_training_metrics(self, project_name: str) -> list:
        """อ่าน results.csv ที่ YOLO/trainer เขียนไว้"""
        results_path = self.runs_dir / project_name / "results.csv"
        if not results_path.exists():
            return []
        try:
            with open(results_path, "r") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            if not rows:
                return []
            # YOLO CSV มักมี whitespace ใน headers
            return [{k.strip(): v.strip() for k, v in row.items()} for row in rows]
        except Exception as e:
            print(f"[TrainingService] Error reading metrics: {e}")
            return []

    def stop_training_container(self, job_id: str) -> str:
        """Stop (cancel) a queued or running job"""
        self._ensure_redis()
        try:
            job = Job.fetch(job_id, connection=self.redis)
            job.cancel()
            print(f"[TrainingService] Job {job_id} cancelled")
            return "cancelled"
        except Exception as e:
            raise RuntimeError(f"Could not stop job {job_id}: {e}")
