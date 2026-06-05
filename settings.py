from __future__ import annotations

import os
from pathlib import Path


def _default_base_dir() -> Path:
    docker_app = Path("/app")
    return docker_app if docker_app.exists() else Path(os.getcwd()).absolute()


BASE_DIR = Path(os.getenv("APP_BASE_DIR", str(_default_base_dir()))).resolve()
STORAGE_DIR = Path(os.getenv("STORAGE_DIR", str(BASE_DIR / "storage"))).resolve()
DATASET_DIR = Path(os.getenv("DATASET_DIR", str(BASE_DIR / "dataset"))).resolve()
RUNS_DIR = Path(os.getenv("RUNS_DIR", str(BASE_DIR / "runs"))).resolve()

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
DATABASE_URL = os.getenv("DATABASE_URL", "")
BACKEND_INTERNAL_TOKEN = os.getenv("BACKEND_INTERNAL_TOKEN", "")
CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "*").split(",")
    if origin.strip()
]


def ensure_runtime_dirs() -> None:
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
