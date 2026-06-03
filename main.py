from __future__ import annotations

import asyncio
import json
import os
import shutil
import time
import zipfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from dataset_utils import (
    find_dataset_yaml,
    format_bytes,
    inspect_dataset,
    safe_dataset_name,
    validate_dataset_for_upload,
)
from model_catalog import get_catalog, get_model, validate_model_params
from security_utils import (
    contained_path,
    replace_directory,
    safe_extract_zip,
    save_upload_to_temp,
    staging_directory,
    validate_slug,
)
from services.training_service import TrainingService
from sse_utils import TERMINAL_STATUSES, heartbeat, sse_event


app = FastAPI(title="No-Code Computer Vision Training Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

training_service = TrainingService()

BASE_DIR = Path("/app") if Path("/app").exists() else Path(os.getcwd()).absolute()
DATASET_DIR = BASE_DIR / "dataset"
RUNS_DIR = BASE_DIR / "runs"
DATASET_DIR.mkdir(exist_ok=True, parents=True)
RUNS_DIR.mkdir(exist_ok=True, parents=True)


class TrainRequest(BaseModel):
    task_type: str | None = None
    model_type: str = "yolo"
    model_name: str | None = None
    dataset_name: str | None = None
    project_name: str = "train_run"
    epochs: int = Field(default=50, ge=1, le=2000)
    batch_size: int = Field(default=16, ge=1, le=256)
    params: dict[str, Any] = Field(default_factory=dict)

    # Legacy YOLO payload support.
    model_size: str | None = None

    class Config:
        extra = "forbid"


def _model_name_from_request(request: TrainRequest, model_entry: dict[str, Any] | None) -> str:
    if request.model_type == "yolo":
        model_size = request.params.get("model_size") or request.model_size or "n"
        model_size = str(model_size).removesuffix(".pt")
        if model_size.startswith("yolo11"):
            model_size = model_size.removeprefix("yolo11")
        if model_size not in {"n", "s", "m", "l", "x"}:
            raise HTTPException(status_code=400, detail="YOLO model_size must be one of: n, s, m, l, x")
        model_name = f"yolo11{model_size}"
        if request.model_name and request.model_name.removesuffix(".pt") != model_name:
            raise HTTPException(status_code=400, detail=f"YOLO model_name must match selected model_size: {model_name}")
        return model_name

    if model_entry:
        if request.model_name and request.model_name.removesuffix(".pt") != str(model_entry["model_name"]):
            architecture = request.params.get("architecture")
            if architecture != request.model_name.removesuffix(".pt"):
                raise HTTPException(status_code=400, detail=f"Unsupported model_name for {request.model_type}")
            return str(architecture)
        return str(model_entry["model_name"])

    if request.model_size:
        return request.model_size.removesuffix(".pt")

    raise HTTPException(status_code=400, detail="model_name is required for this model_type")


def _extra_args_from_request(request: TrainRequest) -> dict[str, Any]:
    extra_args = dict(request.params or {})
    if request.model_type == "yolo" and "model_size" not in extra_args:
        extra_args["model_size"] = request.model_size or "n"
    return validate_model_params(request.model_type, extra_args)


def _safe_extract(zip_file: zipfile.ZipFile, target_dir: Path) -> None:
    try:
        safe_extract_zip(zip_file, target_dir)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _flatten_single_root_folder(target_dir: Path) -> None:
    contents = list(target_dir.iterdir())
    if len(contents) != 1 or not contents[0].is_dir():
        return
    single_dir = contents[0]
    for item in single_dir.iterdir():
        shutil.move(str(item), str(target_dir / item.name))
    single_dir.rmdir()


@app.get("/")
def read_root():
    return {
        "message": "Computer Vision Training Backend is running",
        "catalog_url": "/api/model-catalog",
    }


@app.get("/api/model-catalog")
def model_catalog():
    return get_catalog()


@app.post("/api/train")
def start_train(request: TrainRequest):
    model_entry = get_model(request.model_type)
    if model_entry is None:
        raise HTTPException(status_code=400, detail=f"Unsupported model_type: {request.model_type}")

    task_type = request.task_type or model_entry["task_type"]
    if task_type != model_entry["task_type"]:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{request.model_type}' belongs to task '{model_entry['task_type']}', not '{task_type}'.",
        )
    try:
        validate_slug(request.project_name, "project name")
        if request.dataset_name:
            validate_slug(request.dataset_name, "dataset name")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    for key, value in {"epochs": request.epochs, "batch_size": request.batch_size}.items():
        if key in request.params and request.params[key] != value:
            raise HTTPException(status_code=400, detail=f"Param '{key}' must match top-level '{key}'.")
    model_name = _model_name_from_request(request, model_entry)
    try:
        extra_args = _extra_args_from_request(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        job_id = training_service.start_training_container(
            task_type=task_type,
            model_type=request.model_type,
            model_name=model_name,
            epochs=request.epochs,
            batch_size=request.batch_size,
            project_name=request.project_name,
            dataset_name=request.dataset_name,
            extra_args=extra_args,
        )
        return {"status": "success", "job_id": job_id, "container_id": job_id}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/status/{job_id}")
def get_status(job_id: str):
    status = training_service.get_container_status(job_id)
    return {"job_id": job_id, "container_id": job_id, "status": status}


@app.get("/api/logs/{job_id}")
def get_logs(job_id: str):
    logs = training_service.get_container_logs(job_id)
    return {"job_id": job_id, "container_id": job_id, "logs": logs}


@app.get("/api/metrics/{project_name}")
def get_metrics(project_name: str):
    try:
        metrics = training_service.get_training_metrics(project_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not metrics:
        return {"status": "no_data", "metrics": []}
    return {"status": "success", "metrics": metrics}


@app.post("/api/stop/{job_id}")
def stop_train(job_id: str):
    try:
        training_service.stop_training_container(job_id)
        return {"status": "success"}
    except Exception as exc:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/jobs/{job_id}/events")
async def job_events(job_id: str):
    snapshot = training_service.get_job_snapshot(job_id)
    if snapshot["status"] == "not_found":
        raise HTTPException(status_code=404, detail="Job not found")

    async def stream():
        log_offset = int(snapshot.get("log_offset", 0))
        previous_metrics = json.dumps(snapshot["metrics"], sort_keys=True)
        previous_status = snapshot["status"]
        yield sse_event("snapshot", snapshot)
        if previous_status in TERMINAL_STATUSES:
            yield sse_event("end", {"status": previous_status})
            return

        heartbeat_ticks = 0
        while True:
            await asyncio.sleep(1)
            current = training_service.get_job_snapshot(job_id)
            current_status = current["status"]
            current_metrics = json.dumps(current["metrics"], sort_keys=True)
            sent_event = False

            chunk, log_offset, replace = training_service.read_job_log_chunk(job_id, log_offset)
            if chunk:
                yield sse_event("log", {"text": chunk, "replace": replace})
                sent_event = True
            if current_metrics != previous_metrics:
                yield sse_event("metrics", {"metrics": current["metrics"]})
                previous_metrics = current_metrics
                sent_event = True
            if current_status != previous_status:
                yield sse_event("status", {"status": current_status})
                previous_status = current_status
                sent_event = True
            if current_status in TERMINAL_STATUSES or current_status == "not_found":
                yield sse_event("end", {"status": current_status})
                return

            heartbeat_ticks = 0 if sent_event else heartbeat_ticks + 1
            if heartbeat_ticks >= 15:
                yield heartbeat()
                heartbeat_ticks = 0

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/datasets")
def list_datasets():
    datasets = []
    if not DATASET_DIR.exists():
        return {"datasets": []}

    for item in sorted(DATASET_DIR.iterdir(), key=lambda path: path.name.lower()):
        if not item.is_dir() or item.name.startswith("."):
            continue

        metadata = inspect_dataset(item)
        created = time.strftime("%Y-%m-%d", time.localtime(item.stat().st_ctime))
        datasets.append(
            {
                "id": item.name,
                "name": item.name,
                "images": metadata["image_count"],
                "classes": metadata["classes"],
                "createdAt": created,
                "size": format_bytes(metadata["size_bytes"]),
                "tasks": metadata["tasks"],
                "formats": metadata["formats"],
                "warnings": metadata["warnings"],
                "yamlPath": metadata["yaml_path"],
            }
        )

    return {"datasets": datasets}


@app.post("/api/upload-dataset")
async def upload_dataset(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    if not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Only .zip files are supported")

    dataset_name = safe_dataset_name(file.filename)
    try:
        validate_slug(dataset_name, "dataset name")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    target_dir = contained_path(DATASET_DIR, dataset_name)
    staging_dir = staging_directory(DATASET_DIR)
    temp_zip_path: Path | None = None

    try:
        temp_zip_path = await save_upload_to_temp(file, DATASET_DIR)
        with zipfile.ZipFile(temp_zip_path, "r") as zip_ref:
            extract_dir = staging_dir / "extracted"
            _safe_extract(zip_ref, extract_dir)

        _flatten_single_root_folder(extract_dir)
        metadata = validate_dataset_for_upload(extract_dir)
        replace_directory(extract_dir, target_dir)

        return {
            "status": "success",
            "dataset_name": dataset_name,
            "tasks": metadata["tasks"],
            "formats": metadata["formats"],
            "classes": metadata["classes"],
        }
    except HTTPException:
        raise
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="Invalid ZIP file")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)
        if temp_zip_path and temp_zip_path.exists():
            temp_zip_path.unlink()


@app.delete("/api/datasets/{dataset_name}")
def delete_dataset(dataset_name: str):
    try:
        validate_slug(dataset_name, "dataset name")
        target_dir = contained_path(DATASET_DIR, dataset_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not target_dir.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")
    try:
        shutil.rmtree(target_dir)
        return {"status": "success"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/runs")
def list_runs():
    return {"runs": training_service.list_runs()}


@app.get("/api/runs/{project_name}/files/{file_path:path}")
def download_run_file(project_name: str, file_path: str):
    try:
        validate_slug(project_name, "project name")
        project_dir = contained_path(RUNS_DIR, project_name)
        target = contained_path(project_dir, file_path)
    except ValueError:
        raise HTTPException(status_code=404, detail="File not found")
    if not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(target, filename=target.name)


@app.get("/api/datasets/{dataset_name}/metadata")
def dataset_metadata(dataset_name: str):
    try:
        validate_slug(dataset_name, "dataset name")
        target_dir = contained_path(DATASET_DIR, dataset_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not target_dir.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")
    metadata = inspect_dataset(target_dir)
    metadata["yaml_path"] = str(find_dataset_yaml(target_dir) or "")
    return metadata
