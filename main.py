from __future__ import annotations

import asyncio
import json
import shutil
import time
import zipfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from dataset_utils import (
    dataset_workflow_metadata,
    export_cache_metadata,
    find_dataset_yaml,
    format_bytes,
    inspect_dataset,
    safe_dataset_name,
    validate_dataset_for_upload,
)
from model_catalog import get_catalog, get_model, validate_model_params
from resource_guard import ResourcePlanError, enforce_resource_plan, get_resource_profile
from security_utils import (
    contained_path,
    named_file_lock,
    replace_directory,
    safe_extract_zip,
    save_upload_to_temp,
    staging_directory,
    validate_slug,
)
from services.training_service import TrainingService
from settings import BACKEND_INTERNAL_TOKEN, CORS_ORIGINS, DATASET_DIR, RUNS_DIR, ensure_runtime_dirs
from sse_utils import TERMINAL_STATUSES, heartbeat, sse_event


app = FastAPI(title="No-Code Computer Vision Training Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


PUBLIC_API_PATHS = {"/api/model-catalog", "/api/resource-profile"}


@app.middleware("http")
async def require_internal_token(request: Request, call_next):
    path = request.url.path
    if path.startswith("/api") and path not in PUBLIC_API_PATHS:
        if not BACKEND_INTERNAL_TOKEN:
            return JSONResponse({"detail": "BACKEND_INTERNAL_TOKEN must be set for protected backend API routes"}, status_code=503)
        provided = request.headers.get("x-internal-token")
        if provided != BACKEND_INTERNAL_TOKEN:
            return JSONResponse({"detail": "Unauthorized backend request"}, status_code=401)
    return await call_next(request)

ensure_runtime_dirs()
training_service = TrainingService()


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


def _request_user_id(request: Request) -> str | None:
    value = request.headers.get("x-user-id")
    return value.strip() if value and value.strip() else None


def _request_user_email(request: Request) -> str | None:
    value = request.headers.get("x-user-email")
    return value.strip() if value and value.strip() else None


def _require_request_user_id(request: Request) -> str:
    user_id = _request_user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authenticated user identity is required")
    return user_id


def _read_json_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _dataset_meta_path(dataset_dir: Path) -> Path:
    return dataset_dir / ".ailab_dataset.json"


def _dataset_owner(dataset_dir: Path) -> str | None:
    owner = _read_json_file(_dataset_meta_path(dataset_dir)).get("created_by")
    return str(owner) if owner else None


def _write_dataset_metadata(dataset_dir: Path, request: Request, workflow: dict[str, Any] | None = None) -> None:
    metadata = _read_json_file(_dataset_meta_path(dataset_dir))
    owner_id = _require_request_user_id(request)
    metadata.update(
        {
            "created_by": owner_id,
            "created_by_email": _request_user_email(request),
            "created_at": metadata.get("created_at") or int(time.time()),
        }
    )
    if workflow:
        metadata.update({"workflow": workflow})
    _dataset_meta_path(dataset_dir).write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _assert_owned_resource_visible(owner_id: str | None, request: Request) -> None:
    request_owner = _require_request_user_id(request)
    if not owner_id or owner_id != request_owner:
        raise HTTPException(status_code=404, detail="Resource not found")


def _run_owner(project_name: str) -> str | None:
    config_path = contained_path(RUNS_DIR, project_name, "job_config.json")
    owner = _read_json_file(config_path).get("created_by")
    return str(owner) if owner else None


def _assert_run_visible(project_name: str, request: Request) -> None:
    _assert_owned_resource_visible(_run_owner(project_name), request)


def _assert_job_visible(job_id: str, request: Request) -> None:
    _assert_owned_resource_visible(training_service.get_job_owner(job_id), request)


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


def _dataset_profile(metadata: dict[str, Any]) -> dict[str, Any]:
    workflow = dataset_workflow_metadata(metadata, get_catalog())
    ready_models = [model for model in workflow.get("compatible_models", []) if model.get("ready")]
    return {
        "tasks": metadata["tasks"],
        "formats": metadata["formats"],
        "classes": metadata["classes"],
        "image_count": metadata["image_count"],
        "size_bytes": metadata["size_bytes"],
        "warnings": metadata.get("warnings", []),
        "errors": metadata.get("errors", []),
        "paddleocr_tasks": metadata.get("paddleocr_tasks", []),
        "source_format": workflow["source_format"],
        "dataset_task": workflow["dataset_task"],
        "dataset_tasks": workflow["dataset_tasks"],
        "canonical_task": workflow["canonical_task"],
        "canonical_format": workflow["canonical_format"],
        "normalized_formats": workflow["normalized_formats"],
        "annotation_stats": workflow["annotation_stats"],
        "conversion_warnings": workflow.get("conversion_warnings", []),
        "compatible_models": workflow.get("compatible_models", []),
        "ready_models": ready_models,
        "export_cache": workflow.get("export_cache", []),
    }


def _dataset_response(dataset_name: str, metadata: dict[str, Any]) -> dict[str, Any]:
    profile = _dataset_profile(metadata)
    return {
        "status": "success",
        "dataset_name": dataset_name,
        "tasks": profile["tasks"],
        "formats": profile["formats"],
        "classes": profile["classes"],
        "paddleocrTasks": profile["paddleocr_tasks"],
        "sourceFormat": profile["source_format"],
        "datasetTask": profile["dataset_task"],
        "datasetTasks": profile["dataset_tasks"],
        "canonicalTask": profile["canonical_task"],
        "canonicalFormat": profile["canonical_format"],
        "normalizedFormats": profile["normalized_formats"],
        "annotationStats": profile["annotation_stats"],
        "conversionWarnings": profile["conversion_warnings"],
        "compatibleModels": profile["compatible_models"],
        "readyModels": profile["ready_models"],
        "exportCache": profile["export_cache"],
    }


async def _inspect_uploaded_zip(file: UploadFile) -> tuple[str, dict[str, Any]]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    if not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Only .zip files are supported")

    dataset_name = safe_dataset_name(file.filename)
    staging_dir = staging_directory(DATASET_DIR)
    temp_zip_path: Path | None = None
    try:
        temp_zip_path = await save_upload_to_temp(file, DATASET_DIR)
        with zipfile.ZipFile(temp_zip_path, "r") as zip_ref:
            extract_dir = staging_dir / "extracted"
            _safe_extract(zip_ref, extract_dir)
        _flatten_single_root_folder(extract_dir)
        metadata = inspect_dataset(extract_dir)
        validate_dataset_for_upload(extract_dir)
        return dataset_name, metadata
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="Invalid ZIP file")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)
        if temp_zip_path and temp_zip_path.exists():
            temp_zip_path.unlink()


async def _import_uploaded_zip(request: Request, file: UploadFile) -> dict[str, Any]:
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
        with named_file_lock(DATASET_DIR, dataset_name, "dataset name"):
            target_owner = _dataset_owner(target_dir) if target_dir.exists() else None
            request_owner = _require_request_user_id(request)
            if target_dir.exists() and target_owner != request_owner:
                raise HTTPException(
                    status_code=409,
                    detail="Dataset name already exists. Rename the ZIP and upload again.",
                )

            temp_zip_path = await save_upload_to_temp(file, DATASET_DIR)
            with zipfile.ZipFile(temp_zip_path, "r") as zip_ref:
                extract_dir = staging_dir / "extracted"
                _safe_extract(zip_ref, extract_dir)

            _flatten_single_root_folder(extract_dir)
            metadata = inspect_dataset(extract_dir)
            validate_dataset_for_upload(extract_dir)
            replace_directory(extract_dir, target_dir)
            workflow = dataset_workflow_metadata(metadata, get_catalog())
            _write_dataset_metadata(target_dir, request, workflow)
            return _dataset_response(dataset_name, metadata)
    except HTTPException:
        raise
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="Invalid ZIP file")
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)
        if temp_zip_path and temp_zip_path.exists():
            temp_zip_path.unlink()


@app.get("/")
def read_root():
    return {
        "message": "Computer Vision Training Backend is running",
        "catalog_url": "/api/model-catalog",
    }


@app.get("/api/model-catalog")
def model_catalog():
    return get_catalog()


@app.get("/api/resource-profile")
def resource_profile():
    return get_resource_profile()


@app.post("/api/train")
def start_train(train_request: TrainRequest, request: Request):
    model_entry = get_model(train_request.model_type)
    if model_entry is None:
        raise HTTPException(status_code=400, detail=f"Unsupported model_type: {train_request.model_type}")

    task_type = train_request.task_type or model_entry["task_type"]
    if task_type != model_entry["task_type"]:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{train_request.model_type}' belongs to task '{model_entry['task_type']}', not '{task_type}'.",
        )
    try:
        validate_slug(train_request.project_name, "project name")
        if train_request.dataset_name:
            validate_slug(train_request.dataset_name, "dataset name")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    for key, value in {"epochs": train_request.epochs, "batch_size": train_request.batch_size}.items():
        if key in train_request.params and train_request.params[key] != value:
            raise HTTPException(status_code=400, detail=f"Param '{key}' must match top-level '{key}'.")
    model_name = _model_name_from_request(train_request, model_entry)
    try:
        extra_args = _extra_args_from_request(train_request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        resource_plan = enforce_resource_plan(
            train_request.model_type,
            params=extra_args,
            batch_size=train_request.batch_size,
        )
        extra_args = resource_plan["normalized_params"]
    except ResourcePlanError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        job_id = training_service.start_training_container(
            task_type=task_type,
            model_type=train_request.model_type,
            model_name=model_name,
            epochs=train_request.epochs,
            batch_size=train_request.batch_size,
            project_name=train_request.project_name,
            dataset_name=train_request.dataset_name,
            extra_args=extra_args,
            resource_plan=resource_plan,
            owner_id=_require_request_user_id(request),
            owner_email=_request_user_email(request),
        )
        return {"status": "success", "job_id": job_id, "container_id": job_id}
    except PermissionError as exc:
        raise HTTPException(status_code=401, detail=str(exc))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/status/{job_id}")
def get_status(job_id: str, request: Request):
    _assert_job_visible(job_id, request)
    status = training_service.get_container_status(job_id)
    return {"job_id": job_id, "container_id": job_id, "status": status}


@app.get("/api/logs/{job_id}")
def get_logs(job_id: str, request: Request):
    _assert_job_visible(job_id, request)
    logs = training_service.get_container_logs(job_id)
    return {"job_id": job_id, "container_id": job_id, "logs": logs}


@app.get("/api/metrics/{project_name}")
def get_metrics(project_name: str, request: Request):
    try:
        validate_slug(project_name, "project name")
        _assert_run_visible(project_name, request)
        metrics = training_service.get_training_metrics(project_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not metrics:
        return {"status": "no_data", "metrics": []}
    return {"status": "success", "metrics": metrics}


@app.post("/api/stop/{job_id}")
def stop_train(job_id: str, request: Request):
    _assert_job_visible(job_id, request)
    try:
        training_service.stop_training_container(job_id)
        return {"status": "success"}
    except Exception as exc:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/jobs/{job_id}/events")
async def job_events(job_id: str, request: Request):
    _assert_job_visible(job_id, request)
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
def list_datasets(request: Request):
    request_owner = _require_request_user_id(request)
    datasets = []
    if not DATASET_DIR.exists():
        return {"datasets": []}

    for item in sorted(DATASET_DIR.iterdir(), key=lambda path: path.name.lower()):
        if not item.is_dir() or item.name.startswith("."):
            continue
        owner_id = _dataset_owner(item)
        if owner_id != request_owner:
            continue

        metadata = inspect_dataset(item)
        metadata["export_cache"] = export_cache_metadata(item)
        profile = _dataset_profile(metadata)
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
                "errors": metadata.get("errors", []),
                "yamlPath": metadata["yaml_path"],
                "paddleocrTasks": metadata.get("paddleocr_tasks", []),
                "sourceFormat": profile["source_format"],
                "datasetTask": profile["dataset_task"],
                "datasetTasks": profile["dataset_tasks"],
                "canonicalTask": profile["canonical_task"],
                "canonicalFormat": profile["canonical_format"],
                "normalizedFormats": profile["normalized_formats"],
                "annotationStats": profile["annotation_stats"],
                "conversionWarnings": profile["conversion_warnings"],
                "compatibleModels": profile["compatible_models"],
                "readyModels": profile["ready_models"],
                "exportCache": profile["export_cache"],
                "createdBy": owner_id,
            }
        )

    return {"datasets": datasets}


@app.post("/api/datasets/inspect-upload")
async def inspect_dataset_upload(request: Request, file: UploadFile = File(...)):
    _require_request_user_id(request)
    dataset_name, metadata = await _inspect_uploaded_zip(file)
    return {"status": "success", "dataset_name": dataset_name, "profile": _dataset_profile(metadata)}


@app.post("/api/datasets/import")
async def import_dataset(request: Request, file: UploadFile = File(...)):
    return await _import_uploaded_zip(request, file)


@app.post("/api/upload-dataset")
async def upload_dataset(request: Request, file: UploadFile = File(...)):
    return await _import_uploaded_zip(request, file)


@app.delete("/api/datasets/{dataset_name}")
def delete_dataset(dataset_name: str, request: Request):
    try:
        validate_slug(dataset_name, "dataset name")
        target_dir = contained_path(DATASET_DIR, dataset_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not target_dir.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")
    _assert_owned_resource_visible(_dataset_owner(target_dir), request)
    try:
        shutil.rmtree(target_dir)
        return {"status": "success"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/runs")
def list_runs(request: Request):
    return {"runs": training_service.list_runs(owner_id=_require_request_user_id(request))}


@app.delete("/api/runs/{project_name}")
def delete_run(project_name: str, request: Request):
    try:
        validate_slug(project_name, "project name")
        _assert_run_visible(project_name, request)
        project_dir = contained_path(RUNS_DIR, project_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not project_dir.exists() or not project_dir.is_dir():
        raise HTTPException(status_code=404, detail="Run not found")
    try:
        shutil.rmtree(project_dir)
        return {"status": "success"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/runs/{project_name}/files/{file_path:path}")
def download_run_file(project_name: str, file_path: str, request: Request):
    try:
        validate_slug(project_name, "project name")
        _assert_run_visible(project_name, request)
        project_dir = contained_path(RUNS_DIR, project_name)
        target = contained_path(project_dir, file_path)
    except ValueError:
        raise HTTPException(status_code=404, detail="File not found")
    if not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(target, filename=target.name)


@app.get("/api/datasets/{dataset_name}/metadata")
def dataset_metadata(dataset_name: str, request: Request):
    try:
        validate_slug(dataset_name, "dataset name")
        target_dir = contained_path(DATASET_DIR, dataset_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not target_dir.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")
    _assert_owned_resource_visible(_dataset_owner(target_dir), request)
    metadata = inspect_dataset(target_dir)
    metadata["export_cache"] = export_cache_metadata(target_dir)
    metadata["yaml_path"] = str(find_dataset_yaml(target_dir) or "")
    metadata.update(_dataset_profile(metadata))
    return metadata


@app.get("/api/datasets/{dataset_name}/compatibility")
def dataset_compatibility(dataset_name: str, request: Request):
    try:
        validate_slug(dataset_name, "dataset name")
        target_dir = contained_path(DATASET_DIR, dataset_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not target_dir.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")
    _assert_owned_resource_visible(_dataset_owner(target_dir), request)
    metadata = inspect_dataset(target_dir)
    metadata["export_cache"] = export_cache_metadata(target_dir)
    return {"dataset_name": dataset_name, **_dataset_profile(metadata)}
