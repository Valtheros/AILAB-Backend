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
    DATASET_METADATA_VERSION,
    dataset_workflow_metadata,
    export_cache_metadata,
    find_dataset_yaml,
    format_bytes,
    inspect_dataset,
    inspect_dataset_for_upload,
    safe_dataset_name,
    validate_dataset_for_upload,
)
from dataset_storage import dataset_lock_name, owner_dataset_path, registered_storage_path
from inference_service import (
    MAX_INFERENCE_IMAGE_BYTES,
    InferenceError,
    InferenceUnavailable,
    RateLimitExceeded,
    check_rate_limit,
    predict_image,
)
from model_catalog import get_catalog, get_model, validate_model_params
from resource_guard import (
    ResourcePlanError,
    enforce_resource_plan,
    get_resource_profile,
    validate_resource_plan,
)
from security_utils import (
    contained_path,
    named_file_lock,
    ReversibleDirectoryRemoval,
    ReversibleDirectoryReplace,
    safe_extract_zip,
    save_upload_to_temp,
    staging_directory,
    validate_slug,
)
from services.training_service import TrainingService
from resource_repository import resource_repository
from staged_uploads import StagedUploadStore
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
staged_uploads = StagedUploadStore(DATASET_DIR)
staged_uploads.cleanup()


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


class TaskCreateRequest(BaseModel):
    display_name: str = Field(default="cv_run", min_length=1, max_length=100)


class TaskDraftRequest(BaseModel):
    display_name: str = Field(min_length=1, max_length=100)
    task_type: str
    model_type: str
    model_name: str
    dataset_name: str = ""
    epochs: int = Field(ge=1, le=2000)
    batch_size: int = Field(ge=1, le=256)
    params: dict[str, Any] = Field(default_factory=dict)
    device_selection: str = Field(default="auto", pattern="^(auto|manual)$")

    class Config:
        extra = "forbid"


class ResourcePlanPreviewRequest(BaseModel):
    model_type: str
    params: dict[str, Any] = Field(default_factory=dict)
    batch_size: int = Field(default=16, ge=1, le=256)

    class Config:
        extra = "forbid"


class StagedImportRequest(BaseModel):
    uploadToken: str = Field(min_length=20, max_length=128)


class TransferResourcesRequest(BaseModel):
    targetUserId: str = Field(min_length=1, max_length=255)


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


def _require_admin(request: Request) -> str:
    user_id = _require_request_user_id(request)
    try:
        resource_repository.assert_admin(user_id)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
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
    if resource_repository.enabled:
        return resource_repository.get_run_owner_by_slug(project_name)
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
        model_name = str(
            (request.params.get("architecture") or model_entry["model_name"])
            if request.model_type in {"resnet", "efficientnet"}
            else model_entry["model_name"]
        )
        if request.model_name and request.model_name.removesuffix(".pt") != model_name:
            raise HTTPException(status_code=400, detail=f"Unsupported model_name for {request.model_type}")
        return model_name

    if request.model_size:
        return request.model_size.removesuffix(".pt")

    raise HTTPException(status_code=400, detail="model_name is required for this model_type")


def _extra_args_from_request(request: TrainRequest) -> dict[str, Any]:
    extra_args = dict(request.params or {})
    if request.model_type == "yolo" and "model_size" not in extra_args:
        extra_args["model_size"] = request.model_size or "n"
    return validate_model_params(request.model_type, extra_args)


def _task_response(row: dict[str, Any], run: dict[str, Any] | None = None) -> dict[str, Any]:
    params = dict(row.get("params") or {})
    return {
        "id": str(row["id"]),
        "displayName": row.get("display_name") or "cv_run",
        "status": row.get("status") or "draft",
        "taskType": row.get("task_type") or "object_detection",
        "modelType": row.get("model_type") or "yolo",
        "modelName": row.get("model_name") or "yolo11n",
        "datasetName": row.get("dataset_slug") or "",
        "epochs": int(params.get("epochs", 50)),
        "batchSize": int(params.get("batch_size", 16)),
        "device": str(params.get("device", "cpu")),
        "workers": int(params.get("workers", 4)),
        "amp": bool(params.get("amp", True)),
        "seed": int(params.get("seed", 0)),
        "deviceSelection": str(params.get("_device_selection", "auto")),
        "params": {key: value for key, value in params.items() if not key.startswith("_")},
        "runSlug": row.get("run_slug"),
        "jobId": row.get("rq_job_id"),
        "errorDetail": row.get("error_detail"),
        "createdAt": row.get("created_at"),
        "updatedAt": row.get("updated_at"),
        "finishedAt": row.get("finished_at"),
        "files": (run or {}).get("files", []),
        "latestMetrics": (run or {}).get("latest_metrics"),
    }


def _extract_and_validate_upload(temp_zip_path: Path, staging_dir: Path) -> tuple[Path, dict[str, Any]]:
    with zipfile.ZipFile(temp_zip_path, "r") as zip_ref:
        extract_dir = staging_dir / "extracted"
        safe_extract_zip(zip_ref, extract_dir)
    _flatten_single_root_folder(extract_dir)
    metadata = inspect_dataset_for_upload(extract_dir)
    validate_dataset_for_upload(extract_dir, metadata)
    return extract_dir, metadata


def _flatten_single_root_folder(target_dir: Path) -> None:
    contents = list(target_dir.iterdir())
    if len(contents) != 1 or not contents[0].is_dir():
        return
    single_dir = contents[0]
    structural_names = {"train", "training", "valid", "val", "validation", "test", "images", "labels", "masks", "annotations"}
    if single_dir.name.lower() in structural_names:
        return
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


def _dataset_metadata_is_current(metadata: Any) -> bool:
    required = {"formats", "tasks", "classes", "image_count", "size_bytes", "warnings", "errors"}
    return (
        isinstance(metadata, dict)
        and metadata.get("metadata_version") == DATASET_METADATA_VERSION
        and required.issubset(metadata)
    )


def _dataset_response(dataset_name: str, metadata: dict[str, Any]) -> dict[str, Any]:
    profile = _dataset_profile(metadata)
    return {
        "status": "success",
        "dataset_name": dataset_name,
        "tasks": profile["tasks"],
        "formats": profile["formats"],
        "classes": profile["classes"],
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


async def _inspect_uploaded_zip(request: Request, file: UploadFile) -> dict[str, Any]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    if not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Only .zip files are supported")

    dataset_name = safe_dataset_name(file.filename)
    staging_dir = staging_directory(DATASET_DIR)
    temp_zip_path: Path | None = None
    try:
        temp_zip_path = await save_upload_to_temp(file, DATASET_DIR)
        extract_dir, metadata = await asyncio.to_thread(_extract_and_validate_upload, temp_zip_path, staging_dir)
        profile = _dataset_profile(metadata)
        staged = await asyncio.to_thread(
            staged_uploads.create,
            _require_request_user_id(request),
            dataset_name,
            extract_dir,
            profile,
        )
        return {
            "dataset_name": dataset_name,
            "profile": profile,
            "uploadToken": staged["token"],
            "expiresAt": staged["expires_at"],
        }
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
    request_owner = _require_request_user_id(request)
    target_dir = owner_dataset_path(DATASET_DIR, request_owner, dataset_name)
    staging_dir = staging_directory(DATASET_DIR)
    temp_zip_path: Path | None = None
    replacement: ReversibleDirectoryReplace | None = None

    try:
        temp_zip_path = await save_upload_to_temp(file, DATASET_DIR)
        extract_dir, metadata = await asyncio.to_thread(_extract_and_validate_upload, temp_zip_path, staging_dir)
        with named_file_lock(DATASET_DIR, dataset_lock_name(request_owner, dataset_name), "dataset name"):
            with resource_repository.dataset_guard(request_owner, dataset_name) as connection:
                existing = resource_repository.get_dataset(request_owner, dataset_name, connection)
                if existing:
                    target_dir = registered_storage_path(DATASET_DIR, existing["storage_path"])
                active = resource_repository.active_runs_for_dataset(existing.get("id") if existing else None, connection)
                if active:
                    names = ", ".join(str(run["run_slug"]) for run in active)
                    raise HTTPException(status_code=409, detail=f"Dataset is in use by active training run(s): {names}")
                target_owner = _dataset_owner(target_dir) if target_dir.exists() else None
                if target_dir.exists() and target_owner not in {None, request_owner}:
                    raise HTTPException(status_code=409, detail="Dataset storage ownership is inconsistent. Contact an administrator.")
                target_dir.parent.mkdir(parents=True, exist_ok=True)
                replacement = ReversibleDirectoryReplace(extract_dir, target_dir).apply()
                metadata["yaml_path"] = str(find_dataset_yaml(target_dir) or "")
                workflow = dataset_workflow_metadata(metadata, get_catalog())
                _write_dataset_metadata(target_dir, request, workflow)
                record = resource_repository.upsert_dataset(request_owner, _request_user_email(request), dataset_name, target_dir, metadata, connection)
            replacement.commit()
            return {**_dataset_response(dataset_name, metadata), "id": str(record.get("id", dataset_name))}
    except HTTPException:
        if replacement is not None:
            replacement.rollback()
        raise
    except zipfile.BadZipFile:
        if replacement is not None:
            replacement.rollback()
        raise HTTPException(status_code=400, detail="Invalid ZIP file")
    except FileExistsError as exc:
        if replacement is not None:
            replacement.rollback()
        raise HTTPException(status_code=409, detail=str(exc))
    except ValueError as exc:
        if replacement is not None:
            replacement.rollback()
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        if replacement is not None:
            replacement.rollback()
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


def _resource_plan_summary(plan: dict[str, Any]) -> dict[str, Any]:
    """Surface the numbers resource_guard already computed, for display only.

    This does not re-derive anything: `estimated_vram_mb` and `safe_vram_mb`
    come straight from the plan. `isWithinLimit` mirrors the same comparison
    resource_guard uses internally (CPU runs have no VRAM budget, so they are
    always within limit).
    """
    estimated = int(plan.get("estimated_vram_mb", 0) or 0)
    safe_limit = int(plan.get("safe_vram_mb", 0) or 0)
    is_cpu = str(plan.get("device", "")).startswith("cpu")
    return {
        "estimatedVramMb": estimated,
        "safeLimitMb": safe_limit,
        "isWithinLimit": True if is_cpu else estimated <= safe_limit,
        "device": plan.get("device"),
        "batchSize": plan.get("batch_size"),
        "warnings": plan.get("warnings", []),
        "errors": plan.get("errors", []),
        "suggestions": plan.get("suggestions", []),
    }


@app.post("/api/resource-plan/preview")
def preview_resource_plan(payload: ResourcePlanPreviewRequest, request: Request):
    """Read-only preview of the memory plan for a draft configuration.

    Uses validate_resource_plan (not enforce_*) so an over-budget config is
    reported instead of rejected. The training path keeps using
    enforce_resource_plan unchanged.
    """
    _require_request_user_id(request)
    if get_model(payload.model_type) is None:
        raise HTTPException(status_code=400, detail=f"Unsupported model_type: {payload.model_type}")
    plan = validate_resource_plan(
        payload.model_type,
        params=dict(payload.params or {}),
        batch_size=payload.batch_size,
    )
    return {"status": "success", "ok": plan["ok"], **_resource_plan_summary(plan)}


@app.post("/api/train")
def start_train(train_request: TrainRequest, request: Request):
    return _start_train_request(train_request, request)


def _start_train_request(train_request: TrainRequest, request: Request, task_id: str | None = None):
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
            task_id=task_id,
        )
        return {
            "status": "success",
            "job_id": job_id,
            "container_id": job_id,
            "resourcePlan": _resource_plan_summary(resource_plan),
        }
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
        status = training_service.stop_training_container(job_id)
        return {"status": status}
    except Exception as exc:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/jobs/{job_id}/events")
async def job_events(job_id: str, request: Request):
    _assert_job_visible(job_id, request)
    snapshot = await asyncio.to_thread(training_service.get_job_snapshot, job_id)
    if snapshot["status"] == "not_found":
        raise HTTPException(status_code=404, detail="Job not found")

    async def stream():
        initial_log, log_offset, _ = await asyncio.to_thread(training_service.read_job_log_chunk, job_id, -1)
        snapshot["logs"] = initial_log
        snapshot["log_offset"] = log_offset
        previous_metrics = json.dumps(snapshot["metrics"], sort_keys=True)
        previous_status = snapshot["status"]
        yield sse_event("snapshot", snapshot)
        if previous_status in TERMINAL_STATUSES:
            yield sse_event("end", {"status": previous_status})
            return

        heartbeat_ticks = 0
        while True:
            await asyncio.sleep(1)
            current = await asyncio.to_thread(training_service.get_job_snapshot, job_id)
            current_status = current["status"]
            current_metrics = json.dumps(current["metrics"], sort_keys=True)
            sent_event = False

            chunk, log_offset, replace = await asyncio.to_thread(training_service.read_job_log_chunk, job_id, log_offset)
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

    records = resource_repository.list_datasets(request_owner) if resource_repository.enabled else []
    items = [(registered_storage_path(DATASET_DIR, row["storage_path"]), row) for row in records]
    if not resource_repository.enabled:
        items = [(item, None) for item in DATASET_DIR.iterdir() if item.is_dir() and not item.name.startswith(".")]
    for item, record in sorted(items, key=lambda pair: pair[0].name.lower()):
        if not item.is_dir() or item.name.startswith("."):
            continue
        owner_id = record.get("owner_user_id") if record else _dataset_owner(item)
        if owner_id != request_owner:
            continue

        stored_metadata = record.get("metadata") if record else None
        if isinstance(stored_metadata, str):
            try:
                stored_metadata = json.loads(stored_metadata)
            except json.JSONDecodeError:
                stored_metadata = None
        metadata_is_current = _dataset_metadata_is_current(stored_metadata)
        metadata = dict(stored_metadata) if metadata_is_current else inspect_dataset(item)
        if record and not metadata_is_current:
            resource_repository.update_dataset_metadata(record["id"], metadata)
        metadata["export_cache"] = export_cache_metadata(item)
        profile = _dataset_profile(metadata)
        created = time.strftime("%Y-%m-%d", time.localtime(item.stat().st_ctime))
        datasets.append(
            {
                "id": str(record.get("id")) if record else item.name,
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
    result = await _inspect_uploaded_zip(request, file)
    return {"status": "success", **result}


@app.post("/api/datasets/import")
def import_dataset(payload: StagedImportRequest, request: Request):
    owner_id = _require_request_user_id(request)
    replacement: ReversibleDirectoryReplace | None = None
    try:
        manifest, extracted_dir, _pending_stage_dir = staged_uploads.peek(payload.uploadToken, owner_id)
        dataset_name = str(manifest["dataset_name"])
        validate_slug(dataset_name, "dataset name")
        metadata = inspect_dataset_for_upload(extracted_dir)
        validate_dataset_for_upload(extracted_dir, metadata)
        with resource_repository.dataset_guard(owner_id, dataset_name) as connection:
            existing = resource_repository.get_dataset(owner_id, dataset_name, connection)
            target_dir = (
                registered_storage_path(DATASET_DIR, existing["storage_path"])
                if existing
                else owner_dataset_path(DATASET_DIR, owner_id, dataset_name)
            )
            active = resource_repository.active_runs_for_dataset(existing.get("id") if existing else None, connection)
            if active:
                names = ", ".join(str(run["run_slug"]) for run in active)
                raise HTTPException(status_code=409, detail=f"Dataset is in use by active training run(s): {names}")
            if target_dir.exists() and existing is None:
                raise HTTPException(status_code=409, detail="Dataset storage exists without a registry record. Contact an administrator.")
            _manifest, extracted_dir, stage_dir = staged_uploads.consume(payload.uploadToken, owner_id)
            target_dir.parent.mkdir(parents=True, exist_ok=True)
            replacement = ReversibleDirectoryReplace(extracted_dir, target_dir).apply()
            metadata["yaml_path"] = str(find_dataset_yaml(target_dir) or "")
            workflow = dataset_workflow_metadata(metadata, get_catalog())
            _write_dataset_metadata(target_dir, request, workflow)
            record = resource_repository.upsert_dataset(owner_id, _request_user_email(request), dataset_name, target_dir, metadata, connection)
        replacement.commit()
        shutil.rmtree(stage_dir, ignore_errors=True)
        return {**_dataset_response(dataset_name, metadata), "id": str(record.get("id", dataset_name))}
    except HTTPException:
        if replacement is not None:
            replacement.rollback()
        raise
    except FileNotFoundError as exc:
        if replacement is not None:
            replacement.rollback()
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        if replacement is not None:
            replacement.rollback()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception:
        if replacement is not None:
            replacement.rollback()
        raise


@app.post("/api/upload-dataset")
async def upload_dataset(request: Request, file: UploadFile = File(...)):
    return await _import_uploaded_zip(request, file)


@app.delete("/api/datasets/{dataset_name}")
def delete_dataset(dataset_name: str, request: Request):
    owner_id = _require_request_user_id(request)
    try:
        validate_slug(dataset_name, "dataset name")
        record = resource_repository.get_dataset(owner_id, dataset_name) if resource_repository.enabled else None
        target_dir = registered_storage_path(DATASET_DIR, record["storage_path"]) if record else owner_dataset_path(DATASET_DIR, owner_id, dataset_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not target_dir.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")
    removal: ReversibleDirectoryRemoval | None = None
    try:
        with resource_repository.dataset_guard(owner_id, dataset_name) as connection:
            record = resource_repository.get_dataset(owner_id, dataset_name, connection)
            if resource_repository.enabled and not record:
                raise HTTPException(status_code=404, detail="Dataset not found")
            if not resource_repository.enabled:
                _assert_owned_resource_visible(_dataset_owner(target_dir), request)
            active = resource_repository.active_runs_for_dataset(record.get("id") if record else None, connection)
            if active:
                names = ", ".join(str(run["run_slug"]) for run in active)
                raise HTTPException(status_code=409, detail=f"Dataset is in use by active training run(s): {names}")
            removal = ReversibleDirectoryRemoval(target_dir).apply()
            if record:
                resource_repository.mark_dataset_deleted(record["id"], connection)
        removal.commit()
        return {"status": "success"}
    except HTTPException:
        if removal is not None:
            removal.rollback()
        raise
    except Exception as exc:
        if removal is not None:
            removal.rollback()
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/tasks")
def list_tasks(request: Request):
    owner_id = _require_request_user_id(request)
    runs = {run["id"]: run for run in training_service.list_runs(owner_id=owner_id)}
    return {"tasks": [_task_response(row, runs.get(str(row["id"]))) for row in resource_repository.list_tasks(owner_id)]}


@app.post("/api/tasks", status_code=201)
def create_task(payload: TaskCreateRequest, request: Request):
    display_name = payload.display_name.strip()
    if not display_name:
        raise HTTPException(status_code=400, detail="Project name is required")
    row = resource_repository.create_task(
        _require_request_user_id(request), _request_user_email(request), display_name
    )
    return _task_response(row)


@app.get("/api/tasks/{task_id}")
def get_task(task_id: str, request: Request):
    owner_id = _require_request_user_id(request)
    row = resource_repository.get_task(owner_id, task_id)
    if not row:
        raise HTTPException(status_code=404, detail="Training task not found")
    run = next((item for item in training_service.list_runs(owner_id=owner_id) if item["id"] == task_id), None)
    return _task_response(row, run)


@app.patch("/api/tasks/{task_id}")
def update_task(task_id: str, payload: TaskDraftRequest, request: Request):
    owner_id = _require_request_user_id(request)
    current = resource_repository.get_task(owner_id, task_id)
    if not current:
        raise HTTPException(status_code=404, detail="Training task not found")
    if current["status"] != "draft":
        raise HTTPException(status_code=409, detail="Only draft tasks can be edited")
    model = get_model(payload.model_type)
    if not model or model["task_type"] != payload.task_type:
        raise HTTPException(status_code=400, detail="The selected model does not belong to this task")
    if payload.dataset_name:
        try:
            validate_slug(payload.dataset_name, "dataset name")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    params = dict(payload.params)
    params.update({
        "epochs": payload.epochs,
        "batch_size": payload.batch_size,
        "_device_selection": payload.device_selection,
    })
    row = resource_repository.update_task_draft(owner_id, task_id, {
        "display_name": payload.display_name.strip() or "cv_run",
        "task_type": payload.task_type,
        "model_type": payload.model_type,
        "model_name": payload.model_name,
        "dataset_slug": payload.dataset_name,
        "params": params,
    })
    if not row:
        raise HTTPException(status_code=409, detail="Task changed while it was being saved")
    return _task_response(row)


@app.post("/api/tasks/{task_id}/start")
def start_task(task_id: str, request: Request):
    owner_id = _require_request_user_id(request)
    row = resource_repository.get_task(owner_id, task_id)
    if not row:
        raise HTTPException(status_code=404, detail="Training task not found")
    if row["status"] != "draft":
        raise HTTPException(status_code=409, detail="Training task has already been started")
    if not row.get("dataset_slug"):
        raise HTTPException(status_code=400, detail="Select a compatible dataset before training")
    params = dict(row.get("params") or {})
    run_slug = f"{safe_dataset_name(str(row.get('display_name') or 'cv_run'))}_{str(row['id']).replace('-', '')[:8]}"
    train_request = TrainRequest(
        task_type=row["task_type"], model_type=row["model_type"], model_name=row.get("model_name"),
        dataset_name=row["dataset_slug"], project_name=run_slug,
        epochs=int(params.get("epochs", 50)), batch_size=int(params.get("batch_size", 16)),
        params={key: value for key, value in params.items() if not key.startswith("_")},
    )
    result = _start_train_request(train_request, request, task_id=task_id)
    return {**result, "task_id": task_id}


@app.post("/api/tasks/{task_id}/stop")
def stop_task(task_id: str, request: Request):
    row = resource_repository.get_task(_require_request_user_id(request), task_id)
    if not row:
        raise HTTPException(status_code=404, detail="Training task not found")
    if not row.get("rq_job_id") or row["status"] not in {"queued", "running", "started", "stopping"}:
        raise HTTPException(status_code=409, detail="Training task is not active")
    try:
        return {"status": training_service.stop_training_container(row["rq_job_id"])}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/tasks/{task_id}/logs")
def task_logs(task_id: str, request: Request):
    row = resource_repository.get_task(_require_request_user_id(request), task_id)
    if not row:
        raise HTTPException(status_code=404, detail="Training task not found")
    if row.get("storage_path"):
        run_dir = registered_storage_path(RUNS_DIR, row["storage_path"])
        log_path = contained_path(run_dir, "train.log")
        if log_path.is_file():
            limit = 1024 * 1024
            size = log_path.stat().st_size
            with open(log_path, "rb") as file:
                file.seek(max(0, size - limit))
                content = file.read(limit).decode("utf-8", errors="replace")
            prefix = "[Earlier log output omitted]\n" if size > limit else ""
            return {"logs": prefix + content}
    if row.get("rq_job_id"):
        return {"logs": training_service.get_container_logs(str(row["rq_job_id"]))}
    return {"logs": ""}


@app.delete("/api/tasks/{task_id}")
def delete_task(task_id: str, request: Request):
    owner_id = _require_request_user_id(request)
    row = resource_repository.get_task(owner_id, task_id)
    if not row:
        raise HTTPException(status_code=404, detail="Training task not found")
    if row["status"] in {"queued", "running", "started", "stopping", "recovery_pending"}:
        raise HTTPException(status_code=409, detail="Stop the active training task before deleting it")
    removal: ReversibleDirectoryRemoval | None = None
    try:
        if row.get("storage_path"):
            path = registered_storage_path(RUNS_DIR, row["storage_path"])
            if path.is_dir():
                removal = ReversibleDirectoryRemoval(path).apply()
        deleted = resource_repository.delete_task(owner_id, task_id)
        if not deleted:
            raise HTTPException(status_code=409, detail="Training task could not be deleted")
        if removal:
            removal.commit()
        return {"status": "success"}
    except HTTPException:
        if removal:
            removal.rollback()
        raise
    except Exception as exc:
        if removal:
            removal.rollback()
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/runs")
def list_runs(request: Request):
    return {"runs": training_service.list_runs(owner_id=_require_request_user_id(request))}


@app.delete("/api/runs/{project_name}")
def delete_run(project_name: str, request: Request):
    owner_id = _require_request_user_id(request)
    record = resource_repository.get_run(owner_id, project_name)
    if record and record.get("status") in {"queued", "running", "started", "stopping"}:
        raise HTTPException(status_code=409, detail="Stop the active training run before deleting it")
    try:
        validate_slug(project_name, "project name")
        _assert_run_visible(project_name, request)
        project_dir = contained_path(RUNS_DIR, project_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not project_dir.exists() or not project_dir.is_dir():
        raise HTTPException(status_code=404, detail="Run not found")
    removal: ReversibleDirectoryRemoval | None = None
    try:
        removal = ReversibleDirectoryRemoval(project_dir).apply()
        resource_repository.delete_run_by_slug(owner_id, project_name)
        removal.commit()
        return {"status": "success"}
    except Exception as exc:
        if removal is not None:
            removal.rollback()
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _run_record_for_predict(project_name: str, request: Request) -> dict[str, Any] | None:
    """Resolve a run the caller owns, or raise the appropriate HTTP error."""
    validate_slug(project_name, "project name")
    _assert_run_visible(project_name, request)
    if not resource_repository.enabled:
        return None
    record = resource_repository.get_run(_require_request_user_id(request), project_name)
    if not record:
        raise HTTPException(status_code=404, detail="Run not found")
    return record


@app.post("/api/runs/{project_name}/predict")
async def predict_run(project_name: str, request: Request, file: UploadFile = File(...)):
    """Classify one uploaded image with a completed run's trained model.

    Image classification only. Detection and segmentation runs are rejected
    because their outputs need box/mask rendering that this endpoint does not
    produce.
    """
    owner_id = _require_request_user_id(request)
    try:
        record = _run_record_for_predict(project_name, request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    config = _read_json_file(contained_path(RUNS_DIR, project_name, "job_config.json"))
    task_type = str((record or {}).get("task_type") or config.get("task_type") or "")
    status = str((record or {}).get("status") or "")

    if task_type != "image_classification":
        raise HTTPException(
            status_code=400,
            detail="Model testing currently supports image classification runs only.",
        )
    if record is not None and status != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"This run is '{status or 'unknown'}'. Only completed runs can be tested.",
        )

    try:
        check_rate_limit(owner_id)
    except RateLimitExceeded as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc

    # Read with a hard cap so an oversized upload cannot be buffered in full.
    payload = await file.read(MAX_INFERENCE_IMAGE_BYTES + 1)
    if len(payload) > MAX_INFERENCE_IMAGE_BYTES:
        limit_mb = MAX_INFERENCE_IMAGE_BYTES / 1024 / 1024
        raise HTTPException(status_code=400, detail=f"The image is larger than the {limit_mb:.0f} MB limit.")

    run_dir = contained_path(RUNS_DIR, project_name)
    if not run_dir.is_dir():
        raise HTTPException(status_code=404, detail="Run not found")

    try:
        # Runs off the event loop: loading a checkpoint and the forward pass are
        # both blocking CPU work.
        result = await asyncio.to_thread(predict_image, run_dir, payload)
    except InferenceError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except InferenceUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    return {"status": "success", "runSlug": project_name, **result}


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


@app.get("/api/admin/users/{user_id}/resource-impact")
def admin_user_resource_impact(user_id: str, request: Request):
    _require_admin(request)
    return resource_repository.user_impact(user_id)


@app.post("/api/admin/users/{user_id}/transfer-resources")
def admin_transfer_user_resources(user_id: str, payload: TransferResourcesRequest, request: Request):
    _require_admin(request)
    impact = resource_repository.user_impact(user_id)
    if impact["activeJobs"]:
        raise HTTPException(status_code=409, detail="Stop active training jobs before transferring ownership")
    try:
        return {"status": "success", **resource_repository.transfer_user_resources(user_id, payload.targetUserId)}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.delete("/api/admin/users/{user_id}/resources")
def admin_delete_user_resources(user_id: str, request: Request):
    _require_admin(request)
    datasets, runs = resource_repository.delete_user_resource_records(user_id)
    errors: list[str] = []
    for run in runs:
        job_id = run.get("rq_job_id")
        if job_id and run.get("status") in {"queued", "running", "started", "stopping"}:
            try:
                training_service.stop_training_container(str(job_id))
            except Exception as exc:
                errors.append(f"Could not stop {run.get('run_slug')}: {exc}")
    if errors:
        raise HTTPException(status_code=409, detail=" ".join(errors))
    deadline = time.monotonic() + 15
    for run in runs:
        job_id = run.get("rq_job_id")
        if not job_id:
            continue
        while time.monotonic() < deadline:
            if training_service.get_container_status(str(job_id)) in TERMINAL_STATUSES | {"not_found", "stopped"}:
                break
            time.sleep(0.25)
        else:
            errors.append(f"Training run {run.get('run_slug')} is still stopping; retry cleanup shortly")
    if errors:
        raise HTTPException(status_code=409, detail=" ".join(errors))
    for run in runs:
        if not run.get("storage_path"):
            continue
        try:
            shutil.rmtree(contained_path(RUNS_DIR, Path(str(run["storage_path"]))))
        except FileNotFoundError:
            pass
        except Exception as exc:
            errors.append(f"Could not delete run {run.get('run_slug')}: {exc}")
    for dataset in datasets:
        try:
            shutil.rmtree(contained_path(DATASET_DIR, Path(str(dataset["storage_path"]))))
        except FileNotFoundError:
            pass
        except Exception as exc:
            errors.append(f"Could not delete dataset {dataset.get('slug')}: {exc}")
    if errors:
        raise HTTPException(status_code=500, detail=" ".join(errors))
    resource_repository.finalize_user_resource_delete(user_id)
    return {"status": "success", "datasets": len(datasets), "runs": len(runs)}


@app.get("/api/datasets/{dataset_name}/metadata")
def dataset_metadata(dataset_name: str, request: Request):
    owner_id = _require_request_user_id(request)
    try:
        validate_slug(dataset_name, "dataset name")
        record = resource_repository.get_dataset(owner_id, dataset_name) if resource_repository.enabled else None
        target_dir = registered_storage_path(DATASET_DIR, record["storage_path"]) if record else owner_dataset_path(DATASET_DIR, owner_id, dataset_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not target_dir.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")
    if resource_repository.enabled:
        if not resource_repository.get_dataset(owner_id, dataset_name):
            raise HTTPException(status_code=404, detail="Dataset not found")
    else:
        _assert_owned_resource_visible(_dataset_owner(target_dir), request)
    metadata = inspect_dataset(target_dir)
    metadata["export_cache"] = export_cache_metadata(target_dir)
    metadata["yaml_path"] = str(find_dataset_yaml(target_dir) or "")
    metadata.update(_dataset_profile(metadata))
    return metadata


@app.get("/api/datasets/{dataset_name}/compatibility")
def dataset_compatibility(dataset_name: str, request: Request):
    owner_id = _require_request_user_id(request)
    try:
        validate_slug(dataset_name, "dataset name")
        record = resource_repository.get_dataset(owner_id, dataset_name) if resource_repository.enabled else None
        target_dir = registered_storage_path(DATASET_DIR, record["storage_path"]) if record else owner_dataset_path(DATASET_DIR, owner_id, dataset_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not target_dir.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")
    if resource_repository.enabled:
        if not resource_repository.get_dataset(owner_id, dataset_name):
            raise HTTPException(status_code=404, detail="Dataset not found")
    else:
        _assert_owned_resource_visible(_dataset_owner(target_dir), request)
    metadata = inspect_dataset(target_dir)
    metadata["export_cache"] = export_cache_metadata(target_dir)
    return {"dataset_name": dataset_name, **_dataset_profile(metadata)}
