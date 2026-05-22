from __future__ import annotations

import os
import shutil
import time
import zipfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from dataset_utils import (
    find_dataset_yaml,
    format_bytes,
    inspect_dataset,
    safe_dataset_name,
    validate_dataset_for_upload,
)
from model_catalog import get_catalog, get_model
from services.training_service import TrainingService


app = FastAPI(title="No-Code Computer Vision Training Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
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
    epochs: int = 50
    batch_size: int = 16
    params: dict[str, Any] = Field(default_factory=dict)

    # Legacy YOLO payload support.
    model_size: str | None = None

    class Config:
        extra = "allow"


def _model_name_from_request(request: TrainRequest, model_entry: dict[str, Any] | None) -> str:
    if request.model_name:
        return request.model_name.removesuffix(".pt")

    if request.model_type == "yolo":
        model_size = request.params.get("model_size") or request.model_size or "n"
        model_size = str(model_size).removesuffix(".pt")
        if model_size.startswith("yolo"):
            return model_size
        return f"yolo11{model_size}"

    if model_entry:
        return str(model_entry["model_name"])

    if request.model_size:
        return request.model_size.removesuffix(".pt")

    raise HTTPException(status_code=400, detail="model_name is required for this model_type")


def _extra_args_from_request(request: TrainRequest) -> dict[str, Any]:
    payload = request.dict()
    reserved = {
        "task_type",
        "model_type",
        "model_name",
        "dataset_name",
        "project_name",
        "epochs",
        "batch_size",
        "params",
        "model_size",
    }
    extra_args = dict(request.params or {})
    for key, value in payload.items():
        if key not in reserved and value is not None:
            extra_args[key] = value

    if request.model_type == "yolo" and "model_size" not in extra_args:
        extra_args["model_size"] = request.model_size or "n"

    return extra_args


def _safe_extract(zip_file: zipfile.ZipFile, target_dir: Path) -> None:
    target = target_dir.resolve()
    for member in zip_file.infolist():
        member_path = (target_dir / member.filename).resolve()
        if not str(member_path).startswith(str(target)):
            raise HTTPException(status_code=400, detail="ZIP contains unsafe paths")
    zip_file.extractall(target_dir)


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
    model_name = _model_name_from_request(request, model_entry)
    extra_args = _extra_args_from_request(request)

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
    metrics = training_service.get_training_metrics(project_name)
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


@app.get("/api/datasets")
def list_datasets():
    datasets = []
    if not DATASET_DIR.exists():
        return {"datasets": []}

    for item in sorted(DATASET_DIR.iterdir(), key=lambda path: path.name.lower()):
        if not item.is_dir():
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
    target_dir = DATASET_DIR / dataset_name
    temp_zip_path = DATASET_DIR / file.filename

    if target_dir.exists():
        shutil.rmtree(target_dir)

    try:
        with open(temp_zip_path, "wb") as buffer:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                buffer.write(chunk)

        with zipfile.ZipFile(temp_zip_path, "r") as zip_ref:
            _safe_extract(zip_ref, target_dir)

        _flatten_single_root_folder(target_dir)
        metadata = validate_dataset_for_upload(target_dir)

        return {
            "status": "success",
            "dataset_name": dataset_name,
            "tasks": metadata["tasks"],
            "formats": metadata["formats"],
            "classes": metadata["classes"],
        }
    except HTTPException:
        if target_dir.exists():
            shutil.rmtree(target_dir, ignore_errors=True)
        raise
    except zipfile.BadZipFile:
        if target_dir.exists():
            shutil.rmtree(target_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail="Invalid ZIP file")
    except ValueError as exc:
        if target_dir.exists():
            shutil.rmtree(target_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        if target_dir.exists():
            shutil.rmtree(target_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        if temp_zip_path.exists():
            temp_zip_path.unlink()


@app.delete("/api/datasets/{dataset_name}")
def delete_dataset(dataset_name: str):
    target_dir = DATASET_DIR / dataset_name
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
    project_dir = (RUNS_DIR / project_name).resolve()
    target = (project_dir / file_path).resolve()
    if not str(target).startswith(str(project_dir)) or not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(target, filename=target.name)


@app.get("/api/datasets/{dataset_name}/metadata")
def dataset_metadata(dataset_name: str):
    target_dir = DATASET_DIR / dataset_name
    if not target_dir.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")
    metadata = inspect_dataset(target_dir)
    metadata["yaml_path"] = str(find_dataset_yaml(target_dir) or "")
    return metadata
