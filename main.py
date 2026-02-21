from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import Optional
from services.training_service import TrainingService
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
import zipfile
from pathlib import Path

app = FastAPI()

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

training_service = TrainingService()


class TrainRequest(BaseModel):
    model_size: str   # n, s, m, l, x
    model_type: str = "yolo"  # yolo | efficientdet | rtdetr | ... (เพิ่มได้เรื่อยๆ)
    epochs: int
    batch_size: int = 16
    project_name: str = "train_run"
    # Full config from /config page
    imgsz: int = 640
    device: str = "0"
    workers: int = 2      # DataLoader workers — ลดถ้า OOM (0 = ใช้ main process)
    patience: int = 100
    pretrained: bool = True
    cache: bool = False
    amp: bool = True
    fraction: float = 1.0
    # Optimizer
    optimizer: str = "auto"
    lr0: float = 0.01
    lrf: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 0.0005
    cos_lr: bool = False
    # Warmup
    warmup_epochs: float = 3.0
    warmup_momentum: float = 0.8
    warmup_bias_lr: float = 0.1
    # Loss weights
    box: float = 7.5
    cls: float = 0.5
    dfl: float = 1.5
    # Augmentation
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    degrees: float = 0.0
    translate: float = 0.1
    scale: float = 0.5
    shear: float = 0.0
    perspective: float = 0.0
    flipud: float = 0.0
    fliplr: float = 0.5
    mosaic: float = 1.0
    mixup: float = 0.0
    copy_paste: float = 0.0
    # Advanced
    save_period: int = -1
    close_mosaic: int = 10
    nbs: int = 64
    dropout: float = 0.0
    seed: int = 0
    deterministic: bool = True
    single_cls: bool = False
    rect: bool = False
    multi_scale: bool = False


@app.post("/api/train")
def start_train(request: TrainRequest):
    # สำหรับ YOLO ใช้ชื่อ model เต็ม สำหรับ model อื่นๆ ให้ส่ง model_size เป็นชื่อ model เลย
    if request.model_type == "yolo":
        model_name = f"yolo11{request.model_size}"
    else:
        model_name = request.model_size  # เช่น "efficientdet-d0", "rt-detr-l"
    try:
        # Build the full config dict to pass to training service
        extra_args = {
            "imgsz": request.imgsz,
            "device": request.device,
            "workers": request.workers,
            "patience": request.patience,
            "pretrained": request.pretrained,
            "cache": request.cache,
            "amp": request.amp,
            "fraction": request.fraction,
            "optimizer": request.optimizer,
            "lr0": request.lr0,
            "lrf": request.lrf,
            "momentum": request.momentum,
            "weight_decay": request.weight_decay,
            "cos_lr": request.cos_lr,
            "warmup_epochs": request.warmup_epochs,
            "warmup_momentum": request.warmup_momentum,
            "warmup_bias_lr": request.warmup_bias_lr,
            "box": request.box,
            "cls": request.cls,
            "dfl": request.dfl,
            "hsv_h": request.hsv_h,
            "hsv_s": request.hsv_s,
            "hsv_v": request.hsv_v,
            "degrees": request.degrees,
            "translate": request.translate,
            "scale": request.scale,
            "shear": request.shear,
            "perspective": request.perspective,
            "flipud": request.flipud,
            "fliplr": request.fliplr,
            "mosaic": request.mosaic,
            "mixup": request.mixup,
            "copy_paste": request.copy_paste,
            "save_period": request.save_period,
            "close_mosaic": request.close_mosaic,
            "nbs": request.nbs,
            "dropout": request.dropout,
            "seed": request.seed,
            "deterministic": request.deterministic,
            "single_cls": request.single_cls,
            "rect": request.rect,
            "multi_scale": request.multi_scale,
        }

        container_id = training_service.start_training_container(
            model_name=model_name,
            epochs=request.epochs,
            batch_size=request.batch_size,
            project_name=request.project_name,
            extra_args=extra_args,
            model_type=request.model_type,
        )
        return {"status": "success", "container_id": container_id}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/status/{container_id}")
def get_status(container_id: str):
    status = training_service.get_container_status(container_id)
    return {"container_id": container_id, "status": status}


@app.get("/api/logs/{container_id}")
def get_logs(container_id: str):
    logs = training_service.get_container_logs(container_id)
    return {"container_id": container_id, "logs": logs}


@app.get("/api/metrics/{project_name}")
def get_metrics(project_name: str):
    metrics = training_service.get_training_metrics(project_name)
    if not metrics:
        return {"status": "no_data"}
    return {"status": "success", "metrics": metrics}


@app.post("/api/stop/{container_id}")
def stop_train(container_id: str):
    try:
        training_service.stop_training_container(container_id)
        return {"status": "success"}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ── Dataset Management ──────────────────────────────────────────────

# ใช้ /app/dataset ภายใน container (mount จาก docker-compose.yml)
# Fallback เป็น ./dataset ถ้ารันนอก container (local dev)
DATASET_DIR = Path("/app/dataset") if Path("/app").exists() else Path(os.getcwd()).absolute() / "dataset"
DATASET_DIR.mkdir(exist_ok=True, parents=True)


@app.get("/api/datasets")
def list_datasets():
    """List all datasets in the dataset directory with metadata."""
    datasets = []
    if not DATASET_DIR.exists():
        return {"datasets": []}

    for item in DATASET_DIR.iterdir():
        if item.is_dir():
            # Count images
            image_count = 0
            total_size = 0
            classes = []

            for root, dirs, files in os.walk(item):
                for f in files:
                    fp = Path(root) / f
                    total_size += fp.stat().st_size
                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                        image_count += 1

            # Try to read data.yaml for class info
            yaml_path = None
            for root, dirs, files in os.walk(item):
                if "data.yaml" in files:
                    yaml_path = Path(root) / "data.yaml"
                    break

            if yaml_path and yaml_path.exists():
                try:
                    import yaml
                    with open(yaml_path, 'r') as f:
                        data = yaml.safe_load(f)
                    classes = data.get("names", [])
                    if isinstance(classes, dict):
                        classes = list(classes.values())
                except Exception:
                    pass

            # Format size
            if total_size > 1024 * 1024 * 1024:
                size_str = f"{total_size / (1024 * 1024 * 1024):.1f} GB"
            elif total_size > 1024 * 1024:
                size_str = f"{total_size / (1024 * 1024):.1f} MB"
            else:
                size_str = f"{total_size / 1024:.1f} KB"

            # Get creation time
            import time
            created = time.strftime('%Y-%m-%d', time.localtime(item.stat().st_ctime))

            datasets.append({
                "id": item.name,
                "name": item.name,
                "images": image_count,
                "classes": classes,
                "createdAt": created,
                "size": size_str,
            })

    return {"datasets": datasets}


@app.post("/api/upload-dataset")
async def upload_dataset(file: UploadFile = File(...)):
    """
    Upload a dataset as a ZIP file.
    The ZIP will be extracted into backend/dataset/<zip_name>/
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    if not file.filename.lower().endswith('.zip'):
        raise HTTPException(status_code=400, detail="Only .zip files are supported")

    # Create a temp path to save the uploaded zip
    dataset_name = file.filename.rsplit('.', 1)[0]
    # Sanitize the name
    dataset_name = dataset_name.replace(' ', '_').replace('..', '')

    target_dir = DATASET_DIR / dataset_name
    if target_dir.exists():
        # Overwrite existing
        shutil.rmtree(target_dir)

    temp_zip_path = DATASET_DIR / file.filename
    try:
        # Save the uploaded file
        with open(temp_zip_path, 'wb') as buffer:
            while True:
                chunk = await file.read(1024 * 1024)  # 1MB chunks
                if not chunk:
                    break
                buffer.write(chunk)

        # Extract the ZIP
        with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
            zip_ref.extractall(target_dir)

        # Check if there's a single root folder and flatten if needed
        contents = list(target_dir.iterdir())
        if len(contents) == 1 and contents[0].is_dir():
            # Move contents of the single folder up one level
            single_dir = contents[0]
            for item in single_dir.iterdir():
                shutil.move(str(item), str(target_dir / item.name))
            single_dir.rmdir()

        # Validate Dataset Structure (YOLO Format)
        yaml_path = target_dir / "data.yaml"
        dataset_yaml_path = target_dir / "dataset.yaml"
        
        # ต้องมีไฟล์ yaml อยู่หน้าสุด (root) ของโฟลเดอร์รหัส Dataset เลย ไม่ให้ซ่อนอยู่ข้างใน
        yaml_found = yaml_path.is_file() or dataset_yaml_path.is_file()
        images_found = False
        allowed_img_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

        if yaml_found:
            for root, _, files in os.walk(target_dir):
                for f in files:
                    if f.lower().endswith(tuple(allowed_img_exts)):
                        images_found = True
                        break
                if images_found:
                    break

        if not yaml_found:
            raise HTTPException(status_code=400, detail="รูปแบบไม่ถูกต้อง: ไม่พบไฟล์ data.yaml หรือ dataset.yaml ในโฟลเดอร์หลักของ zip (อาจอยู่ลึกเกินไปหรือไม่มีเลย)")
        
        if not images_found:
            raise HTTPException(status_code=400, detail="รูปแบบไม่ถูกต้อง: ไม่พบรูปภาพ (.jpg, .png) ใน Dataset นี้")

        return {"status": "success", "dataset_name": dataset_name}

    except HTTPException:
        # Re-raise HTTPException to preserve 400 errors
        if target_dir.exists():
            shutil.rmtree(target_dir, ignore_errors=True)
        raise
    except zipfile.BadZipFile:
        if target_dir.exists():
            shutil.rmtree(target_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail="ไฟล์ ZIP ไม่ถูกต้องหรือไม่สามารถแตกไฟล์ได้")
    except Exception as e:
        if target_dir.exists():
            shutil.rmtree(target_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temp zip file
        if temp_zip_path.exists():
            temp_zip_path.unlink()


@app.delete("/api/datasets/{dataset_name}")
def delete_dataset(dataset_name: str):
    """Delete a dataset directory."""
    target_dir = DATASET_DIR / dataset_name
    if not target_dir.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")

    try:
        shutil.rmtree(target_dir)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def read_root():
    return {"message": "YOLO Training Backend is running"}
