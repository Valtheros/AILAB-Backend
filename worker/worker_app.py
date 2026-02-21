"""
worker_app.py — CV Training Worker

ไฟล์นี้คือ entry point ของงาน training ทุกประเภท
RQ Worker จะ import function `run_training` จากไฟล์นี้เพื่อรัน job

วิธีเพิ่ม model type ใหม่:
1. สร้าง <model>_trainer.py ใน trainers/ folder
2. เพิ่ม entry ใน TRAINER_REGISTRY ด้านล่าง
"""

import os
import traceback
from pathlib import Path
from typing import Optional

import rq
from rq import get_current_job

# ── Trainer Registry ─────────────────────────────────────────────────────────
# เพิ่ม model type ใหม่ที่นี่เพียงที่เดียว
# key = ชื่อที่ส่งมาจาก backend ใน config["model_type"]
# value = class ของ Trainer ที่ต้องการใช้
def _build_registry():
    from trainers.yolo_trainer import YOLOTrainer
    registry = {
        "yolo": YOLOTrainer,
        # ── ตัวอย่างการเพิ่ม model ใหม่ในอนาคต ──────────────────────────────
        # "efficientdet": EfficientDetTrainer,
        # "rtdetr":       RTDETRTrainer,
        # "detr":         DETRTrainer,
        # "swin":         SwinTrainer,
    }
    return registry


# ── Main Training Function (called by RQ Worker) ──────────────────────────────
def run_training(config: dict) -> dict:
    """
    Function หลักที่ RQ Worker จะเรียกเมื่อมี training job เข้ามา

    Args:
        config: dict ที่มี fields ต่อไปนี้
            - model_type (str):      "yolo" | "efficientdet" | ...
            - model_name (str):      ชื่อ model เช่น "yolo11n", "yolo11s"
            - epochs (int):          จำนวน epochs
            - batch_size (int):      batch size
            - project_name (str):    ชื่อ project สำหรับบันทึกผล
            - data_yaml_path (str):  absolute path ของ data.yaml ใน container
            - extra_args (dict):     config เพิ่มเติมสำหรับ model นั้นๆ

    Returns:
        dict: { status, model_type, project_name, results_dir }
    """
    job = get_current_job()
    runs_dir = Path("/app/runs")
    runs_dir.mkdir(parents=True, exist_ok=True)

    # เตรียม log file ใน runs/<project_name>/train.log
    project_name = config.get("project_name", "train_run")
    log_dir = runs_dir / project_name
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "train.log"

    def log(msg: str):
        print(msg, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
        # อัปเดต job meta เพื่อให้ backend ดึง log ได้ผ่าน RQ
        if job:
            job.meta["log_path"] = str(log_path)
            job.save_meta()

    model_type = config.get("model_type", "yolo").lower()
    log(f"[Worker] Job ID: {job.id if job else 'N/A'}")
    log(f"[Worker] Model type: {model_type}")
    log(f"[Worker] Config: {config}")

    # ── ดึง Trainer จาก Registry ─────────────────────────────────────────────
    registry = _build_registry()
    TrainerClass = registry.get(model_type)

    if TrainerClass is None:
        error_msg = (
            f"Unknown model_type '{model_type}'. "
            f"Available: {list(registry.keys())}"
        )
        log(f"[Worker] ERROR: {error_msg}")
        raise ValueError(error_msg)

    # ── รัน Training ──────────────────────────────────────────────────────────
    try:
        trainer = TrainerClass()
        log(f"[Worker] Starting training with {TrainerClass.__name__}...")
        result = trainer.train(config=config, log_path=log_path)
        log(f"[Worker] Training completed: {result}")
        return result

    except Exception as e:
        error_detail = traceback.format_exc()
        log(f"[Worker] FAILED:\n{error_detail}")
        raise  # Re-raise เพื่อให้ RQ mark job เป็น failed
