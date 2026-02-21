from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional


class BaseTrainer(ABC):
    """
    Abstract base class สำหรับทุก Computer Vision Trainer.

    วิธีเพิ่ม model ใหม่:
    1. สร้าง file ใหม่ เช่น `efficientdet_trainer.py`
    2. สร้าง class ที่ inherit จาก BaseTrainer
    3. implement method `train(config)` และ `validate_config(config)` ให้ครบ
    4. เพิ่ม entry ใน `TRAINER_REGISTRY` ใน worker_app.py
    """

    @abstractmethod
    def train(self, config: dict, log_path: Optional[Path] = None) -> dict:
        """
        รัน training ด้วย config ที่กำหนด

        Args:
            config: Training configuration dict
            log_path: Path to write logs to (optional)

        Returns:
            dict with keys: status, results_dir, metrics (optional)
        """
        pass

    @abstractmethod
    def validate_config(self, config: dict) -> None:
        """
        ตรวจสอบ config ก่อน training — raise ValueError if invalid.
        """
        pass

    def _write_log(self, log_path: Optional[Path], message: str) -> None:
        """Helper: เขียน log ลง file พร้อม print ด้วย"""
        print(message, flush=True)
        if log_path:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(message + "\n")
