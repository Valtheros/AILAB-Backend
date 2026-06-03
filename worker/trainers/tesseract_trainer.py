from __future__ import annotations

import os
import subprocess
from pathlib import Path

from security_utils import validate_slug
from .base_trainer import BaseTrainer
from .trainer_utils import extra


class TesseractTrainer(BaseTrainer):
    def validate_config(self, config: dict) -> None:
        if not config.get("dataset_path"):
            raise ValueError("TesseractTrainer requires dataset_path")

    def train(self, config: dict, log_path: Path | None = None) -> dict:
        self.validate_config(config)
        args = extra(config)
        tesstrain_dir = Path(os.getenv("TESSTRAIN_DIR", "/opt/tesstrain"))
        if not (tesstrain_dir / "Makefile").exists():
            raise RuntimeError(
                f"tesstrain was not found at {tesstrain_dir}. "
                "Use the ocr-worker image or set TESSTRAIN_DIR."
            )

        model_name = str(args.get("model_name", "custom"))
        start_model = str(args.get("start_model", "eng"))
        validate_slug(model_name, "Tesseract model name")
        validate_slug(start_model, "Tesseract start model")
        dataset_path = Path(config["dataset_path"]).resolve()
        output_dir = Path("/app/runs") / config.get("project_name", "train_run")
        output_dir.mkdir(parents=True, exist_ok=True)

        command = [
            "make",
            "-C",
            str(tesstrain_dir),
            "training",
            f"MODEL_NAME={model_name}",
            f"START_MODEL={start_model}",
            f"GROUND_TRUTH_DIR={dataset_path}",
            f"OUTPUT_DIR={output_dir}",
            f"MAX_ITERATIONS={int(args.get('max_iterations', 10000))}",
            f"TARGET_ERROR_RATE={float(args.get('target_error_rate', 0.01))}",
            f"RATIO_TRAIN={float(args.get('ratio_train', 0.9))}",
        ]
        self._write_log(log_path, "[Tesseract] Running: " + " ".join(command))

        with open(log_path or output_dir / "train.log", "a", encoding="utf-8") as log_file:
            process = subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT)
            return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"Tesseract training failed with exit code {return_code}")

        return {
            "status": "completed",
            "task_type": "ocr",
            "model_type": "tesseract",
            "project_name": config.get("project_name", "train_run"),
            "results_dir": str(output_dir),
        }
