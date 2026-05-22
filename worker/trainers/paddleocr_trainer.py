from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from .base_trainer import BaseTrainer
from .trainer_utils import extra


class PaddleOCRTrainer(BaseTrainer):
    def validate_config(self, config: dict) -> None:
        if not config.get("dataset_path"):
            raise ValueError("PaddleOCRTrainer requires dataset_path")

    def train(self, config: dict, log_path: Path | None = None) -> dict:
        self.validate_config(config)
        args = extra(config)

        paddle_root = Path(os.getenv("PADDLEOCR_ROOT", "/opt/PaddleOCR"))
        train_script = Path(os.getenv("PADDLEOCR_TRAIN_SCRIPT", str(paddle_root / "tools" / "train.py")))
        config_path = Path(str(args.get("config_path", "")))
        if not config_path.is_absolute():
            config_path = paddle_root / config_path

        if not train_script.exists():
            raise RuntimeError(
                f"PaddleOCR training script was not found at {train_script}. "
                "Use the ocr-worker image or set PADDLEOCR_ROOT/PADDLEOCR_TRAIN_SCRIPT."
            )
        if not config_path.exists():
            raise RuntimeError(f"PaddleOCR config file was not found at {config_path}.")

        project_name = config.get("project_name", "train_run")
        output_dir = Path("/app/runs") / project_name
        output_dir.mkdir(parents=True, exist_ok=True)

        overrides = [
            f"Global.epoch_num={int(config.get('epochs', args.get('epochs', 50)))}",
            f"Global.save_model_dir={output_dir}",
            f"Global.eval_batch_step=[0,1000]",
            f"Optimizer.lr.learning_rate={float(args.get('learning_rate', 0.001))}",
        ]
        if args.get("pretrained_model"):
            overrides.append(f"Global.pretrained_model={args['pretrained_model']}")
        if args.get("character_dict_path"):
            overrides.append(f"Global.character_dict_path={args['character_dict_path']}")
        if "use_space_char" in args:
            overrides.append(f"Global.use_space_char={bool(args['use_space_char'])}")
        if "batch_size_per_card" in args:
            overrides.append(f"Train.loader.batch_size_per_card={int(args['batch_size_per_card'])}")

        command = [sys.executable, str(train_script), "-c", str(config_path), "-o", *overrides]
        self._write_log(log_path, "[PaddleOCR] Running: " + " ".join(command))

        with open(log_path or output_dir / "train.log", "a", encoding="utf-8") as log_file:
            process = subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT, cwd=str(paddle_root))
            return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"PaddleOCR training failed with exit code {return_code}")

        return {
            "status": "completed",
            "task_type": "ocr",
            "model_type": "paddleocr",
            "project_name": project_name,
            "results_dir": str(output_dir),
        }
