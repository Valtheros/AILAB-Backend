from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

from security_utils import contained_path, validate_slug
from .base_trainer import BaseTrainer
from .trainer_utils import extra, runs_root


class PaddleOCRTrainer(BaseTrainer):
    def validate_config(self, config: dict) -> None:
        if not config.get("dataset_path"):
            raise ValueError("PaddleOCRTrainer requires dataset_path")

    def train(self, config: dict, log_path: Path | None = None) -> dict:
        self.validate_config(config)
        args = extra(config)

        paddle_root = Path(os.getenv("PADDLEOCR_ROOT", "/opt/PaddleOCR")).resolve()
        dataset_path = Path(config["dataset_path"]).resolve()
        allowed_roots = self._allowed_roots(paddle_root, dataset_path)
        train_script = self._resolve_required_path(
            os.getenv("PADDLEOCR_TRAIN_SCRIPT", str(paddle_root / "tools" / "train.py")),
            paddle_root,
            allowed_roots,
            "PaddleOCR training script",
        )
        config_path = self._resolve_required_path(
            str(args.get("config_path", "")),
            paddle_root,
            allowed_roots,
            "PaddleOCR config file",
        )

        if not train_script.exists():
            raise RuntimeError(
                f"PaddleOCR training script was not found at {train_script}. "
                "Use the ocr-worker image or set PADDLEOCR_ROOT/PADDLEOCR_TRAIN_SCRIPT."
            )
        if not config_path.exists():
            raise RuntimeError(f"PaddleOCR config file was not found at {config_path}.")

        project_name = str(config.get("project_name", "train_run"))
        validate_slug(project_name, "project name")
        output_dir = contained_path(runs_root(), project_name)
        output_dir.mkdir(parents=True, exist_ok=True)

        overrides = self._build_overrides(config, args, output_dir)
        command = [sys.executable, str(train_script), "-c", str(config_path), "-o", *overrides]
        self._write_log(log_path, "[PaddleOCR] Running: " + shlex.join(command))

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

    def _allowed_roots(self, paddle_root: Path, dataset_path: Path) -> list[Path]:
        roots = [paddle_root.resolve(), dataset_path.resolve()]
        for raw_root in os.getenv("PADDLEOCR_ALLOWED_PATHS", "").split(os.pathsep):
            if raw_root.strip():
                roots.append(Path(raw_root).resolve())
        return roots

    def _resolve_contained_path(self, value: str, default_root: Path, allowed_roots: list[Path], label: str) -> Path:
        raw_value = str(value or "").strip()
        if not raw_value:
            raise ValueError(f"{label} is required")
        raw_path = Path(raw_value)
        if raw_path.is_absolute():
            candidate = raw_path.resolve()
        else:
            candidate = contained_path(default_root, raw_path)
        if not any(candidate.is_relative_to(root) for root in allowed_roots):
            allowed = ", ".join(root.as_posix() for root in allowed_roots)
            raise ValueError(f"{label} must be inside an allowed directory: {allowed}")
        return candidate

    def _resolve_required_path(self, value: str, default_root: Path, allowed_roots: list[Path], label: str) -> Path:
        candidate = self._resolve_contained_path(value, default_root, allowed_roots, label)
        if not candidate.exists():
            raise RuntimeError(f"{label} was not found at {candidate}.")
        return candidate

    def _resolve_optional_path(self, value: object, default_root: Path, allowed_roots: list[Path], label: str) -> Path | None:
        raw_value = str(value or "").strip()
        if not raw_value:
            return None
        return self._resolve_contained_path(raw_value, default_root, allowed_roots, label)

    def _override_path(self, path: Path | str) -> str:
        return Path(path).resolve().as_posix()

    def _override_list(self, path: Path | str) -> str:
        escaped = self._override_path(path).replace("'", "\\'")
        return f"['{escaped}']"

    def _build_overrides(self, config: dict, args: dict, output_dir: Path) -> list[str]:
        dataset_path = Path(config["dataset_path"]).resolve()
        paddle_root = Path(os.getenv("PADDLEOCR_ROOT", "/opt/PaddleOCR")).resolve()
        allowed_roots = self._allowed_roots(paddle_root, dataset_path)
        task = str(args.get("ocr_task", "rec"))
        if task not in {"det", "rec"}:
            raise ValueError("PaddleOCR ocr_task must be either 'det' or 'rec'")
        prefix = task
        train_labels = sorted(dataset_path.rglob(f"{prefix}_gt_train.txt"))
        val_labels = sorted(dataset_path.rglob(f"{prefix}_gt_val.txt"))
        if not train_labels:
            raise ValueError(f"PaddleOCR {task} training requires {prefix}_gt_train.txt")

        overrides = [
            f"Global.epoch_num={int(config.get('epochs', args.get('epochs', 50)))}",
            f"Global.save_model_dir={self._override_path(output_dir)}",
            f"Global.eval_batch_step=[0,1000]",
            f"Optimizer.lr.learning_rate={float(args.get('learning_rate', 0.001))}",
            f"Global.max_text_length={int(args.get('max_text_length', 25))}",
            f"Train.dataset.data_dir={self._override_path(dataset_path)}",
            f"Train.dataset.label_file_list={self._override_list(train_labels[0])}",
        ]
        if val_labels:
            overrides.extend(
                [
                    f"Eval.dataset.data_dir={self._override_path(dataset_path)}",
                    f"Eval.dataset.label_file_list={self._override_list(val_labels[0])}",
                ]
            )
        pretrained_model = self._resolve_optional_path(
            args.get("pretrained_model"),
            paddle_root,
            allowed_roots,
            "PaddleOCR pretrained model path",
        )
        if pretrained_model:
            overrides.append(f"Global.pretrained_model={self._override_path(pretrained_model)}")
        character_dict_path = self._resolve_optional_path(
            args.get("character_dict_path"),
            paddle_root,
            allowed_roots,
            "PaddleOCR character dictionary path",
        )
        if character_dict_path:
            overrides.append(f"Global.character_dict_path={self._override_path(character_dict_path)}")
        if "use_space_char" in args:
            overrides.append(f"Global.use_space_char={bool(args['use_space_char'])}")
        if "batch_size_per_card" in args:
            overrides.append(f"Train.loader.batch_size_per_card={int(args['batch_size_per_card'])}")

        return overrides
