from __future__ import annotations

import json
import math
from pathlib import Path

from .base_trainer import BaseTrainer
from .trainer_utils import IMAGE_EXTENSIONS, read_yaml, runs_root


def _format_epoch_log(trainer) -> str:
    values = {}
    try:
        values.update(trainer.label_loss_items(trainer.tloss, prefix="train"))
    except (AttributeError, TypeError, ValueError):
        pass
    values.update(getattr(trainer, "metrics", {}) or {})

    metrics = []
    for key, raw_value in values.items():
        key = str(key)
        if not (key.startswith("train/") or key.startswith("metrics/")):
            continue
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            metrics.append(f"{key}={value:.4f}")

    epoch = int(getattr(trainer, "epoch", 0)) + 1
    epochs = int(getattr(trainer, "epochs", epoch))
    suffix = " " + " ".join(metrics) if metrics else ""
    return f"[YOLO] epoch={epoch}/{epochs}{suffix}"


F1_COLUMN = "metrics/f1(B)"
PRECISION_COLUMN = "metrics/precision(B)"
RECALL_COLUMN = "metrics/recall(B)"


def _f1(precision: float, recall: float) -> float:
    """Harmonic mean of precision and recall; 0 when both are 0 (early epochs)."""
    total = precision + recall
    return 0.0 if total <= 0 else 2.0 * precision * recall / total


def _append_f1_column(results_csv: Path) -> int:
    """Add metrics/f1(B) to a finished ultralytics results.csv.

    Ultralytics already logs precision and recall per epoch, so F1 is derived
    from the numbers on disk rather than re-running validation. Existing columns
    are preserved and the file is rewritten only when the F1 column is missing.
    Returns the number of rows updated (0 when there is nothing to do).
    """
    import csv

    if not results_csv.is_file():
        return 0
    with results_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = [name.strip() for name in (reader.fieldnames or [])]
        rows = [{(key or "").strip(): (value or "").strip() for key, value in row.items()} for row in reader]
    if not rows or PRECISION_COLUMN not in fieldnames or RECALL_COLUMN not in fieldnames:
        return 0
    if F1_COLUMN in fieldnames:
        return 0

    for row in rows:
        try:
            precision = float(row.get(PRECISION_COLUMN, "") or 0.0)
            recall = float(row.get(RECALL_COLUMN, "") or 0.0)
        except (TypeError, ValueError):
            row[F1_COLUMN] = ""
            continue
        row[F1_COLUMN] = f"{_f1(precision, recall):.5f}"

    # Keep F1 next to the precision/recall it is derived from.
    ordered = list(fieldnames)
    ordered.insert(ordered.index(RECALL_COLUMN) + 1, F1_COLUMN)
    with results_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def _class_names(cfg: dict) -> list:
    names = cfg.get("names")
    if isinstance(names, dict):
        return [names[key] for key in sorted(names)]
    if isinstance(names, list):
        return list(names)
    return []


def _resolve_test_dir(data_yaml: str, cfg: dict) -> Path | None:
    """Locate the test split declared in data.yaml, if any. Returns None when the
    dataset has no test split (the held-out evaluation is then skipped)."""
    test_ref = cfg.get("test")
    if not test_ref:
        return None
    base = Path(cfg["path"]) if cfg.get("path") else Path(data_yaml).parent
    candidate = (base / str(test_ref)).resolve()
    return candidate if candidate.exists() else None


def _count_images(path: Path) -> int:
    if path.is_file():  # a .txt listing of image paths
        try:
            return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
        except OSError:
            return 0
    return sum(1 for item in path.rglob("*") if item.suffix.lower() in IMAGE_EXTENSIONS)


def _evaluate_yolo_test_split(best_weights: Path, data_yaml: str, results_dir: str, logger, log_path) -> dict | None:
    """One-shot held-out test evaluation, run once AFTER training completes (not
    part of the training loop). Skips cleanly when the dataset has no test split.
    Mirrors the classification test_evaluation.json contract with detection
    metrics (mAP) so the UI can surface it the same way."""
    from ultralytics import YOLO

    cfg = read_yaml(data_yaml)
    test_dir = _resolve_test_dir(data_yaml, cfg)
    if test_dir is None:
        logger(log_path, "[YOLOTrainer] No test split in data.yaml; skipping held-out test evaluation.")
        return None

    logger(log_path, "[YOLOTrainer] Evaluating best.pt on the held-out test split...")
    model = YOLO(str(best_weights))
    metrics = model.val(
        data=data_yaml,
        split="test",
        project=str(results_dir),
        name="testeval",
        exist_ok=True,
        verbose=False,
        plots=False,
    )
    box = getattr(metrics, "box", None)
    classes = _class_names(cfg)
    result = {
        "task_type": "object_detection",
        "model_type": "yolo",
        "checkpoint": "best.pt",
        "test_images": _count_images(test_dir),
        "test_map50": round(float(box.map50), 6) if box is not None else None,
        "test_map50_95": round(float(box.map), 6) if box is not None else None,
        "test_precision": round(float(box.mp), 6) if box is not None else None,
        "test_recall": round(float(box.mr), 6) if box is not None else None,
        "num_classes": len(classes),
        "classes": classes,
    }
    (Path(results_dir) / "test_evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger(
        log_path,
        f"[YOLOTrainer] Held-out test mAP50={result['test_map50']} mAP50-95={result['test_map50_95']} "
        f"on {result['test_images']} images (from best.pt)",
    )
    return result


class YOLOTrainer(BaseTrainer):
    SUPPORTED_TASKS = {"detect", "segment", "pose", "classify"}

    def validate_config(self, config: dict) -> None:
        if not config.get("data_yaml_path"):
            raise ValueError("YOLOTrainer requires data_yaml_path")
        if not config.get("model_name"):
            raise ValueError("YOLOTrainer requires model_name, for example yolo11n")
        task = config.get("task", "detect")
        if task not in self.SUPPORTED_TASKS:
            raise ValueError(f"Unsupported YOLO task '{task}'. Must be one of {sorted(self.SUPPORTED_TASKS)}")

    def train(self, config: dict, log_path: Path | None = None) -> dict:
        from ultralytics import YOLO

        self.validate_config(config)

        model_name = str(config["model_name"]).removesuffix(".pt")
        data_yaml = config["data_yaml_path"]
        epochs = int(config.get("epochs", 10))
        batch_size = int(config.get("batch_size", 16))
        project_name = config.get("project_name", "train_run")
        extra_args = dict(config.get("extra_args", {}))
        extra_args.pop("model_size", None)
        extra_args.pop("epochs", None)
        extra_args.pop("batch_size", None)

        self._write_log(log_path, f"[YOLOTrainer] model={model_name}, epochs={epochs}, batch={batch_size}")
        self._write_log(log_path, f"[YOLOTrainer] data={data_yaml}")
        self._write_log(log_path, f"[YOLOTrainer] Loading {model_name}.pt; the first run may download or initialize model weights.")

        model = YOLO(f"{model_name}.pt")

        def log_epoch(trainer) -> None:
            self._write_log(log_path, _format_epoch_log(trainer))

        model.add_callback("on_fit_epoch_end", log_epoch)
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch_size,
            project=str(runs_root()),
            name=project_name,
            exist_ok=True,
            task=config.get("task", "detect"),
            **extra_args,
        )

        fallback_dir = runs_root() / project_name
        results_dir = str(results.save_dir) if results and hasattr(results, "save_dir") else str(fallback_dir)
        self._write_log(log_path, f"[YOLOTrainer] results={results_dir}")

        # Derive the F1 column from the precision/recall ultralytics already
        # logged. Runs after training, leaves every existing column intact.
        try:
            updated = _append_f1_column(Path(results_dir) / "results.csv")
            if updated:
                self._write_log(log_path, f"[YOLOTrainer] Added {F1_COLUMN} for {updated} epochs in results.csv")
        except Exception as exc:
            self._write_log(log_path, f"[YOLOTrainer] F1 column skipped: {exc}")

        # Optional one-shot held-out test evaluation on best.pt. Runs after the
        # training loop, never inside it, and a failure here must not fail an
        # otherwise-completed run.
        best_weights = Path(results_dir) / "weights" / "best.pt"
        if best_weights.is_file():
            try:
                _evaluate_yolo_test_split(best_weights, data_yaml, results_dir, self._write_log, log_path)
            except Exception as exc:
                self._write_log(log_path, f"[YOLOTrainer] Held-out test evaluation skipped: {exc}")

        return {
            "status": "completed",
            "task_type": config.get("task_type", "object_detection"),
            "model_type": "yolo",
            "project_name": project_name,
            "results_dir": results_dir,
        }
