from __future__ import annotations

import csv
import os
import random
from pathlib import Path
from typing import Any, Iterable

import yaml


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
MASK_EXTENSIONS = {".png", ".bmp", ".tif", ".tiff"}


def runs_root() -> Path:
    return Path(os.getenv("RUNS_DIR", "/app/runs")).resolve()


def extra(config: dict) -> dict[str, Any]:
    return dict(config.get("extra_args") or {})


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def get_device(device_value: str | None = None):
    import torch

    if device_value and str(device_value).lower() == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        index = str(device_value or "0").split(",")[0]
        return torch.device(f"cuda:{index}")
    return torch.device("cpu")


def read_yaml(data_yaml_path: str | None) -> dict[str, Any]:
    if not data_yaml_path:
        return {}
    path = Path(data_yaml_path)
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def read_classes_from_yaml(data_yaml_path: str | None) -> list[str]:
    data = read_yaml(data_yaml_path)
    if not isinstance(data, dict):
        return []
    names = data.get("names", [])
    if isinstance(names, dict):
        sort_key = lambda item: (0, int(item)) if str(item).isdigit() else (1, str(item))
        return [str(names[key]) for key in sorted(names, key=sort_key)]
    if isinstance(names, list):
        return [str(item) for item in names]
    return []


def split_image_dir(dataset_path: str, split: str) -> Path:
    base = Path(dataset_path)
    candidates = {
        "train": ["train", "training"],
        "val": ["valid", "val", "validation"],
        "test": ["test"],
    }.get(split, [split])
    for candidate in candidates:
        for split_dir in (base / candidate, base / "images" / candidate):
            if not split_dir.exists():
                continue
            images_dir = split_dir / "images"
            return images_dir if images_dir.exists() else split_dir
    return base / candidates[0] / "images"


def list_images(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(path for path in directory.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)


def append_csv_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def optimizer_for(parameters: Iterable, name: str, lr: float, momentum: float, weight_decay: float):
    import torch

    lowered = name.lower()
    if lowered == "adam":
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    if lowered == "adamw":
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    if lowered == "rmsprop":
        return torch.optim.RMSprop(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
    return torch.optim.SGD(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)


def scheduler_for(optimizer, name: str, epochs: int):
    import torch

    lowered = (name or "none").lower()
    if lowered == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    if lowered == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(epochs // 3, 1), gamma=0.1)
    return None


def require_positive_batch_size(value: int, trainer_name: str) -> int:
    if value < 1:
        raise ValueError(f"{trainer_name} requires batch_size >= 1.")
    return value


def collate_detection(batch):
    return tuple(zip(*batch))
