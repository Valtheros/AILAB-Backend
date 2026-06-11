from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable

import yaml


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
MASK_EXTENSIONS = {".png", ".bmp", ".tif", ".tiff"}
PADDLEOCR_TRAIN_LABEL_NAMES = {
    "rec_gt_train.txt",
    "det_gt_train.txt",
}


def safe_dataset_name(filename: str) -> str:
    name = filename.rsplit(".", 1)[0]
    return "".join(ch if ch.isascii() and (ch.isalnum() or ch in ("-", "_", ".")) else "_" for ch in name).strip("._") or "dataset"


def format_bytes(size: int) -> str:
    if size > 1024 * 1024 * 1024:
        return f"{size / (1024 * 1024 * 1024):.1f} GB"
    if size > 1024 * 1024:
        return f"{size / (1024 * 1024):.1f} MB"
    return f"{size / 1024:.1f} KB"


def count_images(dataset_dir: Path) -> int:
    total = 0
    for root, _, files in os.walk(dataset_dir):
        total += sum(1 for file in files if Path(file).suffix.lower() in IMAGE_EXTENSIONS)
    return total


def directory_size(dataset_dir: Path) -> int:
    total = 0
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            path = Path(root) / file
            try:
                total += path.stat().st_size
            except OSError:
                pass
    return total


def find_dataset_yaml(dataset_dir: Path) -> Path | None:
    for name in ("data.yaml", "dataset.yaml"):
        candidate = dataset_dir / name
        if candidate.is_file():
            return candidate
    for root, _, files in os.walk(dataset_dir):
        for name in ("data.yaml", "dataset.yaml"):
            if name in files:
                return Path(root) / name
    return None


def read_yaml_classes(yaml_path: Path | None) -> list[str]:
    if not yaml_path or not yaml_path.exists():
        return []
    try:
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return []
    if not isinstance(data, dict):
        return []

    names = data.get("names", [])
    if isinstance(names, dict):
        sort_key = lambda item: (0, int(item)) if str(item).isdigit() else (1, str(item))
        return [str(names[k]) for k in sorted(names, key=sort_key)]
    if isinstance(names, list):
        return [str(item) for item in names]
    return []


def _iter_yolo_label_files(dataset_dir: Path) -> Iterable[Path]:
    for label_path in dataset_dir.rglob("*.txt"):
        if not label_path.is_file():
            continue
        try:
            relative_parts = {part.lower() for part in label_path.relative_to(dataset_dir).parts}
        except ValueError:
            continue
        if "labels" in relative_parts:
            yield label_path


def _inspect_yolo_labels(dataset_dir: Path) -> tuple[bool, bool]:
    has_label_file = False
    has_segmentation_row = False

    for label_path in _iter_yolo_label_files(dataset_dir):
        has_label_file = True
        try:
            lines = label_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue
        for line in lines:
            if len(line.strip().split()) > 5:
                has_segmentation_row = True

    return has_label_file, has_segmentation_row


def _imagefolder_classes(dataset_dir: Path) -> list[str]:
    for split_name in ("train", "training"):
        split_dir = dataset_dir / split_name
        if not split_dir.is_dir():
            continue
        classes = []
        for item in split_dir.iterdir():
            if item.is_dir() and any(path.suffix.lower() in IMAGE_EXTENSIONS for path in item.rglob("*")):
                classes.append(item.name)
        if classes:
            return sorted(classes)
    return []


def _has_semantic_masks(dataset_dir: Path) -> bool:
    split_names = ("train", "training", "valid", "val", "validation", "test")
    image_dir_names = ("images", "image")
    mask_dir_names = ("masks", "mask", "labels", "label")
    for split_name in split_names:
        split_dir = dataset_dir / split_name
        if not split_dir.is_dir():
            continue
        has_images = any((split_dir / name).is_dir() for name in image_dir_names)
        mask_dirs = [split_dir / name for name in mask_dir_names if (split_dir / name).is_dir()]
        has_masks = any(
            any(path.suffix.lower() in MASK_EXTENSIONS for path in mask_dir.rglob("*"))
            for mask_dir in mask_dirs
        )
        if has_images and has_masks:
            return True
    return False


def _find_coco_files(dataset_dir: Path) -> list[str]:
    patterns = ("*.json",)
    files: list[str] = []
    for pattern in patterns:
        for path in dataset_dir.rglob(pattern):
            lowered = path.name.lower()
            if lowered.startswith("_annotations") or "instances" in lowered or "coco" in lowered:
                files.append(str(path.relative_to(dataset_dir)))
    return sorted(files)


def _has_paddleocr_labels(dataset_dir: Path) -> bool:
    for path in dataset_dir.rglob("*"):
        if not path.is_file():
            continue
        lowered = path.name.lower()
        if lowered in PADDLEOCR_TRAIN_LABEL_NAMES:
            return True
    return False


def _has_tesseract_ground_truth(dataset_dir: Path) -> bool:
    return any(path.is_file() and path.name.lower().endswith(".gt.txt") for path in dataset_dir.rglob("*"))


def inspect_dataset(dataset_dir: Path) -> dict[str, Any]:
    yaml_path = find_dataset_yaml(dataset_dir)
    classes = read_yaml_classes(yaml_path)
    imagefolder_classes = _imagefolder_classes(dataset_dir)
    if imagefolder_classes:
        classes = imagefolder_classes

    formats: list[str] = []
    tasks: list[str] = []

    has_images = count_images(dataset_dir) > 0
    has_yolo_yaml = yaml_path is not None
    has_yolo_labels, has_yolo_segmentation = _inspect_yolo_labels(dataset_dir)
    has_semantic_masks = _has_semantic_masks(dataset_dir)
    coco_files = _find_coco_files(dataset_dir)
    has_paddleocr_labels = _has_paddleocr_labels(dataset_dir)
    has_tesseract_ground_truth = _has_tesseract_ground_truth(dataset_dir)

    if imagefolder_classes:
        formats.append("imagefolder")
        tasks.append("image_classification")
    if has_yolo_yaml and has_yolo_labels and not has_yolo_segmentation:
        formats.append("yolo_detection")
        tasks.append("object_detection")
    if has_yolo_yaml and has_yolo_segmentation:
        formats.append("yolo_segmentation")
        tasks.append("segmentation")
    if has_semantic_masks:
        formats.append("semantic_masks")
        tasks.append("segmentation")
    if coco_files:
        formats.append("coco_instances")
        tasks.extend(["object_detection", "segmentation"])
    if has_paddleocr_labels:
        formats.append("paddleocr_labels")
        tasks.append("ocr")
    if has_tesseract_ground_truth:
        formats.append("tesseract_ground_truth")
        tasks.append("ocr")

    formats = sorted(set(formats))
    tasks = sorted(set(tasks))
    warnings = []
    if has_images and not formats:
        warnings.append("Images were found, but no supported annotation structure was detected.")
    if not has_images:
        warnings.append("No image files were found.")

    return {
        "formats": formats,
        "tasks": tasks,
        "classes": classes,
        "image_count": count_images(dataset_dir),
        "size_bytes": directory_size(dataset_dir),
        "yaml_path": str(yaml_path) if yaml_path else None,
        "coco_files": coco_files,
        "warnings": warnings,
    }


def validate_dataset_for_upload(dataset_dir: Path) -> dict[str, Any]:
    metadata = inspect_dataset(dataset_dir)
    if metadata["image_count"] == 0:
        raise ValueError("No supported image files were found in the uploaded dataset.")
    if not metadata["formats"]:
        raise ValueError(
            "Unsupported dataset structure. Supported formats: YOLO data.yaml + labels, "
            "ImageFolder train/<class>, semantic masks, COCO instances, PaddleOCR labels, "
            "or Tesseract .gt.txt ground truth."
        )
    return metadata
