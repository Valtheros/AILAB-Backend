from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
MASK_EXTENSIONS = {".png", ".bmp", ".tif", ".tiff"}
OCR_LABEL_NAMES = {
    "label.txt",
    "labels.txt",
    "train.txt",
    "val.txt",
    "valid.txt",
    "test.txt",
    "rec_gt_train.txt",
    "rec_gt_val.txt",
    "det_gt_train.txt",
    "det_gt_val.txt",
}


def safe_dataset_name(filename: str) -> str:
    name = filename.rsplit(".", 1)[0]
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in name).strip("._") or "dataset"


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

    names = data.get("names", [])
    if isinstance(names, dict):
        return [str(names[k]) for k in sorted(names, key=lambda item: int(item) if str(item).isdigit() else str(item))]
    if isinstance(names, list):
        return [str(item) for item in names]
    return []


def _has_yolo_labels(dataset_dir: Path) -> bool:
    for root, _, files in os.walk(dataset_dir):
        root_path = Path(root)
        if root_path.name != "labels":
            continue
        if any(Path(file).suffix.lower() == ".txt" for file in files):
            return True
    return False


def _has_yolo_segmentation_labels(dataset_dir: Path) -> bool:
    for root, _, files in os.walk(dataset_dir):
        root_path = Path(root)
        if root_path.name != "labels":
            continue
        for file in files:
            if Path(file).suffix.lower() != ".txt":
                continue
            label_path = root_path / file
            try:
                for line in label_path.read_text(encoding="utf-8").splitlines():
                    if len(line.strip().split()) > 5:
                        return True
            except Exception:
                continue
    return False


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


def _has_ocr_labels(dataset_dir: Path) -> bool:
    for path in dataset_dir.rglob("*"):
        if not path.is_file():
            continue
        lowered = path.name.lower()
        if lowered in OCR_LABEL_NAMES or lowered.endswith(".gt.txt"):
            return True
    return False


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
    has_yolo_labels = _has_yolo_labels(dataset_dir)
    has_yolo_segmentation = _has_yolo_segmentation_labels(dataset_dir)
    has_semantic_masks = _has_semantic_masks(dataset_dir)
    coco_files = _find_coco_files(dataset_dir)
    has_ocr_labels = _has_ocr_labels(dataset_dir)

    if imagefolder_classes:
        formats.append("imagefolder")
        tasks.append("image_classification")
    if has_yolo_yaml and has_yolo_labels:
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
    if has_ocr_labels:
        formats.extend(["paddleocr_labels", "tesseract_ground_truth"])
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
