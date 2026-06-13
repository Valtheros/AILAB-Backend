from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Iterable

import yaml


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
MASK_EXTENSIONS = {".png", ".bmp", ".tif", ".tiff"}
PADDLEOCR_TRAIN_LABEL_NAMES = {
    "rec_gt_train.txt": "rec",
    "rec_gt_val.txt": "rec",
    "det_gt_train.txt": "det",
    "det_gt_val.txt": "det",
}
NORMALIZED_DIR_NAME = ".ailab_normalized"
TRAINABLE_FORMATS = {
    "imagefolder",
    "yolo_detection",
    "semantic_masks",
    "coco_instances",
    "paddleocr_labels",
    "tesseract_ground_truth",
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


def _iter_visible_files(dataset_dir: Path):
    for root, dirs, files in os.walk(dataset_dir):
        dirs[:] = [name for name in dirs if name != NORMALIZED_DIR_NAME]
        for file in files:
            yield Path(root) / file


def count_images(dataset_dir: Path) -> int:
    return sum(1 for path in _iter_visible_files(dataset_dir) if path.suffix.lower() in IMAGE_EXTENSIONS)


def directory_size(dataset_dir: Path) -> int:
    total = 0
    for path in _iter_visible_files(dataset_dir):
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
    normalized_candidate = dataset_dir / NORMALIZED_DIR_NAME / "yolo_detection" / "data.yaml"
    if normalized_candidate.is_file():
        return normalized_candidate
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


def _safe_child(base: Path, *parts: str | Path) -> Path | None:
    base_resolved = base.resolve()
    try:
        candidate = base_resolved.joinpath(*(str(part) for part in parts)).resolve()
    except (OSError, RuntimeError):
        return None
    if candidate == base_resolved or base_resolved in candidate.parents:
        return candidate
    return None


def _is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


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


def _yolo_row_kind(parts: list[str], class_count: int) -> str:
    if len(parts) < 5:
        return "invalid"
    try:
        raw_class_id = float(parts[0])
        class_id = int(raw_class_id)
        coordinates = [float(value) for value in parts[1:]]
    except ValueError:
        return "invalid"
    if raw_class_id != class_id or class_id < 0:
        return "invalid"
    if class_count and class_id >= class_count:
        return "invalid"
    if len(parts) == 5:
        x_center, y_center, width, height = coordinates
        if width <= 0 or height <= 0:
            return "invalid"
        if all(0 <= value <= 1 for value in coordinates):
            return "box"
        return "invalid"
    if len(coordinates) >= 6 and len(coordinates) % 2 == 0 and all(0 <= value <= 1 for value in coordinates):
        return "polygon"
    return "invalid"


def _inspect_yolo_labels(dataset_dir: Path, class_count: int = 0) -> dict[str, int]:
    stats = {"files": 0, "box_rows": 0, "polygon_rows": 0, "invalid_rows": 0}

    for label_path in _iter_yolo_label_files(dataset_dir):
        stats["files"] += 1
        try:
            lines = label_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            kind = _yolo_row_kind(parts, class_count)
            if kind == "box":
                stats["box_rows"] += 1
            elif kind == "polygon":
                stats["polygon_rows"] += 1
            else:
                stats["invalid_rows"] += 1

    return stats


def _imagefolder_classes(dataset_dir: Path) -> list[str]:
    reserved_dir_names = {"image", "images", "label", "labels", "mask", "masks", "annotations"}
    for split_name in ("train", "training"):
        split_dir = dataset_dir / split_name
        if not split_dir.is_dir():
            continue
        if (split_dir / "images").is_dir() and any((split_dir / name).is_dir() for name in ("labels", "masks")):
            continue
        classes = []
        for item in split_dir.iterdir():
            if item.name.lower() in reserved_dir_names:
                continue
            if item.is_dir() and any(path.suffix.lower() in IMAGE_EXTENSIONS for path in item.rglob("*")):
                classes.append(item.name)
        if classes:
            return sorted(classes)
    return []


def _split_candidates(split: str) -> list[str]:
    return {
        "train": ["train", "training"],
        "val": ["valid", "val", "validation"],
        "test": ["test"],
    }.get(split, [split])


def _semantic_split_dirs(dataset_dir: Path, split: str) -> tuple[Path, Path] | None:
    for split_name in _split_candidates(split):
        split_dir = dataset_dir / split_name
        if not split_dir.exists():
            continue
        images_dir = split_dir / "images"
        if not images_dir.exists():
            images_dir = split_dir / "image"
        for mask_name in ("masks", "mask", "labels", "label"):
            masks_dir = split_dir / mask_name
            if images_dir.exists() and masks_dir.exists():
                return images_dir, masks_dir
    return None


def _matching_mask_path(masks_dir: Path, image_path: Path) -> Path | None:
    for extension in MASK_EXTENSIONS:
        candidate = masks_dir / f"{image_path.stem}{extension}"
        if candidate.exists():
            return candidate
    for path in masks_dir.rglob(f"{image_path.stem}.*"):
        if path.suffix.lower() in MASK_EXTENSIONS:
            return path
    return None


def _inspect_semantic_masks(dataset_dir: Path) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "train_images": 0,
        "mask_files": 0,
        "missing_masks": [],
        "splits": [],
    }
    for split in ("train", "val", "test"):
        split_dirs = _semantic_split_dirs(dataset_dir, split)
        if split_dirs is None:
            continue
        images_dir, masks_dir = split_dirs
        images = [path for path in images_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS]
        mask_files = [path for path in masks_dir.rglob("*") if path.suffix.lower() in MASK_EXTENSIONS]
        if not mask_files:
            continue
        if images:
            stats["splits"].append(split)
        if split == "train":
            stats["train_images"] = len(images)
        stats["mask_files"] += len(mask_files)
        for image_path in images:
            if _matching_mask_path(masks_dir, image_path) is None:
                try:
                    stats["missing_masks"].append(str(image_path.relative_to(dataset_dir)))
                except ValueError:
                    stats["missing_masks"].append(image_path.name)
    return stats


def _find_coco_files(dataset_dir: Path) -> list[str]:
    files: list[str] = []
    for path in dataset_dir.rglob("*.json"):
        lowered = path.name.lower()
        if lowered.startswith("_annotations") or "instances" in lowered or "coco" in lowered:
            files.append(str(path.relative_to(dataset_dir)))
    return sorted(files)


def _annotation_split(annotation_path: Path, dataset_dir: Path) -> str:
    try:
        parts = [part.lower() for part in annotation_path.relative_to(dataset_dir).parts]
    except ValueError:
        parts = []
    for part in parts:
        if part in {"train", "training"}:
            return "train"
        if part in {"valid", "val", "validation"}:
            return "val"
        if part == "test":
            return "test"
    return annotation_path.parent.name.lower() if annotation_path.parent != dataset_dir else "train"


def _resolve_coco_image(dataset_dir: Path, annotation_path: Path, file_name: Any) -> Path | None:
    if not isinstance(file_name, str) or not file_name.strip():
        return None
    normalized = file_name.strip().replace("\\", "/")
    variants = [normalized]
    stripped = normalized
    while stripped.startswith("../"):
        stripped = stripped[3:]
        variants.append(stripped)
    if stripped.startswith("./"):
        variants.append(stripped[2:])

    split = _annotation_split(annotation_path, dataset_dir)
    for variant in dict.fromkeys(variants):
        candidates = [
            _safe_child(dataset_dir, variant),
            _safe_child(annotation_path.parent, variant),
            _safe_child(dataset_dir, "images", variant),
            _safe_child(dataset_dir, split, variant),
            _safe_child(dataset_dir, split, "images", variant),
            _safe_child(dataset_dir, "train", "images", variant),
            _safe_child(dataset_dir, "valid", "images", variant),
            _safe_child(dataset_dir, "val", "images", variant),
        ]
        for candidate in candidates:
            if candidate and _is_image_file(candidate):
                return candidate
    basename = Path(stripped).name
    if not basename:
        return None
    for match in dataset_dir.rglob(basename):
        if _is_image_file(match):
            return match
    return None


def _valid_bbox(bbox: Any) -> bool:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return False
    try:
        _x, _y, width, height = [float(value) for value in bbox]
    except (TypeError, ValueError):
        return False
    return width > 0 and height > 0


def _valid_coco_segmentation(segmentation: Any) -> bool:
    if isinstance(segmentation, list):
        for polygon in segmentation:
            if not isinstance(polygon, list) or len(polygon) < 6 or len(polygon) % 2 != 0:
                continue
            try:
                [float(value) for value in polygon]
            except (TypeError, ValueError):
                continue
            return True
        return False
    if isinstance(segmentation, dict):
        return "counts" in segmentation and "size" in segmentation
    return False


def _inspect_coco_files(dataset_dir: Path) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "files": _find_coco_files(dataset_dir),
        "valid_files": [],
        "missing_images": 0,
        "invalid_annotations": 0,
        "box_annotations": 0,
        "mask_annotations": 0,
        "classes": [],
    }
    class_names: dict[Any, str] = {}
    for relative_file in stats["files"]:
        annotation_path = dataset_dir / relative_file
        try:
            data = json.loads(annotation_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        images = data.get("images", [])
        annotations = data.get("annotations", [])
        categories = data.get("categories", [])
        if not isinstance(images, list) or not isinstance(annotations, list):
            continue
        if isinstance(categories, list):
            for category in categories:
                if isinstance(category, dict) and "id" in category:
                    class_names[category.get("id")] = str(category.get("name", category.get("id")))

        found_image_ids = set()
        for image in images:
            if not isinstance(image, dict):
                continue
            image_id = image.get("id")
            image_path = _resolve_coco_image(dataset_dir, annotation_path, image.get("file_name"))
            if image_path is None:
                stats["missing_images"] += 1
                continue
            found_image_ids.add(image_id)

        file_box_annotations = 0
        file_mask_annotations = 0
        for annotation in annotations:
            if not isinstance(annotation, dict):
                stats["invalid_annotations"] += 1
                continue
            if annotation.get("image_id") not in found_image_ids:
                continue
            has_bbox = _valid_bbox(annotation.get("bbox"))
            if has_bbox:
                stats["box_annotations"] += 1
                file_box_annotations += 1
            elif annotation.get("bbox") is not None:
                stats["invalid_annotations"] += 1
            if has_bbox and _valid_coco_segmentation(annotation.get("segmentation")):
                stats["mask_annotations"] += 1
                file_mask_annotations += 1
        if found_image_ids and (file_box_annotations or file_mask_annotations):
            stats["valid_files"].append(relative_file)

    if class_names:
        sort_key = lambda item: (0, int(item[0])) if str(item[0]).isdigit() else (1, str(item[0]))
        stats["classes"] = [name for _category_id, name in sorted(class_names.items(), key=sort_key)]
    return stats


def _inspect_paddleocr_labels(dataset_dir: Path) -> dict[str, Any]:
    stats: dict[str, Any] = {"tasks": [], "files": [], "missing_image_refs": 0}
    tasks = set()
    for path in dataset_dir.rglob("*"):
        if not path.is_file():
            continue
        task = PADDLEOCR_TRAIN_LABEL_NAMES.get(path.name.lower())
        if not task:
            continue
        tasks.add(task)
        stats["files"].append(str(path.relative_to(dataset_dir)))
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            image_ref = stripped.split("\t", 1)[0].strip()
            if not image_ref:
                continue
            normalized_ref = image_ref.replace("\\", "/")
            candidate = _safe_child(path.parent, normalized_ref)
            if candidate is None or not _is_image_file(candidate):
                candidate = _safe_child(dataset_dir, normalized_ref)
            if candidate is None or not _is_image_file(candidate):
                stats["missing_image_refs"] += 1
    stats["tasks"] = sorted(tasks)
    return stats


def _find_tesseract_image_for_gt(gt_path: Path) -> Path | None:
    base_name = gt_path.name[: -len(".gt.txt")]
    for extension in IMAGE_EXTENSIONS:
        candidate = gt_path.with_name(base_name + extension)
        if _is_image_file(candidate):
            return candidate
    return None


def _inspect_tesseract_ground_truth(dataset_dir: Path) -> dict[str, int]:
    stats = {"gt_files": 0, "pairs": 0, "missing_images": 0}
    for gt_path in dataset_dir.rglob("*.gt.txt"):
        if not gt_path.is_file():
            continue
        stats["gt_files"] += 1
        if _find_tesseract_image_for_gt(gt_path):
            stats["pairs"] += 1
        else:
            stats["missing_images"] += 1
    return stats


def inspect_dataset(dataset_dir: Path) -> dict[str, Any]:
    yaml_path = find_dataset_yaml(dataset_dir)
    classes = read_yaml_classes(yaml_path)

    formats: list[str] = []
    tasks: list[str] = []
    warnings: list[str] = []
    errors: list[str] = []

    has_images = count_images(dataset_dir) > 0
    has_yolo_yaml = yaml_path is not None
    yolo_stats = _inspect_yolo_labels(dataset_dir, len(classes))
    has_yolo_labels = yolo_stats["files"] > 0 and (yolo_stats["box_rows"] > 0 or yolo_stats["polygon_rows"] > 0)
    has_yolo_detection = (
        has_yolo_yaml
        and has_yolo_labels
        and yolo_stats["box_rows"] > 0
        and yolo_stats["box_rows"] >= yolo_stats["polygon_rows"]
    )
    has_yolo_segmentation = (
        has_yolo_yaml
        and has_yolo_labels
        and yolo_stats["polygon_rows"] > 0
        and yolo_stats["polygon_rows"] > yolo_stats["box_rows"]
    )
    imagefolder_classes = [] if has_yolo_yaml and has_yolo_labels else _imagefolder_classes(dataset_dir)
    if imagefolder_classes:
        classes = imagefolder_classes

    semantic_stats = _inspect_semantic_masks(dataset_dir)
    has_semantic_masks = semantic_stats["train_images"] > 0 and semantic_stats["mask_files"] > 0
    coco_stats = _inspect_coco_files(dataset_dir)
    paddleocr_stats = _inspect_paddleocr_labels(dataset_dir)
    tesseract_stats = _inspect_tesseract_ground_truth(dataset_dir)

    if imagefolder_classes:
        formats.append("imagefolder")
        tasks.append("image_classification")
    if has_yolo_detection:
        formats.append("yolo_detection")
        tasks.append("object_detection")
    if has_yolo_segmentation:
        formats.append("yolo_segmentation")
        tasks.append("segmentation")
    if has_yolo_yaml and yolo_stats["invalid_rows"]:
        errors.append(f"YOLO labels contain {yolo_stats['invalid_rows']} invalid rows.")
    if has_yolo_yaml and yolo_stats["box_rows"] and yolo_stats["polygon_rows"]:
        dominant = "box" if has_yolo_detection and not has_yolo_segmentation else "polygon"
        warnings.append(f"YOLO labels contain mixed box and polygon rows; using the dominant {dominant} format.")
    if has_semantic_masks:
        formats.append("semantic_masks")
        tasks.append("segmentation")
    if semantic_stats["missing_masks"]:
        preview = ", ".join(semantic_stats["missing_masks"][:5])
        errors.append(f"Semantic masks are missing for {len(semantic_stats['missing_masks'])} images: {preview}")
    if coco_stats["box_annotations"] or coco_stats["mask_annotations"]:
        formats.append("coco_instances")
        if coco_stats["box_annotations"]:
            tasks.append("object_detection")
        if coco_stats["mask_annotations"]:
            tasks.append("segmentation")
        if not classes and coco_stats["classes"]:
            classes = coco_stats["classes"]
    if coco_stats["files"] and coco_stats["missing_images"]:
        errors.append(f"COCO annotations reference {coco_stats['missing_images']} image files that were not found in the dataset.")
    if coco_stats["invalid_annotations"]:
        warnings.append(f"COCO annotations include {coco_stats['invalid_annotations']} invalid annotation rows that will be ignored.")
    if paddleocr_stats["tasks"]:
        formats.append("paddleocr_labels")
        tasks.append("ocr")
    if paddleocr_stats["missing_image_refs"]:
        errors.append(f"PaddleOCR labels reference {paddleocr_stats['missing_image_refs']} missing image files.")
    if tesseract_stats["pairs"]:
        formats.append("tesseract_ground_truth")
        tasks.append("ocr")
    if tesseract_stats["missing_images"]:
        errors.append(f"Tesseract ground truth has {tesseract_stats['missing_images']} .gt.txt files without matching images.")

    formats = sorted(set(formats))
    tasks = sorted(set(tasks))
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
        "coco_files": coco_stats["valid_files"],
        "coco": coco_stats,
        "yolo": yolo_stats,
        "semantic_masks": semantic_stats,
        "paddleocr_tasks": paddleocr_stats["tasks"],
        "tesseract": tesseract_stats,
        "warnings": warnings,
        "errors": errors,
    }


def validate_dataset_for_upload(dataset_dir: Path) -> dict[str, Any]:
    metadata = inspect_dataset(dataset_dir)
    if metadata["image_count"] == 0:
        raise ValueError("No supported image files were found in the uploaded dataset.")
    if metadata.get("errors"):
        raise ValueError(" ".join(metadata["errors"]))
    if not metadata["formats"]:
        raise ValueError(
            "Unsupported dataset structure. Supported trainable formats: YOLO detection data.yaml + labels, "
            "ImageFolder train/<class>, semantic masks, COCO boxes/instances, PaddleOCR labels, "
            "or Tesseract .gt.txt ground truth."
        )
    trainable_formats = sorted(set(metadata["formats"]).intersection(TRAINABLE_FORMATS))
    if not trainable_formats:
        raise ValueError(
            f"Detected formats {metadata['formats']} are not trainable by the current model catalog. "
            f"Supported trainable formats: {sorted(TRAINABLE_FORMATS)}."
        )
    return metadata



def _source_format_from_metadata(metadata: dict[str, Any]) -> str:
    formats = set(metadata.get("formats", []))
    if "yolo_detection" in formats:
        return "yolo"
    if "coco_instances" in formats:
        return "coco"
    if "semantic_masks" in formats:
        return "semantic_masks"
    if "imagefolder" in formats:
        return "imagefolder"
    if "paddleocr_labels" in formats:
        return "paddleocr"
    if "tesseract_ground_truth" in formats:
        return "tesseract"
    return "unknown"


def _canonical_task_from_metadata(metadata: dict[str, Any]) -> str:
    tasks = metadata.get("tasks", [])
    if len(tasks) == 1:
        return str(tasks[0])
    if "object_detection" in tasks and "segmentation" in tasks:
        return "object_detection_or_segmentation"
    return "unknown"


def _annotation_stats(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "images": metadata.get("image_count", 0),
        "classes": len(metadata.get("classes", [])),
        "yolo_boxes": metadata.get("yolo", {}).get("box_rows", 0),
        "yolo_polygons": metadata.get("yolo", {}).get("polygon_rows", 0),
        "coco_boxes": metadata.get("coco", {}).get("box_annotations", 0),
        "coco_masks": metadata.get("coco", {}).get("mask_annotations", 0),
        "semantic_masks": metadata.get("semantic_masks", {}).get("mask_files", 0),
        "paddleocr_tasks": metadata.get("paddleocr_tasks", []),
        "tesseract_pairs": metadata.get("tesseract", {}).get("pairs", 0),
    }


def _model_compatibility_reason(model_id: str, metadata: dict[str, Any], task_id: str) -> tuple[bool, str]:
    formats = set(metadata.get("formats", []))
    tasks = set(metadata.get("tasks", []))
    if task_id not in tasks:
        return False, f"Dataset does not contain {task_id} annotations."
    if model_id == "yolo":
        return ("yolo_detection" in formats, "Requires YOLO detection labels.")
    if model_id == "faster_rcnn":
        ok = bool({"yolo_detection", "coco_instances"}.intersection(formats)) and "object_detection" in tasks
        return ok, "Requires bounding-box annotations in YOLO or COCO format."
    if model_id == "mask_rcnn":
        ok = "coco_instances" in formats and metadata.get("coco", {}).get("mask_annotations", 0) > 0
        return ok, "Requires COCO instance masks."
    if model_id == "deeplabv3plus":
        return ("semantic_masks" in formats, "Requires image/mask semantic segmentation pairs.")
    if model_id in {"resnet", "efficientnet"}:
        return ("imagefolder" in formats, "Requires ImageFolder class folders.")
    if model_id == "paddleocr":
        return ("paddleocr_labels" in formats, "Requires PaddleOCR rec/det label files.")
    if model_id == "tesseract":
        return ("tesseract_ground_truth" in formats, "Requires image and .gt.txt pairs.")
    return False, "No compatibility rule is defined for this model."


def compatible_models_for_metadata(metadata: dict[str, Any], catalog: dict[str, Any]) -> list[dict[str, Any]]:
    compatible: list[dict[str, Any]] = []
    for task in catalog.get("tasks", []):
        task_id = str(task.get("id", ""))
        for model in task.get("models", []):
            ready, reason = _model_compatibility_reason(str(model.get("id", "")), metadata, task_id)
            compatible.append(
                {
                    "id": model.get("id"),
                    "label": model.get("label"),
                    "task": task_id,
                    "ready": ready,
                    "reason": reason,
                    "required_formats": model.get("dataset_formats", task.get("dataset_formats", [])),
                }
            )
    return compatible


def dataset_workflow_metadata(metadata: dict[str, Any], catalog: dict[str, Any] | None = None) -> dict[str, Any]:
    workflow = {
        "source_format": _source_format_from_metadata(metadata),
        "canonical_task": _canonical_task_from_metadata(metadata),
        "normalized_formats": metadata.get("formats", []),
        "annotation_stats": _annotation_stats(metadata),
        "conversion_warnings": metadata.get("conversion_warnings", []),
    }
    if catalog is not None:
        workflow["compatible_models"] = compatible_models_for_metadata(metadata, catalog)
    return workflow


def _normalized_root(dataset_dir: Path) -> Path:
    return dataset_dir / NORMALIZED_DIR_NAME


def _reset_normalized_root(dataset_dir: Path) -> Path:
    root = _normalized_root(dataset_dir)
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _unique_child(directory: Path, filename: str, prefix: str = "") -> Path:
    safe_name = safe_dataset_name(Path(filename).stem)
    suffix = Path(filename).suffix.lower()
    base = f"{prefix}{safe_name}" if prefix else safe_name
    candidate = directory / f"{base}{suffix}"
    index = 1
    while candidate.exists():
        candidate = directory / f"{base}_{index}{suffix}"
        index += 1
    return candidate


def _image_dimensions(image_path: Path, image_record: dict[str, Any]) -> tuple[float, float] | None:
    width = image_record.get("width")
    height = image_record.get("height")
    try:
        if width and height:
            return float(width), float(height)
    except (TypeError, ValueError):
        pass
    try:
        from PIL import Image

        with Image.open(image_path) as image:
            return float(image.width), float(image.height)
    except Exception:
        return None


def _convert_coco_to_yolo(dataset_dir: Path, normalized_root: Path, metadata: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    coco_files = metadata.get("coco_files", [])
    if not coco_files or metadata.get("coco", {}).get("box_annotations", 0) <= 0:
        return warnings

    output_root = normalized_root / "yolo_detection"
    output_root.mkdir(parents=True, exist_ok=True)
    classes = metadata.get("classes", []) or ["object"]
    wrote_labels = False
    splits: set[str] = set()

    for relative_file in coco_files:
        annotation_path = dataset_dir / relative_file
        try:
            data = json.loads(annotation_path.read_text(encoding="utf-8"))
        except Exception:
            warnings.append(f"Could not read COCO annotations from {relative_file}.")
            continue
        images = data.get("images", []) if isinstance(data, dict) else []
        annotations = data.get("annotations", []) if isinstance(data, dict) else []
        categories = data.get("categories", []) if isinstance(data, dict) else []
        if not isinstance(images, list) or not isinstance(annotations, list):
            continue

        category_ids = [category.get("id") for category in categories if isinstance(category, dict) and "id" in category]
        category_to_index = {category_id: index for index, category_id in enumerate(category_ids)}
        if not category_to_index:
            category_to_index = {1: 0}
        annotations_by_image: dict[Any, list[dict[str, Any]]] = {}
        for annotation in annotations:
            if isinstance(annotation, dict):
                annotations_by_image.setdefault(annotation.get("image_id"), []).append(annotation)

        split = _annotation_split(annotation_path, dataset_dir)
        split = "val" if split in {"valid", "validation"} else split
        splits.add(split)
        images_dir = output_root / "images" / split
        labels_dir = output_root / "labels" / split
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        for image_record in images:
            if not isinstance(image_record, dict):
                continue
            source_image = _resolve_coco_image(dataset_dir, annotation_path, image_record.get("file_name"))
            if source_image is None:
                continue
            dimensions = _image_dimensions(source_image, image_record)
            if dimensions is None:
                warnings.append(f"Skipping {source_image.name}: image width/height were not available for YOLO conversion.")
                continue
            width, height = dimensions
            if width <= 0 or height <= 0:
                continue

            target_image = _unique_child(images_dir, source_image.name, prefix=f"{image_record.get('id', '')}_")
            shutil.copy2(source_image, target_image)
            rows: list[str] = []
            for annotation in annotations_by_image.get(image_record.get("id"), []):
                bbox = annotation.get("bbox")
                if not _valid_bbox(bbox):
                    continue
                x, y, box_width, box_height = [float(value) for value in bbox]
                class_index = category_to_index.get(annotation.get("category_id"), 0)
                x_center = (x + box_width / 2) / width
                y_center = (y + box_height / 2) / height
                rows.append(
                    f"{class_index} {x_center:.6f} {y_center:.6f} {box_width / width:.6f} {box_height / height:.6f}"
                )
            (labels_dir / f"{target_image.stem}.txt").write_text("\n".join(rows), encoding="utf-8")
            if rows:
                wrote_labels = True

    if wrote_labels:
        train_split = "train" if "train" in splits else sorted(splits)[0]
        yaml_data: dict[str, Any] = {
            "path": str(dataset_dir),
            "train": f"{NORMALIZED_DIR_NAME}/yolo_detection/images/{train_split}",
            "names": classes,
        }
        if "val" in splits:
            yaml_data["val"] = f"{NORMALIZED_DIR_NAME}/yolo_detection/images/val"
        if "test" in splits:
            yaml_data["test"] = f"{NORMALIZED_DIR_NAME}/yolo_detection/images/test"
        (output_root / "data.yaml").write_text(yaml.dump(yaml_data, default_flow_style=False), encoding="utf-8")
    return warnings


def _convert_tesseract_to_paddleocr_rec(dataset_dir: Path, normalized_root: Path) -> list[str]:
    output_root = normalized_root / "paddleocr_rec"
    images_dir = output_root / "images"
    rows: list[str] = []
    for gt_path in dataset_dir.rglob("*.gt.txt"):
        if NORMALIZED_DIR_NAME in gt_path.parts:
            continue
        image_path = _find_tesseract_image_for_gt(gt_path)
        if image_path is None:
            continue
        images_dir.mkdir(parents=True, exist_ok=True)
        target_image = _unique_child(images_dir, image_path.name)
        shutil.copy2(image_path, target_image)
        text = gt_path.read_text(encoding="utf-8", errors="ignore").strip()
        rows.append(f"images/{target_image.name}\t{text}")
    if rows:
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / "rec_gt_train.txt").write_text("\n".join(rows), encoding="utf-8")
    return []


def _convert_paddleocr_rec_to_tesseract(dataset_dir: Path, normalized_root: Path) -> list[str]:
    output_root = normalized_root / "tesseract_gt"
    rows_written = 0
    for label_path in dataset_dir.rglob("rec_gt_train.txt"):
        if NORMALIZED_DIR_NAME in label_path.parts:
            continue
        try:
            lines = label_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue
        for line in lines:
            if "\t" not in line:
                continue
            image_ref, text = line.split("\t", 1)
            image_path = _safe_child(label_path.parent, image_ref.replace("\\", "/"))
            if image_path is None or not _is_image_file(image_path):
                image_path = _safe_child(dataset_dir, image_ref.replace("\\", "/"))
            if image_path is None or not _is_image_file(image_path):
                continue
            output_root.mkdir(parents=True, exist_ok=True)
            target_image = _unique_child(output_root, image_path.name)
            shutil.copy2(image_path, target_image)
            (output_root / f"{target_image.stem}.gt.txt").write_text(text.strip(), encoding="utf-8")
            rows_written += 1
    return [] if rows_written else []


def normalize_dataset_for_training(dataset_dir: Path) -> dict[str, Any]:
    normalized_root = _reset_normalized_root(dataset_dir)
    source_metadata = inspect_dataset(dataset_dir)
    conversion_warnings: list[str] = []

    if "coco_instances" in source_metadata.get("formats", []):
        conversion_warnings.extend(_convert_coco_to_yolo(dataset_dir, normalized_root, source_metadata))
    if "tesseract_ground_truth" in source_metadata.get("formats", []):
        conversion_warnings.extend(_convert_tesseract_to_paddleocr_rec(dataset_dir, normalized_root))
    if "paddleocr_labels" in source_metadata.get("formats", []) and "rec" in source_metadata.get("paddleocr_tasks", []):
        conversion_warnings.extend(_convert_paddleocr_rec_to_tesseract(dataset_dir, normalized_root))

    metadata = inspect_dataset(dataset_dir)
    metadata["source_format"] = _source_format_from_metadata(source_metadata)
    metadata["canonical_task"] = _canonical_task_from_metadata(metadata)
    metadata["normalized_formats"] = metadata.get("formats", [])
    metadata["annotation_stats"] = _annotation_stats(metadata)
    metadata["conversion_warnings"] = conversion_warnings
    return metadata
