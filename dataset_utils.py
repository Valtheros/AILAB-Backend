from __future__ import annotations

import json
import os
import hashlib
import shutil
import time
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
EXPORTS_DIR_NAME = ".ailab_exports"
SOURCE_FINGERPRINT_SKIP_FILES = {".ailab_dataset.json"}
TRAINABLE_FORMATS = {
    "imagefolder",
    "yolo_detection",
    "semantic_masks",
    "coco_instances",
    "paddleocr_labels",
    "tesseract_ground_truth",
}


def _env_int(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


MAX_LABEL_FILE_BYTES = _env_int("AILAB_MAX_LABEL_FILE_BYTES", 2 * 1024 * 1024)
MAX_LABEL_ROWS = _env_int("AILAB_MAX_LABEL_ROWS", 100_000)
MAX_BOXES_PER_IMAGE = _env_int("AILAB_MAX_BOXES_PER_IMAGE", 10_000)
MAX_OCR_TEXT_CHARS = _env_int("AILAB_MAX_OCR_TEXT_CHARS", 10_000)
MAX_COCO_JSON_BYTES = _env_int("AILAB_MAX_COCO_JSON_BYTES", 64 * 1024 * 1024)
MAX_COCO_IMAGES = _env_int("AILAB_MAX_COCO_IMAGES", 200_000)
MAX_COCO_ANNOTATIONS = _env_int("AILAB_MAX_COCO_ANNOTATIONS", 1_000_000)
MAX_COCO_ANNOTATIONS_PER_IMAGE = _env_int("AILAB_MAX_COCO_ANNOTATIONS_PER_IMAGE", 10_000)
MAX_COCO_POLYGON_POINTS = _env_int("AILAB_MAX_COCO_POLYGON_POINTS", 20_000)
MAX_COCO_MASKS_PER_IMAGE = _env_int("AILAB_MAX_COCO_MASKS_PER_IMAGE", 1_000)
MAX_SOURCE_IMAGE_PIXELS = _env_int("AILAB_MAX_SOURCE_IMAGE_PIXELS", 100_000_000)


def _ensure_file_size(path: Path, max_bytes: int, label: str) -> None:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise ValueError(f"Could not stat {label}: {path}") from exc
    if size > max_bytes:
        raise ValueError(f"{label} exceeds {max_bytes} bytes: {path}")


def _read_text_limited(path: Path, *, max_bytes: int = MAX_LABEL_FILE_BYTES, label: str = "label file") -> str:
    _ensure_file_size(path, max_bytes, label)
    return path.read_text(encoding="utf-8", errors="ignore")


def _iter_text_lines_limited(
    path: Path,
    *,
    max_bytes: int = MAX_LABEL_FILE_BYTES,
    max_rows: int = MAX_LABEL_ROWS,
    label: str = "label file",
):
    _ensure_file_size(path, max_bytes, label)
    with open(path, "r", encoding="utf-8", errors="ignore") as file:
        for index, line in enumerate(file, start=1):
            if index > max_rows:
                raise ValueError(f"{label} exceeds {max_rows} rows: {path}")
            yield line.rstrip("\r\n")


def _read_json_limited(path: Path, *, max_bytes: int = MAX_COCO_JSON_BYTES, label: str = "COCO annotation file") -> Any:
    _ensure_file_size(path, max_bytes, label)
    return json.loads(path.read_text(encoding="utf-8"))


def _ensure_ocr_text_length(text: str, label: str) -> None:
    if len(text) > MAX_OCR_TEXT_CHARS:
        raise ValueError(f"{label} exceeds {MAX_OCR_TEXT_CHARS} characters")


def _ensure_pixel_budget(width: float, height: float, label: str) -> None:
    if width <= 0 or height <= 0:
        raise ValueError(f"{label} has invalid dimensions {width}x{height}")
    pixels = int(width) * int(height)
    if pixels > MAX_SOURCE_IMAGE_PIXELS:
        raise ValueError(f"{label} has {pixels} pixels, above the {MAX_SOURCE_IMAGE_PIXELS} pixel limit")


def _build_image_basename_index(dataset_dir: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in _iter_visible_files(dataset_dir):
        if path.suffix.lower() not in IMAGE_EXTENSIONS or not path.is_file():
            continue
        index.setdefault(path.name, path)
        if len(index) > MAX_COCO_IMAGES:
            raise ValueError(f"Dataset contains more than {MAX_COCO_IMAGES} image basenames")
    return index


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
        dirs[:] = [name for name in dirs if name not in {NORMALIZED_DIR_NAME, EXPORTS_DIR_NAME}]
        for file in files:
            yield Path(root) / file


def _is_export_path(path: Path) -> bool:
    return EXPORTS_DIR_NAME in path.parts


def _iter_source_files(dataset_dir: Path):
    for root, dirs, files in os.walk(dataset_dir):
        dirs[:] = [name for name in dirs if name not in {NORMALIZED_DIR_NAME, EXPORTS_DIR_NAME}]
        for file in files:
            path = Path(root) / file
            if path.name in SOURCE_FINGERPRINT_SKIP_FILES:
                continue
            yield path


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
    for root, dirs, files in os.walk(dataset_dir):
        dirs[:] = [name for name in dirs if name != EXPORTS_DIR_NAME]
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
        if not label_path.is_file() or _is_export_path(label_path):
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
    stats = {"files": 0, "box_rows": 0, "polygon_rows": 0, "invalid_rows": 0, "errors": []}

    for label_path in _iter_yolo_label_files(dataset_dir):
        stats["files"] += 1
        row_count = 0
        try:
            for line in _iter_text_lines_limited(label_path, label="YOLO label file"):
                row_count += 1
                parts = line.strip().split()
                if not parts:
                    continue
                kind = _yolo_row_kind(parts, class_count)
                if kind == "box":
                    if row_count > MAX_BOXES_PER_IMAGE:
                        stats["errors"].append(f"YOLO label file has more than {MAX_BOXES_PER_IMAGE} boxes: {label_path}")
                        break
                    stats["box_rows"] += 1
                elif kind == "polygon":
                    stats["polygon_rows"] += 1
                else:
                    stats["invalid_rows"] += 1
        except (OSError, ValueError) as exc:
            stats["errors"].append(str(exc))
            continue

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
        if _is_export_path(path):
            continue
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


def _resolve_coco_image(dataset_dir: Path, annotation_path: Path, file_name: Any, image_index: dict[str, Path] | None = None) -> Path | None:
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
    if image_index is not None:
        return image_index.get(basename)
    scanned = 0
    for match in dataset_dir.rglob(basename):
        scanned += 1
        if scanned > MAX_COCO_IMAGES:
            return None
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


def _valid_coco_segmentation(segmentation: Any, image_dimensions: tuple[float, float] | None = None) -> bool:
    if isinstance(segmentation, list):
        total_points = 0
        for polygon in segmentation:
            if not isinstance(polygon, list) or len(polygon) < 6 or len(polygon) % 2 != 0:
                continue
            try:
                [float(value) for value in polygon]
            except (TypeError, ValueError):
                continue
            total_points += len(polygon) // 2
            if total_points > MAX_COCO_POLYGON_POINTS:
                return False
            return True
        return False
    if isinstance(segmentation, dict):
        size = segmentation.get("size")
        if "counts" not in segmentation or not isinstance(size, (list, tuple)) or len(size) != 2:
            return False
        try:
            height, width = int(size[0]), int(size[1])
        except (TypeError, ValueError):
            return False
        try:
            _ensure_pixel_budget(width, height, "COCO RLE mask")
        except ValueError:
            return False
        if image_dimensions is not None:
            image_width, image_height = image_dimensions
            if int(round(image_width)) != width or int(round(image_height)) != height:
                return False
        return True
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
        "errors": [],
    }
    class_names: dict[Any, str] = {}
    try:
        image_index = _build_image_basename_index(dataset_dir)
    except ValueError as exc:
        stats["errors"].append(str(exc))
        image_index = {}
    for relative_file in stats["files"]:
        annotation_path = dataset_dir / relative_file
        try:
            data = _read_json_limited(annotation_path)
        except ValueError as exc:
            stats["errors"].append(str(exc))
            continue
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        images = data.get("images", [])
        annotations = data.get("annotations", [])
        categories = data.get("categories", [])
        if not isinstance(images, list) or not isinstance(annotations, list):
            continue
        if len(images) > MAX_COCO_IMAGES:
            stats["errors"].append(f"COCO file {relative_file} has more than {MAX_COCO_IMAGES} images.")
            continue
        if len(annotations) > MAX_COCO_ANNOTATIONS:
            stats["errors"].append(f"COCO file {relative_file} has more than {MAX_COCO_ANNOTATIONS} annotations.")
            continue
        annotations_per_image: dict[Any, int] = {}
        masks_per_image: dict[Any, int] = {}
        for annotation in annotations:
            if isinstance(annotation, dict):
                image_id_for_count = annotation.get("image_id")
                annotations_per_image[image_id_for_count] = annotations_per_image.get(image_id_for_count, 0) + 1
                if annotation.get("segmentation") is not None:
                    masks_per_image[image_id_for_count] = masks_per_image.get(image_id_for_count, 0) + 1
        if any(count > MAX_COCO_ANNOTATIONS_PER_IMAGE for count in annotations_per_image.values()):
            stats["errors"].append(f"COCO file {relative_file} has too many annotations for one image.")
            continue
        if any(count > MAX_COCO_MASKS_PER_IMAGE for count in masks_per_image.values()):
            stats["errors"].append(f"COCO file {relative_file} has too many masks for one image.")
            continue
        if isinstance(categories, list):
            for category in categories:
                if isinstance(category, dict) and "id" in category:
                    class_names[category.get("id")] = str(category.get("name", category.get("id")))

        found_image_ids = set()
        image_dimensions_by_id: dict[Any, tuple[float, float]] = {}
        for image in images:
            if not isinstance(image, dict):
                continue
            image_id = image.get("id")
            image_path = _resolve_coco_image(dataset_dir, annotation_path, image.get("file_name"), image_index)
            if image_path is None:
                stats["missing_images"] += 1
                continue
            try:
                dimensions = _image_dimensions(image_path, image)
            except ValueError as exc:
                stats["errors"].append(str(exc))
                continue
            if dimensions is not None:
                image_dimensions_by_id[image_id] = dimensions
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
            segmentation = annotation.get("segmentation")
            if has_bbox and segmentation is not None:
                if _valid_coco_segmentation(segmentation, image_dimensions_by_id.get(annotation.get("image_id"))):
                    stats["mask_annotations"] += 1
                    file_mask_annotations += 1
                else:
                    stats["errors"].append("COCO segmentation exceeds mask limits or does not match image dimensions.")
        if found_image_ids and (file_box_annotations or file_mask_annotations):
            stats["valid_files"].append(relative_file)

    if class_names:
        sort_key = lambda item: (0, int(item[0])) if str(item[0]).isdigit() else (1, str(item[0]))
        stats["classes"] = [name for _category_id, name in sorted(class_names.items(), key=sort_key)]
    return stats


def _inspect_paddleocr_labels(dataset_dir: Path) -> dict[str, Any]:
    stats: dict[str, Any] = {"tasks": [], "files": [], "missing_image_refs": 0, "errors": []}
    tasks = set()
    for path in dataset_dir.rglob("*"):
        if not path.is_file() or _is_export_path(path):
            continue
        task = PADDLEOCR_TRAIN_LABEL_NAMES.get(path.name.lower())
        if not task:
            continue
        tasks.add(task)
        stats["files"].append(str(path.relative_to(dataset_dir)))
        try:
            for line in _iter_text_lines_limited(path, label="PaddleOCR label file"):
                stripped = line.strip()
                if not stripped:
                    continue
                image_ref, text_value = (stripped.split("\t", 1) + [""])[:2]
                image_ref = image_ref.strip()
                _ensure_ocr_text_length(text_value.strip(), f"PaddleOCR label row in {path}")
                if not image_ref:
                    continue
                normalized_ref = image_ref.replace("\\", "/")
                candidate = _safe_child(path.parent, normalized_ref)
                if candidate is None or not _is_image_file(candidate):
                    candidate = _safe_child(dataset_dir, normalized_ref)
                if candidate is None or not _is_image_file(candidate):
                    stats["missing_image_refs"] += 1
        except (OSError, ValueError) as exc:
            stats["errors"].append(str(exc))
            continue
    stats["tasks"] = sorted(tasks)
    return stats


def _find_tesseract_image_for_gt(gt_path: Path) -> Path | None:
    base_name = gt_path.name[: -len(".gt.txt")]
    for extension in IMAGE_EXTENSIONS:
        candidate = gt_path.with_name(base_name + extension)
        if _is_image_file(candidate):
            return candidate
    return None


def _inspect_tesseract_ground_truth(dataset_dir: Path) -> dict[str, Any]:
    stats: dict[str, Any] = {"gt_files": 0, "pairs": 0, "missing_images": 0, "errors": []}
    for gt_path in dataset_dir.rglob("*.gt.txt"):
        if not gt_path.is_file() or _is_export_path(gt_path):
            continue
        stats["gt_files"] += 1
        try:
            text = _read_text_limited(gt_path, label="Tesseract ground truth file").strip()
            _ensure_ocr_text_length(text, f"Tesseract ground truth file {gt_path}")
        except (OSError, ValueError) as exc:
            stats["errors"].append(str(exc))
            continue
        if _find_tesseract_image_for_gt(gt_path):
            stats["pairs"] += 1
        else:
            stats["missing_images"] += 1
    return stats


def _inspect_source_pixel_budgets(dataset_dir: Path) -> list[str]:
    errors: list[str] = []
    for path in _iter_visible_files(dataset_dir):
        if path.suffix.lower() not in IMAGE_EXTENSIONS.union(MASK_EXTENSIONS):
            continue
        try:
            from PIL import Image

            with Image.open(path) as image:
                _ensure_pixel_budget(float(image.width), float(image.height), path.name)
        except ValueError as exc:
            errors.append(str(exc))
        except Exception:
            continue
    return errors


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
    errors.extend(yolo_stats.get("errors", []))
    errors.extend(coco_stats.get("errors", []))
    errors.extend(paddleocr_stats.get("errors", []))
    errors.extend(tesseract_stats.get("errors", []))
    errors.extend(_inspect_source_pixel_budgets(dataset_dir))
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
    if "coco_instances" in formats:
        return "coco"
    if "yolo_detection" in formats:
        return "yolo"
    if "semantic_masks" in formats:
        return "semantic_masks"
    if "imagefolder" in formats:
        return "imagefolder"
    if "paddleocr_labels" in formats:
        return "paddleocr"
    if "tesseract_ground_truth" in formats:
        return "tesseract"
    return "unknown"


def _dataset_tasks_from_metadata(metadata: dict[str, Any]) -> list[str]:
    formats = set(metadata.get("formats", []))
    dataset_tasks: list[str] = []
    coco = metadata.get("coco", {})
    paddleocr_tasks = set(metadata.get("paddleocr_tasks", []))

    if "imagefolder" in formats:
        dataset_tasks.append("image_classification")
    if "yolo_detection" in formats or coco.get("box_annotations", 0) > 0:
        dataset_tasks.append("object_detection")
    if "semantic_masks" in formats:
        dataset_tasks.append("semantic_segmentation")
    if coco.get("mask_annotations", 0) > 0:
        dataset_tasks.append("instance_segmentation")
    if "rec" in paddleocr_tasks or metadata.get("tesseract", {}).get("pairs", 0) > 0:
        dataset_tasks.append("ocr_recognition")
    if "det" in paddleocr_tasks:
        dataset_tasks.append("ocr_detection")

    return list(dict.fromkeys(dataset_tasks))


def _canonical_task_from_metadata(metadata: dict[str, Any]) -> str:
    dataset_tasks = _dataset_tasks_from_metadata(metadata)
    if len(dataset_tasks) == 1:
        return dataset_tasks[0]
    if "instance_segmentation" in dataset_tasks:
        return "instance_segmentation"
    if "semantic_segmentation" in dataset_tasks:
        return "semantic_segmentation"
    if "object_detection" in dataset_tasks:
        return "object_detection"
    if "ocr_recognition" in dataset_tasks and "ocr_detection" in dataset_tasks:
        return "ocr_recognition_or_detection"
    if dataset_tasks:
        return "multi_task"
    return "unknown"


def _canonical_format_from_metadata(metadata: dict[str, Any]) -> str:
    formats = set(metadata.get("formats", []))
    coco = metadata.get("coco", {})
    paddleocr_tasks = set(metadata.get("paddleocr_tasks", []))

    if "imagefolder" in formats:
        return "imagefolder"
    if "semantic_masks" in formats:
        return "semantic_masks"
    if coco.get("mask_annotations", 0) > 0:
        return "coco_instance_masks"
    if "yolo_detection" in formats or coco.get("box_annotations", 0) > 0:
        return "object_detection_boxes"
    if "det" in paddleocr_tasks:
        return "ocr_detection_labels"
    if "rec" in paddleocr_tasks or metadata.get("tesseract", {}).get("pairs", 0) > 0:
        return "ocr_recognition_labels"
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


def _has_bounding_boxes(metadata: dict[str, Any]) -> bool:
    formats = set(metadata.get("formats", []))
    return "yolo_detection" in formats or metadata.get("coco", {}).get("box_annotations", 0) > 0


def _has_ocr_recognition(metadata: dict[str, Any]) -> bool:
    return "rec" in set(metadata.get("paddleocr_tasks", [])) or metadata.get("tesseract", {}).get("pairs", 0) > 0


def _model_compatibility_reason(model_id: str, metadata: dict[str, Any], task_id: str) -> tuple[bool, str]:
    formats = set(metadata.get("formats", []))
    tasks = set(metadata.get("tasks", []))
    paddleocr_tasks = set(metadata.get("paddleocr_tasks", []))

    if model_id == "yolo":
        ok = _has_bounding_boxes(metadata) and "object_detection" in tasks
        return ok, "Requires bounding boxes; COCO boxes are exported to YOLO at train time."
    if model_id == "faster_rcnn":
        ok = _has_bounding_boxes(metadata) and "object_detection" in tasks
        return ok, "Requires bounding boxes in YOLO or COCO; COCO can be used directly."
    if model_id == "mask_rcnn":
        ok = "coco_instances" in formats and metadata.get("coco", {}).get("mask_annotations", 0) > 0
        return ok, "Requires COCO instance masks, not box-only annotations."
    if model_id == "deeplabv3plus":
        return "semantic_masks" in formats, "Requires image/mask semantic segmentation pairs."
    if model_id in {"resnet", "efficientnet"}:
        return "imagefolder" in formats, "Requires image classification class folders."
    if model_id == "paddleocr":
        ok = bool(paddleocr_tasks) or metadata.get("tesseract", {}).get("pairs", 0) > 0
        return ok, "Requires OCR detection labels or recognition ground truth."
    if model_id == "tesseract":
        ok = _has_ocr_recognition(metadata)
        return ok, "Requires OCR recognition ground truth; PaddleOCR rec labels are exported at train time."
    if task_id not in tasks:
        return False, f"Dataset does not contain {task_id} annotations."
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
                    "dataset_task": model.get("dataset_task"),
                    "required_annotations": model.get("required_annotations", []),
                    "accepted_canonical_formats": model.get("accepted_canonical_formats", []),
                    "train_export_format": model.get("train_export_format"),
                }
            )
    return compatible


def export_cache_metadata(dataset_dir: Path) -> list[dict[str, Any]]:
    root = dataset_dir / EXPORTS_DIR_NAME
    if not root.is_dir():
        return []
    entries: list[dict[str, Any]] = []
    for manifest_path in sorted(root.glob("*/*/manifest.json")):
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        try:
            relative_path = str(manifest_path.parent.relative_to(dataset_dir))
        except ValueError:
            relative_path = str(manifest_path.parent)
        entries.append(
            {
                "model": data.get("model_type"),
                "export_format": data.get("export_format"),
                "fingerprint": data.get("fingerprint"),
                "path": relative_path,
                "created_at": data.get("created_at"),
            }
        )
    return entries


def dataset_workflow_metadata(metadata: dict[str, Any], catalog: dict[str, Any] | None = None) -> dict[str, Any]:
    dataset_tasks = _dataset_tasks_from_metadata(metadata)
    workflow = {
        "source_format": _source_format_from_metadata(metadata),
        "dataset_task": _canonical_task_from_metadata(metadata),
        "dataset_tasks": dataset_tasks,
        "canonical_task": _canonical_task_from_metadata(metadata),
        "canonical_format": _canonical_format_from_metadata(metadata),
        "normalized_formats": metadata.get("formats", []),
        "annotation_stats": _annotation_stats(metadata),
        "conversion_warnings": metadata.get("conversion_warnings", []),
        "export_cache": metadata.get("export_cache", []),
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
    if width and height:
        try:
            width_value, height_value = float(width), float(height)
        except (TypeError, ValueError):
            pass
        else:
            _ensure_pixel_budget(width_value, height_value, f"image {image_path.name}")
            return width_value, height_value
    try:
        from PIL import Image

        with Image.open(image_path) as image:
            width_value, height_value = float(image.width), float(image.height)
            _ensure_pixel_budget(width_value, height_value, f"image {image_path.name}")
            return width_value, height_value
    except ValueError:
        raise
    except Exception:
        return None


def _convert_coco_to_yolo(dataset_dir: Path, normalized_root: Path, metadata: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    coco_files = metadata.get("coco_files", [])
    if not coco_files or metadata.get("coco", {}).get("box_annotations", 0) <= 0:
        return warnings

    output_root = normalized_root / "yolo_detection"
    output_root.mkdir(parents=True, exist_ok=True)
    image_index = _build_image_basename_index(dataset_dir)
    classes = metadata.get("classes", []) or ["object"]
    wrote_labels = False
    splits: set[str] = set()

    for relative_file in coco_files:
        annotation_path = dataset_dir / relative_file
        try:
            data = _read_json_limited(annotation_path)
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
            source_image = _resolve_coco_image(dataset_dir, annotation_path, image_record.get("file_name"), image_index)
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
        text = _read_text_limited(gt_path, label="Tesseract ground truth file").strip()
        _ensure_ocr_text_length(text, f"Tesseract ground truth file {gt_path}")
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
            lines = list(_iter_text_lines_limited(label_path, label="PaddleOCR recognition label file"))
        except OSError:
            continue
        for line in lines:
            if "\t" not in line:
                continue
            image_ref, text = line.split("\t", 1)
            _ensure_ocr_text_length(text.strip(), f"PaddleOCR recognition label row in {label_path}")
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



def _source_fingerprint(dataset_dir: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(_iter_source_files(dataset_dir), key=lambda item: item.relative_to(dataset_dir).as_posix()):
        try:
            stat = path.stat()
            relative = path.relative_to(dataset_dir).as_posix()
        except OSError:
            continue
        digest.update(relative.encode("utf-8", errors="ignore"))
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
    return digest.hexdigest()


def _export_identity(dataset_dir: Path, model_type: str, extra_args: dict[str, Any]) -> tuple[str, Path, str]:
    source_fingerprint = _source_fingerprint(dataset_dir)
    cache_material = {
        "source_fingerprint": source_fingerprint,
        "model_type": model_type,
        "ocr_task": str(extra_args.get("ocr_task", "")) if model_type == "paddleocr" else "",
    }
    fingerprint = hashlib.sha256(json.dumps(cache_material, sort_keys=True).encode("utf-8")).hexdigest()[:20]
    return source_fingerprint, dataset_dir / EXPORTS_DIR_NAME / model_type / fingerprint, fingerprint


def _cache_is_valid(export_root: Path, source_fingerprint: str, export_format: str, required_files: list[str]) -> bool:
    manifest_path = export_root / "manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if manifest.get("source_fingerprint") != source_fingerprint or manifest.get("export_format") != export_format:
        return False
    return all((export_root / required_file).exists() for required_file in required_files)


def _write_export_manifest(
    export_root: Path,
    model_type: str,
    export_format: str,
    source_fingerprint: str,
    fingerprint: str,
    warnings: list[str],
) -> None:
    manifest = {
        "model_type": model_type,
        "export_format": export_format,
        "source_fingerprint": source_fingerprint,
        "fingerprint": fingerprint,
        "warnings": warnings,
        "created_at": int(time.time()),
    }
    (export_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _normalize_yolo_yaml_value(dataset_dir: Path, yaml_key: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"YOLO YAML '{yaml_key}' paths must be non-empty text")

    normalized = value.strip().replace("\\", "/")
    candidates = [normalized]
    stripped = normalized
    while stripped.startswith("../"):
        stripped = stripped[3:]
        candidates.append(stripped)
    if stripped.startswith("./"):
        candidates.append(stripped[2:])

    seen: set[str] = set()
    for candidate_value in candidates:
        if candidate_value in seen:
            continue
        seen.add(candidate_value)
        candidate = _safe_child(dataset_dir, candidate_value)
        if candidate and candidate.exists():
            return candidate_value
    raise ValueError(f"YOLO YAML '{yaml_key}' path was not found inside the dataset: {value}")


def _write_source_yolo_export(dataset_dir: Path, output_root: Path) -> Path:
    original_yaml = find_dataset_yaml(dataset_dir)
    if original_yaml is None:
        raise ValueError(f"Dataset '{dataset_dir.name}' is missing data.yaml for YOLO-style labels.")
    config = yaml.safe_load(original_yaml.read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict):
        raise ValueError("YOLO YAML root must be an object")
    if not isinstance(config.get("names"), (dict, list)) or not config["names"]:
        raise ValueError("YOLO YAML requires a non-empty names list or mapping")
    config["path"] = str(dataset_dir)
    for yaml_key in ("train", "val", "test"):
        value = config.get(yaml_key)
        if value is None:
            continue
        if isinstance(value, list):
            config[yaml_key] = [_normalize_yolo_yaml_value(dataset_dir, yaml_key, item) for item in value]
        else:
            config[yaml_key] = _normalize_yolo_yaml_value(dataset_dir, yaml_key, value)
    if "train" not in config:
        raise ValueError("YOLO YAML requires a train path")
    output_root.mkdir(parents=True, exist_ok=True)
    output_yaml = output_root / "data.yaml"
    output_yaml.write_text(yaml.dump(config, default_flow_style=False), encoding="utf-8")
    return output_yaml


def _write_coco_yolo_export(dataset_dir: Path, output_root: Path, metadata: dict[str, Any]) -> tuple[Path, list[str]]:
    warnings: list[str] = []
    coco_files = metadata.get("coco_files", [])
    if not coco_files or metadata.get("coco", {}).get("box_annotations", 0) <= 0:
        raise ValueError(f"Dataset '{dataset_dir.name}' has no COCO bounding-box annotations to export for YOLO.")

    classes = metadata.get("classes", []) or ["object"]
    wrote_labels = False
    splits: set[str] = set()
    output_root.mkdir(parents=True, exist_ok=True)
    image_index = _build_image_basename_index(dataset_dir)

    for relative_file in coco_files:
        annotation_path = dataset_dir / relative_file
        try:
            data = _read_json_limited(annotation_path)
        except Exception:
            warnings.append(f"Could not read COCO annotations from {relative_file}.")
            continue
        images = data.get("images", []) if isinstance(data, dict) else []
        annotations = data.get("annotations", []) if isinstance(data, dict) else []
        categories = data.get("categories", []) if isinstance(data, dict) else []
        if not isinstance(images, list) or not isinstance(annotations, list):
            continue

        category_ids = [category.get("id") for category in categories if isinstance(category, dict) and "id" in category]
        category_to_index = {category_id: index for index, category_id in enumerate(category_ids)} or {1: 0}
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
            source_image = _resolve_coco_image(dataset_dir, annotation_path, image_record.get("file_name"), image_index)
            if source_image is None:
                continue
            dimensions = _image_dimensions(source_image, image_record)
            if dimensions is None:
                warnings.append(f"Skipping {source_image.name}: image width/height were not available for YOLO export.")
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

    if not wrote_labels:
        raise ValueError(f"Dataset '{dataset_dir.name}' did not produce any YOLO labels from COCO boxes.")
    train_split = "train" if "train" in splits else sorted(splits)[0]
    yaml_data: dict[str, Any] = {
        "path": str(output_root),
        "train": f"images/{train_split}",
        "names": classes,
    }
    if "val" in splits:
        yaml_data["val"] = "images/val"
    if "test" in splits:
        yaml_data["test"] = "images/test"
    output_yaml = output_root / "data.yaml"
    output_yaml.write_text(yaml.dump(yaml_data, default_flow_style=False), encoding="utf-8")
    return output_yaml, warnings


def _write_tesseract_as_paddleocr_rec(dataset_dir: Path, output_root: Path) -> Path:
    images_dir = output_root / "images"
    rows: list[str] = []
    for gt_path in dataset_dir.rglob("*.gt.txt"):
        if _is_export_path(gt_path):
            continue
        image_path = _find_tesseract_image_for_gt(gt_path)
        if image_path is None:
            continue
        images_dir.mkdir(parents=True, exist_ok=True)
        target_image = _unique_child(images_dir, image_path.name)
        shutil.copy2(image_path, target_image)
        text = _read_text_limited(gt_path, label="Tesseract ground truth file").strip()
        _ensure_ocr_text_length(text, f"Tesseract ground truth file {gt_path}")
        rows.append(f"images/{target_image.name}\t{text}")
    if not rows:
        raise ValueError(f"Dataset '{dataset_dir.name}' has no Tesseract ground truth pairs to export for PaddleOCR recognition.")
    output_root.mkdir(parents=True, exist_ok=True)
    label_path = output_root / "rec_gt_train.txt"
    label_path.write_text("\n".join(rows), encoding="utf-8")
    return label_path


def _write_paddleocr_rec_as_tesseract(dataset_dir: Path, output_root: Path) -> None:
    rows_written = 0
    for label_path in dataset_dir.rglob("rec_gt_train.txt"):
        if _is_export_path(label_path):
            continue
        try:
            lines = list(_iter_text_lines_limited(label_path, label="PaddleOCR recognition label file"))
        except OSError:
            continue
        for line in lines:
            if "\t" not in line:
                continue
            image_ref, text = line.split("\t", 1)
            _ensure_ocr_text_length(text.strip(), f"PaddleOCR recognition label row in {label_path}")
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
    if rows_written <= 0:
        raise ValueError(f"Dataset '{dataset_dir.name}' has no PaddleOCR recognition labels to export for Tesseract.")


def _prepared_response(
    dataset_dir: Path,
    model_type: str,
    metadata: dict[str, Any],
    dataset_path: Path,
    export_format: str,
    data_yaml_path: Path | None = None,
    export_path: Path | None = None,
    cache_hit: bool = False,
    fingerprint: str | None = None,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    metadata = dict(metadata)
    metadata["export_cache"] = export_cache_metadata(dataset_dir)
    return {
        "dataset_path": str(dataset_path),
        "source_dataset_path": str(dataset_dir),
        "data_yaml_path": str(data_yaml_path) if data_yaml_path else None,
        "metadata": metadata,
        "export": {
            "model": model_type,
            "export_format": export_format,
            "path": str(export_path) if export_path else None,
            "cache_hit": cache_hit,
            "fingerprint": fingerprint,
            "warnings": warnings or [],
        },
    }


def _cached_generated_export(
    dataset_dir: Path,
    model_type: str,
    extra_args: dict[str, Any],
    export_format: str,
    required_files: list[str],
    writer,
) -> tuple[Path, bool, str, list[str]]:
    source_fingerprint, export_root, fingerprint = _export_identity(dataset_dir, model_type, extra_args)
    if _cache_is_valid(export_root, source_fingerprint, export_format, required_files):
        return export_root, True, fingerprint, []
    shutil.rmtree(export_root, ignore_errors=True)
    export_root.mkdir(parents=True, exist_ok=True)
    warnings = writer(export_root)
    _write_export_manifest(export_root, model_type, export_format, source_fingerprint, fingerprint, warnings)
    return export_root, False, fingerprint, warnings


def prepare_dataset_for_model(dataset_dir: Path, model_type: str, extra_args: dict[str, Any] | None = None) -> dict[str, Any]:
    extra_args = extra_args or {}
    metadata = inspect_dataset(dataset_dir)
    formats = set(metadata.get("formats", []))
    paddleocr_tasks = set(metadata.get("paddleocr_tasks", []))

    if model_type in {"resnet", "efficientnet"}:
        if "imagefolder" not in formats:
            raise ValueError(f"Dataset '{dataset_dir.name}' has no image classification class folders.")
        return _prepared_response(dataset_dir, model_type, metadata, dataset_dir, "imagefolder")

    if model_type == "deeplabv3plus":
        if "semantic_masks" not in formats:
            raise ValueError(f"Dataset '{dataset_dir.name}' has no semantic image/mask pairs for DeepLabV3+.")
        return _prepared_response(dataset_dir, model_type, metadata, dataset_dir, "semantic_masks")

    if model_type == "mask_rcnn":
        if metadata.get("coco", {}).get("mask_annotations", 0) <= 0:
            raise ValueError(f"Dataset '{dataset_dir.name}' has no COCO instance masks for Mask R-CNN.")
        return _prepared_response(dataset_dir, model_type, metadata, dataset_dir, "coco_instances")

    if model_type in {"yolo", "faster_rcnn"}:
        if "yolo_detection" in formats:
            export_root, cache_hit, fingerprint, warnings = _cached_generated_export(
                dataset_dir,
                model_type,
                extra_args,
                "yolo_detection",
                ["data.yaml"],
                lambda output_root: (_write_source_yolo_export(dataset_dir, output_root), [])[1],
            )
            return _prepared_response(
                dataset_dir,
                model_type,
                metadata,
                dataset_dir,
                "yolo_detection",
                data_yaml_path=export_root / "data.yaml",
                export_path=export_root,
                cache_hit=cache_hit,
                fingerprint=fingerprint,
                warnings=warnings,
            )
        if metadata.get("coco", {}).get("box_annotations", 0) > 0:
            if model_type == "faster_rcnn":
                return _prepared_response(dataset_dir, model_type, metadata, dataset_dir, "coco_instances")
            export_root, cache_hit, fingerprint, warnings = _cached_generated_export(
                dataset_dir,
                model_type,
                extra_args,
                "yolo_detection",
                ["data.yaml"],
                lambda output_root: _write_coco_yolo_export(dataset_dir, output_root, metadata)[1],
            )
            return _prepared_response(
                dataset_dir,
                model_type,
                metadata,
                export_root,
                "yolo_detection",
                data_yaml_path=export_root / "data.yaml",
                export_path=export_root,
                cache_hit=cache_hit,
                fingerprint=fingerprint,
                warnings=warnings,
            )
        raise ValueError(f"Dataset '{dataset_dir.name}' has no bounding-box annotations for {model_type}.")

    if model_type == "paddleocr":
        requested_task = str(extra_args.get("ocr_task", "rec"))
        if requested_task not in {"rec", "det"}:
            raise ValueError("PaddleOCR ocr_task must be either 'det' or 'rec'.")
        if requested_task in paddleocr_tasks:
            return _prepared_response(dataset_dir, model_type, metadata, dataset_dir, "paddleocr_labels")
        if requested_task == "rec" and metadata.get("tesseract", {}).get("pairs", 0) > 0:
            export_root, cache_hit, fingerprint, warnings = _cached_generated_export(
                dataset_dir,
                model_type,
                extra_args,
                "paddleocr_recognition_labels",
                ["rec_gt_train.txt"],
                lambda output_root: (_write_tesseract_as_paddleocr_rec(dataset_dir, output_root), [])[1],
            )
            return _prepared_response(
                dataset_dir,
                model_type,
                metadata,
                export_root,
                "paddleocr_recognition_labels",
                export_path=export_root,
                cache_hit=cache_hit,
                fingerprint=fingerprint,
                warnings=warnings,
            )
        raise ValueError(f"Dataset '{dataset_dir.name}' has no PaddleOCR {requested_task} labels.")

    if model_type == "tesseract":
        if metadata.get("tesseract", {}).get("pairs", 0) > 0:
            return _prepared_response(dataset_dir, model_type, metadata, dataset_dir, "tesseract_ground_truth")
        if "rec" in paddleocr_tasks:
            export_root, cache_hit, fingerprint, warnings = _cached_generated_export(
                dataset_dir,
                model_type,
                extra_args,
                "tesseract_ground_truth",
                [],
                lambda output_root: (_write_paddleocr_rec_as_tesseract(dataset_dir, output_root), [])[1],
            )
            return _prepared_response(
                dataset_dir,
                model_type,
                metadata,
                export_root,
                "tesseract_ground_truth",
                export_path=export_root,
                cache_hit=cache_hit,
                fingerprint=fingerprint,
                warnings=warnings,
            )
        raise ValueError(f"Dataset '{dataset_dir.name}' has no OCR recognition ground truth for Tesseract.")

    raise ValueError(f"No dataset export rule is defined for model '{model_type}'.")

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
