from __future__ import annotations

import csv
import json
import os
import hashlib
import math
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

import yaml

from security_utils import named_file_lock, replace_directory


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
MASK_EXTENSIONS = {".png", ".bmp", ".tif", ".tiff"}
EXPORTS_DIR_NAME = ".ailab_exports"
SOURCE_FINGERPRINT_SKIP_FILES = {".ailab_dataset.json"}
TRAINABLE_FORMATS = {
    "imagefolder",
    "yolo_detection",
    "semantic_masks",
    "coco_instances",
}


def _env_int(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


MAX_LABEL_FILE_BYTES = _env_int("AILAB_MAX_LABEL_FILE_BYTES", 2 * 1024 * 1024)
MAX_YAML_FILE_BYTES = _env_int("AILAB_MAX_YAML_FILE_BYTES", 1024 * 1024)
MAX_YAML_EVENTS = _env_int("AILAB_MAX_YAML_EVENTS", 20_000)
MAX_YAML_DEPTH = _env_int("AILAB_MAX_YAML_DEPTH", 32)
MAX_LABEL_ROWS = _env_int("AILAB_MAX_LABEL_ROWS", 100_000)
MAX_BOXES_PER_IMAGE = _env_int("AILAB_MAX_BOXES_PER_IMAGE", 10_000)
MAX_COCO_JSON_BYTES = _env_int("AILAB_MAX_COCO_JSON_BYTES", 64 * 1024 * 1024)
MAX_COCO_IMAGES = _env_int("AILAB_MAX_COCO_IMAGES", 200_000)
MAX_COCO_ANNOTATIONS = _env_int("AILAB_MAX_COCO_ANNOTATIONS", 1_000_000)
MAX_COCO_ANNOTATIONS_PER_IMAGE = _env_int("AILAB_MAX_COCO_ANNOTATIONS_PER_IMAGE", 10_000)
MAX_COCO_POLYGON_POINTS = _env_int("AILAB_MAX_COCO_POLYGON_POINTS", 20_000)
MAX_COCO_MASKS_PER_IMAGE = _env_int("AILAB_MAX_COCO_MASKS_PER_IMAGE", 1_000)
MAX_COCO_RLE_COUNTS = _env_int("AILAB_MAX_COCO_RLE_COUNTS", 2_000_000)
MAX_SOURCE_IMAGE_PIXELS = _env_int("AILAB_MAX_SOURCE_IMAGE_PIXELS", 25_000_000)
MAX_COCO_DECODED_MASK_PIXELS_PER_IMAGE = _env_int(
    "AILAB_MAX_COCO_DECODED_MASK_PIXELS_PER_IMAGE", 256_000_000
)
MAX_DATASET_ISSUES_RETURNED = _env_int("AILAB_MAX_DATASET_ISSUES_RETURNED", 12)


def _ensure_file_size(path: Path, max_bytes: int, label: str) -> None:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise ValueError(f"Could not stat {label}: {path}") from exc
    if size > max_bytes:
        raise ValueError(f"{label} exceeds {max_bytes} bytes: {path}")


def _summarize_issue_list(issues: Iterable[str], *, max_items: int = MAX_DATASET_ISSUES_RETURNED) -> list[str]:
    counts: dict[str, int] = {}
    ordered: list[str] = []
    for issue in issues:
        normalized = " ".join(str(issue).split())
        if not normalized:
            continue
        if normalized not in counts:
            ordered.append(normalized)
            counts[normalized] = 0
        counts[normalized] += 1

    summarized: list[str] = []
    for issue in ordered[:max_items]:
        count = counts[issue]
        if count > 1:
            summarized.append(f"{issue} (repeated {count} times)")
        else:
            summarized.append(issue)
    omitted = len(ordered) - max_items
    if omitted > 0:
        summarized.append(f"{omitted} more unique dataset issues were omitted.")
    return summarized


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


def read_yaml_limited(path: Path, *, label: str = "dataset YAML") -> Any:
    _ensure_file_size(path, MAX_YAML_FILE_BYTES, label)
    text = path.read_text(encoding="utf-8")
    depth = 0
    event_count = 0
    try:
        for event in yaml.parse(text, Loader=yaml.SafeLoader):
            event_count += 1
            if event_count > MAX_YAML_EVENTS:
                raise ValueError(f"{label} exceeds {MAX_YAML_EVENTS} YAML events")
            if isinstance(event, yaml.events.AliasEvent):
                raise ValueError(f"{label} must not contain YAML aliases")
            if isinstance(event, (yaml.events.MappingStartEvent, yaml.events.SequenceStartEvent)):
                depth += 1
                if depth > MAX_YAML_DEPTH:
                    raise ValueError(f"{label} exceeds the maximum YAML nesting depth of {MAX_YAML_DEPTH}")
            elif isinstance(event, (yaml.events.MappingEndEvent, yaml.events.SequenceEndEvent)):
                depth -= 1
        return yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValueError(f"Could not parse {label}: {exc}") from exc


def _ensure_pixel_budget(width: float, height: float, label: str) -> None:
    if not math.isfinite(width) or not math.isfinite(height) or width <= 0 or height <= 0:
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
        dirs[:] = [name for name in dirs if not name.startswith(".")]
        for file in files:
            yield Path(root) / file


def _is_export_path(path: Path) -> bool:
    return EXPORTS_DIR_NAME in path.parts


def _iter_source_files(dataset_dir: Path):
    for root, dirs, files in os.walk(dataset_dir):
        dirs[:] = [name for name in dirs if not name.startswith(".")]
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
    for root, dirs, files in os.walk(dataset_dir):
        dirs[:] = [name for name in dirs if not name.startswith(".")]
        for name in ("data.yaml", "dataset.yaml"):
            if name in files:
                return Path(root) / name
    return None


def read_yaml_classes(yaml_path: Path | None) -> list[str]:
    if not yaml_path or not yaml_path.exists():
        return []
    data = read_yaml_limited(yaml_path) or {}
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


def _yolo_label_image(dataset_dir: Path, label_path: Path) -> Path | None:
    try:
        relative_parts = list(label_path.relative_to(dataset_dir).parts)
    except ValueError:
        return None
    label_indexes = [index for index, part in enumerate(relative_parts) if part.lower() == "labels"]
    if label_indexes:
        image_parts = list(relative_parts)
        image_parts[label_indexes[-1]] = "images"
        image_base = dataset_dir.joinpath(*image_parts).with_suffix("")
        for extension in IMAGE_EXTENSIONS:
            candidate = image_base.with_suffix(extension)
            if candidate.is_file():
                return candidate
    matches = [
        path
        for path in dataset_dir.rglob(f"{label_path.stem}.*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return matches[0] if len(matches) == 1 else None


def _inspect_yolo_labels(
    dataset_dir: Path,
    class_count: int = 0,
    max_files: int | None = None,
) -> dict[str, Any]:
    stats = {
        "files": 0,
        "box_rows": 0,
        "polygon_rows": 0,
        "invalid_rows": 0,
        "orphan_labels": 0,
        "polygon_examples": [],
        "errors": [],
    }

    for label_path in _iter_yolo_label_files(dataset_dir):
        if max_files is not None and stats["files"] >= max_files:
            break
        stats["files"] += 1
        if _yolo_label_image(dataset_dir, label_path) is None:
            stats["orphan_labels"] += 1
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
                    if len(stats["polygon_examples"]) < 5:
                        stats["polygon_examples"].append(
                            f"{label_path.relative_to(dataset_dir).as_posix()}:{row_count}"
                        )
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


def _imagefolder_split_classes(dataset_dir: Path, split_names: tuple[str, ...]) -> list[str]:
    for split_name in split_names:
        split_dir = dataset_dir / split_name
        if not split_dir.is_dir():
            continue
        return sorted(
            item.name
            for item in split_dir.iterdir()
            if item.is_dir() and any(path.suffix.lower() in IMAGE_EXTENSIONS for path in item.rglob("*"))
        )
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


def _roboflow_semantic_pairs(split_dir: Path) -> list[tuple[Path, Path]]:
    if not (split_dir / "_classes.csv").is_file():
        return []
    files = [path for path in split_dir.iterdir() if path.is_file()]
    masks = [path for path in files if path.suffix.lower() == ".png"]
    images = [
        path
        for path in files
        if path.suffix.lower() in IMAGE_EXTENSIONS and not path.stem.lower().endswith("_mask")
    ]
    pairs: list[tuple[Path, Path]] = []
    for image_path in images:
        candidates = [
            split_dir / f"{image_path.stem}_mask.png",
            split_dir / f"{image_path.stem}.png",
        ]
        mask_path = next(
            (candidate for candidate in candidates if candidate != image_path and candidate.is_file()),
            None,
        )
        if mask_path is not None:
            pairs.append((image_path, mask_path))

    # Roboflow's reference loader pairs sorted JPG images and PNG masks by position.
    if not pairs:
        source_images = sorted(path for path in images if path.suffix.lower() != ".png")
        sorted_masks = sorted(masks)
        if source_images and len(source_images) == len(sorted_masks):
            pairs = list(zip(source_images, sorted_masks))
    return pairs


def _roboflow_semantic_classes(dataset_dir: Path) -> list[str]:
    for split in ("train", "training", "valid", "val", "validation", "test"):
        classes_path = dataset_dir / split / "_classes.csv"
        if not classes_path.is_file():
            continue
        try:
            rows = csv.reader(_iter_text_lines_limited(classes_path, label="semantic class mapping"))
            classes: list[tuple[int, str]] = []
            for row in rows:
                if len(row) < 2:
                    continue
                try:
                    class_id = int(row[0].strip())
                except ValueError:
                    continue
                name = row[1].strip()
                if class_id > 0 and name and name.lower() != "background":
                    classes.append((class_id, name))
            if classes:
                return [name for _class_id, name in sorted(classes)]
        except (OSError, ValueError):
            continue
    return []


def _inspect_semantic_masks(dataset_dir: Path, max_pairs: int | None = None) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "train_images": 0,
        "image_files": 0,
        "mask_files": 0,
        "missing_masks": [],
        "splits": [],
        "errors": [],
        "roboflow_png_masks": False,
        "classes": [],
    }
    for split in ("train", "val", "test"):
        split_dirs = _semantic_split_dirs(dataset_dir, split)
        if split_dirs is not None:
            images_dir, masks_dir = split_dirs
            images = [path for path in images_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS]
            mask_files = [path for path in masks_dir.rglob("*") if path.suffix.lower() in MASK_EXTENSIONS]
            pairs = [(image_path, _matching_mask_path(masks_dir, image_path)) for image_path in images]
        else:
            split_dir = next(
                (dataset_dir / name for name in _split_candidates(split) if (dataset_dir / name).is_dir()),
                None,
            )
            roboflow_pairs = _roboflow_semantic_pairs(split_dir) if split_dir is not None else []
            if not roboflow_pairs:
                continue
            stats["roboflow_png_masks"] = True
            pairs = [(image_path, mask_path) for image_path, mask_path in roboflow_pairs]
            images = [image_path for image_path, _mask_path in roboflow_pairs]
            mask_files = [mask_path for _image_path, mask_path in roboflow_pairs]
        if images:
            stats["splits"].append(split)
        if split == "train":
            stats["train_images"] = len(images)
        stats["image_files"] += len(images)
        stats["mask_files"] += len(mask_files)
        inspected_pairs = 0
        for image_path, mask_path in pairs:
            if max_pairs is not None and inspected_pairs >= max_pairs:
                break
            inspected_pairs += 1
            if mask_path is None:
                try:
                    stats["missing_masks"].append(str(image_path.relative_to(dataset_dir)))
                except ValueError:
                    stats["missing_masks"].append(image_path.name)
                continue
            try:
                from PIL import Image

                if not hasattr(Image, "open"):
                    continue

                with Image.open(image_path) as image, Image.open(mask_path) as mask:
                    if image.size != mask.size:
                        stats["errors"].append(
                            f"Semantic image {image_path.name} is {image.width}x{image.height} but mask "
                            f"{mask_path.name} is {mask.width}x{mask.height}."
                        )
            except Exception as exc:
                stats["errors"].append(f"Could not inspect semantic pair {image_path.name}: {exc}")
    if stats["roboflow_png_masks"]:
        stats["classes"] = _roboflow_semantic_classes(dataset_dir)
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
        x, y, width, height = [float(value) for value in bbox]
    except (TypeError, ValueError):
        return False
    return all(math.isfinite(value) for value in (x, y, width, height)) and width > 0 and height > 0


def _validate_coco_structure(data: dict[str, Any], label: str) -> None:
    images = data.get("images", [])
    annotations = data.get("annotations", [])
    categories = data.get("categories", [])
    if not isinstance(images, list) or not isinstance(annotations, list) or not isinstance(categories, list):
        raise ValueError(f"COCO file {label} must contain list images, annotations, and categories.")

    def valid_id(value: Any) -> bool:
        return not isinstance(value, bool) and isinstance(value, (int, str)) and str(value).strip() != ""

    image_ids: set[Any] = set()
    for image in images:
        if not isinstance(image, dict) or not valid_id(image.get("id")):
            raise ValueError(f"COCO file {label} contains an image row without a valid id.")
        if image["id"] in image_ids:
            raise ValueError(f"COCO file {label} contains duplicate image id {image['id']}.")
        if not isinstance(image.get("file_name"), str) or not image["file_name"].strip():
            raise ValueError(f"COCO file {label} image {image['id']} has no file_name.")
        image_ids.add(image["id"])

    category_ids: set[Any] = set()
    for category in categories:
        if not isinstance(category, dict) or not valid_id(category.get("id")):
            raise ValueError(f"COCO file {label} contains a category row without a valid id.")
        if category["id"] in category_ids:
            raise ValueError(f"COCO file {label} contains duplicate category id {category['id']}.")
        category_ids.add(category["id"])
    if not category_ids:
        raise ValueError(f"COCO file {label} requires at least one category.")

    for annotation in annotations:
        if not isinstance(annotation, dict):
            raise ValueError(f"COCO file {label} contains a non-object annotation row.")
        if annotation.get("image_id") not in image_ids:
            raise ValueError(
                f"COCO file {label} annotation references unknown image id {annotation.get('image_id')}."
            )
        if annotation.get("category_id") not in category_ids:
            raise ValueError(
                f"COCO file {label} annotation references unknown category id {annotation.get('category_id')}."
            )


def _valid_coco_segmentation(segmentation: Any, image_dimensions: tuple[float, float] | None = None) -> bool:
    if isinstance(segmentation, list):
        total_points = 0
        for polygon in segmentation:
            if not isinstance(polygon, list) or len(polygon) < 6 or len(polygon) % 2 != 0:
                return False
            try:
                coordinates = [float(value) for value in polygon]
            except (TypeError, ValueError):
                return False
            if not all(math.isfinite(value) for value in coordinates):
                return False
            total_points += len(polygon) // 2
            if total_points > MAX_COCO_POLYGON_POINTS:
                raise ValueError(f"COCO polygon segmentation exceeds {MAX_COCO_POLYGON_POINTS} points")
        return bool(segmentation) and total_points >= 3
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
        counts = segmentation.get("counts")
        if not isinstance(counts, (str, bytes, list)) or not counts:
            return False
        if len(counts) > MAX_COCO_RLE_COUNTS:
            raise ValueError(f"COCO RLE counts exceed {MAX_COCO_RLE_COUNTS} values")
        if isinstance(counts, list):
            if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counts):
                return False
            if sum(counts) != width * height:
                return False
        elif isinstance(counts, str):
            try:
                counts.encode("ascii")
            except UnicodeEncodeError:
                return False
        if image_dimensions is not None:
            image_width, image_height = image_dimensions
            if int(round(image_width)) != width or int(round(image_height)) != height:
                return False
        return True
    return False


def _inspect_coco_files(dataset_dir: Path, sample_records: int | None = None) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "files": _find_coco_files(dataset_dir),
        "valid_files": [],
        "missing_images": 0,
        "invalid_annotations": 0,
        "box_annotations": 0,
        "mask_annotations": 0,
        "polygon_mask_annotations": 0,
        "rle_mask_annotations": 0,
        "classes": [],
        "used_classes": [],
        "errors": [],
        "warnings": [],
        "invalid_segmentations": 0,
        "missing_segmentations": 0,
        "category_ids": [],
        "used_category_ids": [],
    }
    class_names: dict[Any, str] = {}
    used_category_ids: set[Any] = set()
    invalid_coco_segmentations = 0
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
        try:
            _validate_coco_structure(data, relative_file)
        except ValueError as exc:
            stats["errors"].append(str(exc))
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
        inspected_images = images[:sample_records] if sample_records is not None else images
        inspected_annotations = annotations[: sample_records * 20] if sample_records is not None else annotations
        annotations_per_image: dict[Any, int] = {}
        masks_per_image: dict[Any, int] = {}
        for annotation in inspected_annotations:
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
                    category_id = category.get("id")
                    category_name = str(category.get("name", category_id))
                    if category_id in class_names and class_names[category_id] != category_name:
                        stats["errors"].append(
                            f"COCO category id {category_id} has conflicting names '{class_names[category_id]}' and '{category_name}'."
                        )
                    class_names[category_id] = category_name

        found_image_ids = set()
        image_dimensions_by_id: dict[Any, tuple[float, float]] = {}
        for image in inspected_images:
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
        decoded_mask_pixels: dict[Any, int] = {}
        for annotation in inspected_annotations:
            if not isinstance(annotation, dict):
                stats["invalid_annotations"] += 1
                continue
            if annotation.get("image_id") not in found_image_ids:
                continue
            has_bbox = _valid_bbox(annotation.get("bbox"))
            if has_bbox:
                stats["box_annotations"] += 1
                file_box_annotations += 1
                used_category_ids.add(annotation.get("category_id"))
            elif annotation.get("bbox") is not None:
                stats["invalid_annotations"] += 1
            segmentation = annotation.get("segmentation")
            if has_bbox and segmentation is None:
                stats["missing_segmentations"] += 1
            if has_bbox and segmentation is not None:
                if _valid_coco_segmentation(segmentation, image_dimensions_by_id.get(annotation.get("image_id"))):
                    dimensions = image_dimensions_by_id.get(annotation.get("image_id"))
                    if dimensions is not None:
                        width, height = dimensions
                        image_id = annotation.get("image_id")
                        decoded_mask_pixels[image_id] = decoded_mask_pixels.get(image_id, 0) + int(width) * int(height)
                        if decoded_mask_pixels[image_id] > MAX_COCO_DECODED_MASK_PIXELS_PER_IMAGE:
                            stats["errors"].append(
                                f"COCO image {image_id} exceeds the decoded mask memory budget."
                            )
                            continue
                    stats["mask_annotations"] += 1
                    if isinstance(segmentation, list):
                        stats["polygon_mask_annotations"] += 1
                    else:
                        stats["rle_mask_annotations"] += 1
                    file_mask_annotations += 1
                else:
                    invalid_coco_segmentations += 1
        if found_image_ids and (file_box_annotations or file_mask_annotations):
            stats["valid_files"].append(relative_file)

    if invalid_coco_segmentations:
        stats["invalid_segmentations"] = invalid_coco_segmentations
        issue = (
            "COCO segmentation exceeds mask limits or does not match image dimensions "
            f"({invalid_coco_segmentations} annotations)."
        )
        if stats["box_annotations"]:
            stats["warnings"].append(f"{issue} Using COCO bounding boxes for object detection and ignoring invalid masks.")
        else:
            stats["errors"].append(issue)

    if class_names:
        sort_key = lambda item: (0, int(item[0])) if str(item[0]).isdigit() else (1, str(item[0]))
        sorted_categories = sorted(class_names.items(), key=sort_key)
        stats["category_ids"] = [category_id for category_id, _name in sorted_categories]
        stats["classes"] = [name for _category_id, name in sorted_categories]
        used_categories = [
            (category_id, name)
            for category_id, name in sorted_categories
            if category_id in used_category_ids
        ]
        stats["used_category_ids"] = [category_id for category_id, _name in used_categories]
        stats["used_classes"] = list(dict.fromkeys(name for _category_id, name in used_categories))
    return stats


def _has_coco_semantic_masks(metadata: dict[str, Any]) -> bool:
    coco = metadata.get("coco", {})
    return bool(
        coco.get("polygon_mask_annotations", 0)
        and not coco.get("rle_mask_annotations", 0)
        and not coco.get("invalid_segmentations", 0)
        and not coco.get("missing_segmentations", 0)
        and not coco.get("errors")
    )



def _inspect_source_pixel_budgets(dataset_dir: Path, max_files: int | None = None) -> list[str]:
    errors: list[str] = []
    inspected = 0
    for path in _iter_visible_files(dataset_dir):
        if path.suffix.lower() not in IMAGE_EXTENSIONS.union(MASK_EXTENSIONS):
            continue
        if max_files is not None and inspected >= max_files:
            break
        inspected += 1
        try:
            from PIL import Image

            if not hasattr(Image, "open"):
                continue

            with Image.open(path) as image:
                _ensure_pixel_budget(float(image.width), float(image.height), path.name)
                image.verify()
        except ValueError as exc:
            errors.append(str(exc))
        except Exception as exc:
            errors.append(f"Could not decode image or mask {path.name}: {exc}")
    return errors


def inspect_dataset(dataset_dir: Path, *, sample_files: int | None = None) -> dict[str, Any]:
    yaml_path = find_dataset_yaml(dataset_dir)
    yaml_error: str | None = None
    try:
        classes = read_yaml_classes(yaml_path)
    except (OSError, ValueError) as exc:
        classes = []
        yaml_error = str(exc)

    formats: list[str] = []
    tasks: list[str] = []
    warnings: list[str] = []
    errors: list[str] = []
    if yaml_error:
        errors.append(yaml_error)

    has_images = count_images(dataset_dir) > 0
    has_yolo_yaml = yaml_path is not None
    yolo_stats = _inspect_yolo_labels(dataset_dir, len(classes), max_files=sample_files)
    if yaml_path is not None and not yaml_error:
        try:
            yaml_data = read_yaml_limited(yaml_path) or {}
            if not isinstance(yaml_data, dict) or "train" not in yaml_data:
                raise ValueError("YOLO YAML requires a train path.")
            for yaml_key in ("train", "val", "test"):
                raw_value = yaml_data.get(yaml_key)
                if raw_value is None:
                    continue
                values = raw_value if isinstance(raw_value, list) else [raw_value]
                for value in values:
                    normalized = _normalize_yolo_yaml_value(dataset_dir, yaml_key, value)
                    resolved = _safe_child(dataset_dir, normalized)
                    if resolved is None or not resolved.is_dir():
                        raise ValueError(f"YOLO YAML '{yaml_key}' path must be an image directory: {value}")
                    if "images" not in {part.lower() for part in resolved.relative_to(dataset_dir).parts}:
                        raise ValueError(f"YOLO YAML '{yaml_key}' path must include an images directory: {value}")
        except (OSError, ValueError) as exc:
            yolo_stats["errors"].append(str(exc))
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
        if len(imagefolder_classes) < 2:
            errors.append("Image classification requires at least two train class folders.")
        validation_classes = _imagefolder_split_classes(dataset_dir, ("valid", "val", "validation"))
        unknown_validation_classes = sorted(set(validation_classes) - set(imagefolder_classes))
        if unknown_validation_classes:
            errors.append(
                "Validation contains classes not present in training: " + ", ".join(unknown_validation_classes)
            )

    semantic_stats = _inspect_semantic_masks(dataset_dir, max_pairs=sample_files)
    has_semantic_masks = semantic_stats["train_images"] > 0 and semantic_stats["mask_files"] > 0
    coco_stats = _inspect_coco_files(dataset_dir, sample_records=sample_files)

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
    warnings.extend(coco_stats.get("warnings", []))
    errors.extend(_inspect_source_pixel_budgets(dataset_dir, max_files=sample_files))
    if has_yolo_yaml and yolo_stats["invalid_rows"]:
        errors.append(f"YOLO labels contain {yolo_stats['invalid_rows']} invalid rows.")
    if has_yolo_yaml and yolo_stats["box_rows"] and yolo_stats["polygon_rows"]:
        examples = ", ".join(yolo_stats.get("polygon_examples", []))
        suffix = f" Polygon rows: {examples}." if examples else ""
        errors.append("YOLO labels must not mix bounding-box and polygon rows in one dataset." + suffix)
    if has_yolo_yaml and yolo_stats.get("orphan_labels"):
        errors.append(
            f"YOLO contains {yolo_stats['orphan_labels']} label files without a uniquely matching image."
        )
    if has_semantic_masks:
        formats.append("semantic_masks")
        tasks.append("segmentation")
    if semantic_stats["missing_masks"]:
        preview = ", ".join(semantic_stats["missing_masks"][:5])
        errors.append(f"Semantic masks are missing for {len(semantic_stats['missing_masks'])} images: {preview}")
    errors.extend(semantic_stats.get("errors", []))
    has_reliable_coco_masks = (
        coco_stats["mask_annotations"] > 0
        and not coco_stats.get("invalid_segmentations")
        and not coco_stats.get("missing_segmentations")
        and not coco_stats.get("errors")
    )
    if coco_stats["box_annotations"] or coco_stats["mask_annotations"]:
        formats.append("coco_instances")
        if coco_stats["box_annotations"]:
            tasks.append("object_detection")
        if has_reliable_coco_masks:
            tasks.append("segmentation")
        if not classes and coco_stats["classes"]:
            classes = coco_stats.get("used_classes") or coco_stats["classes"]
    if not classes and semantic_stats.get("classes"):
        classes = semantic_stats["classes"]
    if coco_stats["files"] and coco_stats["missing_images"]:
        errors.append(f"COCO annotations reference {coco_stats['missing_images']} image files that were not found in the dataset.")
    if coco_stats["invalid_annotations"]:
        warnings.append(f"COCO annotations include {coco_stats['invalid_annotations']} invalid annotation rows that will be ignored.")
    if coco_stats.get("missing_segmentations"):
        warnings.append(
            f"COCO contains {coco_stats['missing_segmentations']} bounding boxes without instance masks; "
            "Mask R-CNN will not be offered for this dataset."
        )
    formats = sorted(set(formats))
    tasks = sorted(set(tasks))
    if has_images and not formats:
        warnings.append("Images were found, but no supported annotation structure was detected.")
    if not has_images:
        warnings.append("No image files were found.")

    image_count = count_images(dataset_dir)
    if has_semantic_masks and not any(
        format_name in formats for format_name in ("imagefolder", "yolo_detection", "yolo_segmentation", "coco_instances")
    ):
        image_count = semantic_stats["image_files"]

    return {
        "formats": formats,
        "tasks": tasks,
        "classes": classes,
        "image_count": image_count,
        "size_bytes": directory_size(dataset_dir),
        "yaml_path": str(yaml_path) if yaml_path else None,
        "coco_files": coco_stats["valid_files"],
        "coco": coco_stats,
        "yolo": yolo_stats,
        "semantic_masks": semantic_stats,
        "warnings": _summarize_issue_list(warnings),
        "errors": _summarize_issue_list(errors),
    }


def validate_dataset_for_upload(
    dataset_dir: Path,
    metadata: dict[str, Any] | None = None,
    *,
    strict: bool = True,
) -> dict[str, Any]:
    metadata = metadata or inspect_dataset(dataset_dir)
    if metadata["image_count"] == 0:
        raise ValueError("No supported image files were found in the uploaded dataset.")
    if strict and metadata.get("errors"):
        raise ValueError(" ".join(metadata["errors"]))
    if not metadata["formats"]:
        raise ValueError(
            "Unsupported dataset structure. Supported trainable formats: YOLO detection data.yaml + labels, "
            "ImageFolder train/<class>, semantic masks, or COCO boxes/instances."
        )
    trainable_formats = sorted(set(metadata["formats"]).intersection(TRAINABLE_FORMATS))
    if not trainable_formats:
        raise ValueError(
            f"Detected formats {metadata['formats']} are not trainable by the current model catalog. "
            f"Supported trainable formats: {sorted(TRAINABLE_FORMATS)}."
        )
    return metadata


def inspect_dataset_for_upload(dataset_dir: Path) -> dict[str, Any]:
    sample_files = max(1, int(os.getenv("AILAB_UPLOAD_INSPECTION_SAMPLE_FILES", "1")))
    metadata = inspect_dataset(dataset_dir, sample_files=sample_files)
    # Upload only checks the first representative file for each supported layout.
    # A bad sample is rejected; later bad files are handled during training.
    metadata["warnings"] = []
    metadata["validation_deferred"] = True
    metadata["inspection_sample_files"] = sample_files
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
    return "unknown"


def _dataset_tasks_from_metadata(metadata: dict[str, Any]) -> list[str]:
    formats = set(metadata.get("formats", []))
    dataset_tasks: list[str] = []
    coco = metadata.get("coco", {})

    if "imagefolder" in formats:
        dataset_tasks.append("image_classification")
    if "yolo_detection" in formats or coco.get("box_annotations", 0) > 0:
        dataset_tasks.append("object_detection")
    if "semantic_masks" in formats or _has_coco_semantic_masks(metadata):
        dataset_tasks.append("semantic_segmentation")
    if coco.get("mask_annotations", 0) > 0 and not coco.get("invalid_segmentations", 0):
        dataset_tasks.append("instance_segmentation")
    return list(dict.fromkeys(dataset_tasks))


def _canonical_task_from_metadata(metadata: dict[str, Any]) -> str:
    dataset_tasks = _dataset_tasks_from_metadata(metadata)
    if len(dataset_tasks) == 1:
        return dataset_tasks[0]
    if "semantic_segmentation" in dataset_tasks and "instance_segmentation" in dataset_tasks:
        return "segmentation"
    if "instance_segmentation" in dataset_tasks:
        return "instance_segmentation"
    if "semantic_segmentation" in dataset_tasks:
        return "semantic_segmentation"
    if "object_detection" in dataset_tasks:
        return "object_detection"
    if dataset_tasks:
        return "multi_task"
    return "unknown"


def _canonical_format_from_metadata(metadata: dict[str, Any]) -> str:
    formats = set(metadata.get("formats", []))
    coco = metadata.get("coco", {})

    if "imagefolder" in formats:
        return "imagefolder"
    if "semantic_masks" in formats:
        return "semantic_masks"
    if coco.get("mask_annotations", 0) > 0 and not coco.get("invalid_segmentations", 0):
        return "coco_segmentation"
    if "yolo_detection" in formats or coco.get("box_annotations", 0) > 0:
        return "object_detection_boxes"
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
    }


def _has_bounding_boxes(metadata: dict[str, Any]) -> bool:
    formats = set(metadata.get("formats", []))
    return "yolo_detection" in formats or metadata.get("coco", {}).get("box_annotations", 0) > 0


def _model_compatibility_reason(model_id: str, metadata: dict[str, Any], task_id: str) -> tuple[bool, str]:
    formats = set(metadata.get("formats", []))
    tasks = set(metadata.get("tasks", []))

    if model_id == "yolo":
        ok = _has_bounding_boxes(metadata) and "object_detection" in tasks
        return ok, "Requires bounding boxes; COCO boxes are exported to YOLO at train time."
    if model_id == "faster_rcnn":
        ok = _has_bounding_boxes(metadata) and "object_detection" in tasks
        return ok, "Requires bounding boxes in YOLO or COCO; COCO can be used directly."
    if model_id == "mask_rcnn":
        coco = metadata.get("coco", {})
        ok = (
            "coco_instances" in formats
            and coco.get("mask_annotations", 0) > 0
            and not coco.get("invalid_segmentations", 0)
            and not coco.get("missing_segmentations", 0)
            and not coco.get("errors")
        )
        return ok, "Requires COCO instance masks, not box-only annotations."
    if model_id == "deeplabv3plus":
        ok = "semantic_masks" in formats or _has_coco_semantic_masks(metadata)
        return ok, "Requires semantic masks; COCO polygon masks are converted to class-ID masks at train time."
    if model_id in {"resnet", "efficientnet"}:
        return "imagefolder" in formats, "Requires image classification class folders."
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
    config = read_yaml_limited(original_yaml) or {}
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

    classes = metadata.get("coco", {}).get("classes", []) or metadata.get("classes", []) or ["object"]
    canonical_category_ids = metadata.get("coco", {}).get("category_ids", [])
    canonical_category_to_index = {
        category_id: index for index, category_id in enumerate(canonical_category_ids)
    }
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

        category_to_index = canonical_category_to_index or {
            category.get("id"): index
            for index, category in enumerate(categories)
            if isinstance(category, dict) and "id" in category
        }
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
                category_id = annotation.get("category_id")
                if category_id not in category_to_index:
                    raise ValueError(f"COCO annotation references unknown category id {category_id}.")
                class_index = category_to_index[category_id]
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


def _write_coco_semantic_export(dataset_dir: Path, output_root: Path, metadata: dict[str, Any]) -> list[str]:
    from PIL import Image, ImageDraw

    coco_files = metadata.get("coco_files", [])
    documents: list[tuple[Path, dict[str, Any]]] = []
    category_ids: set[Any] = set()
    for relative_file in coco_files:
        annotation_path = dataset_dir / relative_file
        data = _read_json_limited(annotation_path)
        if not isinstance(data, dict):
            continue
        _validate_coco_structure(data, relative_file)
        documents.append((annotation_path, data))
        for annotation in data.get("annotations", []):
            if isinstance(annotation, dict) and isinstance(annotation.get("segmentation"), list):
                category_ids.add(annotation.get("category_id"))
    if not category_ids:
        raise ValueError(f"Dataset '{dataset_dir.name}' has no COCO polygon masks to export for DeepLabV3+.")

    sort_key = lambda value: (0, int(value)) if str(value).isdigit() else (1, str(value))
    category_to_class = {
        category_id: index + 1
        for index, category_id in enumerate(sorted(category_ids, key=sort_key))
    }
    image_index = _build_image_basename_index(dataset_dir)
    wrote_train = False
    warnings: list[str] = []

    for annotation_path, data in documents:
        annotations_by_image: dict[Any, list[dict[str, Any]]] = {}
        for annotation in data.get("annotations", []):
            if isinstance(annotation, dict):
                annotations_by_image.setdefault(annotation.get("image_id"), []).append(annotation)
        split = _annotation_split(annotation_path, dataset_dir)
        split = "val" if split in {"valid", "validation"} else split
        images_dir = output_root / split / "images"
        masks_dir = output_root / split / "masks"

        for image_record in data.get("images", []):
            if not isinstance(image_record, dict):
                continue
            source_image = _resolve_coco_image(
                dataset_dir, annotation_path, image_record.get("file_name"), image_index
            )
            if source_image is None:
                warnings.append(f"Skipping COCO image {image_record.get('file_name')}: file was not found.")
                continue
            dimensions = _image_dimensions(source_image, image_record)
            if dimensions is None:
                warnings.append(f"Skipping {source_image.name}: image dimensions were unavailable.")
                continue
            width, height = (int(round(value)) for value in dimensions)
            images_dir.mkdir(parents=True, exist_ok=True)
            masks_dir.mkdir(parents=True, exist_ok=True)
            target_image = _unique_child(images_dir, source_image.name, prefix=f"{image_record.get('id', '')}_")
            mask = Image.new("I", (width, height), 0)
            draw = ImageDraw.Draw(mask)
            drew_mask = False
            for annotation in annotations_by_image.get(image_record.get("id"), []):
                segmentation = annotation.get("segmentation")
                class_id = category_to_class.get(annotation.get("category_id"))
                if class_id is None or not isinstance(segmentation, list):
                    continue
                if not _valid_coco_segmentation(segmentation, (width, height)):
                    continue
                for polygon in segmentation:
                    points = [(polygon[index], polygon[index + 1]) for index in range(0, len(polygon), 2)]
                    draw.polygon(points, fill=class_id)
                drew_mask = True
            if not drew_mask:
                continue
            shutil.copy2(source_image, target_image)
            mask.convert("I;16").save(masks_dir / f"{target_image.stem}.png")
            wrote_train = wrote_train or split == "train"

    if not wrote_train:
        raise ValueError(f"Dataset '{dataset_dir.name}' did not produce a DeepLabV3+ training split.")
    return warnings


def _write_roboflow_semantic_export(dataset_dir: Path, output_root: Path) -> list[str]:
    wrote_train = False
    for split in ("train", "val", "test"):
        split_dir = next(
            (dataset_dir / name for name in _split_candidates(split) if (dataset_dir / name).is_dir()),
            None,
        )
        if split_dir is None:
            continue
        pairs = _roboflow_semantic_pairs(split_dir)
        if not pairs:
            continue
        images_dir = output_root / split / "images"
        masks_dir = output_root / split / "masks"
        images_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)
        for source_image, source_mask in pairs:
            target_image = _unique_child(images_dir, source_image.name)
            shutil.copy2(source_image, target_image)
            shutil.copy2(source_mask, masks_dir / f"{target_image.stem}{source_mask.suffix.lower()}")
        wrote_train = wrote_train or split == "train"
    if not wrote_train:
        raise ValueError(f"Dataset '{dataset_dir.name}' has no Roboflow semantic training pairs.")
    return []



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
    lock_name = f"export-{model_type}-{fingerprint}"
    with named_file_lock(dataset_dir, lock_name, "dataset export"):
        if _cache_is_valid(export_root, source_fingerprint, export_format, required_files):
            return export_root, True, fingerprint, []
        staging_root = export_root.with_name(f".{export_root.name}-{uuid.uuid4().hex}.tmp")
        shutil.rmtree(staging_root, ignore_errors=True)
        staging_root.mkdir(parents=True, exist_ok=False)
        try:
            warnings = writer(staging_root)
            _write_export_manifest(staging_root, model_type, export_format, source_fingerprint, fingerprint, warnings)
            replace_directory(staging_root, export_root)
        finally:
            shutil.rmtree(staging_root, ignore_errors=True)
        return export_root, False, fingerprint, warnings


def prepare_dataset_for_model(dataset_dir: Path, model_type: str, extra_args: dict[str, Any] | None = None) -> dict[str, Any]:
    extra_args = extra_args or {}
    metadata = inspect_dataset(dataset_dir)
    formats = set(metadata.get("formats", []))

    if model_type in {"resnet", "efficientnet"}:
        if "imagefolder" not in formats:
            raise ValueError(f"Dataset '{dataset_dir.name}' has no image classification class folders.")
        return _prepared_response(dataset_dir, model_type, metadata, dataset_dir, "imagefolder")

    if model_type == "deeplabv3plus":
        if "semantic_masks" in formats:
            if metadata.get("semantic_masks", {}).get("roboflow_png_masks"):
                export_root, cache_hit, fingerprint, warnings = _cached_generated_export(
                    dataset_dir,
                    model_type,
                    extra_args,
                    "semantic_masks",
                    ["train/images", "train/masks"],
                    lambda output_root: _write_roboflow_semantic_export(dataset_dir, output_root),
                )
                return _prepared_response(
                    dataset_dir,
                    model_type,
                    metadata,
                    export_root,
                    "semantic_masks",
                    export_path=export_root,
                    cache_hit=cache_hit,
                    fingerprint=fingerprint,
                    warnings=warnings,
                )
            return _prepared_response(dataset_dir, model_type, metadata, dataset_dir, "semantic_masks")
        if _has_coco_semantic_masks(metadata):
            export_root, cache_hit, fingerprint, warnings = _cached_generated_export(
                dataset_dir,
                model_type,
                extra_args,
                "semantic_masks",
                ["train/images", "train/masks"],
                lambda output_root: _write_coco_semantic_export(dataset_dir, output_root, metadata),
            )
            return _prepared_response(
                dataset_dir,
                model_type,
                metadata,
                export_root,
                "semantic_masks",
                export_path=export_root,
                cache_hit=cache_hit,
                fingerprint=fingerprint,
                warnings=warnings,
            )
        raise ValueError(f"Dataset '{dataset_dir.name}' has no semantic masks or convertible COCO polygons for DeepLabV3+.")

    if model_type == "mask_rcnn":
        coco = metadata.get("coco", {})
        if (
            coco.get("mask_annotations", 0) <= 0
            or coco.get("invalid_segmentations", 0)
            or coco.get("missing_segmentations", 0)
            or coco.get("errors")
        ):
            raise ValueError(f"Dataset '{dataset_dir.name}' does not have a valid instance mask for every COCO box.")
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

    raise ValueError(f"No dataset export rule is defined for model '{model_type}'.")
