from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def _env_int(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


MAX_LABEL_FILE_BYTES = _env_int("AILAB_MAX_LABEL_FILE_BYTES", 2 * 1024 * 1024)
MAX_LABEL_ROWS = _env_int("AILAB_MAX_LABEL_ROWS", 100_000)
MAX_BOXES_PER_IMAGE = _env_int("AILAB_MAX_BOXES_PER_IMAGE", 10_000)
MAX_COCO_JSON_BYTES = _env_int("AILAB_MAX_COCO_JSON_BYTES", 64 * 1024 * 1024)
MAX_COCO_IMAGES = _env_int("AILAB_MAX_COCO_IMAGES", 200_000)
MAX_COCO_ANNOTATIONS = _env_int("AILAB_MAX_COCO_ANNOTATIONS", 1_000_000)
MAX_COCO_ANNOTATIONS_PER_IMAGE = _env_int("AILAB_MAX_COCO_ANNOTATIONS_PER_IMAGE", 10_000)
MAX_COCO_POLYGON_POINTS = _env_int("AILAB_MAX_COCO_POLYGON_POINTS", 20_000)
MAX_COCO_MASKS_PER_IMAGE = _env_int("AILAB_MAX_COCO_MASKS_PER_IMAGE", 1_000)
MAX_SOURCE_IMAGE_PIXELS = _env_int("AILAB_MAX_SOURCE_IMAGE_PIXELS", 100_000_000)


def ensure_file_size(path: Path, max_bytes: int, label: str) -> None:
    size = path.stat().st_size
    if size > max_bytes:
        raise ValueError(f"{label} exceeds {max_bytes} bytes: {path}")


def iter_text_lines_limited(
    path: Path,
    *,
    max_bytes: int = MAX_LABEL_FILE_BYTES,
    max_rows: int = MAX_LABEL_ROWS,
    label: str = "label file",
):
    ensure_file_size(path, max_bytes, label)
    with open(path, "r", encoding="utf-8", errors="ignore") as file:
        for index, line in enumerate(file, start=1):
            if index > max_rows:
                raise ValueError(f"{label} exceeds {max_rows} rows: {path}")
            yield line.rstrip("\r\n")


def read_json_limited(path: Path, *, max_bytes: int = MAX_COCO_JSON_BYTES, label: str = "COCO annotation file") -> Any:
    ensure_file_size(path, max_bytes, label)
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_dimensions_within_budget(width: float, height: float, label: str) -> None:
    if width <= 0 or height <= 0:
        raise ValueError(f"{label} has invalid dimensions {width}x{height}")
    pixels = int(width) * int(height)
    if pixels > MAX_SOURCE_IMAGE_PIXELS:
        raise ValueError(f"{label} has {pixels} pixels, above the {MAX_SOURCE_IMAGE_PIXELS} pixel limit")


def ensure_pil_image_budget(image: Any, label: str) -> None:
    ensure_dimensions_within_budget(float(image.width), float(image.height), label)


def open_rgb_image_checked(path: Path):
    from PIL import Image

    image = Image.open(path)
    try:
        ensure_pil_image_budget(image, path.name)
        return image.convert("RGB")
    finally:
        image.close()


def open_image_checked(path: Path):
    from PIL import Image

    image = Image.open(path)
    ensure_pil_image_budget(image, path.name)
    return image


def validate_coco_document(data: Any, label: str) -> None:
    if not isinstance(data, dict):
        raise ValueError(f"{label} must be a JSON object")
    images = data.get("images", [])
    annotations = data.get("annotations", [])
    if not isinstance(images, list) or not isinstance(annotations, list):
        raise ValueError(f"{label} must contain list images and annotations")
    if len(images) > MAX_COCO_IMAGES:
        raise ValueError(f"{label} has more than {MAX_COCO_IMAGES} images")
    if len(annotations) > MAX_COCO_ANNOTATIONS:
        raise ValueError(f"{label} has more than {MAX_COCO_ANNOTATIONS} annotations")
    annotations_per_image: dict[Any, int] = {}
    masks_per_image: dict[Any, int] = {}
    for annotation in annotations:
        if not isinstance(annotation, dict):
            continue
        image_id = annotation.get("image_id")
        annotations_per_image[image_id] = annotations_per_image.get(image_id, 0) + 1
        if annotation.get("segmentation") is not None:
            masks_per_image[image_id] = masks_per_image.get(image_id, 0) + 1
    if any(count > MAX_COCO_ANNOTATIONS_PER_IMAGE for count in annotations_per_image.values()):
        raise ValueError(f"{label} has too many annotations for one image")
    if any(count > MAX_COCO_MASKS_PER_IMAGE for count in masks_per_image.values()):
        raise ValueError(f"{label} has too many masks for one image")


def validate_coco_segmentation(segmentation: Any, width: int, height: int) -> bool:
    if isinstance(segmentation, list):
        total_points = 0
        for polygon in segmentation:
            if not isinstance(polygon, list) or len(polygon) < 6 or len(polygon) % 2 != 0:
                continue
            total_points += len(polygon) // 2
            if total_points > MAX_COCO_POLYGON_POINTS:
                raise ValueError(f"COCO polygon segmentation exceeds {MAX_COCO_POLYGON_POINTS} points")
            return True
        return False
    if isinstance(segmentation, dict):
        size = segmentation.get("size")
        if "counts" not in segmentation or not isinstance(size, (list, tuple)) or len(size) != 2:
            return False
        mask_height, mask_width = int(size[0]), int(size[1])
        ensure_dimensions_within_budget(mask_width, mask_height, "COCO RLE mask")
        if mask_width != int(width) or mask_height != int(height):
            raise ValueError(
                f"COCO RLE mask size {mask_width}x{mask_height} does not match image size {width}x{height}"
            )
        return True
    return False
