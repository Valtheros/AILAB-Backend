from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import yaml


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


def read_yaml_limited(path: Path, *, label: str = "dataset YAML") -> Any:
    ensure_file_size(path, MAX_YAML_FILE_BYTES, label)
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


def ensure_dimensions_within_budget(width: float, height: float, label: str) -> None:
    if not math.isfinite(width) or not math.isfinite(height) or width <= 0 or height <= 0:
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
    try:
        ensure_pil_image_budget(image, path.name)
        return image
    except Exception:
        image.close()
        raise


def validate_coco_document(data: Any, label: str) -> None:
    if not isinstance(data, dict):
        raise ValueError(f"{label} must be a JSON object")
    images = data.get("images", [])
    annotations = data.get("annotations", [])
    categories = data.get("categories", [])
    if not isinstance(images, list) or not isinstance(annotations, list) or not isinstance(categories, list):
        raise ValueError(f"{label} must contain list images, annotations, and categories")
    if len(images) > MAX_COCO_IMAGES:
        raise ValueError(f"{label} has more than {MAX_COCO_IMAGES} images")
    if len(annotations) > MAX_COCO_ANNOTATIONS:
        raise ValueError(f"{label} has more than {MAX_COCO_ANNOTATIONS} annotations")
    def valid_id(value: Any) -> bool:
        return not isinstance(value, bool) and isinstance(value, (int, str)) and str(value).strip() != ""

    image_ids: set[Any] = set()
    image_pixels: dict[Any, int] = {}
    for image in images:
        if not isinstance(image, dict) or not valid_id(image.get("id")):
            raise ValueError(f"{label} contains an image row without a valid id")
        image_id = image["id"]
        if image_id in image_ids:
            raise ValueError(f"{label} contains duplicate image id {image_id}")
        if not isinstance(image.get("file_name"), str) or not image["file_name"].strip():
            raise ValueError(f"{label} image {image_id} has no file_name")
        image_ids.add(image_id)
        if image.get("width") is not None and image.get("height") is not None:
            width, height = float(image["width"]), float(image["height"])
            ensure_dimensions_within_budget(width, height, f"COCO image {image_id}")
            image_pixels[image_id] = int(width) * int(height)

    category_ids: set[Any] = set()
    for category in categories:
        if not isinstance(category, dict) or not valid_id(category.get("id")):
            raise ValueError(f"{label} contains a category row without a valid id")
        category_id = category["id"]
        if category_id in category_ids:
            raise ValueError(f"{label} contains duplicate category id {category_id}")
        category_ids.add(category_id)
    if not category_ids:
        raise ValueError(f"{label} requires at least one category")

    annotations_per_image: dict[Any, int] = {}
    masks_per_image: dict[Any, int] = {}
    mask_pixels_per_image: dict[Any, int] = {}
    for annotation in annotations:
        if not isinstance(annotation, dict):
            raise ValueError(f"{label} contains a non-object annotation row")
        image_id = annotation.get("image_id")
        if image_id not in image_ids:
            raise ValueError(f"{label} annotation references unknown image id {image_id}")
        if annotation.get("category_id") not in category_ids:
            raise ValueError(f"{label} annotation references unknown category id {annotation.get('category_id')}")
        annotations_per_image[image_id] = annotations_per_image.get(image_id, 0) + 1
        if annotation.get("segmentation") is not None:
            masks_per_image[image_id] = masks_per_image.get(image_id, 0) + 1
            mask_pixels_per_image[image_id] = mask_pixels_per_image.get(image_id, 0) + image_pixels.get(image_id, 0)
    if any(count > MAX_COCO_ANNOTATIONS_PER_IMAGE for count in annotations_per_image.values()):
        raise ValueError(f"{label} has too many annotations for one image")
    if any(count > MAX_COCO_MASKS_PER_IMAGE for count in masks_per_image.values()):
        raise ValueError(f"{label} has too many masks for one image")
    if any(pixels > MAX_COCO_DECODED_MASK_PIXELS_PER_IMAGE for pixels in mask_pixels_per_image.values()):
        raise ValueError(f"{label} exceeds the decoded mask memory budget for one image")


def validate_coco_segmentation(segmentation: Any, width: int, height: int) -> bool:
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
        mask_height, mask_width = int(size[0]), int(size[1])
        counts = segmentation.get("counts")
        if not isinstance(counts, (str, bytes, list)) or not counts:
            return False
        if len(counts) > MAX_COCO_RLE_COUNTS:
            raise ValueError(f"COCO RLE counts exceed {MAX_COCO_RLE_COUNTS} values")
        if isinstance(counts, list):
            if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counts):
                return False
            if sum(counts) != mask_width * mask_height:
                return False
        elif isinstance(counts, str):
            try:
                counts.encode("ascii")
            except UnicodeEncodeError:
                return False
        ensure_dimensions_within_budget(mask_width, mask_height, "COCO RLE mask")
        if mask_width != int(width) or mask_height != int(height):
            raise ValueError(
                f"COCO RLE mask size {mask_width}x{mask_height} does not match image size {width}x{height}"
            )
        return True
    return False
