from __future__ import annotations

import json
import math
from pathlib import Path

from PIL import Image, ImageDraw

from security_utils import contained_path
from .input_limits import (
    MAX_BOXES_PER_IMAGE,
    MAX_COCO_DECODED_MASK_PIXELS_PER_IMAGE,
    iter_text_lines_limited,
    open_rgb_image_checked,
    read_json_limited,
    validate_coco_document,
    validate_coco_segmentation,
)
from .trainer_utils import IMAGE_EXTENSIONS, list_images, read_classes_from_yaml, read_yaml, yolo_split_image_dirs


class YoloBoxDataset:
    def __init__(self, dataset_path: str, data_yaml_path: str | None, split: str = "train"):
        self.dataset_path = Path(dataset_path)
        self.data_yaml_path = data_yaml_path
        self.images_dirs = yolo_split_image_dirs(dataset_path, data_yaml_path, split)
        self.images = sorted({image for directory in self.images_dirs for image in list_images(directory)})
        self.warnings: list[str] = []
        usable_images = []
        for image_path in self.images:
            try:
                image = open_rgb_image_checked(image_path)
                image.close()
                usable_images.append(image_path)
            except Exception as exc:
                self.warnings.append(f"Skipping {split} YOLO image '{image_path}': {exc}")
        self.images = usable_images
        if not self.images:
            raise ValueError(f"No images found for split '{split}' in the YOLO YAML paths")
        self.classes = read_classes_from_yaml(data_yaml_path)
        if not self.classes:
            raise ValueError("YOLO YAML requires at least one class name")
        for image_path in self.images:
            try:
                label_path = self._label_path(image_path)
                if not label_path.exists():
                    continue
                invalid_rows = 0
                for line in iter_text_lines_limited(label_path, label="YOLO label file"):
                    parts = line.strip().split()
                    try:
                        values = [float(value) for value in parts]
                        class_id = int(values[0])
                        valid = (
                            len(values) == 5
                            and values[0] == class_id
                            and 0 <= class_id < len(self.classes)
                            and all(math.isfinite(value) for value in values)
                            and values[3] > 0
                            and values[4] > 0
                        )
                    except (IndexError, TypeError, ValueError):
                        valid = False
                    if not valid:
                        invalid_rows += 1
                if invalid_rows:
                    self.warnings.append(
                        f"Skipping {invalid_rows} invalid label row(s) in '{label_path}'."
                    )
            except Exception as exc:
                self.warnings.append(f"Ignoring unusable YOLO label file for '{image_path}': {exc}")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        import torch
        import torchvision.transforms.functional as F

        image_path = self.images[index]
        image = open_rgb_image_checked(image_path)
        width, height = image.size
        label_path = self._label_path(image_path)

        boxes = []
        labels = []
        if label_path.exists():
            for row_index, line in enumerate(iter_text_lines_limited(label_path, label="YOLO label file"), start=1):
                if row_index > MAX_BOXES_PER_IMAGE:
                    raise ValueError(f"YOLO label file has more than {MAX_BOXES_PER_IMAGE} boxes: {label_path}")
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                try:
                    class_id = int(float(parts[0]))
                    x_center, y_center, box_width, box_height = [float(value) for value in parts[1:5]]
                except (TypeError, ValueError):
                    continue
                if class_id < 0 or class_id >= len(self.classes):
                    continue
                if not all(math.isfinite(value) for value in (x_center, y_center, box_width, box_height)):
                    continue
                x1 = (x_center - box_width / 2) * width
                y1 = (y_center - box_height / 2) * height
                x2 = (x_center + box_width / 2) * width
                y2 = (y_center + box_height / 2) * height
                if x2 > x1 and y2 > y1:
                    boxes.append([x1, y1, x2, y2])
                    labels.append(class_id + 1)

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([index]),
        }
        if boxes:
            target["area"] = (target["boxes"][:, 3] - target["boxes"][:, 1]) * (
                target["boxes"][:, 2] - target["boxes"][:, 0]
            )
        else:
            target["area"] = torch.zeros((0,), dtype=torch.float32)
        target["iscrowd"] = torch.zeros((len(labels),), dtype=torch.int64)
        return F.to_tensor(image), target

    def _label_path(self, image_path: Path) -> Path:
        relative = image_path.resolve().relative_to(self.dataset_path.resolve())
        parts = list(relative.parts)
        image_indexes = [index for index, part in enumerate(parts) if part.lower() == "images"]
        if image_indexes:
            parts[image_indexes[-1]] = "labels"
            return contained_path(self.dataset_path, *parts).with_suffix(".txt")
        raise ValueError(f"YOLO image path must contain an 'images' directory: {image_path}")


class CocoInstanceDataset:
    def __init__(
        self,
        dataset_path: str,
        split: str = "train",
        include_masks: bool = True,
        category_to_label: dict | None = None,
        classes: list[str] | None = None,
    ):
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.include_masks = include_masks
        annotation_path = self._find_annotation_file(split)
        if annotation_path is None:
            raise ValueError(f"No COCO annotation file found for split '{split}'.")

        data = read_json_limited(annotation_path)
        validate_coco_document(data, str(annotation_path))
        self.warnings: list[str] = []
        self.annotation_root = annotation_path.parent
        self.images_by_id = {image["id"]: image for image in data.get("images", [])}
        categories = sorted(
            data.get("categories", []),
            key=lambda category: (0, int(category["id"])) if str(category["id"]).isdigit() else (1, str(category["id"])),
        )
        category_ids = [category["id"] for category in categories]
        own_mapping = {category_id: index + 1 for index, category_id in enumerate(category_ids)}
        if category_to_label is not None:
            unknown = sorted(set(category_ids) - set(category_to_label), key=str)
            if unknown:
                raise ValueError(f"Validation COCO categories are not present in training: {unknown}")
            self.category_to_label = dict(category_to_label)
            self.classes = list(classes or [])
        else:
            self.category_to_label = own_mapping
            self.classes = [category.get("name", str(category.get("id"))) for category in categories]
        annotations_by_image: dict[int, list[dict]] = {}
        for annotation in data.get("annotations", []):
            annotations_by_image.setdefault(annotation["image_id"], []).append(annotation)
        self.items = []
        for image_id, image_info in list(self.images_by_id.items()):
            if not hasattr(Image, "open"):
                self.items.append((image_id, annotations_by_image.get(image_id, [])))
                continue
            try:
                image_path = self._resolve_image_path(image_info["file_name"])
                image = open_rgb_image_checked(image_path)
                image.close()
                self.items.append((image_id, annotations_by_image.get(image_id, [])))
            except Exception as exc:
                self.warnings.append(f"Skipping {split} COCO image id {image_id}: {exc}")
        if not self.items:
            raise ValueError(f"No COCO images found in {annotation_path}.")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        import torch
        import torchvision.transforms.functional as F

        image_id, annotations = self.items[index]
        image_info = self.images_by_id[image_id]
        image_path = self._resolve_image_path(image_info["file_name"])
        image = open_rgb_image_checked(image_path)
        width, height = image.size

        if self.include_masks:
            decoded_mask_pixels = sum(
                width * height
                for annotation in annotations
                if isinstance(annotation.get("bbox"), list) and len(annotation["bbox"]) == 4
            )
            if decoded_mask_pixels > MAX_COCO_DECODED_MASK_PIXELS_PER_IMAGE:
                raise ValueError(
                    f"COCO image {image_id} exceeds the decoded mask memory budget "
                    f"({decoded_mask_pixels} pixels)."
                )

        boxes = []
        labels = []
        masks = []
        iscrowd = []
        for annotation in annotations:
            bbox = annotation.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            x, y, box_width, box_height = bbox
            try:
                x, y, box_width, box_height = [float(value) for value in (x, y, box_width, box_height)]
            except (TypeError, ValueError):
                continue
            if not all(math.isfinite(value) for value in (x, y, box_width, box_height)):
                continue
            if box_width <= 0 or box_height <= 0:
                continue
            category_id = annotation.get("category_id")
            if category_id not in self.category_to_label:
                continue
            if self.include_masks:
                try:
                    mask = self._annotation_mask(annotation, width, height)
                except Exception:
                    continue
                masks.append(mask)
            boxes.append([x, y, x + box_width, y + box_height])
            labels.append(self.category_to_label[category_id])
            iscrowd.append(int(annotation.get("iscrowd", 0)))

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([image_id]),
            "iscrowd": torch.as_tensor(iscrowd, dtype=torch.int64),
        }
        if self.include_masks:
            if masks:
                target["masks"] = torch.stack(masks)
            else:
                target["masks"] = torch.zeros((0, height, width), dtype=torch.uint8)
        if boxes:
            target["area"] = (target["boxes"][:, 3] - target["boxes"][:, 1]) * (
                target["boxes"][:, 2] - target["boxes"][:, 0]
            )
        else:
            target["area"] = torch.zeros((0,), dtype=torch.float32)
        return F.to_tensor(image), target

    def _find_annotation_file(self, split: str) -> Path | None:
        split_names = [split]
        if split == "val":
            split_names.extend(["valid", "validation"])
        names = [name for split_name in split_names for name in (
            f"instances_{split_name}.json",
            f"{split_name}.json",
            "_annotations.coco.json",
        )]
        for split_name in split_names:
            split_dir = self.dataset_path / split_name
            if split_dir.is_dir():
                for name in names:
                    candidate = split_dir / name
                    if candidate.is_file():
                        return candidate
        root_names = [
            f"instances_{split_name}.json"
            for split_name in split_names
        ] + [f"{split_name}.json" for split_name in split_names]
        for name in root_names:
            candidate = self.dataset_path / name
            if candidate.is_file():
                return candidate
        if split == "train":
            candidates = [
                path
                for path in self.dataset_path.rglob("*.json")
                if "instances" in path.name.lower() or "coco" in path.name.lower()
            ]
            if len(candidates) == 1:
                return candidates[0]
        return None

    def _resolve_image_path(self, file_name: str) -> Path:
        direct = contained_path(self.dataset_path, file_name)
        if direct.exists():
            return direct
        for folder in ("images", self.split, f"{self.split}/images", "train/images", "valid/images", "val/images"):
            candidate = contained_path(self.dataset_path, folder, file_name)
            if candidate.exists():
                return candidate
        matches = list(self.dataset_path.rglob(Path(file_name).name))
        if matches:
            return matches[0]
        raise FileNotFoundError(f"COCO image '{file_name}' was not found under {self.dataset_path}")

    def _annotation_mask(self, annotation: dict, width: int, height: int):
        import numpy as np
        import torch

        mask_image = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask_image)
        segmentation = annotation.get("segmentation", [])
        if isinstance(segmentation, list):
            if not validate_coco_segmentation(segmentation, width, height):
                raise ValueError("COCO polygon segmentation is malformed")
            for polygon in segmentation:
                points = [(polygon[i], polygon[i + 1]) for i in range(0, len(polygon), 2)]
                draw.polygon(points, outline=1, fill=1)
        elif isinstance(segmentation, dict):
            validate_coco_segmentation(segmentation, width, height)
            try:
                from pycocotools import mask as coco_mask
            except ImportError as exc:
                raise RuntimeError("pycocotools is required for COCO RLE masks") from exc
            encoded = segmentation
            if isinstance(segmentation.get("counts"), list):
                encoded = coco_mask.frPyObjects(segmentation, height, width)
            try:
                decoded = coco_mask.decode(encoded)
            except Exception as exc:
                raise ValueError("COCO RLE segmentation could not be decoded") from exc
            if getattr(decoded, "ndim", 0) == 3:
                decoded = decoded[:, :, 0]
            return torch.as_tensor(decoded, dtype=torch.uint8)
        return torch.as_tensor(np.array(mask_image, dtype=np.uint8), dtype=torch.uint8)
