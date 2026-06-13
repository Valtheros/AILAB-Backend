from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

from security_utils import contained_path
from .trainer_utils import IMAGE_EXTENSIONS, list_images, read_classes_from_yaml, read_yaml, split_image_dir


class YoloBoxDataset:
    def __init__(self, dataset_path: str, data_yaml_path: str | None, split: str = "train"):
        self.dataset_path = Path(dataset_path)
        self.data_yaml_path = data_yaml_path
        self.images_dir = split_image_dir(dataset_path, split)
        self.images = list_images(self.images_dir)
        if not self.images:
            raise ValueError(f"No images found for split '{split}' in {self.images_dir}")
        self.classes = read_classes_from_yaml(data_yaml_path)
        if not self.classes:
            raise ValueError("YOLO YAML requires at least one class name")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        import torch
        import torchvision.transforms.functional as F

        image_path = self.images[index]
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        label_path = self._label_path(image_path)

        boxes = []
        labels = []
        if label_path.exists():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                class_id = int(float(parts[0]))
                if class_id < 0 or class_id >= len(self.classes):
                    raise ValueError(f"YOLO class ID {class_id} is outside 0..{len(self.classes) - 1} in {label_path}")
                x_center, y_center, box_width, box_height = [float(value) for value in parts[1:5]]
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
        parts = list(image_path.parts)
        for idx, part in enumerate(parts):
            if part == "images":
                parts[idx] = "labels"
                return Path(*parts).with_suffix(".txt")
        return image_path.parent.parent / "labels" / f"{image_path.stem}.txt"


class CocoInstanceDataset:
    def __init__(self, dataset_path: str, split: str = "train", include_masks: bool = True):
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.include_masks = include_masks
        annotation_path = self._find_annotation_file(split)
        if annotation_path is None:
            raise ValueError(f"No COCO annotation file found for split '{split}'.")

        data = json.loads(annotation_path.read_text(encoding="utf-8"))
        self.annotation_root = annotation_path.parent
        self.images_by_id = {image["id"]: image for image in data.get("images", [])}
        categories = sorted(data.get("categories", []), key=lambda category: category["id"])
        category_ids = [category["id"] for category in categories]
        self.category_to_label = {category_id: index + 1 for index, category_id in enumerate(category_ids)}
        self.classes = [category.get("name", str(category.get("id"))) for category in categories]
        annotations_by_image: dict[int, list[dict]] = {}
        for annotation in data.get("annotations", []):
            annotations_by_image.setdefault(annotation["image_id"], []).append(annotation)
        self.items = [(image_id, annotations_by_image.get(image_id, [])) for image_id in self.images_by_id]
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
        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        boxes = []
        labels = []
        masks = []
        iscrowd = []
        for annotation in annotations:
            bbox = annotation.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            x, y, box_width, box_height = bbox
            if box_width <= 0 or box_height <= 0:
                continue
            boxes.append([x, y, x + box_width, y + box_height])
            labels.append(self.category_to_label.get(annotation.get("category_id"), 1))
            if self.include_masks:
                masks.append(self._annotation_mask(annotation, width, height))
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
        for name in names:
            for path in self.dataset_path.rglob(name):
                return path
        for path in self.dataset_path.rglob("*.json"):
            lowered = path.name.lower()
            if "instances" in lowered or "coco" in lowered:
                return path
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
            for polygon in segmentation:
                if len(polygon) >= 6 and len(polygon) % 2 == 0:
                    points = [(polygon[i], polygon[i + 1]) for i in range(0, len(polygon), 2)]
                    draw.polygon(points, outline=1, fill=1)
        elif isinstance(segmentation, dict):
            try:
                from pycocotools import mask as coco_mask
            except ImportError as exc:
                raise RuntimeError("pycocotools is required for COCO RLE masks") from exc
            decoded = coco_mask.decode(segmentation)
            if getattr(decoded, "ndim", 0) == 3:
                decoded = decoded[:, :, 0]
            return torch.as_tensor(decoded, dtype=torch.uint8)
        return torch.as_tensor(np.array(mask_image, dtype=np.uint8), dtype=torch.uint8)
