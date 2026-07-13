from __future__ import annotations

from pathlib import Path

from PIL import Image

from .input_limits import open_image_checked, open_rgb_image_checked
from .trainer_utils import IMAGE_EXTENSIONS, MASK_EXTENSIONS, list_images


class SemanticMaskDataset:
    def __init__(self, dataset_path: str, split: str, image_size: int, num_classes: int, ignore_index: int = 255):
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.image_size = image_size
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.images_dir, self.masks_dir = self._find_split_dirs()
        self.images = list_images(self.images_dir)
        self.warnings: list[str] = []
        usable_images = []
        for image_path in self.images:
            try:
                mask_path = self._mask_path(image_path)
                image = open_rgb_image_checked(image_path)
                mask = open_image_checked(mask_path)
                if image.size != mask.size:
                    raise ValueError(
                        f"image is {image.width}x{image.height}, mask is {mask.width}x{mask.height}"
                    )
                image.close()
                mask.close()
                usable_images.append(image_path)
            except Exception as exc:
                self.warnings.append(f"Skipping {split} semantic image '{image_path}': {exc}")
        self.images = usable_images
        if not self.images:
            raise ValueError(f"No usable semantic segmentation images found in {self.images_dir}")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        import torch
        import torchvision.transforms.functional as F
        from torchvision.transforms import InterpolationMode
        import numpy as np

        image_path = self.images[index]
        mask_path = self._mask_path(image_path)
        image = open_rgb_image_checked(image_path)
        mask = open_image_checked(mask_path)
        if image.size != mask.size:
            raise ValueError(
                f"Semantic image {image_path.name} is {image.width}x{image.height} but mask "
                f"{mask_path.name} is {mask.width}x{mask.height}."
            )
        if mask.mode not in {"L", "P", "I", "I;16"}:
            mask = mask.convert("L")
        image = F.resize(image, [self.image_size, self.image_size], interpolation=InterpolationMode.BILINEAR)
        mask = F.resize(mask, [self.image_size, self.image_size], interpolation=InterpolationMode.NEAREST)
        mask_tensor = torch.from_numpy(np.asarray(mask, dtype=np.int64)).long()
        invalid = (mask_tensor != self.ignore_index) & ((mask_tensor < 0) | (mask_tensor >= self.num_classes))
        if invalid.any():
            invalid_values = sorted(set(mask_tensor[invalid].tolist()))
            raise ValueError(f"Mask {mask_path.name} contains class IDs outside 0..{self.num_classes - 1}: {invalid_values[:10]}")
        return F.to_tensor(image), mask_tensor

    def _find_split_dirs(self) -> tuple[Path, Path]:
        split_candidates = {
            "train": ["train", "training"],
            "val": ["valid", "val", "validation"],
            "test": ["test"],
        }.get(self.split, [self.split])
        for split_name in split_candidates:
            split_dir = self.dataset_path / split_name
            if not split_dir.exists():
                continue
            images_dir = split_dir / "images"
            if not images_dir.exists():
                images_dir = split_dir / "image"
            for mask_name in ("masks", "mask", "labels", "label"):
                masks_dir = split_dir / mask_name
                if images_dir.exists() and masks_dir.exists():
                    return images_dir, masks_dir
        raise ValueError(f"Semantic masks require {self.split}/images and {self.split}/masks folders.")

    def _mask_path(self, image_path: Path) -> Path:
        for extension in MASK_EXTENSIONS:
            candidate = self.masks_dir / f"{image_path.stem}{extension}"
            if candidate.exists():
                return candidate
        for path in self.masks_dir.rglob(f"{image_path.stem}.*"):
            if path.suffix.lower() in MASK_EXTENSIONS:
                return path
        raise FileNotFoundError(f"Mask for image {image_path.name} was not found in {self.masks_dir}")
