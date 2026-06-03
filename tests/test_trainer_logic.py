from __future__ import annotations

import tempfile
import sys
import types
import unittest
from pathlib import Path

if "yaml" not in sys.modules:
    sys.modules["yaml"] = types.SimpleNamespace(safe_load=lambda *_args, **_kwargs: {})
if "PIL" not in sys.modules:
    sys.modules["PIL"] = types.SimpleNamespace(Image=types.SimpleNamespace(), ImageDraw=types.SimpleNamespace())

from dataset_utils import inspect_dataset
from worker.trainers.classification_common import _batch_size_for
from worker.trainers.detection_datasets import CocoInstanceDataset
from worker.trainers.paddleocr_trainer import PaddleOCRTrainer
from worker.trainers.trainer_utils import require_positive_batch_size


class TrainerLogicTests(unittest.TestCase):
    def test_classification_batch_size_uses_model_family(self):
        self.assertEqual(_batch_size_for({"batch_size": 4}, {}, "resnet"), 4)

    def test_ocr_formats_are_detected_independently(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "image.png").write_bytes(b"not-an-image")
            (root / "sample.gt.txt").write_text("hello", encoding="utf-8")
            metadata = inspect_dataset(root)
            self.assertIn("tesseract_ground_truth", metadata["formats"])
            self.assertNotIn("paddleocr_labels", metadata["formats"])

    def test_generic_train_text_is_not_misclassified_as_paddleocr(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "image.png").write_bytes(b"not-an-image")
            (root / "train.txt").write_text("image.png\\thello", encoding="utf-8")
            self.assertNotIn("paddleocr_labels", inspect_dataset(root)["formats"])

    def test_yolo_polygon_labels_are_not_advertised_as_detection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            labels = root / "train" / "labels"
            labels.mkdir(parents=True)
            (root / "data.yaml").write_text("train: train/images", encoding="utf-8")
            (labels / "image.txt").write_text("0 0 0 1 0 1 1 0 1", encoding="utf-8")
            formats = inspect_dataset(root)["formats"]
            self.assertIn("yolo_segmentation", formats)
            self.assertNotIn("yolo_detection", formats)

    def test_paddle_overrides_bind_uploaded_dataset(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            label = root / "rec_gt_train.txt"
            label.write_text("image.png\\thello", encoding="utf-8")
            overrides = PaddleOCRTrainer()._build_overrides(
                {"dataset_path": str(root), "epochs": 2},
                {"ocr_task": "rec", "max_text_length": 12},
                root / "run",
            )
            self.assertIn(f"Train.dataset.data_dir={root.resolve()}", overrides)
            self.assertIn(f"Train.dataset.label_file_list=[{label}]", overrides)
            self.assertIn("Global.max_text_length=12", overrides)

    def test_paddle_overrides_reject_unimplemented_e2e_task(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):
                PaddleOCRTrainer()._build_overrides(
                    {"dataset_path": temp},
                    {"ocr_task": "e2e"},
                    Path(temp) / "run",
                )

    def test_non_yolo_trainers_reject_auto_batch(self):
        with self.assertRaises(ValueError):
            require_positive_batch_size(-1, "deeplabv3plus")

    def test_coco_image_path_rejects_parent_escape(self):
        with tempfile.TemporaryDirectory() as temp:
            dataset = CocoInstanceDataset.__new__(CocoInstanceDataset)
            dataset.dataset_path = Path(temp)
            dataset.split = "train"
            with self.assertRaises(ValueError):
                dataset._resolve_image_path("../outside.png")


if __name__ == "__main__":
    unittest.main()
