from __future__ import annotations

import json
import tempfile
import sys
import types
import unittest
from pathlib import Path

if "yaml" not in sys.modules:
    sys.modules["yaml"] = types.SimpleNamespace(safe_load=lambda *_args, **_kwargs: {})
if "PIL" not in sys.modules:
    sys.modules["PIL"] = types.SimpleNamespace(Image=types.SimpleNamespace(), ImageDraw=types.SimpleNamespace())
if "redis" not in sys.modules:
    sys.modules["redis"] = types.SimpleNamespace(Redis=types.SimpleNamespace)
if "rq" not in sys.modules:
    sys.modules["rq"] = types.SimpleNamespace(Queue=types.SimpleNamespace)
if "rq.command" not in sys.modules:
    sys.modules["rq.command"] = types.SimpleNamespace(send_stop_job_command=lambda *_args, **_kwargs: None)
if "rq.job" not in sys.modules:
    sys.modules["rq.job"] = types.SimpleNamespace(Job=types.SimpleNamespace, JobStatus=types.SimpleNamespace)

from dataset_utils import inspect_dataset
from worker.trainers.classification_common import _batch_size_for
from worker.trainers.detection_datasets import CocoInstanceDataset
from worker.trainers.paddleocr_trainer import PaddleOCRTrainer
from worker.trainers.trainer_utils import require_positive_batch_size
from services.training_service import TrainingService


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


class TrainingServiceOwnershipTests(unittest.TestCase):
    def _service(self, root: Path) -> TrainingService:
        service = TrainingService.__new__(TrainingService)
        service.dataset_dir = root
        service.runs_dir = root / "runs"
        return service

    def _imagefolder_dataset(self, root: Path, name: str, owner_id: str | None = None) -> Path:
        dataset = root / name
        image = dataset / "train" / "class_a" / "sample.jpg"
        image.parent.mkdir(parents=True)
        image.write_bytes(b"not-an-image")
        if owner_id:
            (dataset / ".ailab_dataset.json").write_text(
                json.dumps({"created_by": owner_id}),
                encoding="utf-8",
            )
        return dataset

    def test_named_dataset_rejects_other_owner(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            service = self._service(root)
            self._imagefolder_dataset(root, "private_dataset", owner_id="owner-a")

            with self.assertRaises(FileNotFoundError):
                service._find_dataset_path("private_dataset", owner_id="owner-b")

    def test_latest_compatible_dataset_skips_other_owner(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            service = self._service(root)
            self._imagefolder_dataset(root, "other_dataset", owner_id="owner-b")

            with self.assertRaises(FileNotFoundError):
                service._find_latest_compatible_dataset_path(
                    "resnet",
                    "image_classification",
                    owner_id="owner-a",
                )


if __name__ == "__main__":
    unittest.main()
