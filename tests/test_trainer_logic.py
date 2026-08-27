from __future__ import annotations

import ast
import base64
import json
import os
import tempfile
import sys
import types
import unittest
from unittest.mock import patch
from pathlib import Path


PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def _safe_load_stub(text: str):
    data = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        value = value.strip()
        if value.startswith("["):
            try:
                data[key.strip()] = ast.literal_eval(value)
            except Exception:
                data[key.strip()] = []
        else:
            data[key.strip()] = value
    return data


def _dump_stub(data, *_args, **_kwargs):
    return "\n".join(f"{key}: {value}" for key, value in data.items()) + "\n"


try:
    import yaml as _yaml  # noqa: F401
except Exception:
    sys.modules["yaml"] = types.SimpleNamespace(safe_load=_safe_load_stub, dump=_dump_stub)
else:
    if not getattr(_yaml, "__file__", None):
        sys.modules["yaml"] = types.SimpleNamespace(safe_load=_safe_load_stub, dump=_dump_stub)
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

from dataset_utils import compatible_models_for_metadata, inspect_dataset, inspect_dataset_for_upload, prepare_dataset_for_model, validate_dataset_for_upload
from worker.trainers.classification_common import _batch_size_for
from worker.trainers import detection_datasets as detection_datasets_module
from worker.trainers.detection_datasets import CocoInstanceDataset
from worker.trainers.detection_common import _evaluate_detection_metrics
from worker.trainers.deeplabv3plus_trainer import _segmentation_scores
from worker.trainers.trainer_utils import format_epoch_metrics, require_positive_batch_size, split_image_dir
from worker.trainers.yolo_trainer import _format_epoch_log
from worker.error_utils import concise_error
from model_catalog import get_catalog
from services.training_service import TrainingService


class TrainerLogicTests(unittest.TestCase):
    def test_epoch_log_uses_the_results_row(self):
        row = {
            "epoch": 2,
            "train/loss": 1.23456,
            "val/loss": "",
            "metrics/mAP50(B)": 0.87654,
            "lr": 0.00001,
        }
        self.assertEqual(
            format_epoch_metrics("faster_rcnn", 2, 10, row),
            "[faster_rcnn] epoch=2/10 train/loss=1.2346 metrics/mAP50(B)=0.8765 lr=1e-05",
        )

    def test_non_yolo_quality_metrics(self):
        import torch

        class PerfectDetector:
            def eval(self):
                return self

            def __call__(self, images):
                return [{
                    "boxes": torch.tensor([[1.0, 1.0, 5.0, 5.0]]),
                    "labels": torch.tensor([1]),
                    "scores": torch.tensor([0.99]),
                } for _image in images]

        target = {
            "boxes": torch.tensor([[1.0, 1.0, 5.0, 5.0]]),
            "labels": torch.tensor([1]),
            "image_id": torch.tensor([1]),
            "iscrowd": torch.tensor([0]),
        }
        metrics = _evaluate_detection_metrics(
            PerfectDetector(), [([torch.zeros(3, 8, 8)], [target])], torch.device("cpu"), 2, False
        )
        self.assertAlmostEqual(metrics["metrics/precision(B)"], 1.0)
        self.assertAlmostEqual(metrics["metrics/recall(B)"], 1.0)
        self.assertGreater(metrics["metrics/mAP50(B)"], 0.99)

        class PerfectMaskDetector(PerfectDetector):
            def __call__(self, images):
                mask = torch.zeros((1, 1, 8, 8))
                mask[:, :, 1:5, 1:5] = 1
                return [{**output, "masks": mask} for output in super().__call__(images)]

        mask = torch.zeros((1, 8, 8), dtype=torch.uint8)
        mask[:, 1:5, 1:5] = 1
        mask_metrics = _evaluate_detection_metrics(
            PerfectMaskDetector(),
            [([torch.zeros(3, 8, 8)], [{**target, "masks": mask}])],
            torch.device("cpu"),
            2,
            True,
        )
        self.assertGreater(mask_metrics["metrics/mAP50(M)"], 0.99)

        accuracy, mean_iou, dice = _segmentation_scores(torch.tensor([[8, 1], [1, 10]]))
        self.assertAlmostEqual(accuracy, 0.9)
        self.assertGreater(mean_iou, 0.81)
        self.assertGreater(dice, 0.89)

    def test_yolo_epoch_log_contains_readable_metrics(self):
        class Trainer:
            epoch = 1
            epochs = 5
            tloss = object()
            metrics = {
                "metrics/precision(B)": 0.830251,
                "metrics/mAP50(B)": 0.89618,
                "fitness": 0.75,
            }

            @staticmethod
            def label_loss_items(_loss, prefix="train"):
                return {f"{prefix}/box_loss": 1.251791}

        message = _format_epoch_log(Trainer())

        self.assertEqual(
            message,
            "[YOLO] epoch=2/5 train/box_loss=1.2518 "
            "metrics/precision(B)=0.8303 metrics/mAP50(B)=0.8962",
        )

    def test_concise_error_keeps_root_cause_without_traceback(self):
        error = TypeError(
            "Caught TypeError in DataLoader worker process 0.\n"
            "Original Traceback (most recent call last):\n"
            'File "/app/train.py", line 10, in load\n'
            "ValueError: Mask bad.png contains class IDs outside 0..1: [2]"
        )

        message = concise_error(error)

        self.assertEqual(message, "ValueError: Mask bad.png contains class IDs outside 0..1: [2]")
        self.assertNotIn("Traceback", message)

    def test_classification_batch_size_uses_model_family(self):
        self.assertEqual(_batch_size_for({"batch_size": 4}, {}, "resnet"), 4)



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

    def test_nested_yolo_detection_layout_is_detected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            images = root / "images" / "train"
            labels = root / "labels" / "train"
            images.mkdir(parents=True)
            labels.mkdir(parents=True)
            (root / "data.yaml").write_text("train: images/train\nnames: ['item']", encoding="utf-8")
            (images / "image.jpg").write_bytes(b"not-an-image")
            (labels / "image.txt").write_text("0 0.5 0.5 0.25 0.25", encoding="utf-8")
            formats = inspect_dataset(root)["formats"]
            self.assertIn("yolo_detection", formats)
            self.assertNotIn("yolo_segmentation", formats)

    def test_yolo_labels_are_not_inspected_as_semantic_masks(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            images = root / "train" / "images"
            labels = root / "train" / "labels"
            images.mkdir(parents=True)
            labels.mkdir(parents=True)
            (root / "data.yaml").write_text(
                "train: train/images\nnames: ['item']",
                encoding="utf-8",
            )
            (images / "image.jpg").write_bytes(PNG_1X1)
            (labels / "image.txt").write_text("0 0.5 0.5 0.25 0.25", encoding="utf-8")

            metadata = inspect_dataset(root)

            self.assertIn("yolo_detection", metadata["formats"])
            self.assertEqual(metadata["semantic_masks"]["missing_masks"], [])
            validate_dataset_for_upload(root, metadata)

    def test_split_yolo_detection_is_not_misclassified_as_imagefolder(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            images = root / "train" / "images"
            labels = root / "train" / "labels"
            images.mkdir(parents=True)
            labels.mkdir(parents=True)
            (root / "data.yaml").write_text(
                "train: ../train/images\nnames: ['space-empty', 'space-occupied']",
                encoding="utf-8",
            )
            (images / "image.jpg").write_bytes(b"not-an-image")
            (labels / "image.txt").write_text(
                "0 0.5 0.5 0.25 0.25\n"
                "0 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2",
                encoding="utf-8",
            )
            metadata = inspect_dataset(root)
            self.assertIn("yolo_detection", metadata["formats"])
            self.assertNotIn("imagefolder", metadata["formats"])
            self.assertNotIn("yolo_segmentation", metadata["formats"])
            self.assertTrue(any("must not mix" in error for error in metadata["errors"]))
            self.assertEqual(metadata["classes"], ["space-empty", "space-occupied"])


    def test_yolo_segmentation_upload_is_rejected_until_supported(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            images = root / "train" / "images"
            labels = root / "train" / "labels"
            images.mkdir(parents=True)
            labels.mkdir(parents=True)
            (root / "data.yaml").write_text("train: train/images\nnames: ['item']", encoding="utf-8")
            (images / "image.jpg").write_bytes(b"not-an-image")
            (labels / "image.txt").write_text("0 0 0 1 0 1 1 0 1", encoding="utf-8")
            self.assertIn("yolo_segmentation", inspect_dataset(root)["formats"])
            with self.assertRaises(ValueError):
                validate_dataset_for_upload(root)

    def test_coco_box_only_advertises_detection_not_segmentation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            train = root / "train"
            train.mkdir()
            (train / "image_001.jpg").write_bytes(b"not-an-image")
            (train / "_annotations.coco.json").write_text(
                json.dumps(
                    {
                        "images": [{"id": 1, "file_name": "image_001.jpg"}],
                        "categories": [{"id": 1, "name": "space"}],
                        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 1, 4, 4]}],
                    }
                ),
                encoding="utf-8",
            )
            metadata = inspect_dataset(root)
            self.assertIn("coco_instances", metadata["formats"])
            self.assertIn("object_detection", metadata["tasks"])
            self.assertNotIn("segmentation", metadata["tasks"])
            self.assertEqual(metadata["classes"], ["space"])


    def test_coco_to_yolo_uses_canonical_category_order(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            train = root / "train"
            train.mkdir()
            (train / "image.jpg").write_bytes(b"not-an-image")
            (train / "_annotations.coco.json").write_text(
                json.dumps(
                    {
                        "images": [{"id": 1, "file_name": "image.jpg", "width": 10, "height": 10}],
                        "categories": [{"id": 2, "name": "second"}, {"id": 1, "name": "first"}],
                        "annotations": [{"id": 1, "image_id": 1, "category_id": 2, "bbox": [1, 1, 4, 4]}],
                    }
                ),
                encoding="utf-8",
            )
            prepared = prepare_dataset_for_model(root, "yolo", {"model_size": "n"})
            labels = list(Path(prepared["dataset_path"]).rglob("*.txt"))
            self.assertTrue(labels)
            self.assertTrue(labels[0].read_text(encoding="utf-8").startswith("1 "))
            self.assertEqual(prepared["metadata"]["classes"], ["second"])
            self.assertEqual(prepared["metadata"]["coco"]["classes"], ["first", "second"])

    def test_partial_coco_masks_are_not_mask_rcnn_ready(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            train = root / "train"
            train.mkdir()
            (train / "image.jpg").write_bytes(b"not-an-image")
            (train / "_annotations.coco.json").write_text(
                json.dumps(
                    {
                        "images": [{"id": 1, "file_name": "image.jpg", "width": 10, "height": 10}],
                        "categories": [{"id": 1, "name": "object"}],
                        "annotations": [
                            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 1, 4, 4], "segmentation": [[1, 1, 5, 1, 5, 5]]},
                            {"id": 2, "image_id": 1, "category_id": 1, "bbox": [2, 2, 3, 3]},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            metadata = inspect_dataset(root)
            mask_entry = next(
                item
                for item in compatible_models_for_metadata(metadata, get_catalog())
                if item["id"] == "mask_rcnn"
            )
            self.assertFalse(mask_entry["ready"])

    def test_coco_box_dataset_is_yolo_compatible_before_export(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            train = root / "train"
            train.mkdir()
            (train / "image_001.jpg").write_bytes(b"not-an-image")
            (train / "_annotations.coco.json").write_text(
                json.dumps(
                    {
                        "images": [{"id": 1, "file_name": "image_001.jpg", "width": 10, "height": 10}],
                        "categories": [{"id": 5, "name": "space"}],
                        "annotations": [{"id": 1, "image_id": 1, "category_id": 5, "bbox": [1, 1, 4, 4]}],
                    }
                ),
                encoding="utf-8",
            )
            metadata = inspect_dataset(root)
            ready = {item["id"] for item in compatible_models_for_metadata(metadata, get_catalog()) if item["ready"]}

            self.assertIn("coco_instances", metadata["formats"])
            self.assertNotIn("yolo_detection", metadata["formats"])
            self.assertIn("yolo", ready)
            self.assertIn("faster_rcnn", ready)
            self.assertNotIn("mask_rcnn", ready)

    def test_coco_detection_with_invalid_masks_uploads_without_mask_rcnn(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            train = root / "train"
            train.mkdir()
            (train / "image_001.jpg").write_bytes(PNG_1X1)
            (train / "_annotations.coco.json").write_text(
                json.dumps(
                    {
                        "images": [{"id": 1, "file_name": "image_001.jpg", "width": 10, "height": 10}],
                        "categories": [{"id": 5, "name": "space"}],
                        "annotations": [
                            {
                                "id": 1,
                                "image_id": 1,
                                "category_id": 5,
                                "bbox": [1, 1, 4, 4],
                                "segmentation": {"counts": "abc", "size": [100000, 100000]},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            metadata = inspect_dataset(root)
            ready = {item["id"] for item in compatible_models_for_metadata(metadata, get_catalog()) if item["ready"]}

            validate_dataset_for_upload(root)
            self.assertIn("object_detection", metadata["tasks"])
            self.assertNotIn("segmentation", metadata["tasks"])
            self.assertTrue(any("invalid masks" in warning for warning in metadata["warnings"]))
            self.assertIn("yolo", ready)
            self.assertIn("faster_rcnn", ready)
            self.assertNotIn("mask_rcnn", ready)

    def test_prepare_yolo_from_coco_box_dataset_uses_export_cache(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            train = root / "train"
            train.mkdir()
            (train / "image_001.jpg").write_bytes(b"not-an-image")
            (train / "_annotations.coco.json").write_text(
                json.dumps(
                    {
                        "images": [{"id": 1, "file_name": "image_001.jpg", "width": 10, "height": 10}],
                        "categories": [{"id": 5, "name": "space"}],
                        "annotations": [{"id": 1, "image_id": 1, "category_id": 5, "bbox": [1, 1, 4, 4]}],
                    }
                ),
                encoding="utf-8",
            )

            first = prepare_dataset_for_model(root, "yolo", {"model_size": "n"})
            second = prepare_dataset_for_model(root, "yolo", {"model_size": "s"})

            export_path = Path(first["dataset_path"])
            self.assertIn(".ailab_exports", export_path.parts)
            self.assertTrue((export_path / "data.yaml").is_file())
            self.assertTrue((export_path / "labels" / "train").is_dir())
            self.assertFalse(first["export"]["cache_hit"])
            self.assertTrue(second["export"]["cache_hit"])
            self.assertEqual(first["export"]["fingerprint"], second["export"]["fingerprint"])




    def test_coco_missing_images_are_rejected_on_upload(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            train = root / "train"
            train.mkdir()
            (train / "orphan.jpg").write_bytes(b"not-an-image")
            (train / "_annotations.coco.json").write_text(
                json.dumps(
                    {
                        "images": [{"id": 1, "file_name": "missing.jpg"}],
                        "categories": [{"id": 1, "name": "space"}],
                        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 1, 4, 4]}],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "COCO annotations reference"):
                validate_dataset_for_upload(root)

    def test_semantic_upload_rejects_missing_masks(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            images = root / "train" / "images"
            masks = root / "train" / "masks"
            images.mkdir(parents=True)
            masks.mkdir(parents=True)
            (images / "image_001.jpg").write_bytes(b"not-an-image")
            (images / "image_002.jpg").write_bytes(b"not-an-image")
            (masks / "image_002.png").write_bytes(b"not-a-mask")
            with self.assertRaisesRegex(ValueError, "Semantic masks are missing"):
                validate_dataset_for_upload(root)

    def test_coco_polygon_mask_returns_uint8_tensor(self):
        captured = {}

        def as_tensor(value, dtype=None):
            captured["value"] = value
            return types.SimpleNamespace(
                dtype=dtype,
                shape=getattr(value, "shape", None),
                sum=lambda: types.SimpleNamespace(item=lambda: int(value.sum())),
            )

        class FakeMaskImage:
            def __init__(self, size, fill):
                self.width, self.height = size
                self._data = [fill] * (self.width * self.height)

            def getdata(self):
                return self._data

            def fill_polygon(self):
                self._data[0] = 1

        class FakeImageModule:
            @staticmethod
            def new(_mode, size, fill):
                return FakeMaskImage(size, fill)

        class FakeImageDrawModule:
            @staticmethod
            def Draw(image):
                return types.SimpleNamespace(polygon=lambda *_args, **_kwargs: image.fill_polygon())

        class FakeArray:
            def __init__(self, image):
                self.shape = (image.height, image.width)
                self._sum = sum(image.getdata())

            def sum(self):
                return self._sum

        dataset = CocoInstanceDataset.__new__(CocoInstanceDataset)
        fake_torch = types.SimpleNamespace(uint8="uint8", as_tensor=as_tensor)
        fake_numpy = types.SimpleNamespace(uint8="uint8", array=lambda image, dtype=None: FakeArray(image))
        with patch.object(detection_datasets_module, "Image", FakeImageModule):
            with patch.object(detection_datasets_module, "ImageDraw", FakeImageDrawModule):
                with patch.dict(sys.modules, {"torch": fake_torch, "numpy": fake_numpy}):
                    mask = dataset._annotation_mask({"segmentation": [[0, 0, 2, 0, 2, 2, 0, 2]]}, 4, 4)

        self.assertEqual(mask.dtype, "uint8")
        self.assertEqual(tuple(mask.shape), (4, 4))
        self.assertGreater(int(mask.sum().item()), 0)
        self.assertFalse(isinstance(captured["value"], FakeMaskImage))

    def test_coco_box_dataset_can_omit_masks_for_faster_rcnn(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            train = root / "train"
            train.mkdir()
            image_path = train / "image_001.jpg"
            try:
                from PIL import Image

                Image.new("RGB", (8, 8), "white").save(image_path)
            except Exception:
                image_path.write_bytes(b"not-an-image")
            (train / "_annotations.coco.json").write_text(
                json.dumps(
                    {
                        "images": [{"id": 1, "file_name": "image_001.jpg", "width": 8, "height": 8}],
                        "categories": [{"id": 1, "name": "space"}],
                        "annotations": [
                            {
                                "id": 1,
                                "image_id": 1,
                                "category_id": 1,
                                "bbox": [1, 1, 4, 4],
                                "segmentation": [[1, 1, 5, 1, 5, 5, 1, 5]],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            dataset = CocoInstanceDataset(str(root), "train", include_masks=False)
            self.assertFalse(dataset.include_masks)

    def test_upload_inspection_rejects_invalid_sample(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "train" / "images").mkdir(parents=True)
            (root / "train" / "labels").mkdir(parents=True)
            (root / "data.yaml").write_text("train: train/images\nnames: [car]\n", encoding="utf-8")
            (root / "train" / "images" / "car.jpg").write_bytes(b"not-an-image")
            (root / "train" / "labels" / "car.txt").write_text("0 bad row\n", encoding="utf-8")

            metadata = inspect_dataset_for_upload(root)

            self.assertTrue(metadata["validation_deferred"])
            self.assertTrue(metadata["errors"])
            self.assertEqual(metadata["warnings"], [])
            with self.assertRaises(ValueError):
                validate_dataset_for_upload(root, metadata)



    def test_yolo_split_image_dir_supports_images_train_layout(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            images = root / "images" / "train"
            images.mkdir(parents=True)
            self.assertEqual(split_image_dir(str(root), "train"), images)


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

    def test_coco_validation_does_not_fall_back_to_train_annotations(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "train").mkdir()
            (root / "val").mkdir()
            (root / "train" / "_annotations.coco.json").write_text(
                json.dumps({"images": [], "categories": [], "annotations": []}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "split 'val'"):
                CocoInstanceDataset(str(root), "val")


class _FakeJob:
    id = "job-1"

    def __init__(self):
        self.meta = {}

    def save_meta(self):
        pass


class _FakeQueue:
    def __init__(self):
        self.jobs: list[tuple[str, dict, dict]] = []

    def enqueue(self, target: str, config: dict, **kwargs):
        self.jobs.append((target, config, kwargs))
        return _FakeJob()


class TrainingServiceOwnershipTests(unittest.TestCase):
    def _service(self, root: Path) -> TrainingService:
        service = TrainingService.__new__(TrainingService)
        service.dataset_dir = root
        service.runs_dir = root / "runs"
        return service

    def _queued_service(self, root: Path) -> TrainingService:
        service = self._service(root)
        service.redis = object()
        service.queues = {"cv_training": _FakeQueue()}
        return service

    def _imagefolder_dataset(self, root: Path, name: str, owner_id: str | None = None) -> Path:
        dataset = root / name
        image = dataset / "train" / "class_a" / "sample.jpg"
        image.parent.mkdir(parents=True)
        image.write_bytes(b"not-an-image")
        second_image = dataset / "train" / "class_b" / "sample.jpg"
        second_image.parent.mkdir(parents=True)
        second_image.write_bytes(b"not-an-image")
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



    def test_mask_rcnn_rejects_coco_box_only_dataset(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            service = self._service(root)
            dataset = root / "coco_box"
            train = dataset / "train"
            train.mkdir(parents=True)
            (dataset / ".ailab_dataset.json").write_text(json.dumps({"created_by": "owner-a"}), encoding="utf-8")
            (train / "image_001.jpg").write_bytes(b"not-an-image")
            (train / "_annotations.coco.json").write_text(
                json.dumps(
                    {
                        "images": [{"id": 1, "file_name": "image_001.jpg"}],
                        "categories": [{"id": 1, "name": "space"}],
                        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 1, 4, 4]}],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "COCO instance masks"):
                service._assert_dataset_matches_model(dataset, "mask_rcnn", "segmentation")

    def test_faster_rcnn_coco_dataset_does_not_get_worker_yaml(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            service = self._queued_service(root)
            dataset = root / "coco_box"
            train = dataset / "train"
            train.mkdir(parents=True)
            (dataset / ".ailab_dataset.json").write_text(json.dumps({"created_by": "owner-a"}), encoding="utf-8")
            (train / "image_001.jpg").write_bytes(b"not-an-image")
            (train / "_annotations.coco.json").write_text(
                json.dumps(
                    {
                        "images": [{"id": 1, "file_name": "image_001.jpg"}],
                        "categories": [{"id": 1, "name": "space"}],
                        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 1, 4, 4]}],
                    }
                ),
                encoding="utf-8",
            )
            job_id = service.start_training_container(
                task_type="object_detection",
                model_type="faster_rcnn",
                model_name="fasterrcnn_resnet50_fpn_v2",
                epochs=1,
                batch_size=1,
                project_name="coco_run",
                dataset_name="coco_box",
                owner_id="owner-a",
            )
            self.assertEqual(job_id, "job-1")
            queued_config = service.queues["cv_training"].jobs[0][1]
            self.assertIsNone(queued_config["data_yaml_path"])

    def test_yolo_coco_dataset_is_exported_before_enqueue(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            service = self._queued_service(root)
            dataset = root / "coco_box"
            train = dataset / "train"
            train.mkdir(parents=True)
            (dataset / ".ailab_dataset.json").write_text(json.dumps({"created_by": "owner-a"}), encoding="utf-8")
            (train / "image_001.jpg").write_bytes(b"not-an-image")
            (train / "_annotations.coco.json").write_text(
                json.dumps(
                    {
                        "images": [{"id": 1, "file_name": "image_001.jpg", "width": 10, "height": 10}],
                        "categories": [{"id": 1, "name": "space"}],
                        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 1, 4, 4]}],
                    }
                ),
                encoding="utf-8",
            )

            job_id = service.start_training_container(
                task_type="object_detection",
                model_type="yolo",
                model_name="yolo11n",
                epochs=1,
                batch_size=1,
                project_name="yolo_coco_run",
                dataset_name="coco_box",
                extra_args={"model_size": "n"},
                owner_id="owner-a",
            )

            self.assertEqual(job_id, "job-1")
            queued_config = service.queues["cv_training"].jobs[0][1]
            enqueue_options = service.queues["cv_training"].jobs[0][2]
            self.assertEqual(enqueue_options["job_id"], queued_config["job_id"])
            self.assertEqual(queued_config["source_dataset_path"], str(dataset))
            self.assertIn(".ailab_exports", Path(queued_config["dataset_path"]).parts)
            self.assertTrue(Path(queued_config["data_yaml_path"]).is_file())
            self.assertEqual(queued_config["dataset_export"]["export_format"], "yolo_detection")

    def test_start_training_reserves_project_name_globally(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            service = self._queued_service(root)
            self._imagefolder_dataset(root, "dataset_a", owner_id="owner-a")

            job_id = service.start_training_container(
                task_type="image_classification",
                model_type="resnet",
                model_name="resnet50",
                epochs=1,
                batch_size=1,
                project_name="shared_run",
                dataset_name="dataset_a",
                owner_id="owner-a",
            )
            self.assertEqual(job_id, "job-1")
            self.assertTrue((service.runs_dir / "shared_run" / "job_config.json").is_file())
            with self.assertRaises(FileExistsError):
                service.start_training_container(
                    task_type="image_classification",
                    model_type="resnet",
                    model_name="resnet50",
                    epochs=1,
                    batch_size=1,
                    project_name="shared_run",
                    dataset_name="dataset_a",
                    owner_id="owner-b",
                )

    def test_run_owner_reads_job_config(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            service = self._service(root)
            run_dir = service.runs_dir / "existing_run"
            run_dir.mkdir(parents=True)
            (run_dir / "job_config.json").write_text(
                json.dumps({"created_by": "owner-a"}),
                encoding="utf-8",
            )
            self.assertEqual(service._run_owner("existing_run"), "owner-a")


if __name__ == "__main__":
    unittest.main()
