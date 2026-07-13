from __future__ import annotations

import ast
import json
import os
import tempfile
import sys
import types
import unittest
from unittest.mock import patch
from pathlib import Path


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

from dataset_utils import compatible_models_for_metadata, inspect_dataset, inspect_dataset_for_upload, normalize_dataset_for_training, prepare_dataset_for_model, validate_dataset_for_upload
from worker.trainers.classification_common import _batch_size_for
from worker.trainers import detection_datasets as detection_datasets_module
from worker.trainers.detection_datasets import CocoInstanceDataset
from worker.trainers.paddleocr_trainer import PaddleOCRTrainer
from worker.trainers.trainer_utils import require_positive_batch_size, split_image_dir
from model_catalog import get_catalog
from services.training_service import TrainingService


class TrainerLogicTests(unittest.TestCase):
    def test_classification_batch_size_uses_model_family(self):
        self.assertEqual(_batch_size_for({"batch_size": 4}, {}, "resnet"), 4)

    def test_ocr_formats_are_detected_independently(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "sample.png").write_bytes(b"not-an-image")
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

    def test_coco_box_dataset_normalizes_to_yolo_detection(self):
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
            metadata = normalize_dataset_for_training(root)
            validate_dataset_for_upload(root)
            ready = {item["id"] for item in compatible_models_for_metadata(metadata, get_catalog()) if item["ready"]}

            self.assertIn("coco_instances", metadata["formats"])
            self.assertIn("yolo_detection", metadata["formats"])
            self.assertTrue((root / ".ailab_normalized" / "yolo_detection" / "data.yaml").is_file())
            self.assertIn("yolo", ready)
            self.assertIn("faster_rcnn", ready)
            self.assertNotIn("mask_rcnn", ready)

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
            self.assertEqual(prepared["metadata"]["classes"], ["first", "second"])

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
            (train / "image_001.jpg").write_bytes(b"not-an-image")
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

    def test_prepare_ocr_recognition_exports_between_paddleocr_and_tesseract(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "line_001.png").write_bytes(b"not-an-image")
            (root / "line_001.gt.txt").write_text("hello", encoding="utf-8")

            paddle = prepare_dataset_for_model(root, "paddleocr", {"ocr_task": "rec"})
            paddle_path = Path(paddle["dataset_path"])
            self.assertIn(".ailab_exports", paddle_path.parts)
            self.assertTrue((paddle_path / "rec_gt_train.txt").is_file())

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "line_001.png").write_bytes(b"not-an-image")
            (root / "rec_gt_train.txt").write_text("line_001.png\thello", encoding="utf-8")

            tesseract = prepare_dataset_for_model(root, "tesseract")
            tesseract_path = Path(tesseract["dataset_path"])
            self.assertIn(".ailab_exports", tesseract_path.parts)
            self.assertTrue((tesseract_path / "line_001.gt.txt").is_file())

    def test_tesseract_ground_truth_exports_paddleocr_rec_labels(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "line_001.png").write_bytes(b"not-an-image")
            (root / "line_001.gt.txt").write_text("hello", encoding="utf-8")
            metadata = normalize_dataset_for_training(root)
            validate_dataset_for_upload(root)

            self.assertIn("tesseract_ground_truth", metadata["formats"])
            self.assertIn("paddleocr_labels", metadata["formats"])
            self.assertIn("rec", metadata["paddleocr_tasks"])
            self.assertTrue((root / ".ailab_normalized" / "paddleocr_rec" / "rec_gt_train.txt").is_file())

    def test_paddleocr_rec_labels_export_tesseract_ground_truth(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "line_001.png").write_bytes(b"not-an-image")
            (root / "rec_gt_train.txt").write_text("line_001.png\thello", encoding="utf-8")
            metadata = normalize_dataset_for_training(root)
            validate_dataset_for_upload(root)

            self.assertIn("paddleocr_labels", metadata["formats"])
            self.assertIn("tesseract_ground_truth", metadata["formats"])
            self.assertGreater(metadata["tesseract"]["pairs"], 0)

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
            self.assertIn(f"Train.dataset.data_dir={root.resolve().as_posix()}", overrides)
            self.assertIn(f"Train.dataset.label_file_list=['{label.resolve().as_posix()}']", overrides)
            self.assertIn("Global.max_text_length=12", overrides)


    def test_paddle_overrides_reject_path_outside_allowed_roots(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paddle_root = root / "PaddleOCR"
            dataset = root / "dataset"
            runs = root / "runs"
            paddle_root.mkdir()
            dataset.mkdir()
            runs.mkdir()
            (dataset / "rec_gt_train.txt").write_text("image.png\thello", encoding="utf-8")
            outside = root / "outside_dict.txt"
            outside.write_text("abc", encoding="utf-8")
            with patch.dict(os.environ, {"PADDLEOCR_ROOT": str(paddle_root), "RUNS_DIR": str(runs)}, clear=False):
                with self.assertRaises(ValueError):
                    PaddleOCRTrainer()._build_overrides(
                        {"dataset_path": str(dataset), "epochs": 1},
                        {"ocr_task": "rec", "character_dict_path": str(outside)},
                        runs / "run",
                    )

    def test_yolo_split_image_dir_supports_images_train_layout(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            images = root / "images" / "train"
            images.mkdir(parents=True)
            self.assertEqual(split_image_dir(str(root), "train"), images)

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
        service.queues = {"cv_training": _FakeQueue(), "ocr_training": _FakeQueue()}
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

    def test_worker_yaml_normalizes_roboflow_parent_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset = root / "car-park.yolov11"
            (dataset / "train" / "images").mkdir(parents=True)
            (dataset / "valid" / "images").mkdir(parents=True)
            (dataset / "test" / "images").mkdir(parents=True)
            yaml_path = dataset / "data.yaml"
            yaml_path.write_text(
                "train: ../train/images\n"
                "val: ../valid/images\n"
                "test: ../test/images\n"
                "names: ['space-empty']\n",
                encoding="utf-8",
            )
            service = self._service(root)
            worker_yaml = Path(service._create_worker_yaml(yaml_path, dataset))
            worker_text = worker_yaml.read_text(encoding="utf-8")
            self.assertIn("train: train/images", worker_text)
            self.assertIn("val: valid/images", worker_text)
            self.assertIn("test: test/images", worker_text)


    def test_paddleocr_task_mismatch_is_rejected_before_enqueue(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            service = self._service(root)
            dataset = root / "paddle_det"
            dataset.mkdir()
            (dataset / "image_001.jpg").write_bytes(b"not-an-image")
            (dataset / "det_gt_train.txt").write_text("image_001.jpg\t[]", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "selected PaddleOCR task"):
                service._assert_dataset_matches_model(
                    dataset,
                    "paddleocr",
                    "ocr",
                    extra_args={"ocr_task": "rec"},
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
