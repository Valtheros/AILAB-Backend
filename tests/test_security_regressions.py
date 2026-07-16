from __future__ import annotations

import json
import base64
import tempfile
import unittest
from pathlib import Path
import sys
import types


PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def _install_optional_dependency_stubs() -> None:
    if "fastapi" not in sys.modules:
        class HTTPException(Exception):
            def __init__(self, status_code: int, detail: str = ""):
                super().__init__(detail)
                self.status_code = status_code
                self.detail = detail

        class FastAPI:
            def __init__(self, *_args, **_kwargs):
                pass

            def add_middleware(self, *_args, **_kwargs):
                pass

            def middleware(self, *_args, **_kwargs):
                return lambda fn: fn

            def get(self, *_args, **_kwargs):
                return lambda fn: fn

            def post(self, *_args, **_kwargs):
                return lambda fn: fn

            def patch(self, *_args, **_kwargs):
                return lambda fn: fn

            def delete(self, *_args, **_kwargs):
                return lambda fn: fn

        sys.modules["fastapi"] = types.SimpleNamespace(
            FastAPI=FastAPI,
            File=lambda *args, **kwargs: None,
            HTTPException=HTTPException,
            Request=object,
            UploadFile=object,
        )
        sys.modules["fastapi.middleware"] = types.SimpleNamespace()
        sys.modules["fastapi.middleware.cors"] = types.SimpleNamespace(CORSMiddleware=object)
        sys.modules["fastapi.responses"] = types.SimpleNamespace(
            FileResponse=object,
            JSONResponse=lambda *args, **kwargs: None,
            StreamingResponse=object,
        )
    if "pydantic" not in sys.modules:
        sys.modules["pydantic"] = types.SimpleNamespace(BaseModel=object, Field=lambda default=None, **_kwargs: default)
    if "redis" not in sys.modules:
        sys.modules["redis"] = types.SimpleNamespace(Redis=types.SimpleNamespace(from_url=lambda *_args, **_kwargs: types.SimpleNamespace(ping=lambda: None)))
    if "rq" not in sys.modules:
        sys.modules["rq"] = types.SimpleNamespace(Queue=object)
    if "rq.command" not in sys.modules:
        sys.modules["rq.command"] = types.SimpleNamespace(send_stop_job_command=lambda *_args, **_kwargs: None)
    if "rq.job" not in sys.modules:
        sys.modules["rq.job"] = types.SimpleNamespace(Job=object, JobStatus=types.SimpleNamespace)
    if "PIL" not in sys.modules:
        sys.modules["PIL"] = types.SimpleNamespace(Image=types.SimpleNamespace(), ImageDraw=types.SimpleNamespace())


_install_optional_dependency_stubs()

from fastapi import HTTPException

import dataset_utils
from dataset_utils import inspect_dataset, validate_dataset_for_upload
from main import _assert_owned_resource_visible, _dataset_metadata_is_current, _flatten_single_root_folder, _model_name_from_request, _require_request_user_id, _task_response
from services.training_service import TrainingService
from worker.trainers import input_limits
from worker.trainers.input_limits import iter_text_lines_limited, validate_coco_segmentation


class _FakeRequest:
    def __init__(self, headers: dict[str, str] | None = None):
        self.headers = headers or {}


class SecurityRegressionTests(unittest.TestCase):
    def test_classification_model_name_matches_selected_architecture(self):
        model = {"model_name": "resnet50"}
        request = types.SimpleNamespace(
            model_type="resnet", model_name="resnet18",
            params={"architecture": "resnet18"}, model_size=None,
        )
        self.assertEqual(_model_name_from_request(request, model), "resnet18")

        request.model_name = "resnet50"
        with self.assertRaises(HTTPException):
            _model_name_from_request(request, model)

    def test_task_response_keeps_display_name_separate_from_internal_slug(self):
        task = _task_response({
            "id": "task-1", "display_name": "car_detection", "run_slug": "car_detection_a1b2c3d4",
            "status": "draft", "task_type": "object_detection", "model_type": "yolo", "model_name": "yolo11n",
            "dataset_slug": None, "params": {"epochs": 10, "batch_size": 2, "_device_selection": "manual"},
        })
        self.assertEqual(task["displayName"], "car_detection")
        self.assertEqual(task["runSlug"], "car_detection_a1b2c3d4")
        self.assertEqual(task["deviceSelection"], "manual")
        self.assertNotIn("_device_selection", task["params"])

    def test_old_dataset_metadata_is_reinspected(self):
        metadata = {
            "formats": ["coco_instances"],
            "tasks": ["segmentation"],
            "classes": ["road"],
            "image_count": 1,
            "size_bytes": 1,
            "warnings": [],
            "errors": [],
        }
        self.assertFalse(_dataset_metadata_is_current(metadata))
        metadata["metadata_version"] = dataset_utils.DATASET_METADATA_VERSION
        self.assertTrue(_dataset_metadata_is_current(metadata))

    def test_dataset_yaml_rejects_aliases(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "data.yaml"
            path.write_text("names: &names [car]\ncopy: *names\ntrain: images/train\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "aliases"):
                dataset_utils.read_yaml_limited(path)

    def test_dataset_yaml_size_is_bounded(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "data.yaml"
            path.write_text("names: [car]\n", encoding="utf-8")
            original = dataset_utils.MAX_YAML_FILE_BYTES
            try:
                dataset_utils.MAX_YAML_FILE_BYTES = 4
                with self.assertRaisesRegex(ValueError, "exceeds"):
                    dataset_utils.read_yaml_limited(path)
            finally:
                dataset_utils.MAX_YAML_FILE_BYTES = original

    def test_single_train_directory_is_not_flattened(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            train = root / "train"
            train.mkdir()
            (train / "item.txt").write_text("data", encoding="utf-8")
            _flatten_single_root_folder(root)
            self.assertTrue((train / "item.txt").is_file())

    def test_all_coco_polygons_count_toward_budget_in_upload_and_worker(self):
        polygons = [[0, 0, 1, 0, 1, 1], [2, 2, 3, 2, 3, 3]]
        original_upload = dataset_utils.MAX_COCO_POLYGON_POINTS
        original_worker = input_limits.MAX_COCO_POLYGON_POINTS
        try:
            dataset_utils.MAX_COCO_POLYGON_POINTS = 5
            input_limits.MAX_COCO_POLYGON_POINTS = 5
            with self.assertRaises(ValueError):
                dataset_utils._valid_coco_segmentation(polygons, (10, 10))
            with self.assertRaises(ValueError):
                validate_coco_segmentation(polygons, 10, 10)
        finally:
            dataset_utils.MAX_COCO_POLYGON_POINTS = original_upload
            input_limits.MAX_COCO_POLYGON_POINTS = original_worker

    def test_non_finite_coco_polygon_is_not_valid(self):
        polygon = [[0, 0, 1, 0, float("nan"), 1]]
        self.assertFalse(dataset_utils._valid_coco_segmentation(polygon, (10, 10)))
        self.assertFalse(validate_coco_segmentation(polygon, 10, 10))

    def test_coco_rle_counts_budget_matches_upload_and_worker(self):
        original_upload = dataset_utils.MAX_COCO_RLE_COUNTS
        original_worker = input_limits.MAX_COCO_RLE_COUNTS
        try:
            dataset_utils.MAX_COCO_RLE_COUNTS = 3
            input_limits.MAX_COCO_RLE_COUNTS = 3
            segmentation = {"counts": "abcd", "size": [10, 10]}
            with self.assertRaises(ValueError):
                dataset_utils._valid_coco_segmentation(segmentation, (10, 10))
            with self.assertRaises(ValueError):
                validate_coco_segmentation(segmentation, 10, 10)
        finally:
            dataset_utils.MAX_COCO_RLE_COUNTS = original_upload
            input_limits.MAX_COCO_RLE_COUNTS = original_worker

    def test_log_chunks_are_bounded_and_resume_from_offset(self):
        service = TrainingService.__new__(TrainingService)
        service.redis = object()
        with tempfile.TemporaryDirectory() as temp:
            log_path = Path(temp) / "train.log"
            log_path.write_text("0123456789abcdefghij", encoding="utf-8")
            service._job_log_path = lambda _job_id: log_path
            first, offset, replace = service.read_job_log_chunk("job", -1, max_bytes=8)
            self.assertEqual(first, "cdefghij")
            self.assertEqual(offset, 20)
            self.assertTrue(replace)
            with log_path.open("a", encoding="utf-8") as output:
                output.write("klmnop")
            second, offset, replace = service.read_job_log_chunk("job", offset, max_bytes=3)
            self.assertEqual(second, "klm")
            self.assertEqual(offset, 23)
            self.assertFalse(replace)

    def test_workspace_routes_require_user_identity(self):
        with self.assertRaises(HTTPException) as caught:
            _require_request_user_id(_FakeRequest())
        self.assertEqual(caught.exception.status_code, 401)

    def test_ownerless_resources_are_private(self):
        request = _FakeRequest({"x-user-id": "user-a"})
        with self.assertRaises(HTTPException) as caught:
            _assert_owned_resource_visible(None, request)
        self.assertEqual(caught.exception.status_code, 404)

        with self.assertRaises(HTTPException):
            _assert_owned_resource_visible("user-b", request)
        _assert_owned_resource_visible("user-a", request)

    def test_training_service_filters_ownerless_datasets_and_runs(self):
        service = TrainingService.__new__(TrainingService)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            service.dataset_dir = root / "datasets"
            service.runs_dir = root / "runs"
            service.dataset_dir.mkdir()
            service.runs_dir.mkdir()

            owned_dataset = service.dataset_dir / "owned"
            ownerless_dataset = service.dataset_dir / "ownerless"
            owned_dataset.mkdir()
            ownerless_dataset.mkdir()
            (owned_dataset / ".ailab_dataset.json").write_text(json.dumps({"created_by": "user-a"}), encoding="utf-8")

            self.assertTrue(service._is_dataset_visible(owned_dataset, "user-a"))
            self.assertFalse(service._is_dataset_visible(owned_dataset, "user-b"))
            self.assertFalse(service._is_dataset_visible(ownerless_dataset, "user-a"))
            self.assertFalse(service._is_dataset_visible(owned_dataset, None))

            owned_run = service.runs_dir / "run-owned"
            ownerless_run = service.runs_dir / "run-ownerless"
            owned_run.mkdir()
            ownerless_run.mkdir()
            (owned_run / "job_config.json").write_text(json.dumps({"created_by": "user-a"}), encoding="utf-8")
            (ownerless_run / "job_config.json").write_text(json.dumps({}), encoding="utf-8")

            runs = service.list_runs(owner_id="user-a")
            self.assertEqual([run["project_name"] for run in runs], ["run-owned"])
            self.assertEqual(service.list_runs(owner_id=None), [])

    def test_yolo_label_budget_rejects_oversized_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            labels = root / "train" / "labels"
            images = root / "train" / "images"
            labels.mkdir(parents=True)
            images.mkdir(parents=True)
            (root / "data.yaml").write_text("train: train/images\nnames: ['item']", encoding="utf-8")
            (images / "image.jpg").write_bytes(b"not-an-image")
            (labels / "image.txt").write_text("0 0.5 0.5 0.1 0.1\n" + ("x" * (dataset_utils.MAX_LABEL_FILE_BYTES + 1)), encoding="utf-8")

            metadata = inspect_dataset(root)
            self.assertTrue(any("YOLO label file exceeds" in error for error in metadata["errors"]))
            with self.assertRaises(ValueError):
                validate_dataset_for_upload(root)

    def test_line_iterator_rejects_too_many_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "labels.txt"
            path.write_text("a\nb\nc\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                list(iter_text_lines_limited(path, max_rows=2, label="test label"))

    def test_coco_rle_size_budget_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_coco_segmentation({"counts": "abc", "size": [100000, 100000]}, 100000, 100000)

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            train = root / "train"
            train.mkdir()
            (train / "image_001.jpg").write_bytes(PNG_1X1)
            (train / "_annotations.coco.json").write_text(
                json.dumps(
                    {
                        "images": [{"id": 1, "file_name": "image_001.jpg", "width": 10, "height": 10}],
                        "categories": [{"id": 1, "name": "space"}],
                        "annotations": [
                            {
                                "id": 1,
                                "image_id": 1,
                                "category_id": 1,
                                "bbox": [1, 1, 4, 4],
                                "segmentation": {"counts": "abc", "size": [100000, 100000]},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            metadata = inspect_dataset(root)
            self.assertFalse(metadata["errors"])
            self.assertTrue(any("COCO segmentation exceeds" in warning for warning in metadata["warnings"]))
            self.assertIn("object_detection", metadata["tasks"])
            self.assertNotIn("segmentation", metadata["tasks"])
            validate_dataset_for_upload(root)

    def test_source_image_pixel_budget_uses_coco_dimensions(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            train = root / "train"
            train.mkdir()
            (train / "image_001.jpg").write_bytes(b"not-an-image")
            (train / "_annotations.coco.json").write_text(
                json.dumps(
                    {
                        "images": [{"id": 1, "file_name": "image_001.jpg", "width": 100000, "height": 100000}],
                        "categories": [{"id": 1, "name": "space"}],
                        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 1, 4, 4]}],
                    }
                ),
                encoding="utf-8",
            )
            metadata = inspect_dataset(root)
            self.assertTrue(any("above the" in error and "pixel limit" in error for error in metadata["errors"]))
            with self.assertRaises(ValueError):
                validate_dataset_for_upload(root)


if __name__ == "__main__":
    unittest.main()
