from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
import sys
import types


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
from main import _assert_owned_resource_visible, _require_request_user_id
from services.training_service import TrainingService
from worker.trainers.input_limits import iter_text_lines_limited, validate_coco_segmentation


class _FakeRequest:
    def __init__(self, headers: dict[str, str] | None = None):
        self.headers = headers or {}


class SecurityRegressionTests(unittest.TestCase):
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
            (train / "image_001.jpg").write_bytes(b"not-an-image")
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
            self.assertTrue(any("COCO segmentation exceeds" in error for error in metadata["errors"]))
            with self.assertRaises(ValueError):
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
