from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from dataset_utils import DATASET_METADATA_VERSION, dataset_workflow_metadata, inspect_dataset, normalize_dataset_metadata, prepare_dataset_for_model
from model_catalog import get_catalog


class CocoSemanticTests(unittest.TestCase):
    def test_coco_polygons_are_prepared_for_semantic_or_instance_training(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "arbitrary-name"
            split = root / "train"
            split.mkdir(parents=True)
            Image.new("RGB", (8, 6), "white").save(split / "sample.png")
            (split / "_annotations.coco.json").write_text(
                json.dumps(
                    {
                        "images": [{"id": 1, "file_name": "sample.png", "width": 8, "height": 6}],
                        "categories": [
                            {"id": 0, "name": "project", "supercategory": "none"},
                            {"id": 1, "name": "road", "supercategory": "project"},
                        ],
                        "annotations": [
                            {
                                "id": 1,
                                "image_id": 1,
                                "category_id": 1,
                                "bbox": [1, 1, 5, 3],
                                "segmentation": [[1, 1, 6, 1, 6, 4, 1, 4]],
                                "area": 15,
                                "iscrowd": 0,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            metadata = inspect_dataset(root)
            self.assertEqual(metadata["metadata_version"], DATASET_METADATA_VERSION)
            workflow = dataset_workflow_metadata(metadata, get_catalog())
            ready = {item["id"] for item in workflow["compatible_models"] if item["ready"]}
            self.assertEqual(metadata["classes"], ["road"])
            self.assertEqual(metadata["image_count"], 1)
            self.assertEqual(workflow["canonical_task"], "segmentation")
            self.assertEqual(workflow["canonical_format"], "coco_segmentation")
            self.assertNotIn("object_detection", metadata["tasks"])
            self.assertNotIn("object_detection", workflow["dataset_tasks"])
            self.assertNotIn("yolo", ready)
            self.assertNotIn("faster_rcnn", ready)
            self.assertIn("deeplabv3plus", ready)
            self.assertIn("mask_rcnn", ready)

            legacy = {**metadata, "metadata_version": 1, "tasks": ["object_detection", "segmentation"]}
            self.assertEqual(normalize_dataset_metadata(legacy)["tasks"], ["segmentation"])

            prepared = prepare_dataset_for_model(root, "deeplabv3plus")
            export_root = Path(prepared["dataset_path"])
            mask = Image.open(next((export_root / "train" / "masks").glob("*.png")))
            self.assertEqual(mask.getpixel((0, 0)), 0)
            self.assertEqual(mask.getpixel((2, 2)), 1)

if __name__ == "__main__":
    unittest.main()
