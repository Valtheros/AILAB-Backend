import json
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

from PIL import Image

import annotation_service as module


class AnnotationExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1])
        self.old_dataset_dir = module.DATASET_DIR
        module.DATASET_DIR = Path(self.temp.name)
        self.owner = "owner-1"
        self.project_id = str(uuid.uuid4())
        self.classes = [
            {"id": "road", "name": "Road", "color": "#22c55e"},
            {"id": "car", "name": "Car", "color": "#3b82f6"},
        ]
        source_dir = module._project_root(self.owner, self.project_id) / "images"
        source_dir.mkdir(parents=True)
        self.source = source_dir / "source.png"
        Image.new("RGB", (20, 10), "white").save(self.source)

    def tearDown(self):
        module.DATASET_DIR = self.old_dataset_dir
        self.temp.cleanup()

    def project(self, task_type):
        return {
            "id": self.project_id,
            "owner_user_id": self.owner,
            "task_type": task_type,
            "classes": self.classes,
        }

    def row(self, annotations, image_id="image-1"):
        return {
            "id": image_id,
            "file_name": "source.png",
            "storage_path": str(self.source),
            "width": 20,
            "height": 10,
            "sort_order": 0,
            "annotations": annotations,
        }

    def test_all_task_exports(self):
        service = module.AnnotationService()
        split = {"image-1": "train", "image-2": "train"}

        classification = Path(self.temp.name) / "classification"
        service._write_dataset(classification, self.project("image_classification"), [
            self.row([{"id": "a", "type": "classification", "classId": "road"}]),
            self.row([{"id": "b", "type": "classification", "classId": "car"}], "image-2"),
        ], split)
        self.assertTrue(any((classification / "train" / "0000_Road").iterdir()))
        self.assertEqual(self._ready_models(classification), {"resnet", "efficientnet"})

        detection = Path(self.temp.name) / "detection"
        service._write_dataset(detection, self.project("object_detection"), [self.row([
            {"id": "a", "type": "rectangle", "classId": "car", "x": 0.1, "y": 0.2, "width": 0.5, "height": 0.4},
        ])], split)
        detection_coco = json.loads((detection / "train" / "_annotations.coco.json").read_text())
        self.assertEqual(detection_coco["annotations"][0]["bbox"], [2.0, 2.0, 10.0, 4.0])
        self.assertNotIn("segmentation", detection_coco["annotations"][0])
        self.assertEqual(self._ready_models(detection), {"yolo", "faster_rcnn"})

        instance = Path(self.temp.name) / "instance"
        polygon = {"id": "a", "type": "polygon", "classId": "road", "points": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]}
        service._write_dataset(instance, self.project("instance_segmentation"), [self.row([polygon])], split)
        instance_coco = json.loads((instance / "train" / "_annotations.coco.json").read_text())
        self.assertEqual(len(instance_coco["annotations"][0]["segmentation"][0]), 8)
        self.assertGreater(instance_coco["annotations"][0]["area"], 0)
        self.assertEqual(self._ready_models(instance), {"mask_rcnn", "deeplabv3plus"})

        semantic = Path(self.temp.name) / "semantic"
        brush = {
            "id": "brush", "type": "brush", "classId": "car", "mode": "paint", "radius": 0.1,
            "points": [[0.2, 0.5], [0.8, 0.5]],
        }
        service._write_dataset(semantic, self.project("semantic_segmentation"), [self.row([polygon, brush])], split)
        with Image.open(semantic / "train" / "masks" / "image-1.png") as mask:
            self.assertEqual(mask.getpixel((10, 2)), 1)
            self.assertEqual(mask.getpixel((4, 5)), 2)
        self.assertEqual(self._ready_models(semantic), {"deeplabv3plus"})

        instance_brush = Path(self.temp.name) / "instance-brush"
        service._write_dataset(instance_brush, self.project("instance_segmentation"), [self.row([
            {**brush, "instanceId": "car-1"},
        ])], split)
        brush_coco = json.loads((instance_brush / "train" / "_annotations.coco.json").read_text())
        self.assertIsInstance(brush_coco["annotations"][0]["segmentation"], dict)
        self.assertGreater(brush_coco["annotations"][0]["area"], 0)

    @staticmethod
    def _ready_models(dataset: Path) -> set[str]:
        workflow = module.dataset_workflow_metadata(module.inspect_dataset(dataset), module.get_catalog())
        return {str(item["id"]) for item in workflow["compatible_models"] if item["ready"]}

    def test_validation_and_split_are_stable(self):
        with self.assertRaises(ValueError):
            module._validate_class_count("image_classification", self.classes[:1])
        with self.assertRaisesRegex(ValueError, "cannot be marked empty"):
            module._validate_annotations("image_classification", [], True, {"road", "car"})
        with self.assertRaisesRegex(ValueError, "finite numbers"):
            module._validate_annotations("object_detection", [{
                "id": "bad", "type": "rectangle", "classId": "car",
                "x": float("nan"), "y": 0, "width": 0.5, "height": 0.5,
            }], False, {"road", "car"})
        with self.assertRaisesRegex(ValueError, "inside the image"):
            module._validate_annotations("semantic_segmentation", [{
                "id": "bad", "type": "polygon", "classId": "road",
                "points": [[0, 0], [1.1, 0], [0, 1]],
            }], False, {"road", "car"})
        rows = [{"id": str(index), "file_name": f"{index}.jpg", "annotations": [{"classId": "road"}]} for index in range(10)]
        project = {"task_type": "image_classification", "train_ratio": 80, "val_ratio": 10, "test_ratio": 10, "split_seed": "seed"}
        first = module._assign_splits(rows, project)
        self.assertEqual(first, module._assign_splits(list(reversed(rows)), project))
        self.assertEqual(list(first.values()).count("train"), 8)

        initial = module._assign_initial_splits(rows, project)
        self.assertEqual(list(initial.values()).count("train"), 8)
        for row in rows:
            row["split"] = "test" if row["id"] == "0" else initial[row["id"]]
            row["split_source"] = "manual"
        preserved = module._assign_splits(rows, project)
        self.assertEqual(preserved["0"], "test")

        rows[1]["split"] = None
        partially_assigned = module._assign_splits(rows, project)
        self.assertEqual(partially_assigned["0"], "test")
        self.assertIn(partially_assigned["1"], module.SPLITS)

    def test_classification_split_keeps_each_class_in_train(self):
        rows = [
            {"id": "road-1", "file_name": "road.jpg", "annotations": [{"classId": "road"}]},
            {"id": "car-1", "file_name": "car.jpg", "annotations": [{"classId": "car"}]},
        ]
        project = {
            "task_type": "image_classification",
            "train_ratio": 80,
            "val_ratio": 10,
            "test_ratio": 10,
            "split_seed": "seed",
        }
        split = module._assign_initial_splits(rows, project)
        self.assertEqual(split, {"road-1": "train", "car-1": "train"})

    def test_mutations_reject_publishing_and_unknown_owners(self):
        service = module.AnnotationService()
        connection = MagicMock()
        connection.__enter__.return_value = connection
        actions = [
            lambda: service.save_image(self.owner, self.project_id, "image-1", 0, [], True),
            lambda: service.set_split(self.owner, self.project_id, "image-1", "train"),
            lambda: service.bulk_update(self.owner, self.project_id, ["image-1"], excluded=True),
            lambda: service.rebalance(self.owner, self.project_id),
            lambda: service.update_project(self.owner, self.project_id, "Project", self.classes, (80, 10, 10)),
            lambda: service.restore_revision(self.owner, self.project_id, "image-1", "revision-1"),
        ]
        with patch.object(service, "_connect", return_value=connection):
            for project, error in [({"status": "publishing"}, RuntimeError), (None, FileNotFoundError)]:
                for action in actions:
                    connection.reset_mock()
                    connection.execute.return_value.fetchone.return_value = project
                    with self.assertRaises(error):
                        action()
                    self.assertEqual(connection.execute.call_count, 1)
                    self.assertEqual(connection.execute.call_args.args[1], (self.owner, self.project_id))

    def test_publish_preparation_failure_restores_draft(self):
        service = module.AnnotationService()
        connection = MagicMock()
        connection.__enter__.return_value = connection
        connection.execute.return_value.fetchone.side_effect = [
            {**self.project("object_detection"), "published_version": 2},
            {"count": 0}, {"count": 1},
        ]
        connection.execute.return_value.fetchall.return_value = [self.row([])]
        with patch.object(service, "_connect", return_value=connection), patch.object(
            module, "_assign_splits", return_value={"image-1": "test"}
        ), patch.object(module, "staging_directory") as stage:
            with self.assertRaisesRegex(ValueError, "at least one usable image"):
                service.publish(self.owner, None, self.project_id)
            stage.assert_not_called()
        self.assertIn("set status = 'draft'", connection.execute.call_args.args[0])

    def test_source_pixel_limit_is_checked_before_decode(self):
        opened = MagicMock()
        opened.__enter__.return_value = opened
        opened.size = (module.MAX_SOURCE_IMAGE_PIXELS + 1, 1)
        with patch.object(module.Image, "open", return_value=opened), patch.object(
            module.ImageOps, "exif_transpose"
        ) as transpose:
            with self.assertRaises(ValueError):
                module._save_normalized_image(self.source, self.source, self.source)
            transpose.assert_not_called()
            opened.load.assert_not_called()

    def test_late_cancel_does_not_discard_committed_publish(self):
        import annotation_operations as operations

        connection = MagicMock()
        connection.__enter__.return_value = connection
        connection.execute.return_value.fetchone.side_effect = [
            {"kind": "publish", "owner_user_id": self.owner, "project_id": self.project_id},
            {"email": "owner@example.com"},
        ]
        result = {"datasetId": "dataset-1", "datasetName": "project-v1", "version": 1}
        with patch.object(operations.annotation_operations, "_connect", return_value=connection), patch.object(
            operations.annotation_service, "publish", return_value=result
        ), patch.object(operations.annotation_operations, "is_cancelled", return_value=True):
            self.assertEqual(operations.run_operation("operation-1"), result)
        self.assertIn("status = 'completed'", connection.execute.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
