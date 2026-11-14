import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

import dataset_utils
import main
from dataset_utils import inspect_dataset_for_upload
from dataset_versioning import encode_manifest, source_manifest, verify_source, write_manifest


class DatasetClassMetadataTests(unittest.TestCase):
    def test_sampled_coco_uses_all_declared_categories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / 'train'
            train.mkdir()
            Image.new('RGB', (2, 2)).save(train / 'red.png')
            Image.new('RGB', (2, 2)).save(train / 'blue.png')
            (train / '_annotations.coco.json').write_text(json.dumps({
                'images': [{'id': 1, 'file_name': 'red.png', 'width': 2, 'height': 2},
                           {'id': 2, 'file_name': 'blue.png', 'width': 2, 'height': 2}],
                'categories': [{'id': 9, 'name': 'blue'}, {'id': 3, 'name': 'red'}],
                'annotations': [{'id': 1, 'image_id': 1, 'category_id': 3, 'bbox': [0, 0, 1, 1]},
                                {'id': 2, 'image_id': 2, 'category_id': 9, 'bbox': [0, 0, 1, 1]}],
            }))
            metadata = inspect_dataset_for_upload(root)
            self.assertEqual(metadata['classes'], ['red', 'blue'])
            self.assertEqual(main._dataset_profile(metadata)['classes'], ['red', 'blue'])
            from worker.trainers.detection_datasets import CocoInstanceDataset
            trainer = CocoInstanceDataset(str(root), include_masks=False)
            self.assertEqual(trainer.classes, ['red', 'blue'])
            self.assertEqual(trainer.category_to_label, {3: 1, 9: 2})

    def test_semantic_ids_across_masks_and_bounded_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images, masks = root / 'train' / 'images', root / 'train' / 'masks'
            images.mkdir(parents=True)
            masks.mkdir(parents=True)
            for index, value in enumerate((1, 5)):
                Image.new('RGB', (2, 2)).save(images / f'{index}.png')
                Image.new('L', (2, 2), value).save(masks / f'{index}.png')
            metadata = inspect_dataset_for_upload(root)
            self.assertEqual(metadata['semantic_masks']['class_ids'], [1, 5])
            self.assertEqual(main._dataset_profile(metadata)['classes'], ['class ID 1', 'class ID 5'])
            self.assertEqual(main._dataset_profile(metadata)['mask_class_ids'], [1, 5])
            self.assertTrue(main._dataset_profile(metadata)['classes_known'])
            with patch.object(dataset_utils, 'MAX_SEMANTIC_CLASS_SCAN_PIXELS', 4):
                partial = inspect_dataset_for_upload(root)
            self.assertFalse(main._dataset_profile(partial)['classes_known'])
            indexed = {'train': {'classes': {'background': {}, 'class ID 1': {}, 'class ID 5': {}}}}
            self.assertEqual(main._dataset_profile(partial, indexed)['classes'], ['class ID 1', 'class ID 5'])
            self.assertEqual(main._dataset_profile(partial, indexed)['mask_class_ids'], [1, 5])

            from services.training_service import TrainingService
            service = TrainingService.__new__(TrainingService)
            with self.assertRaisesRegex(ValueError, 'Mask classes must be at least 6'):
                service._assert_dataset_matches_model(root, 'deeplabv3plus', 'segmentation', {'num_classes': 2})
            service._assert_dataset_matches_model(root, 'deeplabv3plus', 'segmentation', {'num_classes': 6})

    def test_display_upgrade_does_not_change_existing_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'image.txt').write_text('original')
            metadata = {'formats': ['coco_instances'], 'tasks': ['object_detection'], 'classes': ['red'],
                        'image_count': 1, 'size_bytes': 1, 'warnings': [], 'errors': [],
                        'coco': {'classes': ['red', 'blue'], 'box_annotations': 1}}
            manifest = source_manifest(root, metadata)
            digest = write_manifest(root, manifest)
            self.assertEqual(main._dataset_profile(metadata)['classes'], ['red', 'blue'])
            self.assertEqual(metadata['classes'], ['red'])
            self.assertEqual(encode_manifest(manifest)[1], digest)
            self.assertTrue(verify_source(root, digest))

    def test_bad_json_identifies_the_annotation_file(self):
        with tempfile.TemporaryDirectory() as directory:
            annotation = Path(directory) / 'broken.json'
            annotation.write_text('{bad')
            with self.assertRaisesRegex(ValueError, 'broken.json.*JSON'):
                dataset_utils._read_json_limited(annotation)


if __name__ == '__main__':
    unittest.main()
