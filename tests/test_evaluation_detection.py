from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as coco_mask

from evaluation_detection import _coco_test_data, _yolo_test_data, evaluate_detection


class FakeDetector:
    def __init__(self, instance=False):
        self.roi_heads = SimpleNamespace()
        self.instance = instance

    def __call__(self, images):
        if images[0].mean().item() < .5:
            output = {'boxes': torch.tensor([[1., 1., 5., 5.]]), 'labels': torch.tensor([1]),
                      'scores': torch.tensor([.9])}
            if self.instance:
                mask = torch.zeros((1, 1, 8, 8))
                mask[:, :, 1:5, 1:5] = 1
                output['masks'] = mask
            return [output]
        output = {'boxes': torch.tensor([[0., 0., 2., 2.]]), 'labels': torch.tensor([2]),
                  'scores': torch.tensor([.9])}
        if self.instance:
            mask = torch.zeros((1, 1, 8, 8))
            mask[:, :, :2, :2] = 1
            output['masks'] = mask
        return [output]


class EvaluationDetectionTests(unittest.TestCase):
    def test_coco_boxes_masks_negative_images_and_reordered_classes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images = root / 'test' / 'images'
            images.mkdir(parents=True)
            Image.new('RGB', (8, 8), 'black').save(images / 'a.png')
            Image.new('RGB', (8, 8), 'white').save(images / 'b.png')
            binary = np.zeros((8, 8), dtype=np.uint8)
            binary[1:5, 1:5] = 1
            encoded = coco_mask.encode(np.asfortranarray(binary))
            document = {'images': [{'id': 9, 'file_name': 'images/a.png', 'width': 8, 'height': 8},
                                   {'id': 10, 'file_name': 'images/b.png', 'width': 8, 'height': 8}],
                        'categories': [{'id': 8, 'name': 'bus'}, {'id': 4, 'name': 'car'}],
                        'annotations': [{'id': 1, 'image_id': 9, 'category_id': 4, 'bbox': [1, 1, 4, 4],
                                         'area': 16, 'iscrowd': 0,
                                         'segmentation': {'size': encoded['size'], 'counts': encoded['counts'].decode()}}]}
            (root / 'test' / '_annotations.coco.json').write_text(json.dumps(document))
            entry = {'kind': 'instance', 'model': FakeDetector(instance=True), 'classes': ['car', 'bus'], 'device': 'cpu'}
            writer = MagicMock()
            writer.save_mask.side_effect = lambda kind, mask: f'{kind}.png'
            writer.add.side_effect = lambda source, outcome, score, detail, **kwargs: json.dumps(detail, allow_nan=False)
            with patch('segmentation_inference.load_segmenter', return_value=entry):
                result = evaluate_detection({}, root, {'formats': ['coco_instances']}, root, 'mask_rcnn',
                                            writer, 'cpu', MagicMock())
            self.assertEqual((result['status'], result['evaluated'], result['skipped']), ('completed', 2, 0))
            self.assertAlmostEqual(result['metrics']['boxes']['map50'], 1.0)
            self.assertAlmostEqual(result['metrics']['masks']['map50'], 1.0)
            self.assertEqual(result['classMapping'], {'8': 'bus', '4': 'car'})
            self.assertEqual([call.args[1] for call in writer.add.call_args_list], ['correct', 'incorrect'])
            self.assertEqual(writer.add.call_args_list[1].args[3]['counts']['fp'], 1)

    def test_coco_polygon_results_remain_json_serializable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'test').mkdir()
            Image.new('RGB', (8, 8), 'black').save(root / 'test' / 'a.png')
            document = {'images': [{'id': 1, 'file_name': 'a.png', 'width': 8, 'height': 8}],
                        'categories': [{'id': 1, 'name': 'car'}],
                        'annotations': [{'id': 1, 'image_id': 1, 'category_id': 1,
                                         'bbox': [1, 1, 4, 4], 'area': 16, 'iscrowd': 0,
                                         'segmentation': [[1, 1, 5, 1, 5, 5, 1, 5]]}]}
            (root / 'test' / '_annotations.coco.json').write_text(json.dumps(document))
            entry = {'kind': 'instance', 'model': FakeDetector(instance=True), 'classes': ['car'], 'device': 'cpu'}
            writer = MagicMock()
            writer.save_mask.side_effect = lambda kind, mask: f'{kind}.png'
            writer.add.side_effect = lambda source, outcome, score, detail, **kwargs: json.dumps(detail, allow_nan=False)
            with patch('segmentation_inference.load_segmenter', return_value=entry):
                result = evaluate_detection({}, root, {'formats': ['coco_instances']}, root, 'mask_rcnn',
                                            writer, 'cpu', MagicMock())
            self.assertEqual(result['status'], 'completed')
            self.assertEqual(writer.add.call_args.args[3]['actual'][0]['segmentation'],
                             document['annotations'][0]['segmentation'])

    def test_no_test_does_not_fall_back_to_train(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'train').mkdir()
            (root / 'train' / '_annotations.coco.json').write_text('{}')
            self.assertIsNone(_coco_test_data(root, ['car']))

    def test_model_errors_are_not_reported_as_corrupt_detection_images(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'test').mkdir()
            Image.new('RGB', (8, 8)).save(root / 'test' / 'a.png')
            (root / 'test' / '_annotations.coco.json').write_text(json.dumps({
                'images': [{'id': 1, 'file_name': 'a.png', 'width': 8, 'height': 8}],
                'categories': [{'id': 1, 'name': 'car'}], 'annotations': []}))
            entry = {'kind': 'detector', 'model': FakeDetector(), 'classes': ['car'], 'device': 'cpu'}
            writer = MagicMock()
            with patch('detection_inference.load_detector', return_value=entry), \
                 patch('evaluation_detection._prediction', side_effect=ValueError('model is broken')):
                with self.assertRaisesRegex(ValueError, 'model is broken'):
                    evaluate_detection({}, root, {'formats': ['coco']}, root, 'faster_rcnn', writer, 'cpu', MagicMock())
            writer.add.assert_not_called()

    def test_coco_unknown_classes_and_train_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'test').mkdir()
            path = root / 'test' / '_annotations.coco.json'
            document = {'images': [{'id': 1, 'file_name': 'train/image.png'}],
                        'categories': [{'id': 1, 'name': 'unknown'}], 'annotations': []}
            path.write_text(json.dumps(document))
            with self.assertRaisesRegex(ValueError, 'does not know'):
                _coco_test_data(root, ['car'])
            document['categories'][0]['name'] = 'car'
            path.write_text(json.dumps(document))
            with self.assertRaisesRegex(ValueError, 'training or validation'):
                _coco_test_data(root, ['car'])

    def test_yolo_uses_importer_path_normalization_without_using_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'test' / 'images').mkdir(parents=True)
            (root / 'valid' / 'images').mkdir(parents=True)
            Image.new('RGB', (8, 8)).save(root / 'test' / 'images' / 'test.png')
            yaml = root / 'data.yaml'
            yaml.write_text('train: ../train/images\ntest: ../test/images\nnames: [car]\n')
            items, mapping = _yolo_test_data(root, ['car'])
            self.assertEqual(len(items), 1)
            self.assertEqual(mapping, {'0': 'car'})
            yaml.write_text('train: ../train/images\ntest: ../valid/images\nnames: [car]\n')
            with self.assertRaisesRegex(ValueError, 'training or validation'):
                _yolo_test_data(root, ['car'])


if __name__ == '__main__':
    unittest.main()
