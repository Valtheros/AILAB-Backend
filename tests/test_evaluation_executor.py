from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from PIL import Image

from evaluation_executor import (_classification_items, _evaluate_classification,
                                 _evaluate_semantic, _semantic_class_mapping, _semantic_coco_mask)


class EvaluationExecutorTests(unittest.TestCase):
    def test_label_studio_checkpoint_names_map_to_prefixed_test_folders(self):
        import torch

        class ConstantClassifier(torch.nn.Module):
            def forward(self, batch):
                return torch.tensor([[3.0, 1.0]], device=batch.device).repeat(len(batch), 1)

        schema = {'label_schema': {'origin': 'label_studio', 'task': 'image_classification',
                                   'classes': ['curve', 'junction']}}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / 'test' / '000_curve'
            second = root / 'test' / '001_junction'
            first.mkdir(parents=True)
            second.mkdir(parents=True)
            Image.new('RGB', (8, 8)).save(first / 'image.png')
            (second / 'corrupt.jpg').write_bytes(b'not an image')
            writer = MagicMock()
            entry = {'model': ConstantClassifier(), 'classes': ['curve', 'junction'],
                     'image_size': 8, 'device': 'cpu'}
            with patch('inference_service.load_classifier', return_value=entry), \
                 patch('evaluation_executor._record_progress'):
                report = _evaluate_classification({}, root, schema, root,
                                                  {'dataset_metadata': schema}, writer, 'cpu')
            self.assertEqual((report['status'], report['evaluated'], report['skipped']), ('partial', 1, 1))
            self.assertEqual(report['metrics']['accuracy'], 1.0)
            self.assertEqual([call.args[1] for call in writer.add.call_args_list], ['correct', 'error'])

    def test_unknown_test_class_is_rejected_instead_of_remapped_by_index(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            unknown = root / 'test' / 'unseen'
            unknown.mkdir(parents=True)
            (unknown / 'one.jpg').write_bytes(b'content')
            with self.assertRaisesRegex(ValueError, 'does not know'):
                _classification_items(root, {}, ['known'])

    def test_result_storage_errors_are_not_treated_as_broken_images(self):
        import torch

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root / 'test' / 'known'
            folder.mkdir(parents=True)
            Image.new('RGB', (8, 8)).save(folder / 'image.png')
            writer = MagicMock()
            writer.add.side_effect = OSError('disk full')
            entry = {'model': lambda batch: torch.ones((1, 1)), 'classes': ['known'],
                     'image_size': 8, 'device': 'cpu'}
            with patch('inference_service.load_classifier', return_value=entry), \
                 patch('evaluation_executor._record_progress'):
                with self.assertRaisesRegex(OSError, 'disk full'):
                    _evaluate_classification({}, root, {}, root, {}, writer, 'cpu')

    def test_model_errors_are_not_reported_as_corrupt_classification_images(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root / 'test' / 'known'
            folder.mkdir(parents=True)
            Image.new('RGB', (8, 8)).save(folder / 'image.png')
            model = MagicMock(side_effect=ValueError('model is broken'))
            writer = MagicMock()
            entry = {'model': model, 'classes': ['known'], 'image_size': 8, 'device': 'cpu'}
            with patch('inference_service.load_classifier', return_value=entry), \
                 patch('evaluation_executor._record_progress'):
                with self.assertRaisesRegex(ValueError, 'model is broken'):
                    _evaluate_classification({}, root, {}, root, {}, writer, 'cpu')
            writer.add.assert_not_called()

    def test_semantic_classes_are_mapped_by_name_not_source_index(self):
        import numpy as np
        import torch

        class ConstantSegmenter(torch.nn.Module):
            def forward(self, batch):
                result = torch.zeros((1, 3, batch.shape[-2], batch.shape[-1]), device=batch.device)
                result[:, 1] = 1
                return result

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images, masks = root / 'test' / 'images', root / 'test' / 'masks'
            images.mkdir(parents=True)
            masks.mkdir()
            (root / 'test' / '_classes.csv').write_text('id,name\n0,background\n1,junction\n2,curve\n')
            Image.new('RGB', (8, 8)).save(images / 'one.png')
            Image.fromarray(np.full((8, 8), 2, dtype=np.uint8)).save(masks / 'one.png')
            (images / 'broken.jpg').write_bytes(b'bad')
            writer = MagicMock()
            entry = {'model': ConstantSegmenter(), 'classes': ['background', 'curve', 'junction'],
                     'image_size': 8, 'device': 'cpu', 'class_schema_verified': True}
            with patch('segmentation_inference.load_segmenter', return_value=entry), \
                 patch('evaluation_executor._record_progress'):
                result = _evaluate_semantic({}, root, {}, root, writer, 'cpu')
            self.assertEqual((result['status'], result['evaluated'], result['skipped']), ('partial', 1, 1))
            self.assertEqual(result['metrics']['pixelAccuracy'], 1.0)
            self.assertEqual(result['classMapping']['2'], 'curve')

    def test_semantic_evaluation_rejects_checkpoint_with_unknown_class_names(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images, masks = root / 'test' / 'images', root / 'test' / 'masks'
            images.mkdir(parents=True)
            masks.mkdir()
            Image.new('RGB', (8, 8)).save(images / 'one.png')
            Image.new('L', (8, 8)).save(masks / 'one.png')
            entry = {'classes': ['background', 'class 1'], 'class_schema_verified': False}
            with patch('segmentation_inference.load_segmenter', return_value=entry):
                with self.assertRaisesRegex(ValueError, 'does not record a complete semantic class mapping'):
                    _evaluate_semantic({}, root, {}, root, MagicMock(), 'cpu')

    def test_semantic_rejects_unmapped_or_unknown_classes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'test').mkdir()
            with self.assertRaisesRegex(ValueError, 'explicit class mapping'):
                _semantic_class_mapping(root, {}, ['background', 'curve'])
            (root / 'test' / '_classes.csv').write_text('id,name\n0,background\n1,unknown\n')
            with self.assertRaisesRegex(ValueError, 'does not know'):
                _semantic_class_mapping(root, {}, ['background', 'curve'])

    def test_semantic_coco_crowd_pixels_are_ignored(self):
        import numpy as np
        from pycocotools import mask as coco_mask

        binary = np.zeros((4, 4), dtype=np.uint8)
        binary[:2, :2] = 1
        crowd = coco_mask.encode(np.asfortranarray(binary))
        encoded = {'size': crowd['size'], 'counts': crowd['counts'].decode()}
        target = _semantic_coco_mask([
            {'segmentation': [[0, 0, 4, 0, 4, 4, 0, 4]], 'category_id': 1},
            {'segmentation': encoded, 'category_id': 1, 'iscrowd': 1},
        ], 4, 4)
        self.assertTrue((target[:2, :2] == 255).all())
        self.assertTrue((target[2:, :] == 1).all())


if __name__ == '__main__':
    unittest.main()
