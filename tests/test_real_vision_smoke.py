"""Optional CPU or CUDA smoke tests with disposable data and real trainers."""
from __future__ import annotations

import asyncio
import io
import json
import os
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

from PIL import Image, ImageDraw

DEVICE = os.getenv('AILAB_REAL_MODEL_DEVICE', 'cpu')
TRAIN_DEVICE = '0' if DEVICE == 'cuda:0' else 'cpu'


@unittest.skipUnless(os.getenv('AILAB_REAL_MODEL_TEST') == '1', 'Requires compute image and real model dependencies')
class RealVisionSmokeTests(unittest.TestCase):
    def setUp(self):
        self.assertIn(DEVICE, ('cpu', 'cuda:0'))
        if DEVICE == 'cuda:0':
            import torch
            self.assertTrue(torch.cuda.is_available(), 'CUDA is not available for the requested smoke test')

    def _batch(self, root, run, family, task):
        import batch_prediction
        from batch_upload import stage_batch
        from tests.test_batch_prediction import Upload

        image = io.BytesIO()
        Image.new('RGB', (64, 64)).save(image, format='PNG')
        directory = root / f'{family}-batch'
        asyncio.run(stage_batch([Upload('image.png', image.getvalue()), Upload('bad.jpg', b'bad')], directory))
        row = {'id': uuid.uuid4(), 'dispatch_token': uuid.uuid4(), 'payload': {
            'project_name': family, 'task_type': task, 'model_type': family, 'threshold': .5}}
        with patch.object(batch_prediction, 'job_directory', return_value=directory), \
             patch.object(batch_prediction, 'RUNS_DIR', run.parent), patch.object(batch_prediction, '_write_progress'):
            batch_prediction.execute_batch(row, DEVICE)
        result = json.loads((directory / 'result.json').read_text())
        self.assertEqual((result['succeeded'], result['failed']), (1, 1))
        self.assertEqual(result['images'][0]['result']['model']['device'], DEVICE)

    def _coco(self, root):
        source = root / 'dataset'
        for split in ('train', 'test'):
            directory = source / split
            directory.mkdir(parents=True)
            Image.new('RGB', (64, 64), 'white').save(directory / 'one.png')
            document = {'images': [{'id': 1, 'file_name': 'one.png', 'width': 64, 'height': 64}],
                        'categories': [{'id': 1, 'name': 'object'}],
                        'annotations': [{'id': 1, 'image_id': 1, 'category_id': 1,
                                         'bbox': [8, 8, 32, 32], 'area': 1024, 'iscrowd': 0,
                                         'segmentation': [[8, 8, 40, 8, 40, 40, 8, 40]]}]}
            (directory / '_annotations.coco.json').write_text(json.dumps(document))
        return source

    def _detection(self, family):
        from evaluation_detection import evaluate_detection
        from worker.trainers.detection_common import train_detection_model

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, runs = self._coco(root), root / 'runs'
            task = 'segmentation' if family == 'mask_rcnn' else 'object_detection'
            metadata = {'classes': ['object'], 'formats': ['coco_instances'], 'tasks': [task]}
            config = {'dataset_path': str(source), 'dataset_metadata': metadata, 'task_type': task,
                      'model_type': family, 'model_name': family, 'project_name': family,
                      'epochs': 1, 'batch_size': 1, 'compute_job_id': str(uuid.uuid4()),
                      'extra_args': {'pretrained': False, 'image_size': 64, 'max_size': 64,
                                     'workers': 0, 'device': TRAIN_DEVICE, 'amp': False}}
            with patch.dict(os.environ, {'RUNS_DIR': str(runs)}):
                train_detection_model(config, family, None, lambda *_: None)
            self.assertTrue((runs / family / 'best.pt').is_file())
            report = evaluate_detection({}, source, metadata, runs / family, family, MagicMock(), DEVICE, MagicMock())
            self.assertEqual((report['status'], report['evaluated']), ('completed', 1))
            self.assertIn('map50', report['metrics']['boxes'])
            if family == 'mask_rcnn':
                self.assertIn('map50', report['metrics']['masks'])
            self._batch(root, runs / family, family, task)

    def test_faster_rcnn_train_evaluate_and_batch(self):
        self._detection('faster_rcnn')

    def test_mask_rcnn_train_evaluate_and_batch(self):
        self._detection('mask_rcnn')

    def test_yolo_train_evaluate_and_batch(self):
        from ultralytics import YOLO
        from evaluation_detection import evaluate_detection
        from worker.trainers.yolo_trainer import YOLOTrainer

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, runs = root / 'dataset', root / 'runs'
            for split in ('train', 'valid', 'test'):
                images, labels = source / split / 'images', source / split / 'labels'
                images.mkdir(parents=True)
                labels.mkdir()
                Image.new('RGB', (64, 64), 'white').save(images / 'one.png')
                (labels / 'one.txt').write_text('0 0.5 0.5 0.5 0.5\n')
                (source / split / 'labels.cache').write_bytes(b'untrusted cache')
            yaml = source / 'data.yaml'
            yaml.write_text(f'path: {source}\ntrain: train/images\nval: valid/images\ntest: test/images\nnames: [object]\n')
            metadata = {'classes': ['object'], 'formats': ['yolo_detection'], 'tasks': ['object_detection']}
            config = {'dataset_path': str(source), 'data_yaml_path': str(yaml), 'dataset_metadata': metadata,
                      'task_type': 'object_detection', 'model_type': 'yolo', 'model_name': 'yolo11n', 'project_name': 'yolo',
                      'epochs': 1, 'batch_size': 1, 'compute_job_id': str(uuid.uuid4()),
                      'extra_args': {'imgsz': 64, 'pretrained': False, 'workers': 0, 'device': TRAIN_DEVICE, 'amp': False,
                                     'plots': False, 'verbose': False}}
            YOLO('yolo11n.yaml').save(root / 'yolo11n.pt')
            previous = Path.cwd()
            try:
                os.chdir(root)
                with patch.dict(os.environ, {'RUNS_DIR': str(runs)}):
                    YOLOTrainer().train(config)
            finally:
                os.chdir(previous)
            self.assertTrue((runs / 'yolo' / 'weights' / 'best.pt').is_file())
            for split in ('train', 'valid', 'test'):
                self.assertEqual((source / split / 'labels.cache').read_bytes(), b'untrusted cache')
            report = evaluate_detection({}, source, metadata, runs / 'yolo', 'yolo', MagicMock(), DEVICE, MagicMock())
            self.assertEqual((report['status'], report['evaluated']), ('completed', 1))
            self.assertIn('map50', report['metrics']['boxes'])
            self._batch(root, runs / 'yolo', 'yolo', 'object_detection')

    def test_deeplab_train_evaluate_and_batch(self):
        from evaluation_executor import _evaluate_semantic
        from worker.trainers.deeplabv3plus_trainer import DeepLabV3PlusTrainer

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, runs = root / 'dataset', root / 'runs'
            for split in ('train', 'test'):
                images, masks = source / split / 'images', source / split / 'masks'
                images.mkdir(parents=True)
                masks.mkdir()
                (source / split / '_classes.csv').write_text('id,name\n0,background\n1,object\n')
                for index in range(2):
                    Image.new('RGB', (64, 64), 'white').save(images / f'{index}.png')
                    mask = Image.new('L', (64, 64))
                    ImageDraw.Draw(mask).rectangle((8, 8, 40, 40), fill=1)
                    mask.save(masks / f'{index}.png')
            metadata = {'classes': ['object'], 'formats': ['semantic_masks'], 'tasks': ['semantic_segmentation']}
            config = {'dataset_path': str(source), 'dataset_metadata': metadata, 'task_type': 'segmentation',
                      'model_type': 'deeplabv3plus', 'model_name': 'deeplabv3plus', 'project_name': 'deeplabv3plus',
                      'epochs': 1, 'batch_size': 2, 'compute_job_id': str(uuid.uuid4()),
                      'extra_args': {'encoder_name': 'resnet18', 'encoder_weights': 'none', 'decoder_channels': 32,
                                     'image_size': 64, 'num_classes': 2, 'workers': 0, 'device': TRAIN_DEVICE, 'amp': False}}
            with patch.dict(os.environ, {'RUNS_DIR': str(runs)}):
                DeepLabV3PlusTrainer().train(config)
            self.assertTrue((runs / 'deeplabv3plus' / 'best.pt').is_file())
            with patch('evaluation_executor._record_progress'):
                report = _evaluate_semantic({}, source, metadata, runs / 'deeplabv3plus', MagicMock(), DEVICE)
            self.assertEqual((report['status'], report['evaluated']), ('completed', 2))
            self.assertEqual({item['name'] for item in report['metrics']['classes']}, {'background', 'object'})
            self._batch(root, runs / 'deeplabv3plus', 'deeplabv3plus', 'segmentation')


if __name__ == '__main__':
    unittest.main()
