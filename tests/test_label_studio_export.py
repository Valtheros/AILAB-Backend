import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from dataset_utils import inspect_dataset, validate_dataset_for_upload
from label_studio_export import SnapshotExporter


class ExportTests(unittest.TestCase):
    def test_all_four_layouts_pass_dataset_inspection(self):
        def download(owner, action, payload, target):
            Image.new('L' if action == 'mask-download' else 'RGB', (16, 12), 255).save(target)

        for task in ('image_classification', 'object_detection', 'semantic_segmentation', 'instance_segmentation'):
            with self.subTest(task=task), tempfile.TemporaryDirectory() as directory, patch('label_studio_export.download', download):
                root = Path(directory)
                exporter = SnapshotExporter(root, 'owner', 'snapshot', {'kind': task, 'classes': [{'name': 'one'}, {'name': 'two'}]})
                for index, label in enumerate(('one', 'two'), 1):
                    kind = 'choices' if task == 'image_classification' else 'rectanglelabels' if task == 'object_detection' else 'polygonlabels'
                    value = {kind: [label]}
                    if kind == 'rectanglelabels':
                        value.update(x=10, y=10, width=40, height=40)
                    elif kind == 'polygonlabels':
                        value['points'] = [[10, 10], [80, 10], [80, 80], [10, 80]]
                    result = [{'id': str(index), 'type': kind, 'value': value}]
                    if task.endswith('segmentation'):
                        result.append({'id': f'brush-{index}', 'type': 'brushlabels', 'value': {'brushlabels': [label]}})
                    exporter.write_image({'id': index, 'width': 16, 'height': 12, 'split': 'train', 'result': result})
                schema = exporter.finish()
                metadata = inspect_dataset(root)
                validate_dataset_for_upload(root, metadata)
                self.assertTrue(metadata['formats'])
                if task == 'semantic_segmentation':
                    self.assertEqual(schema['num_classes'], 3)
                    self.assertEqual(metadata['classes'], ['one', 'two'])


if __name__ == '__main__':
    unittest.main()
