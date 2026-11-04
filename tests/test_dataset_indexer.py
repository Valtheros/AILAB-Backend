from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from dataset_indexer import add_stats, index_image, source_images


class DatasetIndexerTests(unittest.TestCase):
    def test_classification_pixel_hash_and_statistics_across_splits(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for split in ('train', 'test'):
                folder = root / split / '000_curve'
                folder.mkdir(parents=True)
                Image.new('RGB', (600, 300), 'red').save(folder / 'image.png')
            metadata = {'formats': ['imagefolder'], 'label_schema': {'origin': 'label_studio',
                'task': 'image_classification', 'classes': ['curve']}}
            items = source_images(root, metadata)
            summaries, stats = [], {}
            for number, (split, image, annotation) in enumerate(items):
                summary = index_image(root, image, annotation, root / f'thumb-{number}.jpg', root / f'mask-{number}.png')
                summaries.append(summary)
                add_stats(stats, split, summary)
                with Image.open(root / f'thumb-{number}.jpg') as thumbnail:
                    self.assertLessEqual(max(thumbnail.size), 512)
            self.assertEqual(summaries[0]['pixelSha256'], summaries[1]['pixelSha256'])
            self.assertEqual(stats['train']['classes']['curve']['images'], 1)
            self.assertEqual(stats['test']['classes']['curve']['images'], 1)

    def test_coco_negative_images_and_objects_are_counted_separately(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images = root / 'test' / 'images'
            images.mkdir(parents=True)
            for name in ('one.png', 'empty.png'):
                Image.new('RGB', (8, 8)).save(images / name)
            data = {'images': [{'id': 1, 'file_name': 'images/one.png'}, {'id': 2, 'file_name': 'images/empty.png'}],
                    'categories': [{'id': 4, 'name': 'car'}], 'annotations': [
                        {'id': 1, 'image_id': 1, 'category_id': 4, 'bbox': [1, 1, 3, 3]},
                        {'id': 2, 'image_id': 1, 'category_id': 4, 'bbox': [2, 2, 3, 3]}]}
            (root / 'test' / '_annotations.coco.json').write_text(json.dumps(data))
            items = source_images(root, {'formats': ['coco_instances']})
            stats = {}
            for number, (split, image, annotation) in enumerate(items):
                summary = index_image(root, image, annotation, root / f'{number}.jpg', root / f'{number}.png')
                add_stats(stats, split, summary)
            self.assertEqual(stats['test']['images'], 2)
            self.assertEqual(stats['test']['classes']['car'], {'images': 1, 'objects': 2, 'pixels': 0})

    def test_semantic_uses_explicit_sparse_class_ids_for_pixel_counts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images, masks = root / 'train' / 'images', root / 'train' / 'masks'
            images.mkdir(parents=True)
            masks.mkdir()
            Image.new('RGB', (4, 4)).save(images / 'one.png')
            pixels = np.zeros((4, 4), dtype=np.uint8)
            pixels[:2] = 3
            palette_mask = Image.fromarray(pixels, mode='P')
            palette = [0] * 768
            palette[9:12] = [255, 0, 0]
            palette_mask.putpalette(palette)
            palette_mask.save(masks / 'one.png')
            (root / 'train' / '_classes.csv').write_text('id,name\n0,background\n3,road\n')
            split, image, annotation = source_images(root, {'formats': ['semantic_masks']})[0]
            summary = index_image(root, image, annotation, root / 'thumb.jpg', root / 'mask.png')
            self.assertEqual(summary['pixels'], {'background': 8, 'road': 8})
            with Image.open(root / 'mask.png') as thumbnail:
                self.assertEqual(thumbnail.getpixel((0, 0)), 3)
            self.assertTrue(summary['hasObjects'])
            self.assertEqual(split, 'train')

    def test_coco_empty_segmentation_image_counts_background_pixels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images = root / 'test' / 'images'
            images.mkdir(parents=True)
            Image.new('RGB', (4, 4)).save(images / 'empty.png')
            data = {'images': [{'id': 1, 'file_name': 'images/empty.png', 'width': 4, 'height': 4}],
                    'categories': [{'id': 9, 'name': 'road'}], 'annotations': []}
            (root / 'test' / '_annotations.coco.json').write_text(json.dumps(data))
            items = source_images(root, {'formats': ['coco_instances'], 'coco': {'mask_annotations': 1}})
            _, image, annotation = items[0]
            summary = index_image(root, image, annotation, root / 'thumb.jpg', root / 'mask.png')
            self.assertEqual(summary['pixels'], {'background': 16})
            self.assertFalse(summary['hasObjects'])
            self.assertTrue(summary['mask'].is_file())


if __name__ == '__main__':
    unittest.main()
