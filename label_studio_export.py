"""Convert a frozen Label Studio snapshot to existing AILAB trainer formats."""
from __future__ import annotations

import json
import math
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from pycocotools import mask as coco_mask

from label_studio_client import download


class SnapshotExporter:
    def __init__(self, root: Path, owner: str, snapshot: str, config: dict):
        self.root, self.owner, self.snapshot, self.config = root, owner, snapshot, config
        self.kind = config['kind']
        self.classes = [c['name'] for c in config['classes']]
        self.train_classes = set()
        self.coco = {split: {'images': [], 'annotations': [], 'categories': [
            {'id': index + 1, 'name': name, 'supercategory': 'none'} for index, name in enumerate(self.classes)
        ]} for split in ('train', 'val', 'test')}
        self.annotation_id = 0

    def write_image(self, item: dict):
        image_id = int(item['id'])
        width, height, split = int(item['width']), int(item['height']), item['split']
        if split not in self.coco or not 0 < width * height <= 25_000_000:
            raise ValueError('Invalid snapshot image size or split.')
        result = item['result']
        if self.kind == 'image_classification':
            selected = result[0]['value']['choices']
            if len(selected) != 1 or selected[0] not in self.classes:
                raise ValueError('Classification images must have exactly one valid class.')
            class_id = self.classes.index(selected[0])
            # Numeric prefixes preserve the declared class order in ImageFolder.
            folder = f'{class_id:03d}_' + ''.join(c if c.isalnum() or c in '-_' else '_' for c in selected[0])[:80]
            target = self.root / split / folder / f'{image_id}.png'
            if split == 'train':
                self.train_classes.add(class_id)
        else:
            target = self.root / split / 'images' / f'{image_id}.png'
        target.parent.mkdir(parents=True, exist_ok=True)
        download(self.owner, 'image-download', {'snapshot': self.snapshot, 'image': image_id}, target)
        with Image.open(target) as original:
            if original.size != (width, height):
                raise ValueError('Snapshot image dimensions changed.')
            original.verify()
        if self.kind == 'image_classification':
            return
        document = self.coco[split]
        document['images'].append({'id': image_id, 'file_name': f'images/{image_id}.png', 'width': width, 'height': height})
        semantic = np.zeros((height, width), dtype=np.uint8) if self.kind == 'semantic_segmentation' else None
        for region in result:
            kind, value = region['type'], region['value']
            class_id = self.classes.index(value[kind][0]) + 1
            if kind == 'rectanglelabels':
                box = [value['x'] * width / 100, value['y'] * height / 100,
                       value['width'] * width / 100, value['height'] * height / 100]
                if any(not math.isfinite(v) or v < 0 for v in box) or box[2] <= 0 or box[3] <= 0:
                    raise ValueError('Invalid bounding box.')
                self.add_annotation(document, image_id, class_id, box, box[2] * box[3])
                continue
            if kind == 'polygonlabels':
                points = [(min(width - 1, max(0, x * width / 100)), min(height - 1, max(0, y * height / 100))) for x, y in value['points']]
                image = Image.new('L', (width, height), 0)
                ImageDraw.Draw(image).polygon(points, fill=1)
                binary = np.asarray(image, dtype=np.uint8)
                segmentation = [[coord for point in points for coord in point]]
            elif kind == 'brushlabels':
                temporary = self.root / '.brush.png'
                download(self.owner, 'mask-download', {'snapshot': self.snapshot, 'image': image_id, 'region': region['id']}, temporary)
                with Image.open(temporary) as image:
                    if image.size != (width, height):
                        raise ValueError('Brush mask dimensions changed.')
                    binary = (np.asarray(image.convert('L')) > 0).astype(np.uint8)
                temporary.unlink()
                segmentation = None
            else:
                raise ValueError('Unsupported segmentation region.')
            if semantic is not None:
                semantic[binary != 0] = class_id
            elif binary.any():
                encoded = coco_mask.encode(np.asfortranarray(binary))
                area = float(coco_mask.area(encoded))
                box = coco_mask.toBbox(encoded).tolist()
                if segmentation is None:
                    segmentation = {'size': encoded['size'], 'counts': encoded['counts'].decode('ascii')}
                self.add_annotation(document, image_id, class_id, box, area, segmentation)
        if semantic is not None:
            masks = self.root / split / 'masks'
            masks.mkdir(parents=True, exist_ok=True)
            Image.fromarray(semantic).save(masks / f'{image_id}.png')

    def add_annotation(self, document, image_id, class_id, box, area, segmentation=None):
        self.annotation_id += 1
        annotation = {'id': self.annotation_id, 'image_id': image_id, 'category_id': class_id,
                      'bbox': box, 'area': area, 'iscrowd': 0}
        if segmentation is not None:
            annotation['segmentation'] = segmentation
        document['annotations'].append(annotation)

    def finish(self):
        if self.kind == 'semantic_segmentation':
            for split in ('train', 'val', 'test'):
                if (self.root / split).is_dir():
                    with (self.root / split / '_classes.csv').open('w', newline='') as output:
                        writer = csv.writer(output)
                        writer.writerow(['id', 'name'])
                        writer.writerow([0, 'background'])
                        writer.writerows(enumerate(self.classes, start=1))
        if self.kind == 'image_classification' and len(self.train_classes) != len(self.classes):
            missing = [name for i, name in enumerate(self.classes) if i not in self.train_classes]
            raise ValueError('Move at least one image of every class to Train. Missing: ' + ', '.join(missing))
        if self.kind in {'object_detection', 'instance_segmentation'}:
            if not self.coco['train']['annotations']:
                raise ValueError('At least one labeled object is required in Train; other images may be empty.')
            for split, document in self.coco.items():
                if document['images']:
                    (self.root / split / '_annotations.coco.json').write_text(json.dumps(document, allow_nan=False))
        return {'origin': 'label_studio', 'task': self.kind, 'classes': self.classes,
                'background_id': 0 if self.kind == 'semantic_segmentation' else None,
                'num_classes': len(self.classes) + (1 if self.kind in {'semantic_segmentation', 'instance_segmentation'} else 0)}
