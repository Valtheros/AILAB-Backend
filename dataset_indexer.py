"""Build read-only image summaries and thumbnails for one immutable dataset."""
from __future__ import annotations

import hashlib
import csv
import json
import math
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from dataset_utils import (IMAGE_EXTENSIONS, _matching_mask_path, _normalize_yolo_yaml_value,
                           _roboflow_semantic_pairs, _semantic_split_dirs, find_dataset_yaml)
from evaluation_executor import _classification_names
from security_utils import contained_path
from worker.trainers.input_limits import (MAX_BOXES_PER_IMAGE, iter_text_lines_limited, open_image_checked,
                                          read_json_limited, read_yaml_limited, validate_coco_document)
from worker.trainers.trainer_utils import read_classes_from_yaml, yolo_split_image_dirs

SPLITS = {'train': ('train', 'training'), 'val': ('val', 'valid', 'validation'), 'test': ('test',)}


def _name(value):
    return unicodedata.normalize('NFC', str(value)).strip()


def _images(directory):
    if not directory.is_dir():
        return []
    return sorted(path for path in directory.rglob('*') if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def _classification(root, metadata):
    result = []
    for split, aliases in SPLITS.items():
        directory = next((root / name for name in aliases if (root / name).is_dir()), None)
        if not directory:
            continue
        folders = sorted(path for path in directory.iterdir() if path.is_dir() and not path.name.startswith('.'))
        names = _classification_names([folder.name for folder in folders], metadata)
        for folder, name in zip(folders, names):
            result.extend((split, path, {'kind': 'classification', 'className': name}) for path in _images(folder))
    return result


def _semantic(root, metadata):
    result = []
    for split, aliases in SPLITS.items():
        schema = metadata.get('label_schema') or {}
        declared = schema.get('classes') if schema.get('task') == 'semantic_segmentation' else None
        class_map = {0: 'background', **{index: _name(name) for index, name in enumerate(declared or [], 1)}}
        if not declared:
            csv_paths = [root / alias / '_classes.csv' for alias in aliases] + [root / 'train' / '_classes.csv']
            class_file = next((path for path in csv_paths if path.is_file()), None)
            if class_file:
                with class_file.open(encoding='utf-8-sig', newline='') as stream:
                    for row in csv.reader(stream):
                        if len(row) >= 2 and row[0].strip().isdigit():
                            class_id = int(row[0].strip())
                            name = _name(row[1])
                            if not 0 <= class_id < 255 or not name or (class_id in class_map and class_id != 0):
                                raise ValueError('Semantic class IDs are invalid or duplicated.')
                            class_map[class_id] = name
        if len(set(class_map.values())) != len(class_map):
            raise ValueError('Semantic class names are ambiguous.')
        directories = _semantic_split_dirs(root, split)
        if directories:
            image_root, mask_root = directories
            pairs = [(image, _matching_mask_path(mask_root, image)) for image in _images(image_root)]
        else:
            directory = next((root / name for name in aliases if (root / name).is_dir()), None)
            pairs = _roboflow_semantic_pairs(directory) if directory else []
        result.extend((split, image, {'kind': 'semantic', 'mask': mask, 'classMap': class_map}) for image, mask in pairs)
    return result


def _coco_path(root, annotation, filename, split):
    normalized = str(filename).replace('\\', '/')
    relative = Path(normalized)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('COCO image path is outside the dataset.')
    other_splits = {part for name, aliases in SPLITS.items() if name != split for part in aliases}
    if any(part.lower() in other_splits for part in relative.parts[:-1]):
        raise ValueError('COCO annotation refers to a different split.')
    candidates = [contained_path(root, annotation.parent, normalized), contained_path(root, normalized)]
    for alias in SPLITS[split]:
        candidates.append(contained_path(root, alias, 'images', normalized))
    candidates.append(contained_path(root, 'images', normalized))
    return next((path for path in candidates if path.is_file()), candidates[0])


def _coco(root, metadata):
    result = []
    for split, aliases in SPLITS.items():
        candidates = [root / alias / name for alias in aliases for name in
                      ('_annotations.coco.json', f'instances_{alias}.json', f'{alias}.json')]
        candidates += [root / f'instances_{alias}.json' for alias in aliases] + [root / f'{alias}.json' for alias in aliases]
        files = [path for path in candidates if path.is_file()]
        if not files:
            known = [contained_path(root, value) for value in metadata.get('coco_files', [])]
            files = [path for path in known if path.is_file() and any(part.lower() in aliases for part in path.relative_to(root).parts[:-1])]
            if not files and split == 'train' and len(known) == 1 and known[0].is_file():
                files = known
        if len(files) > 1:
            raise ValueError('More than one COCO annotation file matches the same split.')
        if not files:
            continue
        annotation = files[0]
        data = read_json_limited(annotation)
        validate_coco_document(data, str(annotation))
        categories = {item['id']: _name(item.get('name', '')) for item in data['categories']}
        if not all(categories.values()) or len(set(categories.values())) != len(categories):
            raise ValueError('COCO categories have missing or ambiguous names.')
        class_ids = {category_id: index for index, category_id in enumerate(categories, 1)}
        segmented = bool((metadata.get('coco') or {}).get('mask_annotations')) or any(item.get('segmentation') for item in data.get('annotations', []))
        by_image = {}
        for item in data.get('annotations', []):
            bbox = item.get('bbox')
            if not isinstance(bbox, list) or len(bbox) != 4 or any(not math.isfinite(float(value)) for value in bbox):
                raise ValueError('COCO annotation contains an invalid bounding box.')
            by_image.setdefault(item['image_id'], []).append({'className': categories[item['category_id']],
                'classId': class_ids[item['category_id']], 'bbox': bbox, 'segmentation': item.get('segmentation'),
                'iscrowd': int(bool(item.get('iscrowd')))})
        for image in data['images']:
            path = _coco_path(root, annotation, image['file_name'], split)
            result.append((split, path, {'kind': 'objects', 'objects': by_image.get(image['id'], []),
                                         'segmented': segmented, 'classMap': {0: 'background', **{class_ids[key]: value for key, value in categories.items()}}}))
    return result


def _yolo(root):
    yaml = find_dataset_yaml(root)
    if not yaml:
        return []
    config = read_yaml_limited(yaml) or {}
    classes = [_name(value) for value in read_classes_from_yaml(str(yaml))]
    if not classes or not all(classes) or len(set(classes)) != len(classes):
        raise ValueError('YOLO class names are missing or ambiguous.')
    result = []
    for split, aliases in SPLITS.items():
        value = config.get('val' if split == 'val' else split)
        if value is None:
            directories = yolo_split_image_dirs(str(root), None, split)
        else:
            values = value if isinstance(value, list) else [value]
            directories = [contained_path(root, _normalize_yolo_yaml_value(root, split, item)) for item in values]
        for directory in directories:
            others = {part for name, names in SPLITS.items() if name != split for part in names}
            if any(part.lower() in others for part in directory.relative_to(root).parts):
                raise ValueError('YOLO split refers to a different split.')
            for image in _images(directory):
                parts = list(image.resolve().relative_to(root).parts)
                positions = [index for index, part in enumerate(parts) if part.lower() == 'images']
                if not positions:
                    raise ValueError('YOLO images must be inside an images directory.')
                parts[positions[-1]] = 'labels'
                result.append((split, image, {'kind': 'yolo', 'label': contained_path(root, *parts).with_suffix('.txt'), 'classes': classes}))
    return result


def source_images(root: Path, metadata: dict):
    formats = set(metadata.get('formats') or [])
    if 'imagefolder' in formats:
        return _classification(root, metadata)
    if 'semantic_masks' in formats:
        return _semantic(root, metadata)
    if 'coco_instances' in formats:
        return _coco(root, metadata)
    if 'yolo_detection' in formats:
        return _yolo(root)
    raise ValueError('This dataset format cannot be indexed.')


def _yolo_objects(label, classes, width, height):
    objects = []
    if label.is_file():
        for index, line in enumerate(iter_text_lines_limited(label), 1):
            if index > MAX_BOXES_PER_IMAGE:
                raise ValueError('Too many labels in one image.')
            values = [float(value) for value in line.split()]
            if len(values) != 5 or not all(math.isfinite(value) for value in values):
                raise ValueError('Invalid YOLO annotation.')
            class_id = int(values[0])
            cx, cy, bw, bh = values[1:]
            if values[0] != class_id or not 0 <= class_id < len(classes) or not 0 <= cx <= 1 or not 0 <= cy <= 1 or not 0 < bw <= 1 or not 0 < bh <= 1:
                raise ValueError('Invalid YOLO annotation.')
            objects.append({'className': classes[class_id], 'bbox': [(cx - bw / 2) * width, (cy - bh / 2) * height, bw * width, bh * height]})
    return objects


def index_image(root: Path, path: Path, source: dict, thumbnail: Path, mask_thumbnail: Path):
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError('Image is outside the dataset.')
    with open_image_checked(resolved) as image:
        width, height = image.size
        normalized = ImageOps.exif_transpose(image).convert('RGB')
        digest = hashlib.sha256(f'{normalized.width}x{normalized.height}:'.encode())
        digest.update(normalized.tobytes())
        raw = image.convert('RGB')
    raw.thumbnail((512, 512), Image.Resampling.LANCZOS)
    thumbnail.parent.mkdir(parents=True, exist_ok=True)
    raw.save(thumbnail, format='JPEG', quality=85)
    kind = source['kind']
    annotation = {'kind': kind}
    counts, pixels = Counter(), Counter()
    classes = []
    semantic_objects = False
    generated_mask = None
    if kind == 'classification':
        annotation['className'] = source['className']
        classes = [source['className']]
    elif kind in {'objects', 'yolo'}:
        objects = _yolo_objects(source['label'], source['classes'], width, height) if kind == 'yolo' else source['objects']
        counts.update(item['className'] for item in objects)
        classes = sorted(counts)
        annotation.update(kind='objects', objects=objects)
        if kind == 'objects' and source.get('segmented') and len(source['classMap']) <= 255 and all(item.get('segmentation') for item in objects):
            from evaluation_executor import _semantic_coco_mask
            from worker.trainers.input_limits import MAX_COCO_DECODED_MASK_PIXELS_PER_IMAGE
            if width * height * len(objects) > MAX_COCO_DECODED_MASK_PIXELS_PER_IMAGE:
                raise ValueError('COCO masks exceed the image memory budget.')
            regions = [{'category_id': item['classId'], 'segmentation': item['segmentation'], 'iscrowd': item['iscrowd']}
                       for item in objects]
            mask_array = _semantic_coco_mask(regions, width, height)
            found, amounts = np.unique(mask_array, return_counts=True)
            for value, amount in zip(found.tolist(), amounts.tolist()):
                if value != 255:
                    pixels[source['classMap'].get(value, f'class ID {value}')] += amount
            classes = sorted(set(classes).union(pixels))
            mask_thumbnail.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(mask_array, mode='L').resize(raw.size, Image.Resampling.NEAREST).save(mask_thumbnail, format='PNG')
            generated_mask = mask_thumbnail
            annotation.update(mask=True, classes=[{'id': key, 'name': value} for key, value in sorted(source['classMap'].items())])
        elif kind == 'objects' and source.get('segmented'):
            annotation['maskUnavailable'] = True
    elif kind == 'semantic':
        mask_path = source['mask']
        if not mask_path or not mask_path.resolve().is_relative_to(root):
            raise ValueError('Semantic mask is unavailable.')
        with open_image_checked(mask_path) as mask:
            if mask.size != (width, height) or mask.mode not in {'L', 'P', 'I', 'I;16'}:
                raise ValueError('Semantic image and indexed mask are incompatible.')
            values = np.asarray(mask, dtype=np.int64)
            found, amounts = np.unique(values, return_counts=True)
            for value, amount in zip(found.tolist(), amounts.tolist()):
                if value == 255:
                    continue
                if not 0 <= value < 255:
                    raise ValueError('Semantic mask class ID is out of range.')
                name = source['classMap'].get(value, f'class ID {value}')
                pixels[name] += amount
                semantic_objects = semantic_objects or value > 0
            classes = sorted(pixels)
            mask_thumbnail.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(values.astype(np.uint8), mode='L').resize(raw.size, Image.Resampling.NEAREST).save(mask_thumbnail, format='PNG')
            generated_mask = mask_thumbnail
        names = {value: source['classMap'].get(value, f'class ID {value}') for value in found.tolist() if value != 255}
        annotation.update(classes=[{'id': key, 'name': value} for key, value in sorted(names.items())], mask=True)
    return {'width': width, 'height': height, 'pixelSha256': digest.hexdigest(), 'classes': classes,
            'objects': dict(counts), 'pixels': dict(pixels), 'annotation': annotation,
            'hasObjects': bool(counts) if kind in {'objects', 'yolo'} else semantic_objects if kind == 'semantic' else bool(classes),
            'mask': generated_mask}


def add_stats(stats, split, summary):
    entry = stats.setdefault(split, {'images': 0, 'classes': {}, 'pixels': 0})
    entry['images'] += 1
    entry['pixels'] += sum(summary['pixels'].values())
    for name in summary['classes']:
        cls = entry['classes'].setdefault(name, {'images': 0, 'objects': 0, 'pixels': 0})
        cls['images'] += 1
        cls['objects'] += summary['objects'].get(name, 0)
        cls['pixels'] += summary['pixels'].get(name, 0)
