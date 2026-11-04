"""Evaluate detection and instance models without reusing validation data."""
from __future__ import annotations

import math
import unicodedata
from pathlib import Path

import numpy as np

from dataset_utils import _normalize_yolo_yaml_value, find_dataset_yaml
from evaluation_metrics import detection_report
from security_utils import contained_path
from worker.trainers.input_limits import (MAX_BOXES_PER_IMAGE, iter_text_lines_limited,
                                          read_json_limited, read_yaml_limited, validate_coco_document,
                                          validate_coco_segmentation)
from worker.trainers.trainer_utils import read_classes_from_yaml, yolo_split_image_dirs


def _name(value):
    return unicodedata.normalize('NFC', str(value)).strip()


def _class_map(names, model_classes):
    source = [_name(name) for name in names]
    target = [_name(name) for name in model_classes]
    if not source or not target or not all(source) or not all(target) or len(set(source)) != len(source) or len(set(target)) != len(target):
        raise ValueError('Dataset or checkpoint has ambiguous class names.')
    unknown = [name for name in source if name not in target]
    if unknown:
        raise ValueError('Dataset contains classes that the trained model does not know: ' + ', '.join(unknown[:10]))
    return [target.index(name) + 1 for name in source]


def _test_annotation_file(root):
    for path in (root / 'test' / '_annotations.coco.json', root / 'test' / 'instances_test.json',
                 root / 'test' / 'test.json', root / 'instances_test.json', root / 'test.json'):
        if path.is_file():
            return path
    return None


def _coco_image_path(root, annotation_path, file_name):
    normalized = str(file_name).replace('\\', '/')
    path = Path(normalized)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('COCO image path is outside the dataset.')
    if any(part.lower() in {'train', 'training', 'val', 'valid', 'validation'} for part in path.parts[:-1]):
        raise ValueError('Test annotations cannot refer to training or validation images.')
    candidates = [contained_path(root, annotation_path.parent, normalized), contained_path(root, normalized),
                  contained_path(root, 'test', 'images', normalized), contained_path(root, 'images', normalized)]
    return next((candidate for candidate in candidates if candidate.is_file()), candidates[0])


def _coco_test_data(root, model_classes):
    annotation_path = _test_annotation_file(root)
    if annotation_path is None:
        return None
    data = read_json_limited(annotation_path)
    validate_coco_document(data, str(annotation_path))
    categories = data.get('categories', [])
    mapping = _class_map([category.get('name', '') for category in categories], model_classes)
    ids = {category['id']: mapped for category, mapped in zip(categories, mapping)}
    by_image = {}
    for annotation in data.get('annotations', []):
        by_image.setdefault(annotation['image_id'], []).append({**annotation, 'category_id': ids[annotation['category_id']]})
    return [(dict(image), _coco_image_path(root, annotation_path, image['file_name']), by_image.get(image['id'], []))
            for image in data.get('images', [])], {str(category['id']): model_classes[value - 1]
                                                    for category, value in zip(categories, mapping)}


def _yolo_test_data(root, model_classes):
    yaml_path = find_dataset_yaml(root)
    if yaml_path is None:
        return None
    names = read_classes_from_yaml(str(yaml_path))
    mapping = _class_map(names, model_classes)
    config = read_yaml_limited(yaml_path) or {}
    test = config.get('test')
    if test is None:
        directories = yolo_split_image_dirs(str(root), None, 'test')
    else:
        values = test if isinstance(test, list) else [test]
        directories = [contained_path(root, _normalize_yolo_yaml_value(root, 'test', value)) for value in values]
    for directory in directories:
        if any(part.lower() in {'train', 'training', 'valid', 'val', 'validation'} for part in directory.relative_to(root).parts):
            raise ValueError('The test split cannot use training or validation images.')
    paths = sorted({path for directory in directories for path in directory.rglob('*')
                    if path.is_file() and path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}})
    items = []
    for index, path in enumerate(paths, 1):
        parts = list(path.resolve().relative_to(root).parts)
        indices = [position for position, part in enumerate(parts) if part.lower() == 'images']
        if not indices:
            raise ValueError('YOLO test images must be inside an images directory.')
        parts[indices[-1]] = 'labels'
        label_path = contained_path(root, *parts).with_suffix('.txt')
        items.append(({'id': index}, path.resolve(), (label_path, mapping)))
    return items, {str(index): model_classes[value - 1] for index, value in enumerate(mapping)}


def _ground_truth(raw, image_id, width, height, annotation_id, instance):
    if isinstance(raw, tuple):
        label_path, mapping = raw
        converted = []
        if label_path.is_file():
            for line_number, line in enumerate(iter_text_lines_limited(label_path), 1):
                if line_number > MAX_BOXES_PER_IMAGE:
                    raise ValueError('Too many ground-truth boxes in one image.')
                values = [float(value) for value in line.split()]
                if len(values) != 5 or not all(math.isfinite(value) for value in values):
                    raise ValueError('Invalid YOLO box annotation.')
                class_id = int(values[0])
                if values[0] != class_id or not 0 <= class_id < len(mapping):
                    raise ValueError('YOLO annotation refers to an unknown class.')
                cx, cy, bw, bh = values[1:]
                if not 0 <= cx <= 1 or not 0 <= cy <= 1 or not 0 < bw <= 1 or not 0 < bh <= 1:
                    raise ValueError('YOLO annotation has an invalid bounding box.')
                converted.append({'category_id': mapping[class_id], 'bbox': [(cx - bw / 2) * width,
                                  (cy - bh / 2) * height, bw * width, bh * height]})
        raw = converted
    result = []
    for source in raw:
        box = source.get('bbox')
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            raise ValueError('Invalid ground-truth bounding box.')
        x, y, bw, bh = [float(value) for value in box]
        if not all(math.isfinite(value) for value in (x, y, bw, bh)) or bw <= 0 or bh <= 0:
            raise ValueError('Invalid ground-truth bounding box.')
        item = {'id': annotation_id + len(result), 'image_id': image_id, 'category_id': source['category_id'],
                'bbox': [x, y, bw, bh], 'iscrowd': int(bool(source.get('iscrowd', 0)))}
        area = float(source.get('area') or bw * bh)
        if not math.isfinite(area) or area <= 0:
            raise ValueError('Invalid ground-truth area.')
        item['area'] = area
        if instance:
            segmentation = source.get('segmentation')
            if not validate_coco_segmentation(segmentation, width, height):
                raise ValueError('Instance annotations require valid masks.')
            item['segmentation'] = segmentation
        result.append(item)
    return result


def _prediction(entry, image, image_id, instance):
    import torch
    import torchvision.transforms.functional as F
    from pycocotools import mask as coco_mask

    if entry['kind'] == 'yolo':
        results = entry['model'].predict(source=image, conf=.001, iou=.7, max_det=100,
                                         verbose=False, device=entry['device'])
        boxes = results[0].boxes if results else None
        if boxes is None:
            return []
        values = zip(boxes.xyxy.tolist(), boxes.cls.tolist(), boxes.conf.tolist())
        return [{'image_id': image_id, 'category_id': int(label) + 1,
                 'bbox': [float(box[0]), float(box[1]), float(box[2] - box[0]), float(box[3] - box[1])],
                 'score': float(score)} for box, label, score in values]
    entry['model'].roi_heads.score_thresh = .001
    entry['model'].roi_heads.detections_per_img = 100
    with torch.no_grad():
        result = entry['model']([F.to_tensor(image).to(entry['device'])])[0]
    boxes, labels, scores = result['boxes'].tolist(), result['labels'].tolist(), result['scores'].tolist()
    predictions = []
    for index, (box, label, score) in enumerate(zip(boxes, labels, scores)):
        if not 1 <= int(label) <= len(entry['classes']):
            raise ValueError('The model predicted an unknown class.')
        prediction = {'image_id': image_id, 'category_id': int(label), 'score': float(score),
                      'bbox': [float(box[0]), float(box[1]), float(box[2] - box[0]), float(box[3] - box[1])]}
        if instance:
            binary = (result['masks'][index, 0].detach().cpu().numpy() > .5).astype(np.uint8)
            encoded = coco_mask.encode(np.asfortranarray(binary))
            prediction['segmentation'] = {'size': encoded['size'], 'counts': encoded['counts'].decode('ascii')}
        predictions.append(prediction)
    return predictions


def evaluate_detection(row, root, metadata, run_dir, model_type, writer, device, record_progress):
    from detection_inference import load_detector
    from segmentation_inference import load_segmenter
    from worker.trainers.input_limits import open_rgb_image_checked

    instance = model_type == 'mask_rcnn'
    entry = load_segmenter(run_dir, model_type, device=device) if instance else load_detector(run_dir, model_type, device=device)
    classes = [_name(name) for name in entry['classes']]
    source = (_yolo_test_data(root, classes) if 'yolo_detection' in metadata.get('formats', []) and not instance
              else _coco_test_data(root, classes))
    if source is None or not source[0]:
        return {'status': 'no_test', 'metrics': None, 'evaluated': 0, 'skipped': 0}
    items, mapping = source
    record_progress(row, len(items))
    images, annotations, predictions, records = [], [], [], []
    for index, (image_info, path, raw_annotations) in enumerate(items, 1):
        image = None
        try:
            image = open_rgb_image_checked(path)
            width, height = image.size
            if image_info.get('width') is not None and int(image_info['width']) != width:
                raise ValueError('Image width does not match its annotation.')
            if image_info.get('height') is not None and int(image_info['height']) != height:
                raise ValueError('Image height does not match its annotation.')
            truth = _ground_truth(raw_annotations, index, width, height, len(annotations) + 1, instance)
        except (OSError, ValueError):
            if image is not None:
                image.close()
            records.append((path, None, {'error': 'Image or annotation could not be evaluated.'}))
            if index % 10 == 0:
                record_progress(row, len(items), index)
            continue
        with image:
            predicted = _prediction(entry, image, index, instance)
        images.append({'id': index, 'width': width, 'height': height})
        annotations.extend(truth)
        predictions.extend(predicted)
        records.append((path, index, {'actual': truth, 'predictions': predicted, 'width': width, 'height': height}))
        if index % 10 == 0:
            record_progress(row, len(items), index)
    if not images:
        raise ValueError('No usable detection test images were found.')
    categories = [{'id': index, 'name': name} for index, name in enumerate(classes, 1)]
    bbox, by_image = detection_report(images, annotations, categories, predictions, 'bbox')
    metrics = {'boxes': bbox}
    if instance:
        metrics['masks'], _ = detection_report(images, annotations, categories, predictions, 'segm')
    for path, image_id, detail in records:
        if image_id is None:
            writer.add(path, 'error', None, detail)
        else:
            scores = by_image[image_id]
            detail['counts'] = scores
            if instance:
                from evaluation_executor import _semantic_coco_mask
                actual_mask = _semantic_coco_mask(detail['actual'], detail['width'], detail['height'])
                predicted_mask = _semantic_coco_mask([item for item in detail['predictions'] if item['score'] >= .25],
                                                     detail['width'], detail['height'])
                detail['actualMask'] = writer.save_mask('actual', actual_mask)
                detail['predictedMask'] = writer.save_mask('predicted', predicted_mask)
            writer.add(path, 'incorrect' if scores['fp'] or scores['fn'] else 'correct', None, detail,
                       fp=scores['fp'], fn=scores['fn'])
    return {'status': 'partial' if len(images) != len(items) else 'completed', 'metrics': metrics,
            'classMapping': mapping, 'evaluated': len(images), 'skipped': len(items) - len(images)}
