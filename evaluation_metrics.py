"""Task-specific evaluation metrics with explicit undefined values."""
from __future__ import annotations

import contextlib
from copy import deepcopy
import io

import numpy as np


def _ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def classification_report(confusion, classes: list[str]) -> dict:
    matrix = np.asarray(confusion, dtype=np.int64)
    if matrix.shape != (len(classes), len(classes)) or (matrix < 0).any():
        raise ValueError('Invalid classification confusion matrix.')
    per_class = []
    for index, name in enumerate(classes):
        tp = int(matrix[index, index])
        support, predicted = int(matrix[index].sum()), int(matrix[:, index].sum())
        fp, fn = predicted - tp, support - tp
        per_class.append({'name': name, 'support': support, 'precision': _ratio(tp, tp + fp),
                          'recall': _ratio(tp, tp + fn), 'f1': _ratio(2 * tp, 2 * tp + fp + fn)})
    return {'accuracy': _ratio(int(np.trace(matrix)), int(matrix.sum())), 'images': int(matrix.sum()), 'classes': per_class}


def semantic_report(confusion, classes: list[str]) -> dict:
    matrix = np.asarray(confusion, dtype=np.int64)
    if matrix.shape != (len(classes), len(classes)) or (matrix < 0).any():
        raise ValueError('Invalid semantic confusion matrix.')
    per_class = []
    for index, name in enumerate(classes):
        tp = int(matrix[index, index])
        support, predicted = int(matrix[index].sum()), int(matrix[:, index].sum())
        per_class.append({'name': name, 'pixels': support, 'iou': _ratio(tp, support + predicted - tp),
                          'dice': _ratio(2 * tp, support + predicted)})
    values = [entry['iou'] for entry in per_class if entry['iou'] is not None]
    return {'pixelAccuracy': _ratio(int(np.trace(matrix)), int(matrix.sum())),
            'meanIoU': sum(values) / len(values) if values else None, 'classes': per_class}


def detection_report(images: list[dict], annotations: list[dict], categories: list[dict],
                     predictions: list[dict], iou_type: str = 'bbox') -> tuple[dict, dict[int, dict]]:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    if iou_type not in {'bbox', 'segm'}:
        raise ValueError('Unsupported COCO evaluation type.')
    with contextlib.redirect_stdout(io.StringIO()):
        ground_truth = COCO()
        # COCOeval mutates mask annotations into RLE containing bytes.
        ground_truth.dataset = {'images': images, 'annotations': deepcopy(annotations), 'categories': categories, 'info': {}}
        ground_truth.createIndex()
        if predictions:
            result = ground_truth.loadRes(deepcopy(predictions))
        else:
            result = COCO()
            result.dataset = {'images': images, 'annotations': [], 'categories': categories}
            result.createIndex()
        evaluator = COCOeval(ground_truth, result, iou_type)
        evaluator.params.imgIds = [image['id'] for image in images]
        evaluator.params.catIds = [category['id'] for category in categories]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()

    precision = evaluator.eval['precision']
    thresholds = evaluator.params.iouThrs
    iou_index = int(np.argmin(np.abs(thresholds - .5)))
    by_class = {category['id']: {'name': category['name'], 'tp': 0, 'fp': 0, 'support': 0} for category in categories}
    by_image = {image['id']: {'tp': 0, 'fp': 0, 'fn': 0} for image in images}
    for item in evaluator.evalImgs:
        if not item or item['aRng'] != evaluator.params.areaRng[0]:
            continue
        category = by_class[item['category_id']]
        selected = np.asarray(item['dtScores']) >= .25
        matches = np.asarray(item['dtMatches'][iou_index]) > 0
        ignored = np.asarray(item['dtIgnore'][iou_index], dtype=bool)
        tp = int((selected & matches & ~ignored).sum())
        fp = int((selected & ~matches & ~ignored).sum())
        support = int((~np.asarray(item['gtIgnore'], dtype=bool)).sum())
        category['tp'] += tp
        category['fp'] += fp
        category['support'] += support
        result = by_image[item['image_id']]
        result['tp'] += tp
        result['fp'] += fp
        result['fn'] += max(0, support - tp)

    per_class = []
    for category in categories:
        index = evaluator.params.catIds.index(category['id'])
        stats = by_class[category['id']]
        values = precision[:, :, index, 0, -1]
        values50 = precision[iou_index, :, index, 0, -1]
        valid, valid50 = values[values >= 0], values50[values50 >= 0]
        tp, fp, support = stats['tp'], stats['fp'], stats['support']
        per_class.append({'name': stats['name'], 'support': support, 'ap': float(valid.mean()) if valid.size else None,
                          'ap50': float(valid50.mean()) if valid50.size else None,
                          'precision': _ratio(tp, tp + fp), 'recall': _ratio(tp, support),
                          'f1': _ratio(2 * tp, tp + fp + support)})
    def score(value):
        return float(value) if value >= 0 else None
    return {'map': score(evaluator.stats[0]), 'map50': score(evaluator.stats[1]),
            'confidenceThreshold': .25, 'iouThreshold': .5, 'classes': per_class}, by_image
