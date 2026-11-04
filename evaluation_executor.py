"""Evaluate a trained checkpoint against one immutable dataset version."""
from __future__ import annotations

import json
import os
import shutil
import unicodedata
import uuid
import csv
from pathlib import Path

import numpy as np

from dataset_storage import registered_storage_path
from dataset_utils import IMAGE_EXTENSIONS
from dataset_versioning import sha256_stream, verify_source
from evaluation_metrics import classification_report, semantic_report
from resource_repository import resource_repository as registry
from security_utils import contained_path
from settings import DATASET_DIR, RUNS_DIR


def _name(value):
    return unicodedata.normalize('NFC', str(value)).strip()


def _classification_names(raw_names: list[str], metadata: dict) -> list[str]:
    schema = metadata.get('label_schema') or {}
    if schema.get('origin') == 'label_studio' and schema.get('task') == 'image_classification':
        declared = list(schema.get('classes') or [])
        encoded = [f"{index:03d}_" + ''.join(c if c.isalnum() or c in '-_' else '_' for c in value)[:80]
                   for index, value in enumerate(declared)]
        mapping = dict(zip(encoded, declared))
        if any(name not in mapping for name in raw_names):
            raise ValueError('Classification folders do not match their declared class schema.')
        names = [_name(mapping[name]) for name in raw_names]
    else:
        names = [_name(name) for name in raw_names]
    if not all(names) or len(set(names)) != len(names):
        raise ValueError('Class names are empty or ambiguous.')
    return names


def _classification_items(root: Path, metadata: dict, model_classes: list[str]):
    split = next((root / name for name in ('test', 'testing') if (root / name).is_dir()), None)
    if split is None:
        return []
    folders = sorted(path for path in split.iterdir() if path.is_dir() and not path.name.startswith('.'))
    names = _classification_names([folder.name for folder in folders], metadata)
    mapping = dict(zip(names, folders))
    unknown = [name for name in names if name not in model_classes and any(
        path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS for path in mapping[name].rglob('*'))]
    if unknown:
        raise ValueError('Dataset contains classes that the trained model does not know: ' + ', '.join(unknown[:10]))
    items = []
    for name, folder in mapping.items():
        for path in sorted(folder.rglob('*')):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                resolved = path.resolve()
                if not resolved.is_relative_to(root):
                    raise ValueError('Dataset image is outside the dataset directory.')
                items.append((resolved, name))
    return items


class EvaluationWriter:
    def __init__(self, row, run_dir: Path, dataset_root: Path):
        self.row = row
        self.dataset_root = dataset_root
        self.evaluation_id = str(row['payload']['evaluation_id'])
        uuid_name = str(uuid.UUID(self.evaluation_id))
        base = contained_path(run_dir, 'evaluations')
        base.mkdir(parents=True, exist_ok=True)
        self.target = contained_path(base, uuid_name)
        self.stage = contained_path(base, f'.{uuid_name}-{row["dispatch_token"]}')
        self.stage.mkdir(exist_ok=False)
        self.stream = (self.stage / 'images.jsonl').open('wb')
        self.pending = []
        self.total = 0
        with registry._connect() as connection:
            connection.execute('delete from evaluation_images where evaluation_id=%s', (self.evaluation_id,))

    def add(self, source: Path, outcome: str, score, detail: dict, *, fp=0, fn=0):
        encoded = json.dumps(detail, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode('utf-8') + b'\n'
        if len(encoded) > 10 * 1024 * 1024:
            raise ValueError('An image evaluation result exceeds the size limit.')
        offset = self.stream.tell()
        self.stream.write(encoded)
        relative = source.resolve().relative_to(self.dataset_root).as_posix()
        self.pending.append((self.evaluation_id, self.total, source.name[:512], relative, outcome, score,
                             offset, len(encoded), fp, fn))
        self.total += 1
        if len(self.pending) >= 50:
            self.flush()

    def save_mask(self, kind: str, mask):
        from PIL import Image

        if kind not in {'actual', 'predicted'}:
            raise ValueError('Invalid evaluation mask type.')
        directory = self.stage / 'masks'
        directory.mkdir(exist_ok=True)
        name = f'{self.total}-{kind}.png'
        Image.fromarray(mask.astype(np.uint8), mode='L').save(directory / name, format='PNG')
        return name

    def flush(self):
        if not self.pending:
            return
        self.stream.flush()
        with registry._connect() as connection:
            connection.cursor().executemany('''insert into evaluation_images(evaluation_id,ordinal,filename,source_path,
                outcome,score,record_offset,record_length,fp_count,fn_count) values(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)''', self.pending)
            connection.execute('''update compute_jobs set progress_current=greatest(progress_current,%s) where id=%s and dispatch_token=%s
                                  and status='running' ''', (self.total, self.row['id'], self.row['dispatch_token']))
        self.pending.clear()

    def finish(self, report: dict) -> Path:
        self.flush()
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.stream.close()
        (self.stage / 'report.json').write_text(json.dumps(report, ensure_ascii=False, allow_nan=False), encoding='utf-8')
        if self.target.exists():
            raise FileExistsError('Evaluation results already exist.')
        self.stage.replace(self.target)
        return self.target / 'report.json'

    def close(self):
        if not self.stream.closed:
            self.stream.close()
        if self.stage.exists():
            shutil.rmtree(self.stage)


def _record_progress(row, total, current=0):
    with registry._connect() as connection:
        connection.execute('''update compute_jobs set progress_total=%s,progress_current=greatest(progress_current,%s)
                              where id=%s and dispatch_token=%s and status='running' ''',
                           (total, current, row['id'], row['dispatch_token']))


def _evaluate_classification(row, dataset_root, metadata, run_dir, config, writer, device):
    import torch
    import torchvision.transforms as transforms
    from inference_service import IMAGENET_MEAN, IMAGENET_STD, load_classifier, model_forward
    from worker.trainers.input_limits import open_rgb_image_checked

    split = next((dataset_root / name for name in ('test', 'testing') if (dataset_root / name).is_dir()), None)
    if split is None:
        return {'status': 'no_test', 'metrics': None, 'evaluated': 0, 'skipped': 0}
    entry = load_classifier(run_dir, device=device)
    # Classification checkpoints already contain the original class names.
    classes = _classification_names(entry['classes'], {})
    items = _classification_items(dataset_root, metadata, classes)
    if not items:
        return {'status': 'no_test', 'metrics': None, 'evaluated': 0, 'skipped': 0}
    _record_progress(row, len(items))
    transform = transforms.Compose([transforms.Resize((entry['image_size'], entry['image_size'])),
                                    transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    confusion = np.zeros((len(classes), len(classes)), dtype=np.int64)
    skipped = 0
    for path, actual in items:
        try:
            with open_rgb_image_checked(path) as image:
                tensor = transform(image).unsqueeze(0)
        except (OSError, ValueError):
            skipped += 1
            writer.add(path, 'error', None, {'actual': actual, 'error': 'Image could not be evaluated.'})
            continue
        tensor = tensor.to(entry['device'])
        with torch.no_grad():
            probabilities = torch.softmax(model_forward(entry['model'], tensor), dim=1)[0]
        predicted = int(probabilities.argmax().item())
        truth = classes.index(actual)
        confidence = float(probabilities[predicted].item())
        confusion[truth, predicted] += 1
        writer.add(path, 'correct' if truth == predicted else 'incorrect', confidence,
                   {'actual': actual, 'predicted': classes[predicted], 'confidence': confidence})
    if not confusion.sum():
        raise ValueError('No usable test images were found.')
    return {'status': 'partial' if skipped else 'completed', 'metrics': classification_report(confusion, classes),
            'classMapping': {name: index for index, name in enumerate(classes)},
            'evaluated': int(confusion.sum()), 'skipped': skipped}


def _semantic_class_mapping(root: Path, metadata: dict, model_classes: list[str]) -> dict[int, int]:
    schema = metadata.get('label_schema') or {}
    declared = None
    if schema.get('task') == 'semantic_segmentation' and schema.get('classes'):
        declared = {0: 'background', **{index: value for index, value in enumerate(schema['classes'], 1)}}
    else:
        for split in ('test', 'train', 'training'):
            path = root / split / '_classes.csv'
            if not path.is_file():
                continue
            declared = {}
            with path.open(encoding='utf-8-sig', newline='') as stream:
                for row in csv.reader(stream):
                    if len(row) < 2 or not row[0].strip().isdigit():
                        continue
                    class_id, name = int(row[0].strip()), _name(row[1])
                    if class_id in declared or not 0 <= class_id < 255 or not name:
                        raise ValueError('Semantic class mapping contains duplicate or invalid IDs.')
                    declared[class_id] = name
            break
    if not declared or _name(declared.get(0, '')).casefold() != 'background':
        raise ValueError('Semantic masks require an explicit class mapping including background ID 0.')
    names = [_name(value) for value in model_classes]
    if not all(names) or len(set(names)) != len(names) or names[0].casefold() != 'background':
        raise ValueError('The checkpoint has an ambiguous semantic class mapping.')
    normalized = {class_id: _name(name) for class_id, name in declared.items()}
    if len(set(normalized.values())) != len(normalized):
        raise ValueError('Semantic class names are ambiguous.')
    unknown = [name for class_id, name in normalized.items() if class_id and name not in names]
    if unknown:
        raise ValueError('Dataset contains classes that the trained model does not know: ' + ', '.join(unknown[:10]))
    return {class_id: 0 if class_id == 0 else names.index(name) for class_id, name in normalized.items()}


def _semantic_test_items(root: Path):
    from dataset_utils import _matching_mask_path, _roboflow_semantic_pairs, _semantic_split_dirs

    split = root / 'test'
    if not split.is_dir():
        return None
    directories = _semantic_split_dirs(root, 'test')
    if directories:
        images, masks = directories
        return [(path, _matching_mask_path(masks, path)) for path in sorted(images.rglob('*'))
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
    return _roboflow_semantic_pairs(split)


def _semantic_coco_mask(annotations, width, height):
    from pycocotools import mask as coco_mask
    from worker.trainers.input_limits import validate_coco_segmentation

    actual = np.zeros((height, width), dtype=np.uint8)
    ignored = np.zeros((height, width), dtype=bool)
    for annotation in annotations:
        if not 0 < int(annotation['category_id']) < 255:
            raise ValueError('Semantic mask class IDs must be between 1 and 254.')
        segmentation = annotation.get('segmentation')
        if not validate_coco_segmentation(segmentation, width, height):
            raise ValueError('COCO semantic annotations require valid masks.')
        if isinstance(segmentation, list):
            encoded = coco_mask.merge(coco_mask.frPyObjects(segmentation, height, width))
        elif isinstance(segmentation.get('counts'), list):
            encoded = coco_mask.frPyObjects(segmentation, height, width)
        else:
            encoded = segmentation
        binary = coco_mask.decode(encoded).astype(bool)
        if binary.ndim == 3:
            binary = binary.any(axis=2)
        if annotation.get('iscrowd'):
            ignored |= binary
        else:
            actual[binary] = annotation['category_id']
    actual[ignored] = 255
    return actual


def _evaluate_semantic(row, dataset_root, metadata, run_dir, writer, device):
    import torch
    import torchvision.transforms.functional as F
    from PIL import Image
    from torchvision.transforms import InterpolationMode
    from segmentation_inference import load_segmenter
    from worker.trainers.input_limits import open_image_checked, open_rgb_image_checked

    indexed_items = _semantic_test_items(dataset_root)
    coco_items = None
    if not indexed_items:
        from evaluation_detection import _coco_test_data, _test_annotation_file
        if _test_annotation_file(dataset_root):
            # Resolve categories only after loading the checkpoint's recorded names.
            coco_items = True
        elif indexed_items is None:
            return {'status': 'no_test', 'metrics': None, 'evaluated': 0, 'skipped': 0}
        else:
            raise ValueError('The test split does not contain usable semantic images and masks.')
    entry = load_segmenter(run_dir, 'deeplabv3plus', device=device)
    if not entry.get('class_schema_verified'):
        raise ValueError('The checkpoint does not record a complete semantic class mapping. Its classes cannot be matched reliably.')
    classes = [_name(name) for name in entry['classes']]
    if coco_items:
        coco_items, class_mapping = _coco_test_data(dataset_root, classes[1:])
        items = [(path, (info, annotations)) for info, path, annotations in coco_items]
        mapping = None
    else:
        items = indexed_items
        mapping = _semantic_class_mapping(dataset_root, metadata, classes)
        class_mapping = {str(source): classes[destination] for source, destination in mapping.items()}
    if not items:
        return {'status': 'no_test', 'metrics': None, 'evaluated': 0, 'skipped': 0}
    _record_progress(row, len(items))
    confusion = np.zeros((len(classes), len(classes)), dtype=np.int64)
    evaluated = skipped = 0
    for path, mask_source in items:
        try:
            if not path.resolve().is_relative_to(dataset_root):
                raise ValueError('Image is outside the dataset.')
            with open_rgb_image_checked(path) as image:
                size = image.size
                if coco_items:
                    info, annotations = mask_source
                    if (info.get('width') is not None and int(info['width']) != size[0]) or (info.get('height') is not None and int(info['height']) != size[1]):
                        raise ValueError('Image dimensions do not match the COCO annotation.')
                    target = _semantic_coco_mask(annotations, *size)
                else:
                    mask_path = mask_source
                    if not mask_path or not mask_path.resolve().is_relative_to(dataset_root):
                        raise ValueError('Mask is missing or outside the dataset.')
                    with open_image_checked(mask_path) as mask:
                        if image.size != mask.size or mask.mode not in {'L', 'P', 'I', 'I;16'}:
                            raise ValueError('Image and indexed mask are incompatible.')
                        target = np.asarray(mask, dtype=np.int64)
                resized = F.resize(image, [entry['image_size'], entry['image_size']], interpolation=InterpolationMode.BILINEAR)
                tensor = F.to_tensor(resized).unsqueeze(0)
        except (OSError, ValueError):
            skipped += 1
            writer.add(path, 'error', None, {'error': 'Image or mask could not be evaluated.'})
            continue
        tensor = tensor.to(entry['device'])
        with torch.no_grad():
            logits = entry['model'](tensor)
        predicted = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
        predicted = np.asarray(Image.fromarray(predicted, mode='L').resize(size, Image.Resampling.NEAREST))
        if coco_items:
            actual = target
        else:
            unknown = set(np.unique(target).tolist()) - set(mapping) - {255}
            if unknown:
                raise ValueError('A semantic mask contains class IDs missing from its declared mapping.')
            actual = np.full(target.shape, 255, dtype=np.uint8)
            for source, destination in mapping.items():
                actual[target == source] = destination
        valid = actual != 255
        if not valid.any():
            skipped += 1
            writer.add(path, 'error', None, {'error': 'The mask has no pixels eligible for evaluation.'})
            continue
        indices = actual[valid].astype(np.int64) * len(classes) + predicted[valid].astype(np.int64)
        image_confusion = np.bincount(indices, minlength=len(classes) ** 2).reshape(len(classes), len(classes))
        per_image = semantic_report(image_confusion, classes)
        score = per_image['meanIoU']
        actual_file, predicted_file = writer.save_mask('actual', actual), writer.save_mask('predicted', predicted)
        writer.add(path, 'correct' if score == 1 else 'incorrect', score,
                   {'meanIoU': score, 'metrics': per_image, 'actualMask': actual_file, 'predictedMask': predicted_file})
        confusion += image_confusion
        evaluated += 1
    if not evaluated:
        raise ValueError('No usable semantic test images were found.')
    return {'status': 'partial' if skipped else 'completed', 'metrics': semantic_report(confusion, classes),
            'classMapping': class_mapping,
            'evaluated': evaluated, 'skipped': skipped}


def evaluate_job(row: dict, device: str):
    evaluation_id = row['payload']['evaluation_id']
    with registry._connect() as connection:
        evaluation = connection.execute('''select e.*,d.storage_path,d.metadata,d.content_sha256,t.model_type,t.task_type,t.run_slug
            from evaluations e join datasets d on d.id=e.dataset_version_id join training_tasks t on t.id=e.run_id
            where e.id=%s and e.compute_job_id=%s and e.owner_user_id=%s and e.status='running' and d.status='active' ''',
            (evaluation_id, row['id'], row['owner_user_id'])).fetchone()
    if not evaluation:
        raise ValueError('Evaluation or dataset is no longer available.')
    dataset_root = registered_storage_path(DATASET_DIR, evaluation['storage_path'])
    run_dir = contained_path(RUNS_DIR, evaluation['run_slug'])
    config = json.loads((run_dir / 'job_config.json').read_text(encoding='utf-8'))
    source_verified = verify_source(dataset_root, evaluation['dataset_content_sha256'])
    from detection_inference import _yolo_checkpoint_path
    from inference_service import _checkpoint_path
    checkpoint = _yolo_checkpoint_path(run_dir) if evaluation['model_type'] == 'yolo' else _checkpoint_path(run_dir)
    with checkpoint.open('rb') as stream:
        checkpoint_hash = sha256_stream(stream)
    writer = EvaluationWriter(row, run_dir, dataset_root)
    try:
        if evaluation['task_type'] == 'image_classification':
            outcome = _evaluate_classification(row, dataset_root, evaluation['metadata'] or {}, run_dir, config, writer, device)
        elif evaluation['model_type'] == 'deeplabv3plus':
            outcome = _evaluate_semantic(row, dataset_root, evaluation['metadata'] or {}, run_dir, writer, device)
        elif evaluation['model_type'] in {'yolo', 'faster_rcnn', 'mask_rcnn'}:
            from evaluation_detection import evaluate_detection
            outcome = evaluate_detection(row, dataset_root, evaluation['metadata'] or {}, run_dir,
                                         evaluation['model_type'], writer, device, _record_progress)
        else:
            raise ValueError('This model evaluation is not supported yet.')
        with checkpoint.open('rb') as stream:
            if sha256_stream(stream) != checkpoint_hash:
                raise ValueError('Checkpoint changed during evaluation. Run the evaluation again.')
        verify_source(dataset_root, evaluation['dataset_content_sha256'])
        report = {**outcome, 'evaluatorVersion': 1, 'taskType': evaluation['task_type'],
                  'modelType': evaluation['model_type'], 'datasetVersionId': str(evaluation['dataset_version_id']),
                  'datasetContentSha256': evaluation['dataset_content_sha256'], 'sourceVerified': source_verified,
                  'checkpointSha256': checkpoint_hash}
        result_path = writer.finish(report)
        with registry._connect() as connection:
            connection.execute("update evaluations set checkpoint_sha256=%s where id=%s and compute_job_id=%s and status='running'",
                               (checkpoint_hash, evaluation_id, row['id']))
        return str(result_path), outcome['status']
    finally:
        writer.close()
