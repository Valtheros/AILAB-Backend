"""Run a bounded batch in the compute worker and persist per-image results."""
from __future__ import annotations

import base64
import csv
import io
import json
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw

from compute_repository import job_directory, registry
from security_utils import contained_path
from settings import RUNS_DIR


def _csv_text(value):
    text = str(value if value is not None else '')
    return "'" + text if text.lstrip().startswith(('=', '+', '-', '@')) else text


def _write_progress(row, current: int, total: int) -> None:
    with registry._connect() as connection:
        connection.execute('''update compute_jobs set progress_current=%s,progress_total=%s
            where id=%s and dispatch_token=%s and status='running' ''',
            (current, total, row['id'], row['dispatch_token']))


def _detection_overlay(image_path: Path, predictions: list[dict], output: Path) -> None:
    from inference_service import _open_verified_image
    image = _open_verified_image(image_path.read_bytes())
    draw = ImageDraw.Draw(image)
    for prediction in predictions:
        box = prediction.get('box')
        if isinstance(box, list) and len(box) == 4:
            draw.rectangle(box, outline=(0, 220, 115), width=max(1, min(image.size) // 250))
    image.save(output, format='PNG')


def execute_batch(row: dict, device: str) -> None:
    directory = job_directory(row['id'])
    run_dir = contained_path(RUNS_DIR, row['payload']['project_name'])
    entries = json.loads((directory / 'input-manifest.json').read_text(encoding='utf-8'))['images']
    results, succeeded = [], 0
    _write_progress(row, 0, len(entries))
    for index, entry in enumerate(entries):
        item = {'filename': entry['name'], 'status': 'error'}
        from inference_service import InferenceError, _open_verified_image

        try:
            if entry.get('error'):
                raise InferenceError(entry['error'])
            image_path = contained_path(directory, 'inputs', str(index))
            payload = image_path.read_bytes()
            with _open_verified_image(payload):
                pass
        except InferenceError as exc:
            item['error'] = str(exc)[:300]
            results.append(item)
            _write_progress(row, index + 1, len(entries))
            continue
        task, model = row['payload']['task_type'], row['payload']['model_type']
        threshold = row['payload'].get('threshold')
        if task == 'image_classification':
            from inference_service import predict_image
            prediction = predict_image(run_dir, payload, device=device)
        elif task == 'object_detection':
            from detection_inference import predict_detection
            prediction = predict_detection(run_dir, payload, model, threshold, device=device)
            overlay = directory / 'overlays' / f'{index:03d}.png'
            overlay.parent.mkdir(parents=True, exist_ok=True)
            _detection_overlay(image_path, prediction.get('predictions', []), overlay)
            prediction['overlayFile'] = overlay.relative_to(directory).as_posix()
        elif task == 'segmentation':
            from segmentation_inference import predict_segmentation

            def save_mask(mask_id, image):
                mask = directory / 'masks' / f'{index:03d}-{mask_id}.png'
                mask.parent.mkdir(parents=True, exist_ok=True)
                image.save(mask, format='PNG')
                return mask.relative_to(directory).as_posix()

            prediction = predict_segmentation(run_dir, payload, model, threshold, device=device, mask_sink=save_mask)
            segmentation = prediction.get('segmentation', {})
            encoded = segmentation.pop('overlay', '')
            if encoded.startswith('data:image/png;base64,'):
                overlay_image = Image.open(io.BytesIO(base64.b64decode(encoded.split(',', 1)[1], validate=True))).convert('RGBA')
                with Image.open(image_path) as original:
                    image = Image.alpha_composite(original.convert('RGBA'), overlay_image).convert('RGB')
                overlay = directory / 'overlays' / f'{index:03d}.png'
                overlay.parent.mkdir(parents=True, exist_ok=True)
                image.save(overlay, format='PNG')
                segmentation['overlayFile'] = overlay.relative_to(directory).as_posix()
        else:
            raise ValueError('Unsupported model task.')
        item.update(status='completed', result=prediction)
        succeeded += 1
        results.append(item)
        _write_progress(row, index + 1, len(entries))

    report = {'status': 'completed' if succeeded else 'failed', 'total': len(entries),
              'succeeded': succeeded, 'failed': len(entries) - succeeded, 'images': results}
    temporary = directory / 'result.json.tmp'
    temporary.write_text(json.dumps(report, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temporary.replace(directory / 'result.json')
    with (directory / 'results.csv').open('w', newline='', encoding='utf-8-sig') as output:
        writer = csv.writer(output)
        writer.writerow(['filename', 'status', 'top_class', 'confidence', 'object_count', 'error'])
        for item in results:
            result = item.get('result') or {}
            top = result.get('top') or {}
            writer.writerow([_csv_text(item['filename']), item['status'], _csv_text(top.get('className')),
                             top.get('confidence', ''), result.get('count', ''), _csv_text(item.get('error'))])
    with zipfile.ZipFile(directory / 'results.zip', 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name in ('result.json', 'results.csv'):
            archive.write(directory / name, name)
        for folder in ('overlays', 'masks'):
            path = directory / folder
            if path.is_dir():
                for item in sorted(path.iterdir()):
                    if item.is_file():
                        archive.write(item, item.relative_to(directory).as_posix())
    if not succeeded:
        raise ValueError('No usable images were found in this batch.')
