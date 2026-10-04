"""Real CUDA smoke, disposable data only. Run after the isolated integration suite."""
import json
import os
import time
from pathlib import Path

from PIL import Image, ImageDraw
import yaml

import compute_repository as compute
from dataset_storage import owner_dataset_path
from dataset_utils import inspect_dataset
from services.training_service import TrainingService
from settings import DATASET_DIR, RUNS_DIR


def wait(job_id, timeout=600):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        row = compute.get_job('alice', job_id)
        if row['status'] in compute.TERMINAL:
            if row['status'] != 'completed':
                raise RuntimeError(json.dumps(row, default=str))
            return row
        time.sleep(2)
    raise TimeoutError(compute.get_job('alice', job_id))


def fixtures():
    paths = {kind: owner_dataset_path(DATASET_DIR, 'alice', f'smoke-{kind}') for kind in ('classification','boxes','instances','semantic')}
    for split in ('train','val'):
        coco = {'images': [], 'annotations': [], 'categories': [{'id':1,'name':'square'}]}
        for i in range(4):
            image = Image.new('RGB', (128,128), '#446688')
            ImageDraw.Draw(image).rectangle((32,32,96,96), fill='white' if i%2 else 'black')
            class_path = paths['classification'] / split / ('light' if i%2 else 'dark') / f'{i}.png'
            class_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(class_path)
            for kind in ('boxes','instances','semantic'):
                path = paths[kind] / split / 'images' / f'{i}.png'
                path.parent.mkdir(parents=True, exist_ok=True)
                image.save(path)
            label = paths['boxes'] / split / 'labels' / f'{i}.txt'
            label.parent.mkdir(parents=True, exist_ok=True)
            label.write_text('0 0.5 0.5 0.5 0.5\n')
            mask = Image.new('L', (128,128), 0)
            ImageDraw.Draw(mask).rectangle((32,32,96,96), fill=1)
            mask_path = paths['semantic'] / split / 'masks' / f'{i}.png'
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            mask.save(mask_path)
            coco['images'].append({'id':i,'file_name':f'images/{i}.png','height':128,'width':128})
            coco['annotations'].append({'id':i,'image_id':i,'category_id':1,'bbox':[32,32,64,64],
                   'area':4096,'iscrowd':0,'segmentation':[[32,32,96,32,96,96,32,96]]})
        (paths['instances'] / split / '_annotations.coco.json').write_text(json.dumps(coco))
    (paths['boxes'] / 'data.yaml').write_text(yaml.safe_dump({'path': str(paths['boxes']), 'train':'train/images','val':'val/images','names':{0:'square'}}))
    for path in paths.values():
        metadata = inspect_dataset(path)
        compute.registry.upsert_dataset('alice', 'alice@test.invalid', path.name, path, metadata)
    return paths


def main():
    if os.getenv('COMPUTE_TESTS') != '1' or 'ailab_compute_test_pg' not in compute.registry.database_url:
        raise RuntimeError('This smoke must not use production storage or database')
    paths = fixtures()
    service = TrainingService()
    models = [
      ('resnet','resnet18','classification',{'architecture':'resnet18','image_size':64,'pretrained':False}),
      ('efficientnet','efficientnet_b0','classification',{'architecture':'efficientnet_b0','image_size':64,'pretrained':False}),
      ('yolo','yolo11n','boxes',{'model_size':'n','imgsz':128,'amp':False,'plots':False}),
      ('faster_rcnn','fasterrcnn_resnet50_fpn_v2','boxes',{'image_size':128,'max_size':128,'pretrained':False}),
      ('deeplabv3plus','deeplabv3plus','semantic',{'image_size':64,'num_classes':2,'encoder_name':'resnet18','encoder_weights':'none','decoder_channels':64}),
      ('mask_rcnn','maskrcnn_resnet50_fpn_v2','instances',{'image_size':128,'max_size':128,'pretrained':False}),
    ]
    for model, architecture, kind, args in models:
        task = 'image_classification' if kind == 'classification' else 'object_detection' if kind == 'boxes' else 'segmentation'
        slug = f'smoke_{model}_{int(time.time())}'
        job_id = service.start_training_container(task, model, architecture, 1, 2, slug,
                     dataset_name=paths[kind].name, extra_args={**args,'workers':0,'amp':False},
                     owner_id='alice', execution={'mode':'auto'}, resource_plan={'estimated_vram_mb':4000})
        train = wait(job_id)
        assert train['assignedGpu'], train
        print(f'CUDA TRAIN OK: {model} {train["assignedGpu"]}', flush=True)
        record = compute.training_record(job_id)
        prediction_id = __import__('uuid').uuid4()
        directory = compute.job_directory(prediction_id)
        directory.mkdir(parents=True, exist_ok=True)
        image = next((paths[kind] / 'val').rglob('*.png'))
        (directory / 'input').write_bytes(image.read_bytes())
        compute.create_job('alice', record['id'], 'predict', {'project_name':slug,'task_type':task,'model_type':model},
                           {'mode':'auto'}, job_id=prediction_id, estimated_vram_mb=4000)
        wait(prediction_id)
        result = compute.read_result('alice', prediction_id)
        assert result['model']['device'] == 'cuda:0', result['model']
        if kind in ('semantic','instances'):
            assert result['segmentation']['overlay'].endswith('/overlay'), result.keys()
            assert (directory / 'overlay.png').is_file()
        print(f'CUDA PREDICT OK: {model} {result["timingMs"]}', flush=True)
    print('All six trainer and prediction paths passed on the real CUDA GPU.', flush=True)


if __name__ == '__main__': main()
