# Computer Vision Training Backend

FastAPI backend and Redis/RQ workers for the no-code Computer Vision training platform.

The backend owns:

- Dataset upload, extraction, format detection, and compatibility checks.
- A model catalog consumed by the frontend.
- Training job submission and queue routing.
- Job status, logs, metrics, and result artifact downloads.
- Docker services for the API, Redis, CV workers, OCR workers, and RQ dashboard.

## Supported tasks and models

| Task | Models | Primary dataset formats |
| --- | --- | --- |
| Image Classification | ResNet, EfficientNet | ImageFolder |
| Semantic / Instance Segmentation | DeepLabV3+, Mask R-CNN | Semantic masks, COCO instances |
| OCR / Document Vision | PaddleOCR, Tesseract | PaddleOCR labels, Tesseract ground truth |
| Object Detection | YOLOv11, Faster R-CNN | YOLO detection |

`model_catalog.py` is the backend source of truth for task IDs, model IDs, selectable parameters, and expected dataset formats.

## Architecture

```text
backend/
|-- main.py                    # FastAPI routes
|-- model_catalog.py           # Task/model/parameter catalog
|-- dataset_utils.py           # Dataset inspection and validation
|-- services/
|   `-- training_service.py    # RQ job orchestration and run metadata
|-- worker/
|   |-- worker_app.py          # RQ worker entry point and trainer registry
|   |-- Dockerfile             # PyTorch/CUDA CV worker
|   |-- Dockerfile.ocr         # PaddleOCR and Tesseract worker
|   |-- requirements.txt
|   |-- requirements-ocr.txt
|   `-- trainers/
|       |-- yolo_trainer.py
|       |-- resnet_trainer.py
|       |-- efficientnet_trainer.py
|       |-- deeplabv3plus_trainer.py
|       |-- mask_rcnn_trainer.py
|       |-- faster_rcnn_trainer.py
|       |-- paddleocr_trainer.py
|       `-- tesseract_trainer.py
|-- dataset/                    # Extracted uploaded datasets
`-- runs/                       # Logs, weights, CSV metrics, job config files
```

Training is queue-based:

1. FastAPI validates the request and selected dataset.
2. `TrainingService` enqueues the job into Redis.
3. `cv_training` handles PyTorch/TorchVision/Ultralytics/SMP trainers.
4. `ocr_training` handles PaddleOCR and Tesseract trainers.
5. Workers write artifacts into `runs/<project_name>/`.

## Run with Docker

From `backend/`:

```powershell
docker compose up --build
```

Docker Compose starts:

| Service | Purpose | Port |
| --- | --- | --- |
| `backend` | FastAPI API | `8000` |
| `redis` | Queue and job store | `6379` |
| `worker` | CV training queue | none |
| `ocr-worker` | OCR training queue | none |
| `rq-dashboard` | Queue dashboard | `9181` |

Useful URLs:

- API root: `http://localhost:8000`
- Swagger UI: `http://localhost:8000/docs`
- Model catalog: `http://localhost:8000/api/model-catalog`
- RQ dashboard: `http://localhost:9181`

The CV worker is configured for NVIDIA GPU reservations in `docker-compose.yml`. CPU-only environments can still use the API and may run compatible training jobs after adjusting Docker/GPU configuration and model parameters.

## Local API development

Docker is the intended path because Redis and worker images are part of the runtime. For API-only local work:

```powershell
python -m pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Redis must be reachable through `REDIS_URL` for job submission.

Default local value:

```text
redis://localhost:6379
```

Compose value:

```text
redis://redis:6379
```

## API contract

### Start training

`POST /api/train`

```json
{
  "task_type": "object_detection",
  "model_type": "yolo",
  "model_name": "yolo11n",
  "dataset_name": "traffic_signs",
  "project_name": "traffic_signs_run_001",
  "epochs": 50,
  "batch_size": 16,
  "params": {
    "device": "0",
    "workers": 4,
    "amp": true,
    "imgsz": 640,
    "optimizer": "auto"
  }
}
```

Response:

```json
{
  "status": "success",
  "job_id": "rq-job-id",
  "container_id": "rq-job-id"
}
```

`container_id` is kept as a compatibility alias for older frontend code. New code should treat it as an RQ job ID.

### Core routes

| Route | Purpose |
| --- | --- |
| `GET /api/model-catalog` | Return tasks, models, and parameter specs |
| `POST /api/train` | Enqueue a training job |
| `GET /api/status/{job_id}` | Read normalized job status |
| `GET /api/logs/{job_id}` | Read worker training logs |
| `GET /api/metrics/{project_name}` | Read `results.csv` rows |
| `POST /api/stop/{job_id}` | Cancel a queued or running job |
| `GET /api/datasets` | List uploaded datasets and detected formats |
| `POST /api/upload-dataset` | Upload a dataset ZIP |
| `DELETE /api/datasets/{dataset_name}` | Delete a dataset directory |
| `GET /api/datasets/{dataset_name}/metadata` | Inspect one dataset |
| `GET /api/runs` | List run folders, artifacts, and latest metrics |
| `GET /api/runs/{project_name}/files/{file_path}` | Download a run artifact |

## Dataset formats

Uploaded ZIP files are extracted into `dataset/<dataset_name>/`. The backend detects these formats:

| Format ID | Expected shape | Typical model use |
| --- | --- | --- |
| `imagefolder` | `train/<class_name>/image.jpg` | ResNet, EfficientNet |
| `yolo_detection` | `data.yaml`, image folders, YOLO box labels | YOLO, Faster R-CNN |
| `yolo_segmentation` | YOLO polygon labels | YOLO segmentation compatibility |
| `semantic_masks` | `train/images`, `train/masks` | DeepLabV3+ |
| `coco_instances` | COCO JSON annotations and images | Mask R-CNN |
| `paddleocr_labels` | OCR label text files | PaddleOCR |
| `tesseract_ground_truth` | `*.gt.txt` files | Tesseract |

The selected model must accept at least one detected dataset format before the job is queued.

## Result files

Workers write under:

```text
runs/<project_name>/
```

Depending on the trainer, a run can include:

- `job_config.json`
- `train.log`
- `results.csv`
- `best.pt`
- `last.pt`
- OCR-specific exported files such as `.traineddata`

## Adding a model

1. Add the model entry and parameter specs to `model_catalog.py`.
2. Add a dedicated trainer file under `worker/trainers/`.
3. Register the trainer in `worker/worker_app.py`.
4. Declare dependencies in the correct worker requirements file.
5. Set compatible `dataset_formats` so dataset validation remains explicit.

Keep model-specific logic in its own trainer file. Shared data loading or optimization helpers belong in focused helper modules, not in one oversized base trainer.

## Verification

Syntax check used during development:

```powershell
python -m py_compile main.py model_catalog.py dataset_utils.py services\training_service.py worker\worker_app.py worker\trainers\*.py
```

For runtime verification, rebuild and start the Compose stack before submitting training jobs:

```powershell
docker compose up --build
```
