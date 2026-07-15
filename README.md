# AILAB Backend

FastAPI backend for AILAB, a no-code Computer Vision training platform. It
accepts uploaded datasets, detects supported formats, queues training jobs with
Redis/RQ, streams progress, and serves trained artifacts back to the frontend.

## What It Does

- Upload and validate dataset ZIP files.
- Expose the model catalog used by the UI.
- Queue training jobs for the Computer Vision worker.
- Stream live logs, metrics, status, and heartbeats through SSE.
- Store run outputs such as logs, metrics, weights, and reports.
- Protect user-owned datasets and runs when requests include authenticated user
  headers from the frontend proxy.

## Supported Tasks

| Task | Models | Dataset formats |
| --- | --- | --- |
| Image Classification | ResNet, EfficientNet | ImageFolder |
| Segmentation | DeepLabV3+, Mask R-CNN | semantic masks, COCO instances |
| Object Detection | YOLOv11, Faster R-CNN | YOLO detection |

## Run Locally

Docker Compose is the recommended local runtime because the API depends on
Redis, PostgreSQL, and worker services.

```powershell
docker compose up --build
```

Useful URLs:

- API root: `http://localhost:8000`
- Swagger UI: `http://localhost:8000/docs`
- Model catalog: `http://localhost:8000/api/model-catalog`
- RQ dashboard: `http://localhost:9181`

## Environment

Copy `.env.example` and adjust values when needed.

```text
APP_BASE_DIR=/app
STORAGE_DIR=/app/storage
DATASET_DIR=/app/dataset
RUNS_DIR=/app/runs
REDIS_URL=redis://redis:6379
DATABASE_URL=postgresql://ailab:ailab_dev_password@postgres:5432/ailab
BACKEND_INTERNAL_TOKEN=
CORS_ORIGINS=http://localhost:3000
```

Set the same `BACKEND_INTERNAL_TOKEN` in the frontend and backend for
production-like deployments so browser traffic goes through the authenticated
Next.js proxy instead of calling FastAPI directly.

## Important Routes

| Route | Purpose |
| --- | --- |
| `GET /api/model-catalog` | Tasks, models, and parameter specs |
| `POST /api/upload-dataset` | Upload a dataset ZIP |
| `GET /api/datasets` | List visible datasets from the PostgreSQL resource registry |
| `POST /api/datasets/inspect-upload` | Upload, validate, and stage a ZIP once |
| `POST /api/datasets/import` | Commit a staged upload token |
| `POST /api/train` | Queue a training job |
| `GET /api/jobs/{job_id}/events` | Live SSE updates |
| `GET /api/status/{job_id}` | Job snapshot |
| `GET /api/logs/{job_id}` | Training logs |
| `GET /api/metrics/{project_name}` | Metrics rows |
| `GET /api/runs` | Run artifacts |

## Verification

```powershell
python -m unittest discover -s tests -v
python -m py_compile main.py security_utils.py sse_utils.py model_catalog.py dataset_utils.py services\training_service.py worker\worker_app.py worker\security_utils.py worker\trainers\*.py
```

Keep Redis internal to Docker Compose and bind local-only services to
`127.0.0.1` unless the stack is behind proper authentication and network
controls.
